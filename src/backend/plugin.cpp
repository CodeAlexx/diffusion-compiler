#include "dif/backend/plugin.hpp"

#include "dif/backend/abi.h"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dif::backend {
namespace {

class Library {
public:
  explicit Library(const std::filesystem::path &path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      fail("cannot load backend plugin: " + std::string(dlerror()));
  }
  ~Library() {
    if (handle_)
      (void)dlclose(handle_);
  }
  Library(const Library &) = delete;
  Library &operator=(const Library &) = delete;

  void *symbol(const char *name) const {
    dlerror();
    void *value = dlsym(handle_, name);
    if (const char *error = dlerror())
      fail("backend plugin symbol lookup failed: " + std::string(error));
    return value;
  }

  void *optional_symbol(const char *name) const {
    dlerror();
    void *value = dlsym(handle_, name);
    (void)dlerror();
    return value;
  }

private:
  void *handle_{};
};

std::string plugin_error(dif_backend_status status,
                         const std::array<char, 1024> &buffer) {
  const auto message = buffer[0] == '\0' ? "no plugin error text" : buffer.data();
  return "backend plugin status " + std::to_string(static_cast<int>(status)) +
         ": " + message;
}

class PluginRuntime {
public:
  PluginRuntime(const std::filesystem::path &path, int device) : library_(path) {
    if (const auto symbol = library_.optional_symbol("dif_backend_get_v2")) {
      const auto getter = reinterpret_cast<dif_backend_get_v2_fn>(symbol);
      api_v2_ = getter();
      constexpr auto minimum_size = offsetof(dif_backend_api_v2, execute) +
                                    sizeof(dif_backend_api_v2::execute);
      if (!api_v2_ || api_v2_->abi_version != DIF_BACKEND_ABI_VERSION_V2 ||
          api_v2_->struct_size < minimum_size || !api_v2_->backend_name ||
          !api_v2_->create || !api_v2_->destroy || !api_v2_->compile ||
          !api_v2_->destroy_executable || !api_v2_->execute)
        fail("backend plugin does not implement the complete v2 ABI");
    } else {
      const auto getter = reinterpret_cast<dif_backend_get_v1_fn>(
          library_.symbol("dif_backend_get_v1"));
      api_v1_ = getter();
      constexpr auto minimum_size = offsetof(dif_backend_api_v1, execute) +
                                    sizeof(dif_backend_api_v1::execute);
      if (!api_v1_ ||
          api_v1_->abi_version != DIF_BACKEND_ABI_VERSION_V1 ||
          api_v1_->struct_size < minimum_size || !api_v1_->backend_name ||
          !api_v1_->create || !api_v1_->destroy || !api_v1_->compile ||
          !api_v1_->destroy_executable || !api_v1_->execute)
        fail("backend plugin does not implement the complete v1 ABI");
    }
    std::array<char, 1024> buffer{};
    dif_backend_error error{buffer.data(), buffer.size()};
    const auto status = api_v2_ ? api_v2_->create(device, &context_, &error)
                                : api_v1_->create(device, &context_, &error);
    if (status != DIF_BACKEND_OK)
      fail(plugin_error(status, buffer));
    if (!context_)
      fail("backend plugin create returned a null context");
  }

  ~PluginRuntime() {
    if (context_)
      api_v2_ ? api_v2_->destroy(context_) : api_v1_->destroy(context_);
  }

  PluginRuntime(const PluginRuntime &) = delete;
  PluginRuntime &operator=(const PluginRuntime &) = delete;

  bool is_v2() const { return api_v2_ != nullptr; }
  const dif_backend_api_v1 *api_v1() const { return api_v1_; }
  const dif_backend_api_v2 *api_v2() const { return api_v2_; }
  const char *name() const {
    return api_v2_ ? api_v2_->backend_name : api_v1_->backend_name;
  }
  dif_backend_context context() const { return context_; }

private:
  Library library_;
  const dif_backend_api_v1 *api_v1_{};
  const dif_backend_api_v2 *api_v2_{};
  dif_backend_context context_{};
};

class PluginPreparedExecution final : public runtime::PreparedExecution {
public:
  PluginPreparedExecution(std::shared_ptr<PluginRuntime> runtime,
                          ir::Program program,
                          const runtime::TensorMap &bindings,
                          const runtime::RunOptions &options)
      : runtime_(std::move(runtime)), program_(std::move(program)) {
    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Constant))
        continue;
      const auto found = bindings.find(desc.id);
      if (found == bindings.end())
        fail("missing plugin constant tensor " + std::to_string(desc.id));
      found->second.validate();
      if (found->second.dtype != desc.dtype || found->second.dims != desc.dims)
        fail("plugin constant tensor shape/dtype mismatch");
      constants_.emplace(desc.id, found->second);
    }
    const auto encoded = ir::encode(program_);
    std::array<char, 1024> buffer{};
    dif_backend_error error{buffer.data(), buffer.size()};
    std::vector<dif_backend_tensor_view> constant_views;
    constant_views.reserve(constants_.size());
    for (const auto &description : program_.tensors) {
      if (!description.has_role(ir::TensorRole::Constant))
        continue;
      const auto &tensor = constants_.at(description.id);
      constant_views.push_back(
          {description.id, static_cast<std::uint32_t>(description.dtype),
           static_cast<std::uint32_t>(description.dims.size()),
           description.dims.data(), const_cast<std::uint8_t *>(tensor.data()),
           tensor.byte_size()});
    }
    const auto start = std::chrono::steady_clock::now();
    dif_backend_status status{};
    if (runtime_->is_v2()) {
      dif_backend_compile_options_v2 compile_options{
          options.minimum_free_bytes, DIF_BACKEND_COMPILE_COPY_CONSTANTS};
      dif_backend_compile_telemetry_v2 telemetry{};
      status = runtime_->api_v2()->compile(
          runtime_->context(), encoded.data(), encoded.size(),
          constant_views.data(), constant_views.size(), &compile_options,
          &executable_, &telemetry, &error);
      preparation_milliseconds_ = telemetry.preparation_milliseconds;
      resident_bytes_ = telemetry.resident_bytes;
      if (telemetry.device_name[0] != '\0')
        device_name_ = telemetry.device_name;
    } else {
      status = runtime_->api_v1()->compile(runtime_->context(), encoded.data(),
                                           encoded.size(), &executable_, &error);
    }
    if (status != DIF_BACKEND_OK)
      fail(plugin_error(status, buffer));
    if (!executable_)
      fail("backend plugin compile returned a null executable");
    const auto stop = std::chrono::steady_clock::now();
    if (preparation_milliseconds_ == 0.0)
      preparation_milliseconds_ =
          std::chrono::duration<double, std::milli>(stop - start).count();
    if (runtime_->is_v2()) {
      for (const auto &[id, tensor] : constants_) {
        (void)id;
        tensor.discard_mapped_pages();
      }
    }
  }

  ~PluginPreparedExecution() override {
    if (executable_)
      runtime_->is_v2()
          ? runtime_->api_v2()->destroy_executable(executable_)
          : runtime_->api_v1()->destroy_executable(executable_);
  }

  PluginPreparedExecution(const PluginPreparedExecution &) = delete;
  PluginPreparedExecution &operator=(const PluginPreparedExecution &) = delete;

  runtime::RunResult run(const ir::Program &program,
                         const runtime::TensorMap &inputs,
                         const runtime::RunOptions &options) = delete;

  runtime::RunResult run(const runtime::TensorMap &inputs,
                         const runtime::RunOptions &options) override {
    runtime::TensorMap bindings = constants_;
    for (const auto &[id, tensor] : inputs) {
      const auto *desc = program_.tensor(id);
      if (desc && desc->has_role(ir::TensorRole::Input))
        bindings.insert_or_assign(id, tensor);
    }
    std::array<char, 1024> buffer{};
    dif_backend_error error{buffer.data(), buffer.size()};
    runtime::RunResult result;
    std::vector<dif_backend_tensor_view> input_views;
    std::vector<dif_backend_tensor_view> output_views;
    for (const auto &desc : program_.tensors) {
      if (desc.has_role(ir::TensorRole::Input) ||
          (!runtime_->is_v2() &&
           desc.has_role(ir::TensorRole::Constant))) {
        const auto found = bindings.find(desc.id);
        if (found == bindings.end())
          fail("missing plugin input tensor " + std::to_string(desc.id));
        found->second.validate();
        if (found->second.dtype != desc.dtype || found->second.dims != desc.dims)
          fail("plugin input tensor shape/dtype mismatch");
        input_views.push_back(
            {desc.id, static_cast<std::uint32_t>(desc.dtype),
             static_cast<std::uint32_t>(desc.dims.size()), desc.dims.data(),
             const_cast<std::uint8_t *>(found->second.data()),
             found->second.byte_size()});
      }
      if (desc.has_role(ir::TensorRole::Output)) {
        auto [found, inserted] = result.outputs.emplace(desc.id, runtime::zeros(desc));
        (void)inserted;
        output_views.push_back(
            {desc.id, static_cast<std::uint32_t>(desc.dtype),
             static_cast<std::uint32_t>(desc.dims.size()), desc.dims.data(),
             found->second.mutable_data(), found->second.byte_size()});
      }
    }
    dif_backend_run_options run_options{options.warmups, options.iterations,
                                        options.minimum_free_bytes};
    dif_backend_telemetry telemetry{};
    buffer.fill('\0');
    const auto status = runtime_->is_v2()
                            ? runtime_->api_v2()->execute(
                                  executable_, input_views.data(),
                                  input_views.size(), output_views.data(),
                                  output_views.size(), &run_options, &telemetry,
                                  &error)
                            : runtime_->api_v1()->execute(
                                  executable_, input_views.data(),
                                  input_views.size(), output_views.data(),
                                  output_views.size(), &run_options, &telemetry,
                                  &error);
    if (status != DIF_BACKEND_OK)
      fail(plugin_error(status, buffer));
    result.mean_milliseconds = telemetry.mean_milliseconds;
    result.minimum_milliseconds = telemetry.minimum_milliseconds;
    result.maximum_milliseconds = telemetry.maximum_milliseconds;
    result.free_bytes_before = telemetry.free_bytes_before;
    result.free_bytes_after = telemetry.free_bytes_after;
    result.device_name = telemetry.device_name[0] != '\0'
                             ? telemetry.device_name
                             : device_name_;
    result.backend_name = runtime_->name();
    result.preparation_milliseconds = preparation_milliseconds_;
    result.resident_bytes = resident_bytes_;
    return result;
  }

  std::string name() const override { return runtime_->name(); }
  double preparation_milliseconds() const override {
    return preparation_milliseconds_;
  }
  std::uint64_t resident_bytes() const override { return resident_bytes_; }

private:
  std::shared_ptr<PluginRuntime> runtime_;
  ir::Program program_;
  runtime::TensorMap constants_;
  dif_backend_executable executable_{};
  double preparation_milliseconds_{};
  std::uint64_t resident_bytes_{};
  std::string device_name_;
};

class PluginExecutor final : public runtime::Executor {
public:
  PluginExecutor(const std::filesystem::path &path, int device)
      : runtime_(std::make_shared<PluginRuntime>(path, device)) {}

  std::unique_ptr<runtime::PreparedExecution>
  prepare(const ir::Program &program, const runtime::TensorMap &bindings,
          const runtime::RunOptions &options) override {
    ir::verify(program);
    return std::make_unique<PluginPreparedExecution>(runtime_, program, bindings,
                                                     options);
  }

  std::string name() const override { return runtime_->name(); }

private:
  std::shared_ptr<PluginRuntime> runtime_;
};

} // namespace

std::unique_ptr<runtime::Executor>
make_plugin_executor(const std::filesystem::path &library, int device) {
  return std::make_unique<PluginExecutor>(library, device);
}

} // namespace dif::backend
