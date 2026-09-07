#pragma once

#include "dif/runtime/residual_cache.hpp"
#include "dif/support/error.hpp"

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <string>

namespace dif::runtime {

struct ResidualCacheKernelHandles {
  CUfunction gather{}, quantize{}, store{}, apply{};
};
// The executor supplies its existing counted submissions, preserving telemetry.
struct ResidualCacheCudaDispatch {
  CUresult (*launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                    unsigned, unsigned, CUstream, void **, void **){};
  CUresult (*htod)(CUdeviceptr, const void *, std::size_t, CUstream){};
  CUresult (*dtoh)(void *, CUdeviceptr, std::size_t, CUstream){};
  CUresult (*synchronize)(CUstream){};
};

// Nonowning device helper: no context, allocation, module or compilation path.
// Its entire device state is a sealed slice of the shared runtime's arena.
class CudaResidualCache {
 public:
  CudaResidualCache(const ResidualCacheRegion &region, CUdeviceptr workspace,
                    CUstream stream, ResidualCacheKernelHandles kernels,
                    ResidualCacheCudaDispatch dispatch,
                    std::shared_ptr<ResidualCacheCallbacks> callbacks)
      : region_(region), layout_(residual_cache_device_layout(region)),
        workspace_(workspace), stream_(stream), kernels_(kernels), dispatch_(dispatch),
        callbacks_(std::move(callbacks)) {
    if (!workspace || !kernels.gather || !kernels.quantize || !kernels.store || !kernels.apply ||
        !dispatch.launch || !dispatch.htod || !dispatch.dtoh || !dispatch.synchronize ||
        !callbacks_ || !callbacks_->decide || !callbacks_->refreshed)
      fail("residual-cache device helper requires the shared arena/module/submission hooks");
    indices_ = region.probe_rows;
    indices_.insert(indices_.end(), region.audio_probe_rows.begin(), region.audio_probe_rows.end());
    before_.resize(layout_.main_probe_elements + layout_.audio_probe_elements);
    after_.resize(before_.size());
    checked(dispatch_.htod(pointer(layout_.indices), indices_.data(), indices_.size() * 4, stream_),
            "upload probe indices");
  }
  const ResidualCacheRegion &region() const { return region_; }
  std::uint64_t required_bytes() const { return layout_.required_bytes; }
  bool residual_ready() const { return residual_ready_; }

  void before_front(CUdeviceptr hidden) {
    action_ = ResidualCacheAction::Exact;
    need_probes_ = !callbacks_->needs_probes || callbacks_->needs_probes();
    if (need_probes_) gather(hidden, pointer(layout_.before_probe));
  }
  bool before_middle(CUdeviceptr front, CUdeviceptr reconstructed_middle) {
    if (need_probes_) {
      gather(front, pointer(layout_.after_probe));
      checked(dispatch_.dtoh(before_.data(), pointer(layout_.before_probe), before_.size() * 4, stream_),
              "read before-front probe");
      checked(dispatch_.dtoh(after_.data(), pointer(layout_.after_probe), after_.size() * 4, stream_),
              "read after-front probe");
      checked(dispatch_.synchronize(stream_), "synchronize small probes");
      const auto main = layout_.main_probe_elements;
      action_ = callbacks_->decide(
          std::span<const float>(before_).first(main), std::span<const float>(after_).first(main),
          std::span<const float>(before_).subspan(main), std::span<const float>(after_).subspan(main));
    } else action_ = callbacks_->decide({}, {}, {}, {});
    if (action_ == ResidualCacheAction::Exact) {
      residual_ready_ = false;
      return false;
    }
    if (!need_probes_) fail("residual cache cannot refresh/reuse without current probes");
    if (action_ == ResidualCacheAction::Refresh) {
      residual_ready_ = false;
      auto codes = pointer(layout_.front_codes), scales = pointer(layout_.front_scales);
      auto groups = layout_.groups;
      void *args[]{&front, &codes, &scales, &groups};
      launch(kernels_.quantize, groups, args, "quantize front snapshot");
      return false;
    }
    if (action_ != ResidualCacheAction::Reuse || !residual_ready_)
      fail("residual-cache reuse requested without a completed device residual");
    // Exact in-place and disjoint buffers are safe. A partially overlapping
    // memory-plan reuse would otherwise race between elementwise GPU lanes.
    const auto bytes = layout_.elements * 2;
    if (front != reconstructed_middle &&
        (front < reconstructed_middle ? reconstructed_middle - front < bytes :
                                        front - reconstructed_middle < bytes))
      fail("residual-cache reconstruction buffers partially overlap");
    auto codes = pointer(layout_.residual_codes), scales = pointer(layout_.residual_scales);
    auto elements = layout_.elements;
    void *args[]{&front, &codes, &scales, &reconstructed_middle, &elements};
    launch(kernels_.apply, elements, args, "apply compressed residual");
    return true;
  }
  void before_back(CUdeviceptr middle) {
    if (action_ != ResidualCacheAction::Refresh) return;
    auto front = pointer(layout_.front_codes), front_scales = pointer(layout_.front_scales);
    auto codes = pointer(layout_.residual_codes), scales = pointer(layout_.residual_scales);
    auto groups = layout_.groups;
    void *args[]{&middle, &front, &front_scales, &codes, &scales, &groups};
    launch(kernels_.store, groups, args, "store compressed middle residual");
    residual_ready_ = true;
    callbacks_->refreshed();
  }
 private:
  static void checked(CUresult status, const char *action) {
    if (status != CUDA_SUCCESS)
      fail(std::string("residual-cache CUDA ") + action + " failed, code=" + std::to_string(status));
  }
  CUdeviceptr pointer(std::uint64_t offset) const { return workspace_ + offset; }
  void launch(CUfunction kernel, std::uint64_t elements, void **args, const char *label) {
    const auto grid = static_cast<unsigned>(std::min<std::uint64_t>((elements + 127) / 128, 65535));
    checked(dispatch_.launch(kernel, grid, 1, 1, 128, 1, 1, 0, stream_, args, nullptr), label);
  }
  void gather(CUdeviceptr hidden, CUdeviceptr output) {
    auto rows = pointer(layout_.indices);
    auto width = region_.hidden;
    auto count = static_cast<std::uint64_t>(before_.size());
    void *args[]{&hidden, &rows, &output, &width, &count};
    launch(kernels_.gather, count, args, "gather BF16 row probes");
  }
  ResidualCacheRegion region_;
  ResidualCacheDeviceLayout layout_;
  CUdeviceptr workspace_{};
  CUstream stream_{};
  ResidualCacheKernelHandles kernels_;
  ResidualCacheCudaDispatch dispatch_;
  std::shared_ptr<ResidualCacheCallbacks> callbacks_;
  std::vector<std::int32_t> indices_;
  std::vector<float> before_, after_;
  ResidualCacheAction action_{ResidualCacheAction::Exact};
  bool need_probes_{}, residual_ready_{};
};
} // namespace dif::runtime
