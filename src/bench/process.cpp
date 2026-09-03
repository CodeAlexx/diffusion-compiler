#include "dif/bench/process.hpp"

#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <fcntl.h>
#include <map>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace dif::bench {
namespace {

struct Running {
  std::size_t index{};
  pid_t pid{};
  std::chrono::steady_clock::time_point start;
};

std::vector<std::string>
environment_for(const ResolvedStage &stage,
                const std::vector<std::pair<std::string, std::string>> &extra) {
  std::map<std::string, std::string> merged;
  for (char **entry = environ; entry && *entry; ++entry) {
    const std::string text = *entry;
    const auto split = text.find('=');
    if (split == std::string::npos)
      continue;
    merged[text.substr(0, split)] = text.substr(split + 1U);
  }
  for (const auto &[key, value] : stage.environment)
    merged[key] = value;
  for (const auto &[key, value] : extra)
    merged[key] = value;
  std::vector<std::string> out;
  out.reserve(merged.size());
  for (const auto &[key, value] : merged)
    out.push_back(key + "=" + value);
  return out;
}

[[noreturn]] void child_exec(const ResolvedStage &stage,
                             const std::vector<std::string> &environment,
                             const std::filesystem::path &log) {
  const int descriptor =
      ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (descriptor >= 0) {
    ::dup2(descriptor, STDOUT_FILENO);
    ::dup2(descriptor, STDERR_FILENO);
  }
  if (!stage.cwd.empty() && ::chdir(stage.cwd.c_str()) != 0)
    ::_exit(126);
  std::vector<char *> argv;
  for (const auto &argument : stage.argv)
    argv.push_back(const_cast<char *>(argument.c_str()));
  argv.push_back(nullptr);
  std::vector<char *> envp;
  for (const auto &entry : environment)
    envp.push_back(const_cast<char *>(entry.c_str()));
  envp.push_back(nullptr);
  ::execvpe(argv.front(), argv.data(), envp.data());
  ::_exit(127);
}

} // namespace

std::string stage_cache_key(const ResolvedStage &stage,
                            const StageCachePolicy &policy,
                            const std::filesystem::path &work_directory) {
  const auto workdir = work_directory.string();
  const auto normalize = [&](std::string text) {
    if (workdir.empty())
      return text;
    for (auto at = text.find(workdir); at != std::string::npos;
         at = text.find(workdir, at + 10U))
      text.replace(at, workdir.size(), "${workdir}");
    return text;
  };
  std::string material = "diffusion-compiler-stage-cache-v1\n";
  material += "stage=" + stage.name + "\n";
  for (const auto &argument : stage.argv)
    material += "argv=" + normalize(argument) + "\n";
  for (const auto &file : stage.cache_key_files) {
    std::error_code error;
    const auto size = std::filesystem::file_size(file, error);
    if (error)
      fail("stage cache key file is missing: " + file.string());
    const auto written = std::filesystem::last_write_time(file, error);
    const auto mtime = static_cast<long long>(
        written.time_since_epoch().count());
    material += "file=" + normalize(file.string()) +
                " size=" + std::to_string(size);
    // Files inside the work directory are per-run products: identify them
    // by content only. Files outside it (models, prompts, keyframes) carry
    // their mtime so a rewritten model file invalidates the key even when
    // it is too large to digest.
    if (file.string().rfind(workdir, 0U) != 0U)
      material += " mtime=" + std::to_string(mtime);
    if (size <= policy.digest_limit_bytes)
      material += " sha256=" + hex_digest(sha256_file(file));
    material += "\n";
  }
  return hex_digest(sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(material.data()),
      material.size())));
}

namespace {

std::filesystem::path cache_entry_path(const StageCachePolicy &policy,
                                       const std::string &key,
                                       const std::filesystem::path &output) {
  return policy.directory / key / output.filename();
}

bool restore_from_cache(const ResolvedStage &stage,
                        const StageCachePolicy &policy,
                        const std::string &key) {
  const auto entry = policy.directory / key;
  std::error_code error;
  if (!std::filesystem::is_regular_file(entry / "manifest.json", error))
    return false;
  for (const auto &output : stage.cache_outputs)
    if (!std::filesystem::is_regular_file(cache_entry_path(policy, key, output),
                                          error))
      return false;
  for (const auto &output : stage.cache_outputs) {
    std::filesystem::create_directories(output.parent_path(), error);
    std::filesystem::copy_file(cache_entry_path(policy, key, output), output,
                               std::filesystem::copy_options::overwrite_existing,
                               error);
    if (error)
      fail("stage cache restore failed for " + output.string() + ": " +
           error.message());
  }
  return true;
}

bool store_to_cache(const ResolvedStage &stage, const StageCachePolicy &policy,
                    const std::string &key) {
  const auto entry = policy.directory / key;
  std::error_code error;
  std::filesystem::create_directories(entry, error);
  if (error)
    return false;
  for (const auto &output : stage.cache_outputs) {
    if (!std::filesystem::is_regular_file(output, error))
      return false;
    std::filesystem::copy_file(output, cache_entry_path(policy, key, output),
                               std::filesystem::copy_options::overwrite_existing,
                               error);
    if (error)
      return false;
  }
  std::string manifest = "{\"kind\": \"diffusion-compiler-stage-cache\", "
                         "\"version\": 1, \"stage\": \"" +
                         stage.name + "\", \"key\": \"" + key +
                         "\", \"outputs\": [";
  for (std::size_t index = 0; index < stage.cache_outputs.size(); ++index)
    manifest += std::string(index == 0U ? "" : ", ") + "\"" +
                stage.cache_outputs[index].filename().string() + "\"";
  manifest += "], \"key_files\": [";
  for (std::size_t index = 0; index < stage.cache_key_files.size(); ++index)
    manifest += std::string(index == 0U ? "" : ", ") + "\"" +
                stage.cache_key_files[index].string() + "\"";
  manifest += "]}\n";
  std::ofstream out(entry / "manifest.json", std::ios::binary);
  out << manifest;
  return static_cast<bool>(out);
}

} // namespace

ChainResult run_chain(
    const ResolvedRecipe &resolved, const std::filesystem::path &log_directory,
    const std::function<std::vector<std::pair<std::string, std::string>>(
        std::size_t)> &before_stage,
    const StageCachePolicy &cache) {
  std::filesystem::create_directories(log_directory);
  ChainResult result;
  result.stages.resize(resolved.stages.size());
  for (std::size_t index = 0; index < resolved.stages.size(); ++index) {
    result.stages[index].name = resolved.stages[index].name;
    result.stages[index].log =
        log_directory / (resolved.stages[index].name + ".log");
    result.stages[index].cache_status =
        !cache.enabled() ? "disabled"
        : resolved.stages[index].cacheable ? "miss" : "none";
  }
  if (cache.enabled())
    std::filesystem::create_directories(cache.directory);
  std::vector<bool> succeeded(resolved.stages.size(), false);
  std::vector<bool> failed(resolved.stages.size(), false);
  std::vector<Running> running;
  bool aborted = false;
  const auto chain_start = std::chrono::steady_clock::now();
  const auto seconds_since = [](std::chrono::steady_clock::time_point from,
                                std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
  };
  std::size_t remaining = resolved.stages.size();
  while (remaining != 0U) {
    if (!aborted) {
      for (std::size_t index = 0; index < resolved.stages.size(); ++index) {
        auto &record = result.stages[index];
        if (record.started)
          continue;
        bool ready = true;
        for (const auto dependency : resolved.stages[index].after)
          ready = ready && succeeded[dependency];
        if (!ready)
          continue;
        if (cache.enabled() && resolved.stages[index].cacheable) {
          // Key files must exist by now (they are inputs or earlier
          // outputs); a hit restores the outputs and never starts the
          // process, and the stage wall is the restore time.
          const auto start = std::chrono::steady_clock::now();
          record.cache_key = stage_cache_key(resolved.stages[index], cache,
                                             resolved.work_directory);
          if (restore_from_cache(resolved.stages[index], cache,
                                 record.cache_key)) {
            record.started = true;
            record.finished = true;
            record.exit_status = 0;
            record.cache_status = "hit";
            record.start_offset_seconds = seconds_since(chain_start, start);
            record.wall_seconds =
                seconds_since(start, std::chrono::steady_clock::now());
            succeeded[index] = true;
            --remaining;
            continue;
          }
        }
        const auto extra = before_stage ? before_stage(index)
                                        : std::vector<std::pair<std::string,
                                                                std::string>>{};
        const auto environment =
            environment_for(resolved.stages[index], extra);
        const auto start = std::chrono::steady_clock::now();
        const pid_t pid = ::fork();
        if (pid < 0)
          fail(std::string("fork failed: ") + std::strerror(errno));
        if (pid == 0)
          child_exec(resolved.stages[index], environment, record.log);
        record.started = true;
        record.start_offset_seconds = seconds_since(chain_start, start);
        running.push_back({index, pid, start});
      }
    }
    if (running.empty()) {
      // Nothing can start and nothing is running: a dependency failed.
      break;
    }
    int status = 0;
    struct rusage usage {};
    const pid_t pid = ::wait4(-1, &status, 0, &usage);
    if (pid < 0) {
      if (errno == EINTR)
        continue;
      fail(std::string("wait4 failed: ") + std::strerror(errno));
    }
    const auto stop = std::chrono::steady_clock::now();
    for (auto it = running.begin(); it != running.end(); ++it) {
      if (it->pid != pid)
        continue;
      auto &record = result.stages[it->index];
      record.finished = true;
      record.wall_seconds = seconds_since(it->start, stop);
      if (WIFEXITED(status)) {
        record.exit_status = WEXITSTATUS(status);
      } else if (WIFSIGNALED(status)) {
        record.signaled = true;
        record.signal = WTERMSIG(status);
        record.exit_status = 128 + record.signal;
      }
      record.user_cpu_seconds =
          static_cast<double>(usage.ru_utime.tv_sec) +
          static_cast<double>(usage.ru_utime.tv_usec) / 1.0e6;
      record.system_cpu_seconds =
          static_cast<double>(usage.ru_stime.tv_sec) +
          static_cast<double>(usage.ru_stime.tv_usec) / 1.0e6;
      record.max_rss_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
      record.major_faults = static_cast<std::uint64_t>(usage.ru_majflt);
      record.minor_faults = static_cast<std::uint64_t>(usage.ru_minflt);
      record.filesystem_input_blocks =
          static_cast<std::uint64_t>(usage.ru_inblock);
      record.filesystem_output_blocks =
          static_cast<std::uint64_t>(usage.ru_oublock);
      if (record.exit_status == 0) {
        succeeded[it->index] = true;
        if (cache.enabled() && resolved.stages[it->index].cacheable &&
            !store_to_cache(resolved.stages[it->index], cache,
                            record.cache_key))
          record.cache_status = "store-failed";
      } else {
        failed[it->index] = true;
        if (!aborted) {
          aborted = true;
          result.failed_stage = record.name;
        }
      }
      --remaining;
      running.erase(it);
      break;
    }
  }
  result.wall_seconds =
      seconds_since(chain_start, std::chrono::steady_clock::now());
  result.success = !aborted && remaining == 0U;
  for (std::size_t index = 0; index < result.stages.size(); ++index) {
    auto &record = result.stages[index];
    if (!record.started)
      continue;
    const auto start = record.start_offset_seconds;
    const auto stop = start + record.wall_seconds;
    for (std::size_t other = 0; other < result.stages.size(); ++other) {
      if (other == index || !result.stages[other].started)
        continue;
      const auto other_start = result.stages[other].start_offset_seconds;
      const auto other_stop = other_start + result.stages[other].wall_seconds;
      if (other_start < stop && start < other_stop)
        record.concurrent_with.push_back(result.stages[other].name);
    }
  }
  return result;
}

CaptureResult run_capture(const std::vector<std::string> &argv) {
  if (argv.empty())
    fail("run_capture requires a program");
  int pipe_descriptors[2];
  if (::pipe(pipe_descriptors) != 0)
    fail(std::string("pipe failed: ") + std::strerror(errno));
  const pid_t pid = ::fork();
  if (pid < 0)
    fail(std::string("fork failed: ") + std::strerror(errno));
  if (pid == 0) {
    ::close(pipe_descriptors[0]);
    ::dup2(pipe_descriptors[1], STDOUT_FILENO);
    ::close(pipe_descriptors[1]);
    std::vector<char *> arguments;
    for (const auto &argument : argv)
      arguments.push_back(const_cast<char *>(argument.c_str()));
    arguments.push_back(nullptr);
    ::execvp(arguments.front(), arguments.data());
    ::_exit(127);
  }
  ::close(pipe_descriptors[1]);
  CaptureResult result;
  char buffer[4096];
  while (true) {
    const auto count = ::read(pipe_descriptors[0], buffer, sizeof(buffer));
    if (count < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    if (count == 0)
      break;
    result.output.append(buffer, static_cast<std::size_t>(count));
  }
  ::close(pipe_descriptors[0]);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  result.exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return result;
}

} // namespace dif::bench
