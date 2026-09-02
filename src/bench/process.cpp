#include "dif/bench/process.hpp"

#include "dif/support/error.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
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

ChainResult run_chain(
    const ResolvedRecipe &resolved, const std::filesystem::path &log_directory,
    const std::function<std::vector<std::pair<std::string, std::string>>(
        std::size_t)> &before_stage) {
  std::filesystem::create_directories(log_directory);
  ChainResult result;
  result.stages.resize(resolved.stages.size());
  for (std::size_t index = 0; index < resolved.stages.size(); ++index) {
    result.stages[index].name = resolved.stages[index].name;
    result.stages[index].log =
        log_directory / (resolved.stages[index].name + ".log");
  }
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
