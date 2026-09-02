// difregress: tiered regression runner. Correctness is strict (exit status
// and JSON assertions); performance is compared against recorded baselines
// with a noise-aware tolerance. A check that cannot run on this host is
// BLOCKED, never PASS. Suites and baselines are data; the runner is
// model-neutral.

#include "dif/bench/process.hpp"
#include "dif/build_info.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/telemetry/schema.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: difregress run SUITE.json --tier smoke|full|model NAME\n"
         "                     [--build DIR] [--workdir DIR] [--baseline FILE]\n"
         "                     [--samples N] [--json] [--report FILE]\n"
         "       difregress record SUITE.json --tier smoke|full|model NAME --baseline FILE\n"
         "                     [--build DIR] [--workdir DIR] [--samples N] [--json]\n"
         "       difregress show SUITE.json [--tier ...] [--json]\n"
         "\n"
         "Check verdicts: PASS, FAIL (exit status or JSON assertion), BLOCKED\n"
         "(a declared blocked exit status, or the program cannot start),\n"
         "REGRESSED (median above baseline median by more than max(tolerance,\n"
         "baseline noise)). Tier verdict: FAIL > REGRESSED > BLOCKED > PASS.\n"
         "Exit status: 0 PASS, 1 FAIL/REGRESSED, 3 BLOCKED.\n";
}

const dif::json::Value &required(const dif::json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value)
    dif::fail(std::string("regression suite is missing field '") + key + "'");
  return *value;
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    dif::fail("cannot open " + path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    dif::fail("cannot write " + path.string());
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

struct Performance {
  bool present{};
  std::string metric{"wall_seconds"};
  std::uint32_t samples{1U};
  double tolerance{0.10};
};

struct Check {
  std::string name;
  std::string tier;
  std::string model;
  std::vector<std::string> argv;
  int expect_exit{0};
  std::vector<int> blocked_exit;
  std::vector<std::pair<std::string, std::string>> expect_json;
  Performance performance;
};

struct Suite {
  std::filesystem::path path;
  std::string name;
  std::map<std::string, std::string> variables;
  std::vector<Check> checks;
};

Suite parse_suite(const std::filesystem::path &path) {
  const auto document = dif::json::parse(read_file(path));
  if (!document.is_object() ||
      required(document, "kind").string() != "diffusion-compiler-regression-suite" ||
      required(document, "version").number() != 1.0)
    dif::fail("file is not a version-1 diffusion-compiler regression suite");
  Suite suite;
  suite.path = std::filesystem::absolute(path);
  suite.name = required(document, "name").string();
  if (const auto *variables = document.find("variables"))
    for (const auto &[key, value] : variables->object())
      suite.variables.emplace(key, value.string());
  for (const auto &entry : required(document, "checks").array()) {
    Check check;
    check.name = required(entry, "name").string();
    check.tier = required(entry, "tier").string();
    if (const auto *model = entry.find("model"))
      check.model = model->string();
    for (const auto &argument : required(entry, "argv").array())
      check.argv.push_back(argument.string());
    if (check.argv.empty())
      dif::fail("check '" + check.name + "' has an empty argv");
    if (const auto *exit_code = entry.find("expect_exit"))
      check.expect_exit = static_cast<int>(exit_code->number());
    if (const auto *blocked = entry.find("blocked_exit"))
      for (const auto &code : blocked->array())
        check.blocked_exit.push_back(static_cast<int>(code.number()));
    if (const auto *expect = entry.find("expect_json"))
      for (const auto &[key, value] : expect->object()) {
        std::string text;
        if (std::holds_alternative<std::string>(value.storage))
          text = value.string();
        else if (std::holds_alternative<bool>(value.storage))
          text = value.boolean() ? "true" : "false";
        else
          text = dif::telemetry::number_text(value.number());
        check.expect_json.emplace_back(key, text);
      }
    if (const auto *performance = entry.find("performance")) {
      check.performance.present = true;
      if (const auto *metric = performance->find("metric"))
        check.performance.metric = metric->string();
      if (const auto *samples = performance->find("samples"))
        check.performance.samples = static_cast<std::uint32_t>(samples->number());
      if (const auto *tolerance = performance->find("tolerance"))
        check.performance.tolerance = tolerance->number();
    }
    suite.checks.push_back(std::move(check));
  }
  return suite;
}

std::string substitute(const std::string &text, const Suite &suite,
                       const std::filesystem::path &build,
                       const std::filesystem::path &workdir) {
  std::string out;
  std::size_t index = 0U;
  while (index < text.size()) {
    const auto start = text.find("${", index);
    if (start == std::string::npos) {
      out += text.substr(index);
      break;
    }
    const auto stop = text.find('}', start);
    if (stop == std::string::npos)
      dif::fail("unterminated ${ reference: " + text);
    out += text.substr(index, start - index);
    const auto name = text.substr(start + 2U, stop - start - 2U);
    if (name == "build")
      out += build.string();
    else if (name == "suite_dir")
      out += suite.path.parent_path().string();
    else if (name == "workdir")
      out += workdir.string();
    else if (const auto found = suite.variables.find(name); found != suite.variables.end())
      // Variables may reference the builtins and each other (bounded by the
      // number of variables so a cycle cannot recurse forever).
      out += substitute(found->second, suite, build, workdir);
    else
      dif::fail("regression suite references unknown variable '" + name + "'");
    index = stop + 1U;
  }
  return out;
}

const dif::json::Value *lookup(const dif::json::Value &root, const std::string &path) {
  const dif::json::Value *current = &root;
  std::stringstream parts(path);
  std::string part;
  while (std::getline(parts, part, '.')) {
    if (!current->is_object())
      return nullptr;
    current = current->find(part);
    if (!current)
      return nullptr;
  }
  return current;
}

std::string scalar_text(const dif::json::Value &value) {
  if (std::holds_alternative<std::string>(value.storage))
    return value.string();
  if (std::holds_alternative<bool>(value.storage))
    return value.boolean() ? "true" : "false";
  if (std::holds_alternative<double>(value.storage))
    return dif::telemetry::number_text(value.number());
  return "<non-scalar>";
}

struct Baseline {
  std::vector<double> samples;
  double median{};
  double minimum{};
  double maximum{};
  std::string recorded_at;
  std::string revision;
};

std::map<std::string, Baseline> read_baselines(const std::filesystem::path &path) {
  std::map<std::string, Baseline> out;
  std::error_code error;
  if (path.empty() || !std::filesystem::exists(path, error))
    return out;
  const auto document = dif::json::parse(read_file(path));
  const auto *checks = document.find("checks");
  if (!checks || !checks->is_object())
    return out;
  for (const auto &[name, entry] : checks->object()) {
    Baseline baseline;
    if (const auto *samples = entry.find("samples"))
      for (const auto &sample : samples->array())
        baseline.samples.push_back(sample.number());
    if (const auto *median = entry.find("median"))
      baseline.median = median->number();
    if (const auto *minimum = entry.find("min"))
      baseline.minimum = minimum->number();
    if (const auto *maximum = entry.find("max"))
      baseline.maximum = maximum->number();
    if (const auto *at = entry.find("recorded_at"))
      baseline.recorded_at = at->string();
    if (const auto *revision = entry.find("revision"))
      baseline.revision = revision->string();
    out.emplace(name, std::move(baseline));
  }
  return out;
}

double median_of(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  if (values.empty())
    return 0.0;
  return values.size() % 2U == 1U
             ? values[values.size() / 2U]
             : 0.5 * (values[values.size() / 2U - 1U] + values[values.size() / 2U]);
}

struct CheckResult {
  std::string name;
  std::string verdict; // PASS | FAIL | BLOCKED | REGRESSED
  std::string detail;
  std::vector<int> exit_codes;
  std::vector<double> metric_samples;
  std::string metric;
  std::optional<double> median;
  std::optional<Baseline> baseline;
  std::string performance_verdict; // unbaselined | within | regressed | improved | not-measured
  double effective_tolerance{};
};

CheckResult run_check(const Check &check, const Suite &suite,
                      const std::filesystem::path &build,
                      const std::filesystem::path &workdir_root,
                      std::uint32_t sample_override,
                      const std::map<std::string, Baseline> &baselines) {
  CheckResult result;
  result.name = check.name;
  result.metric = check.performance.metric;
  const auto samples = check.performance.present
                           ? (sample_override ? sample_override
                                              : check.performance.samples)
                           : 1U;
  bool failed = false;
  bool blocked = false;
  for (std::uint32_t sample = 0; sample < samples; ++sample) {
    const auto workdir =
        workdir_root / (check.name + "-" + std::to_string(sample + 1U));
    std::vector<std::string> argv;
    for (const auto &argument : check.argv)
      argv.push_back(substitute(argument, suite, build, workdir));
    std::error_code error;
    if (argv.front().find('/') != std::string::npos &&
        !std::filesystem::exists(argv.front(), error)) {
      blocked = true;
      result.detail = "program not found: " + argv.front();
      result.exit_codes.push_back(-1);
      break;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto capture = dif::bench::run_capture(argv);
    const auto wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.exit_codes.push_back(capture.exit_status);
    if (capture.exit_status == 127) {
      blocked = true;
      result.detail = "program could not be executed: " + argv.front();
      break;
    }
    if (std::find(check.blocked_exit.begin(), check.blocked_exit.end(),
                  capture.exit_status) != check.blocked_exit.end()) {
      blocked = true;
      result.detail = "declared blocked exit status " +
                      std::to_string(capture.exit_status);
      break;
    }
    if (capture.exit_status != check.expect_exit) {
      failed = true;
      result.detail = "exit status " + std::to_string(capture.exit_status) +
                      ", expected " + std::to_string(check.expect_exit);
      break;
    }
    std::optional<dif::json::Value> parsed;
    if (!check.expect_json.empty() ||
        check.performance.metric.rfind("json:", 0U) == 0U) {
      try {
        parsed = dif::json::parse(capture.output);
      } catch (const std::exception &error) {
        failed = true;
        result.detail = std::string("stdout is not JSON: ") + error.what();
        break;
      }
    }
    for (const auto &[path, expected] : check.expect_json) {
      const auto *value = lookup(*parsed, path);
      if (!value || scalar_text(*value) != expected) {
        failed = true;
        result.detail = "JSON assertion failed: " + path + " expected " + expected +
                        " got " + (value ? scalar_text(*value) : "<absent>");
        break;
      }
    }
    if (failed)
      break;
    if (check.performance.present) {
      if (check.performance.metric == "wall_seconds") {
        result.metric_samples.push_back(wall);
      } else if (check.performance.metric.rfind("json:", 0U) == 0U) {
        const auto *value = lookup(*parsed, check.performance.metric.substr(5U));
        if (!value || !std::holds_alternative<double>(value->storage)) {
          failed = true;
          result.detail = "performance metric absent from JSON: " +
                          check.performance.metric;
          break;
        }
        result.metric_samples.push_back(value->number());
      } else {
        dif::fail("unknown performance metric " + check.performance.metric);
      }
    }
  }
  if (failed) {
    result.verdict = "FAIL";
    result.performance_verdict = "not-measured";
    return result;
  }
  if (blocked) {
    result.verdict = "BLOCKED";
    result.performance_verdict = "not-measured";
    return result;
  }
  result.verdict = "PASS";
  if (!check.performance.present) {
    result.performance_verdict = "not-measured";
    return result;
  }
  result.median = median_of(result.metric_samples);
  const auto found = baselines.find(check.name);
  if (found == baselines.end()) {
    result.performance_verdict = "unbaselined";
    return result;
  }
  result.baseline = found->second;
  const auto &baseline = found->second;
  const double noise = baseline.median > 0.0
                           ? (baseline.maximum - baseline.minimum) / baseline.median
                           : 0.0;
  result.effective_tolerance = std::max(check.performance.tolerance, noise);
  if (*result.median > baseline.median * (1.0 + result.effective_tolerance)) {
    result.performance_verdict = "regressed";
    result.verdict = "REGRESSED";
    result.detail = "median " + dif::telemetry::number_text(*result.median) +
                    " exceeds baseline median " +
                    dif::telemetry::number_text(baseline.median) + " by more than " +
                    dif::telemetry::number_text(result.effective_tolerance * 100.0) +
                    " percent";
  } else if (*result.median < baseline.median * (1.0 - result.effective_tolerance)) {
    result.performance_verdict = "improved";
  } else {
    result.performance_verdict = "within";
  }
  return result;
}

dif::telemetry::Object result_section(const CheckResult &result, const Check &check) {
  dif::telemetry::Object out;
  out.set("name", result.name);
  out.set("tier", check.tier);
  out.set("model", check.model);
  out.set("verdict", result.verdict);
  out.set("detail", result.detail);
  dif::telemetry::Array exits;
  for (const auto code : result.exit_codes)
    exits.push_back(code);
  out.set("exit_codes", std::move(exits));
  dif::telemetry::Object performance;
  performance.set("present", check.performance.present);
  performance.set("metric", result.metric);
  performance.set("verdict", result.performance_verdict);
  dif::telemetry::Array samples;
  for (const auto sample : result.metric_samples)
    samples.push_back(sample);
  performance.set("samples", std::move(samples));
  performance.set("median", result.median ? dif::telemetry::Value(*result.median)
                                          : dif::telemetry::Value(nullptr));
  performance.set("tolerance", check.performance.tolerance);
  performance.set("effective_tolerance", result.effective_tolerance);
  if (result.baseline) {
    dif::telemetry::Object baseline;
    baseline.set("median", result.baseline->median);
    baseline.set("min", result.baseline->minimum);
    baseline.set("max", result.baseline->maximum);
    baseline.set("recorded_at", result.baseline->recorded_at);
    baseline.set("revision", result.baseline->revision);
    performance.set("baseline", std::move(baseline));
  } else {
    performance.set("baseline", nullptr);
  }
  out.set("performance", std::move(performance));
  return out;
}

bool selected(const Check &check, const std::string &tier, const std::string &model) {
  if (tier == "full")
    return true;
  if (tier == "model")
    return check.tier == "model" && check.model == model;
  return check.tier == tier;
}

std::filesystem::path default_build() {
  char buffer[4096];
  const auto count = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1U);
  if (count > 0) {
    buffer[count] = '\0';
    return std::filesystem::path(buffer).parent_path();
  }
  return std::filesystem::current_path();
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    const std::filesystem::path suite_path = argv[2];
    std::string tier;
    std::string model;
    std::filesystem::path build;
    std::filesystem::path workdir;
    std::filesystem::path baseline_path;
    std::uint32_t samples = 0U;
    bool json = false;
    std::filesystem::path report;
    for (int index = 3; index < argc; ++index) {
      const std::string option = argv[index];
      const auto value = [&]() -> std::string {
        if (index + 1 >= argc)
          dif::fail("missing value after " + option);
        return argv[++index];
      };
      if (option == "--tier") {
        tier = value();
        if (tier == "model") {
          if (index + 1 >= argc)
            dif::fail("--tier model needs a model name");
          model = argv[++index];
        } else if (tier != "smoke" && tier != "full")
          dif::fail("--tier accepts smoke, full, or model NAME");
      } else if (option == "--build")
        build = value();
      else if (option == "--workdir")
        workdir = value();
      else if (option == "--baseline")
        baseline_path = value();
      else if (option == "--samples")
        samples = static_cast<std::uint32_t>(std::stoul(value()));
      else if (option == "--json")
        json = true;
      else if (option == "--report")
        report = value();
      else
        dif::fail("unknown difregress option: " + option);
    }
    if (build.empty())
      build = default_build();
    const auto suite = parse_suite(suite_path);
    if (command == "show") {
      auto document = dif::telemetry::make_document("regression-suite");
      document.set("suite", suite.name);
      document.set("path", suite.path.string());
      dif::telemetry::Array checks;
      for (const auto &check : suite.checks) {
        if (!tier.empty() && !selected(check, tier, model))
          continue;
        dif::telemetry::Object entry;
        entry.set("name", check.name);
        entry.set("tier", check.tier);
        entry.set("model", check.model);
        entry.set("performance", check.performance.present);
        checks.push_back(std::move(entry));
      }
      document.set("checks", std::move(checks));
      if (json)
        std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
      else
        for (const auto &check : document.find("checks")->array())
          std::cout << check.object().find("tier")->string() << "  "
                    << check.object().find("name")->string() << "\n";
      return 0;
    }
    if (tier.empty())
      dif::fail("difregress " + command + " requires --tier");
    if (workdir.empty())
      workdir = std::filesystem::temp_directory_path() / "difregress";
    workdir = std::filesystem::absolute(workdir);
    std::filesystem::create_directories(workdir);
    const auto baselines = command == "run" ? read_baselines(baseline_path)
                                            : std::map<std::string, Baseline>{};
    std::vector<std::pair<const Check *, CheckResult>> results;
    for (const auto &check : suite.checks) {
      if (!selected(check, tier, model))
        continue;
      if (!json)
        std::cerr << "difregress: " << check.name << " ...\n";
      results.emplace_back(&check, run_check(check, suite, build, workdir, samples,
                                             baselines));
      if (!json)
        std::cerr << "difregress: " << check.name << " "
                  << results.back().second.verdict
                  << (results.back().second.detail.empty()
                          ? ""
                          : " (" + results.back().second.detail + ")")
                  << "\n";
    }
    if (command == "record") {
      if (baseline_path.empty())
        dif::fail("difregress record requires --baseline FILE");
      auto existing = read_baselines(baseline_path);
      dif::telemetry::Object checks;
      const auto now = dif::telemetry::utc_timestamp_now();
      std::size_t recorded = 0U;
      for (const auto &[check, result] : results) {
        if (result.verdict != "PASS" || result.metric_samples.empty())
          continue;
        Baseline baseline;
        baseline.samples = result.metric_samples;
        baseline.median = median_of(result.metric_samples);
        baseline.minimum = *std::min_element(result.metric_samples.begin(),
                                             result.metric_samples.end());
        baseline.maximum = *std::max_element(result.metric_samples.begin(),
                                             result.metric_samples.end());
        baseline.recorded_at = now;
        baseline.revision = std::string(dif::build::compiler_revision());
        existing[check->name] = baseline;
        ++recorded;
      }
      for (const auto &[name, baseline] : existing) {
        dif::telemetry::Object entry;
        dif::telemetry::Array values;
        for (const auto sample : baseline.samples)
          values.push_back(sample);
        entry.set("samples", std::move(values));
        entry.set("median", baseline.median);
        entry.set("min", baseline.minimum);
        entry.set("max", baseline.maximum);
        entry.set("recorded_at", baseline.recorded_at);
        entry.set("revision", baseline.revision);
        checks.set(name, std::move(entry));
      }
      auto document = dif::telemetry::make_document("regression-baselines");
      document.set("suite", suite.name);
      document.set("checks", std::move(checks));
      write_text(baseline_path, dif::telemetry::serialize(dif::telemetry::Value(document)));
      if (json)
        std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
      else
        std::cout << "DIFREGRESS recorded " << recorded << " baseline(s) into "
                  << baseline_path.string() << "\n";
      return 0;
    }
    if (command != "run") {
      usage();
      return 2;
    }
    auto document = dif::telemetry::make_document("regression-report");
    document.set("suite", suite.name);
    document.set("tier", tier == "model" ? "model " + model : tier);
    document.set("build_directory", build.string());
    document.set("baseline_file", baseline_path.string());
    dif::telemetry::Array entries;
    std::size_t passed = 0U;
    std::size_t failed = 0U;
    std::size_t blocked = 0U;
    std::size_t regressed = 0U;
    for (const auto &[check, result] : results) {
      entries.push_back(result_section(result, *check));
      if (result.verdict == "PASS")
        ++passed;
      else if (result.verdict == "FAIL")
        ++failed;
      else if (result.verdict == "BLOCKED")
        ++blocked;
      else if (result.verdict == "REGRESSED")
        ++regressed;
    }
    document.set("checks", std::move(entries));
    std::string verdict = "PASS";
    if (failed != 0U)
      verdict = "FAIL";
    else if (regressed != 0U)
      verdict = "REGRESSED";
    else if (blocked != 0U)
      verdict = "BLOCKED";
    if (results.empty())
      verdict = "BLOCKED";
    dif::telemetry::Object summary;
    summary.set("checks", results.size());
    summary.set("passed", passed);
    summary.set("failed", failed);
    summary.set("blocked", blocked);
    summary.set("regressed", regressed);
    summary.set("verdict", verdict);
    document.set("summary", std::move(summary));
    const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
    if (!report.empty())
      write_text(report, text);
    if (json) {
      std::cout << text;
    } else {
      for (const auto &[check, result] : results)
        std::cout << std::left << std::setw(10) << result.verdict << " "
                  << std::setw(28) << result.name << " perf="
                  << result.performance_verdict
                  << (result.detail.empty() ? "" : "  " + result.detail) << "\n";
      std::cout << "DIFREGRESS " << verdict << " checks=" << results.size()
                << " passed=" << passed << " failed=" << failed
                << " blocked=" << blocked << " regressed=" << regressed << "\n";
    }
    if (verdict == "PASS")
      return 0;
    if (verdict == "BLOCKED")
      return 3;
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "difregress: " << error.what() << "\n";
    return 1;
  }
}
