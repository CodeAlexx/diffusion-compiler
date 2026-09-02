#include "dif/telemetry/trace_sink.hpp"

#include "dif/ir/codec.hpp"
#include "dif/support/sha256.hpp"
#include "dif/telemetry/schema.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace dif::telemetry {

std::filesystem::path trace_sink_path() {
  const char *value = std::getenv(kTraceFileVariable);
  if (!value || *value == '\0')
    return {};
  return std::filesystem::path(value);
}

bool trace_events_requested(const runtime::RunOptions &options) {
  return options.trace_events || !trace_sink_path().empty();
}

bool nvtx_ranges_requested(const runtime::RunOptions &options) {
  if (options.nvtx_ranges)
    return true;
  const char *value = std::getenv(kNvtxVariable);
  return value && *value != '\0' && std::string(value) != "0";
}

void append_runtime_trace(const runtime::RunResult &result,
                          const ir::Program &program,
                          const runtime::RunOptions &options) {
  const auto path = trace_sink_path();
  if (path.empty())
    return;
  try {
    const auto document = runtime_trace_document(
        result, hex_digest(ir::fingerprint(program)),
        program.operations.size(), options);
    std::ofstream stream(path, std::ios::app | std::ios::binary);
    if (!stream)
      throw std::runtime_error("cannot open trace sink " + path.string());
    const auto text = serialize_compact(Value(document));
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.put('\n');
    if (!stream)
      throw std::runtime_error("cannot write trace sink " + path.string());
  } catch (const std::exception &error) {
    std::cerr << "diffusion-compiler trace sink: " << error.what() << "\n";
  }
}

} // namespace dif::telemetry
