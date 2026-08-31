// diftokenize: native prompt -> token-id tool for the Qwen-family byte-level
// BPE tokenizer shipped with a checkpoint's processor/ directory.
//
// Reproduces `transformers` Qwen2TokenizerFast with add_special_tokens=False
// (tokenizer.json + tokenizer_config.json additional_special_tokens; no chat
// template). Reports the id count, the SHA-256 of the comma-joined decimal
// id sequence, and can emit the ids as text and as an I32 .diftensor.
//
// --battery mode reads a JSON array of strings and prints a JSON array of
// id arrays — used by the oracle-parity harness.

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/text/qwen_bpe_tokenizer.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr << "diftokenize: " << message << "\n"
            << "usage: diftokenize --processor <dir> | "
               "(--tokenizer-json <f> [--tokenizer-config <f>])\n"
            << "  --prompt-file <path> | --prompt <text> | --battery <json>\n"
            << "  [--strip-trailing-newline] [--ids-out <path>]\n"
            << "  [--diftensor-out <path>] [--quiet]\n";
  std::exit(2);
}

std::string read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    dif::fail("cannot open " + path.string());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return std::move(buffer).str();
}

std::string joined_decimal(const std::vector<std::int32_t> &ids) {
  std::string joined;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i != 0U)
      joined.push_back(',');
    joined += std::to_string(ids[i]);
  }
  return joined;
}

std::string ids_sha256_hex(const std::vector<std::int32_t> &ids) {
  const std::string joined = joined_decimal(ids);
  return dif::hex_digest(dif::sha256(
      {reinterpret_cast<const std::uint8_t *>(joined.data()), joined.size()}));
}

void write_json_escaped_ascii(std::ostream &out, const std::string &text) {
  out << '"';
  for (const char c : text) {
    const auto u = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (u < 0x20U) {
      char buffer[8];
      std::snprintf(buffer, sizeof buffer, "\\u%04x", unsigned{u});
      out << buffer;
    } else {
      out << c;
    }
  }
  out << '"';
}

} // namespace

int main(int argc, char **argv) {
  try {
    fs::path processor_dir;
    fs::path tokenizer_json;
    fs::path tokenizer_config;
    fs::path prompt_file;
    fs::path battery_file;
    fs::path ids_out;
    fs::path diftensor_out;
    std::string prompt_text;
    bool have_prompt_text = false;
    bool strip_trailing_newline = false;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      const auto value = [&](const char *name) -> std::string {
        if (i + 1 >= argc)
          usage_error(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (option == "--processor")
        processor_dir = value("--processor");
      else if (option == "--tokenizer-json")
        tokenizer_json = value("--tokenizer-json");
      else if (option == "--tokenizer-config")
        tokenizer_config = value("--tokenizer-config");
      else if (option == "--prompt-file")
        prompt_file = value("--prompt-file");
      else if (option == "--prompt") {
        prompt_text = value("--prompt");
        have_prompt_text = true;
      } else if (option == "--battery")
        battery_file = value("--battery");
      else if (option == "--ids-out")
        ids_out = value("--ids-out");
      else if (option == "--diftensor-out")
        diftensor_out = value("--diftensor-out");
      else if (option == "--strip-trailing-newline")
        strip_trailing_newline = true;
      else if (option == "--quiet")
        quiet = true;
      else
        usage_error("unknown option " + option);
    }

    if (!processor_dir.empty()) {
      if (!tokenizer_json.empty() || !tokenizer_config.empty())
        usage_error("--processor conflicts with explicit tokenizer paths");
      tokenizer_json = processor_dir / "tokenizer.json";
      tokenizer_config = processor_dir / "tokenizer_config.json";
    }
    if (tokenizer_json.empty())
      usage_error("need --processor or --tokenizer-json");

    const auto tokenizer =
        dif::text::QwenBpeTokenizer::load(tokenizer_json, tokenizer_config);

    if (!battery_file.empty()) {
      const dif::json::Value cases = dif::json::parse(read_file(battery_file));
      if (!cases.is_array())
        dif::fail("--battery file must hold a JSON array of strings");
      std::cout << "[\n";
      for (std::size_t c = 0; c < cases.array().size(); ++c) {
        const auto ids = tokenizer.encode(cases.array()[c].string());
        std::cout << "  [" << joined_decimal(ids) << "]"
                  << (c + 1U < cases.array().size() ? "," : "") << "\n";
      }
      std::cout << "]\n";
      return 0;
    }

    if (have_prompt_text == !prompt_file.empty())
      usage_error("need exactly one of --prompt / --prompt-file / --battery");
    std::string prompt =
        have_prompt_text ? prompt_text : read_file(prompt_file);
    if (strip_trailing_newline)
      while (!prompt.empty() && prompt.back() == '\n')
        prompt.pop_back();

    const std::vector<std::int32_t> ids = tokenizer.encode(prompt);

    if (!ids_out.empty()) {
      std::ofstream out(ids_out);
      if (!out)
        dif::fail("cannot open " + ids_out.string());
      for (const std::int32_t id : ids)
        out << id << "\n";
    }
    if (!diftensor_out.empty()) {
      dif::runtime::Tensor tensor{
          dif::ir::DType::I32,
          {static_cast<std::uint64_t>(ids.size())},
          {}};
      tensor.bytes.resize(ids.size() * sizeof(std::int32_t));
      std::memcpy(tensor.bytes.data(), ids.data(), tensor.bytes.size());
      tensor.validate();
      dif::runtime::write_tensor(tensor, diftensor_out);
    }

    if (!quiet) {
      std::cout << "{\n  \"prompt_bytes\": " << prompt.size()
                << ",\n  \"token_count\": " << ids.size()
                << ",\n  \"ids_sha256\": ";
      write_json_escaped_ascii(std::cout, ids_sha256_hex(ids));
      std::cout << ",\n  \"id_space\": " << tokenizer.id_space()
                << ",\n  \"added_tokens\": " << tokenizer.added_tokens().size()
                << ",\n  \"first_ids\": [";
      for (std::size_t i = 0; i < ids.size() && i < 8U; ++i)
        std::cout << (i ? "," : "") << ids[i];
      std::cout << "],\n  \"last_ids\": [";
      const std::size_t tail = ids.size() > 8U ? ids.size() - 8U : 0U;
      for (std::size_t i = tail; i < ids.size(); ++i)
        std::cout << (i != tail ? "," : "") << ids[i];
      std::cout << "]\n}\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "diftokenize: " << error.what() << "\n";
    return 1;
  }
}
