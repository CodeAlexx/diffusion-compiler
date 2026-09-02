// diftokenize: native prompt -> token-id tool for the Qwen-family byte-level
// BPE tokenizer shipped with a checkpoint's processor/ directory.
//
// Reproduces `transformers` Qwen2TokenizerFast with add_special_tokens=False.
// Model chat policy stays in its frontend: --flux2-inputs-out applies the
// exact FLUX.2 single-user Qwen3 template and 512-token creator padding.
//
// --battery mode reads a JSON array of strings and prints a JSON array of
// id arrays — used by the oracle-parity harness.

#include "dif/frontend/flux2_prompt.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/text/qwen_bpe_tokenizer.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
            << "  [--diftensor-out <path>] [--krea2-inputs-out <path>] "
               "[--flux2-inputs-out <path>] "
               "[--quiet]\n";
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
    fs::path krea2_inputs_out;
    fs::path flux2_inputs_out;
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
      else if (option == "--krea2-inputs-out")
        krea2_inputs_out = value("--krea2-inputs-out");
      else if (option == "--flux2-inputs-out")
        flux2_inputs_out = value("--flux2-inputs-out");
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

    if (!krea2_inputs_out.empty() && !flux2_inputs_out.empty())
      usage_error("--krea2-inputs-out conflicts with --flux2-inputs-out");

    std::vector<std::int32_t> ids = tokenizer.encode(prompt);
    std::size_t valid_tokens = ids.size();

    if (!flux2_inputs_out.empty()) {
      const auto flux2 =
          dif::frontend::make_flux2_qwen_prompt_inputs(tokenizer, prompt);
      ids = flux2.input_ids;
      valid_tokens = flux2.valid_tokens;
      std::vector<float> positions(ids.size());
      for (std::size_t index = 0U; index < positions.size(); ++index)
        positions[index] = static_cast<float>(index);
      std::vector<dif::weights::SafeTensorWriteSpec> specs{
          {"input_ids", dif::ir::DType::I32, {1U, ids.size()}},
          {"attention_mask", dif::ir::DType::Bool,
           {1U, flux2.attention_mask.size()}},
          {"position_ids", dif::ir::DType::F32, {positions.size(), 1U}},
      };
      dif::weights::SafeTensorWriter writer(flux2_inputs_out,
                                             std::move(specs));
      writer.append(
          "input_ids",
          {reinterpret_cast<const std::uint8_t *>(ids.data()),
           ids.size() * sizeof(std::int32_t)});
      writer.append("attention_mask",
                    {flux2.attention_mask.data(),
                     flux2.attention_mask.size()});
      writer.append(
          "position_ids",
          {reinterpret_cast<const std::uint8_t *>(positions.data()),
           positions.size() * sizeof(float)});
      (void)writer.finish();
    }

    if (!krea2_inputs_out.empty()) {
      constexpr std::string_view krea_prefix =
          "<|im_start|>system\nDescribe the image by detailing the color, shape, "
          "size, texture, quantity, text, spatial relationships of the objects "
          "and background:<|im_end|>\n<|im_start|>user\n";
      constexpr std::size_t prefix_length = 541U;
      constexpr std::int32_t pad_id = 151643;
      constexpr std::int32_t suffix[] = {151645, 198, 151644, 77091, 198};
      auto krea_ids = tokenizer.encode(std::string(krea_prefix) + prompt);
      if (krea_ids.size() > prefix_length)
        krea_ids.resize(prefix_length);
      const auto valid_prefix = krea_ids.size();
      krea_ids.resize(prefix_length, pad_id);
      krea_ids.insert(krea_ids.end(), std::begin(suffix), std::end(suffix));
      std::vector<std::uint8_t> mask(krea_ids.size(), 0U);
      std::fill(mask.begin(), mask.begin() + valid_prefix, 1U);
      std::fill(mask.begin() + prefix_length, mask.end(), 1U);
      std::vector<float> positions(krea_ids.size());
      std::int32_t position = -1;
      for (std::size_t index = 0U; index < krea_ids.size(); ++index) {
        if (mask[index])
          ++position;
        positions[index] = mask[index] ? static_cast<float>(position) : 1.0F;
      }
      std::vector<dif::weights::SafeTensorWriteSpec> specs{
          {"input_ids", dif::ir::DType::I32, {1U, krea_ids.size()}},
          {"attention_mask", dif::ir::DType::Bool, {1U, mask.size()}},
          {"position_ids", dif::ir::DType::F32, {positions.size(), 1U}},
      };
      dif::weights::SafeTensorWriter writer(krea2_inputs_out,
                                             std::move(specs));
      writer.append(
          "input_ids",
          {reinterpret_cast<const std::uint8_t *>(krea_ids.data()),
           krea_ids.size() * sizeof(std::int32_t)});
      writer.append("attention_mask", {mask.data(), mask.size()});
      writer.append(
          "position_ids",
          {reinterpret_cast<const std::uint8_t *>(positions.data()),
           positions.size() * sizeof(float)});
      (void)writer.finish();
    }

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
                << ",\n  \"valid_token_count\": " << valid_tokens
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
