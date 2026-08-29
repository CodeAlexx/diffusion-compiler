#include "dif/support/json.hpp"

#include "dif/support/error.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dif::json {

bool Value::is_object() const { return std::holds_alternative<Object>(storage); }
bool Value::is_array() const { return std::holds_alternative<Array>(storage); }

const Value::Object &Value::object() const {
  if (!is_object())
    fail("JSON value is not an object");
  return std::get<Object>(storage);
}

const Value::Array &Value::array() const {
  if (!is_array())
    fail("JSON value is not an array");
  return std::get<Array>(storage);
}

const std::string &Value::string() const {
  if (!std::holds_alternative<std::string>(storage))
    fail("JSON value is not a string");
  return std::get<std::string>(storage);
}

double Value::number() const {
  if (!std::holds_alternative<double>(storage))
    fail("JSON value is not a number");
  return std::get<double>(storage);
}

bool Value::boolean() const {
  if (!std::holds_alternative<bool>(storage))
    fail("JSON value is not a boolean");
  return std::get<bool>(storage);
}

const Value *Value::find(std::string_view key) const {
  if (!is_object())
    return nullptr;
  const auto &values = std::get<Object>(storage);
  const auto found = values.find(key);
  return found == values.end() ? nullptr : &found->second;
}

namespace {

class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}

  Value parse_document() {
    skip_space();
    auto value = parse_value(0U);
    skip_space();
    if (offset_ != input_.size())
      fail("trailing characters after JSON document");
    return value;
  }

private:
  static bool digit(char value) { return value >= '0' && value <= '9'; }

  void skip_space() {
    while (offset_ < input_.size() &&
           (input_[offset_] == ' ' || input_[offset_] == '\t' ||
            input_[offset_] == '\n' || input_[offset_] == '\r'))
      ++offset_;
  }

  char take() {
    if (offset_ >= input_.size())
      fail("unexpected end of JSON");
    return input_[offset_++];
  }

  bool consume(char value) {
    if (offset_ < input_.size() && input_[offset_] == value) {
      ++offset_;
      return true;
    }
    return false;
  }

  void literal(std::string_view value) {
    if (input_.substr(offset_, value.size()) != value)
      fail("invalid JSON literal");
    offset_ += value.size();
  }

  static unsigned hex(char value) {
    if (value >= '0' && value <= '9')
      return static_cast<unsigned>(value - '0');
    if (value >= 'a' && value <= 'f')
      return static_cast<unsigned>(value - 'a') + 10U;
    if (value >= 'A' && value <= 'F')
      return static_cast<unsigned>(value - 'A') + 10U;
    fail("invalid JSON unicode escape");
  }

  std::uint32_t unicode_escape() {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4U; ++i)
      value = (value << 4U) | hex(take());
    return value;
  }

  static void append_utf8(std::string &output, std::uint32_t value) {
    if (value <= 0x7fU) {
      output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (value >> 6U)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else if (value <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (value >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else if (value <= 0x10ffffU) {
      output.push_back(static_cast<char>(0xf0U | (value >> 18U)));
      output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else {
      fail("JSON unicode code point is out of range");
    }
  }

  std::string parse_string() {
    if (take() != '"')
      fail("expected JSON string");
    std::string output;
    while (true) {
      const auto value = take();
      if (value == '"')
        return output;
      if (static_cast<unsigned char>(value) < 0x20U)
        fail("unescaped control character in JSON string");
      if (value != '\\') {
        output.push_back(value);
        continue;
      }
      const auto escaped = take();
      switch (escaped) {
      case '"': output.push_back('"'); break;
      case '\\': output.push_back('\\'); break;
      case '/': output.push_back('/'); break;
      case 'b': output.push_back('\b'); break;
      case 'f': output.push_back('\f'); break;
      case 'n': output.push_back('\n'); break;
      case 'r': output.push_back('\r'); break;
      case 't': output.push_back('\t'); break;
      case 'u': {
        auto codepoint = unicode_escape();
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
          if (take() != '\\' || take() != 'u')
            fail("JSON high surrogate lacks low surrogate");
          const auto low = unicode_escape();
          if (low < 0xdc00U || low > 0xdfffU)
            fail("invalid JSON low surrogate");
          codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                      (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
          fail("unexpected JSON low surrogate");
        }
        append_utf8(output, codepoint);
        break;
      }
      default: fail("invalid JSON string escape");
      }
    }
  }

  Value parse_number() {
    const auto start = offset_;
    (void)consume('-');
    if (consume('0')) {
      if (offset_ < input_.size() && digit(input_[offset_]))
        fail("JSON number has a leading zero");
    } else {
      if (offset_ >= input_.size() || !digit(input_[offset_]))
        fail("invalid JSON number");
      while (offset_ < input_.size() && digit(input_[offset_]))
        ++offset_;
    }
    if (consume('.')) {
      if (offset_ >= input_.size() || !digit(input_[offset_]))
        fail("invalid JSON fraction");
      while (offset_ < input_.size() && digit(input_[offset_]))
        ++offset_;
    }
    if (offset_ < input_.size() &&
        (input_[offset_] == 'e' || input_[offset_] == 'E')) {
      ++offset_;
      if (offset_ < input_.size() &&
          (input_[offset_] == '+' || input_[offset_] == '-'))
        ++offset_;
      if (offset_ >= input_.size() || !digit(input_[offset_]))
        fail("invalid JSON exponent");
      while (offset_ < input_.size() && digit(input_[offset_]))
        ++offset_;
    }
    double output = 0.0;
    const auto first = input_.data() + start;
    const auto last = input_.data() + offset_;
    const auto parsed = std::from_chars(first, last, output);
    if (parsed.ec != std::errc{} || parsed.ptr != last || !std::isfinite(output))
      fail("invalid or non-finite JSON number");
    return Value{output};
  }

  Value parse_array(unsigned depth) {
    (void)take();
    Value::Array output;
    skip_space();
    if (consume(']'))
      return Value{std::move(output)};
    while (true) {
      skip_space();
      output.push_back(parse_value(depth + 1U));
      skip_space();
      if (consume(']'))
        return Value{std::move(output)};
      if (!consume(','))
        fail("expected comma in JSON array");
    }
  }

  Value parse_object(unsigned depth) {
    (void)take();
    Value::Object output;
    skip_space();
    if (consume('}'))
      return Value{std::move(output)};
    while (true) {
      skip_space();
      if (offset_ >= input_.size() || input_[offset_] != '"')
        fail("expected string key in JSON object");
      auto key = parse_string();
      skip_space();
      if (!consume(':'))
        fail("expected colon in JSON object");
      skip_space();
      if (!output.emplace(std::move(key), parse_value(depth + 1U)).second)
        fail("duplicate JSON object key");
      skip_space();
      if (consume('}'))
        return Value{std::move(output)};
      if (!consume(','))
        fail("expected comma in JSON object");
    }
  }

  Value parse_value(unsigned depth) {
    if (depth > 128U)
      fail("JSON nesting is too deep");
    skip_space();
    if (offset_ >= input_.size())
      fail("unexpected end of JSON");
    switch (input_[offset_]) {
    case '{': return parse_object(depth);
    case '[': return parse_array(depth);
    case '"': return Value{parse_string()};
    case 't': literal("true"); return Value{true};
    case 'f': literal("false"); return Value{false};
    case 'n': literal("null"); return Value{};
    default:
      if (input_[offset_] == '-' || digit(input_[offset_]))
        return parse_number();
      fail("invalid JSON value");
    }
  }

  std::string_view input_;
  std::size_t offset_{};
};

} // namespace

Value parse(std::string_view input) { return Parser(input).parse_document(); }

} // namespace dif::json
