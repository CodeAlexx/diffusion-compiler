#include "dif/telemetry/document.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"

#include <cmath>
#include <cstdio>

namespace dif::telemetry {

Object &Object::set(std::string key, Value value) {
  for (auto &member : members_) {
    if (member.first == key) {
      member.second = std::move(value);
      return *this;
    }
  }
  members_.emplace_back(std::move(key), std::move(value));
  return *this;
}

const Value *Object::find(std::string_view key) const {
  for (const auto &member : members_)
    if (member.first == key)
      return &member.second;
  return nullptr;
}

Value *Object::find(std::string_view key) {
  for (auto &member : members_)
    if (member.first == key)
      return &member.second;
  return nullptr;
}

bool Value::is_null() const {
  return std::holds_alternative<std::nullptr_t>(storage_);
}
bool Value::is_object() const { return std::holds_alternative<Object>(storage_); }
bool Value::is_array() const { return std::holds_alternative<Array>(storage_); }
bool Value::is_string() const {
  return std::holds_alternative<std::string>(storage_);
}
bool Value::is_number() const {
  return std::holds_alternative<std::int64_t>(storage_) ||
         std::holds_alternative<std::uint64_t>(storage_) ||
         std::holds_alternative<double>(storage_);
}

const Object &Value::object() const {
  if (!is_object())
    fail("telemetry value is not an object");
  return std::get<Object>(storage_);
}

Object &Value::object() {
  if (!is_object())
    fail("telemetry value is not an object");
  return std::get<Object>(storage_);
}

const Array &Value::array() const {
  if (!is_array())
    fail("telemetry value is not an array");
  return std::get<Array>(storage_);
}

Array &Value::array() {
  if (!is_array())
    fail("telemetry value is not an array");
  return std::get<Array>(storage_);
}

const std::string &Value::string() const {
  if (!is_string())
    fail("telemetry value is not a string");
  return std::get<std::string>(storage_);
}

double Value::number() const {
  if (const auto *value = std::get_if<std::int64_t>(&storage_))
    return static_cast<double>(*value);
  if (const auto *value = std::get_if<std::uint64_t>(&storage_))
    return static_cast<double>(*value);
  if (const auto *value = std::get_if<double>(&storage_))
    return *value;
  fail("telemetry value is not a number");
}

bool Value::boolean() const {
  if (!std::holds_alternative<bool>(storage_))
    fail("telemetry value is not a boolean");
  return std::get<bool>(storage_);
}

std::string quote(std::string_view text) {
  std::string out = "\"";
  for (const unsigned char character : text) {
    switch (character) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (character < 0x20U) {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                      static_cast<unsigned>(character));
        out += buffer;
      } else {
        out += static_cast<char>(character);
      }
    }
  }
  out += "\"";
  return out;
}

Value nullable_string(const std::string &text) {
  if (text.empty())
    return Value(nullptr);
  return Value(text);
}

Value nullable_number(bool present, double value) {
  if (!present)
    return Value(nullptr);
  return Value(value);
}

std::string number_text(double value) {
  if (!std::isfinite(value))
    return "null";
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

namespace {

bool scalar(const Value &value) {
  return !value.is_object() && !value.is_array();
}

void write(std::string &out, const Value &value, std::size_t depth);
void write_compact(std::string &out, const Value &value);

void indent(std::string &out, std::size_t depth) {
  out.append(depth * 2U, ' ');
}

void write_scalar(std::string &out, const Value &value) {
  const auto &storage = value.storage();
  if (std::holds_alternative<std::nullptr_t>(storage))
    out += "null";
  else if (const auto *flag = std::get_if<bool>(&storage))
    out += *flag ? "true" : "false";
  else if (const auto *integer = std::get_if<std::int64_t>(&storage))
    out += std::to_string(*integer);
  else if (const auto *unsigned_integer = std::get_if<std::uint64_t>(&storage))
    out += std::to_string(*unsigned_integer);
  else if (const auto *real = std::get_if<double>(&storage))
    out += number_text(*real);
  else if (const auto *text = std::get_if<std::string>(&storage))
    out += quote(*text);
  else
    fail("telemetry scalar has an unexpected storage");
}

void write(std::string &out, const Value &value, std::size_t depth) {
  if (scalar(value)) {
    write_scalar(out, value);
    return;
  }
  if (value.is_array()) {
    const auto &items = value.array();
    if (items.empty()) {
      out += "[]";
      return;
    }
    bool all_scalars = true;
    for (const auto &item : items)
      all_scalars = all_scalars && scalar(item);
    if (all_scalars) {
      out += "[";
      for (std::size_t index = 0; index < items.size(); ++index) {
        if (index != 0U)
          out += ", ";
        write_scalar(out, items[index]);
      }
      out += "]";
      return;
    }
    out += "[\n";
    for (std::size_t index = 0; index < items.size(); ++index) {
      indent(out, depth + 1U);
      write(out, items[index], depth + 1U);
      out += index + 1U == items.size() ? "\n" : ",\n";
    }
    indent(out, depth);
    out += "]";
    return;
  }
  const auto &members = value.object().members();
  if (members.empty()) {
    out += "{}";
    return;
  }
  out += "{\n";
  for (std::size_t index = 0; index < members.size(); ++index) {
    indent(out, depth + 1U);
    out += quote(members[index].first);
    out += ": ";
    write(out, members[index].second, depth + 1U);
    out += index + 1U == members.size() ? "\n" : ",\n";
  }
  indent(out, depth);
  out += "}";
}

void write_compact(std::string &out, const Value &value) {
  if (scalar(value)) {
    write_scalar(out, value);
    return;
  }
  if (value.is_array()) {
    out += "[";
    const auto &items = value.array();
    for (std::size_t index = 0; index < items.size(); ++index) {
      if (index != 0U)
        out += ",";
      write_compact(out, items[index]);
    }
    out += "]";
    return;
  }
  out += "{";
  const auto &members = value.object().members();
  for (std::size_t index = 0; index < members.size(); ++index) {
    if (index != 0U)
      out += ",";
    out += quote(members[index].first);
    out += ":";
    write_compact(out, members[index].second);
  }
  out += "}";
}

} // namespace

std::string serialize(const Value &value) {
  std::string out;
  write(out, value, 0U);
  out += "\n";
  return out;
}

std::string serialize_compact(const Value &value) {
  std::string out;
  write_compact(out, value);
  return out;
}

Value from_parsed(const json::Value &value) {
  if (std::holds_alternative<std::nullptr_t>(value.storage))
    return Value(nullptr);
  if (const auto *flag = std::get_if<bool>(&value.storage))
    return Value(*flag);
  if (const auto *real = std::get_if<double>(&value.storage)) {
    // Restore exact integers so re-serialized counters and byte sizes do not
    // pick up a fractional or exponent form.
    if (std::isfinite(*real) && std::floor(*real) == *real &&
        std::fabs(*real) < 9007199254740992.0) {
      if (*real < 0.0)
        return Value(static_cast<long long>(*real));
      return Value(static_cast<unsigned long long>(*real));
    }
    return Value(*real);
  }
  if (const auto *text = std::get_if<std::string>(&value.storage))
    return Value(*text);
  if (const auto *items = std::get_if<json::Value::Array>(&value.storage)) {
    Array out;
    out.reserve(items->size());
    for (const auto &item : *items)
      out.push_back(from_parsed(item));
    return Value(std::move(out));
  }
  const auto &members = value.object();
  Object out;
  for (const auto &[key, member] : members)
    out.set(key, from_parsed(member));
  return Value(std::move(out));
}

} // namespace dif::telemetry
