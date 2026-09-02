#pragma once

// Ordered JSON document model shared by every agent-facing telemetry
// document: probe, benchmark, trace, and plan reports. Member order is
// insertion order so a document serializes the same way on every run, and
// integers are carried exactly rather than through double.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace dif::json {
struct Value;
}

namespace dif::telemetry {

class Value;

using Array = std::vector<Value>;

class Object {
public:
  using Member = std::pair<std::string, Value>;

  Object() = default;

  // Replaces an existing member in place or appends a new one, so a builder
  // can revise a field without disturbing the recorded order.
  Object &set(std::string key, Value value);
  const Value *find(std::string_view key) const;
  Value *find(std::string_view key);
  bool empty() const { return members_.empty(); }
  std::size_t size() const { return members_.size(); }
  const std::vector<Member> &members() const { return members_; }

private:
  std::vector<Member> members_;
};

class Value {
public:
  using Storage = std::variant<std::nullptr_t, bool, std::int64_t,
                               std::uint64_t, double, std::string, Array,
                               Object>;

  Value() = default;
  Value(std::nullptr_t) : storage_(nullptr) {}
  Value(bool value) : storage_(value) {}
  Value(int value) : storage_(static_cast<std::int64_t>(value)) {}
  Value(long value) : storage_(static_cast<std::int64_t>(value)) {}
  Value(long long value) : storage_(static_cast<std::int64_t>(value)) {}
  Value(unsigned value) : storage_(static_cast<std::uint64_t>(value)) {}
  Value(unsigned long value) : storage_(static_cast<std::uint64_t>(value)) {}
  Value(unsigned long long value)
      : storage_(static_cast<std::uint64_t>(value)) {}
  Value(double value) : storage_(value) {}
  Value(const char *value) : storage_(std::string(value)) {}
  Value(std::string value) : storage_(std::move(value)) {}
  Value(std::string_view value) : storage_(std::string(value)) {}
  Value(Array value) : storage_(std::move(value)) {}
  Value(Object value) : storage_(std::move(value)) {}

  bool is_null() const;
  bool is_object() const;
  bool is_array() const;
  bool is_string() const;
  bool is_number() const;
  const Object &object() const;
  Object &object();
  const Array &array() const;
  Array &array();
  const std::string &string() const;
  // Numeric view for any of the three numeric storages.
  double number() const;
  bool boolean() const;
  const Storage &storage() const { return storage_; }

private:
  Storage storage_;
};

// Deterministic serialization: two-space indentation, members in insertion
// order, scalars-only arrays on one line, `%.17g` doubles, and null for a
// non-finite measurement so a document is always valid JSON.
std::string serialize(const Value &value);
// Single-line form for JSON-lines sinks; same member order and number text.
std::string serialize_compact(const Value &value);
std::string quote(std::string_view text);
// null for an empty string / absent number, otherwise the value. Kept out
// of line so conditional temporaries never trip the compiler's variant
// initialization analysis.
Value nullable_string(const std::string &text);
Value nullable_number(bool present, double value);
std::string number_text(double value);

// Converts a parsed document (whose members are key-sorted) into the ordered
// model so one tool can embed another tool's JSON output verbatim.
Value from_parsed(const json::Value &value);

} // namespace dif::telemetry
