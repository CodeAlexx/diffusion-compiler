#pragma once

#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace dif::json {

struct Value {
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value, std::less<>>;
  using Storage =
      std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

  Storage storage{nullptr};

  bool is_object() const;
  bool is_array() const;
  const Object &object() const;
  const Array &array() const;
  const std::string &string() const;
  double number() const;
  bool boolean() const;
  const Value *find(std::string_view key) const;
};

Value parse(std::string_view input);

} // namespace dif::json
