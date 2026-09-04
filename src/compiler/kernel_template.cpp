#include "dif/compiler/kernel_template.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace dif::compiler {

// Generated at build time from src/compiler/kernels/*.cu by
// cmake/embed_kernels.cmake: name -> template text.
const std::unordered_map<std::string_view, std::string_view> &
embedded_kernel_templates();

std::string_view kernel_template(std::string_view name) {
  const auto &table = embedded_kernel_templates();
  const auto found = table.find(name);
  if (found == table.end())
    fail("no embedded kernel template named '" + std::string(name) +
         "' (expected src/compiler/kernels/" + std::string(name) + ".cu)");
  return found->second;
}

std::vector<std::string_view> kernel_template_names() {
  std::vector<std::string_view> names;
  for (const auto &entry : embedded_kernel_templates())
    names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

std::string render_kernel_template(
    std::string_view name, const std::vector<KernelTemplateValue> &values) {
  const auto text = kernel_template(name);
  std::unordered_map<std::string_view, const std::string *> lookup;
  for (const auto &[key, value] : values) {
    if (key.empty())
      fail("kernel template '" + std::string(name) + "': empty placeholder name");
    if (!lookup.emplace(key, &value).second)
      fail("kernel template '" + std::string(name) + "': placeholder '" +
           std::string(key) + "' supplied twice");
  }
  std::unordered_set<std::string_view> used;
  std::string out;
  out.reserve(text.size() + 64U);
  std::size_t index = 0U;
  while (index < text.size()) {
    const auto start = text.find("${", index);
    if (start == std::string_view::npos) {
      out.append(text.substr(index));
      break;
    }
    out.append(text.substr(index, start - index));
    const auto end = text.find('}', start + 2U);
    if (end == std::string_view::npos)
      fail("kernel template '" + std::string(name) +
           "': unterminated placeholder");
    const auto key = text.substr(start + 2U, end - start - 2U);
    const auto found = lookup.find(key);
    if (found == lookup.end())
      fail("kernel template '" + std::string(name) + "': placeholder '" +
           std::string(key) + "' has no value");
    out.append(*found->second);
    used.insert(key);
    index = end + 1U;
  }
  for (const auto &[key, value] : values)
    if (!used.contains(key))
      fail("kernel template '" + std::string(name) + "': value for '" +
           std::string(key) + "' is not used by the template");
  return out;
}

std::string render_kernel_template(
    std::string_view name, std::initializer_list<KernelTemplateValue> values) {
  return render_kernel_template(
      name, std::vector<KernelTemplateValue>(values.begin(), values.end()));
}

} // namespace dif::compiler
