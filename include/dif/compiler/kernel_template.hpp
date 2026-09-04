#pragma once
// Kernel templates: CUDA kernel bodies live as real source files under
// src/compiler/kernels/*.cu, embedded into the binary at build time
// (cmake/embed_kernels.cmake). An emitter renders one by name, substituting
// `${placeholder}` occurrences with the operation's concrete values (shapes,
// literals, optional code fragments). Rendering fails closed: a placeholder
// with no value, or a value for a placeholder the template does not use, is
// an error — so a template and its emitter cannot drift apart silently.
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dif::compiler {

using KernelTemplateValue = std::pair<std::string_view, std::string>;

// The embedded template text for `name` (the file stem under
// src/compiler/kernels/, e.g. "silu" for silu.cu). Throws dif::Error when no
// such template was embedded.
std::string_view kernel_template(std::string_view name);

// Names of every embedded template, sorted.
std::vector<std::string_view> kernel_template_names();

// Render `name` with the given placeholder values.
std::string render_kernel_template(
    std::string_view name,
    std::initializer_list<KernelTemplateValue> values);
std::string render_kernel_template(std::string_view name,
                                   const std::vector<KernelTemplateValue> &values);

} // namespace dif::compiler
