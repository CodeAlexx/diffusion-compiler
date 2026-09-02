#pragma once

#include <string_view>

namespace dif::build {

std::string_view compiler_name();
std::string_view compiler_version();
std::string_view compiler_revision();

} // namespace dif::build
