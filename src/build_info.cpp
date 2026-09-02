#include "dif/build_info.hpp"

namespace dif::build {

std::string_view compiler_name() { return "diffusion-compiler"; }

std::string_view compiler_version() { return DIF_COMPILER_VERSION; }

std::string_view compiler_revision() { return DIF_COMPILER_REVISION; }

} // namespace dif::build
