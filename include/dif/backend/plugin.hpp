#pragma once

#include "dif/runtime/executor.hpp"

#include <filesystem>
#include <memory>

namespace dif::backend {

std::unique_ptr<runtime::Executor>
make_plugin_executor(const std::filesystem::path &library, int device = 0);

} // namespace dif::backend
