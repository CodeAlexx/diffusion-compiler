#pragma once

#include "dif/ir/ir.hpp"
#include "dif/support/sha256.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace dif::ir {

std::vector<std::uint8_t> encode(const Program &program);
Program decode(std::span<const std::uint8_t> bytes);
void write_file(const Program &program, const std::filesystem::path &path);
Program read_file(const std::filesystem::path &path);
Sha256Digest fingerprint(const Program &program);

} // namespace dif::ir
