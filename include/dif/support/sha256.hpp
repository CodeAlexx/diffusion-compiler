#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace dif {

using Sha256Digest = std::array<std::uint8_t, 32>;

Sha256Digest sha256(std::span<const std::uint8_t> bytes);
Sha256Digest sha256_file(const std::filesystem::path &path);
Sha256Digest sha256_file_prefix(const std::filesystem::path &path,
                                std::uint64_t bytes);
std::string hex_digest(const Sha256Digest &digest);

} // namespace dif
