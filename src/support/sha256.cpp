#include "dif/support/sha256.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace dif {
namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::uint32_t rotr(std::uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32U - amount));
}

std::uint32_t load_be32(const std::uint8_t *ptr) {
  return (static_cast<std::uint32_t>(ptr[0]) << 24U) |
         (static_cast<std::uint32_t>(ptr[1]) << 16U) |
         (static_cast<std::uint32_t>(ptr[2]) << 8U) |
         static_cast<std::uint32_t>(ptr[3]);
}

void store_be32(std::uint8_t *ptr, std::uint32_t value) {
  ptr[0] = static_cast<std::uint8_t>(value >> 24U);
  ptr[1] = static_cast<std::uint8_t>(value >> 16U);
  ptr[2] = static_cast<std::uint8_t>(value >> 8U);
  ptr[3] = static_cast<std::uint8_t>(value);
}

} // namespace

class Sha256State {
public:
  void update(std::span<const std::uint8_t> bytes) {
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - total_bytes_)
      fail("SHA-256 input is too large");
    total_bytes_ += bytes.size();
    if (buffered_ != 0U) {
      const auto take = std::min(bytes.size(), buffer_.size() - buffered_);
      std::memcpy(buffer_.data() + buffered_, bytes.data(), take);
      buffered_ += take;
      bytes = bytes.subspan(take);
      if (buffered_ == buffer_.size()) {
        transform(buffer_.data());
        buffered_ = 0U;
      }
    }
    while (bytes.size() >= buffer_.size()) {
      transform(bytes.data());
      bytes = bytes.subspan(buffer_.size());
    }
    if (!bytes.empty()) {
      std::memcpy(buffer_.data(), bytes.data(), bytes.size());
      buffered_ = bytes.size();
    }
  }

  Sha256Digest finish() {
    if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U)
      fail("SHA-256 bit length overflow");
    const auto bit_length = total_bytes_ * 8U;
    buffer_[buffered_++] = 0x80U;
    if (buffered_ > 56U) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                buffer_.end(), 0U);
      transform(buffer_.data());
      buffered_ = 0U;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
              buffer_.begin() + 56, 0U);
    for (unsigned byte = 0; byte < 8U; ++byte)
      buffer_[63U - byte] = static_cast<std::uint8_t>(bit_length >> (byte * 8U));
    transform(buffer_.data());
    Sha256Digest result{};
    for (std::size_t i = 0; i < state_.size(); ++i)
      store_be32(result.data() + i * 4U, state_[i]);
    return result;
  }

private:
  void transform(const std::uint8_t *block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16U; ++i)
      words[i] = load_be32(block + i * 4U);
    for (std::size_t i = 16U; i < words.size(); ++i) {
      const auto s0 = rotr(words[i - 15U], 7U) ^ rotr(words[i - 15U], 18U) ^
                      (words[i - 15U] >> 3U);
      const auto s1 = rotr(words[i - 2U], 17U) ^ rotr(words[i - 2U], 19U) ^
                      (words[i - 2U] >> 10U);
      words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
    }
    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto sum1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
      const auto choose = (e & f) ^ ((~e) & g);
      const auto temp1 = h + sum1 + choose + kRound[i] + words[i];
      const auto sum0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_{};
  std::uint64_t total_bytes_{};
};

Sha256Digest sha256(std::span<const std::uint8_t> bytes) {
  Sha256State state;
  state.update(bytes);
  return state.finish();
}

Sha256Digest sha256_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open file for SHA-256: " + path.string());
  Sha256State state;
  std::array<std::uint8_t, 1024U * 1024U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0)
      state.update(std::span<const std::uint8_t>(
          buffer.data(), static_cast<std::size_t>(count)));
  }
  if (!input.eof())
    fail("cannot read file for SHA-256: " + path.string());
  return state.finish();
}

Sha256Digest sha256_file_prefix(const std::filesystem::path &path,
                                std::uint64_t bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open file for SHA-256: " + path.string());
  Sha256State state;
  std::array<std::uint8_t, 1024U * 1024U> buffer{};
  auto remaining = bytes;
  while (remaining > 0U) {
    const auto requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(requested));
    if (input.gcount() != static_cast<std::streamsize>(requested))
      fail("cannot read requested file prefix for SHA-256: " + path.string());
    state.update(std::span<const std::uint8_t>(buffer.data(), requested));
    remaining -= requested;
  }
  return state.finish();
}

std::string hex_digest(const Sha256Digest &digest) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const auto byte : digest)
    stream << std::setw(2) << static_cast<unsigned>(byte);
  return stream.str();
}

} // namespace dif
