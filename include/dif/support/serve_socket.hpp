// Persistent-worker protocol over a Unix stream socket, shared by the tools
// that offer --serve SOCKET / --connect SOCKET (difh3infer, difvaedecode).
//
// Request: u32 token count, then per token u32 length + bytes (the tool's
// argv without the program name). Reply: i32 status, u32 length, the
// request's stdout text. A request consisting of the single token
// --shutdown stops the server. One request at a time; the served process
// keeps its prepared execution (resident weights, plans, scratch) between
// requests and refuses a request whose prepare-affecting flags differ.
#pragma once

#include "dif/support/error.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dif::serve {

inline void write_all(int descriptor, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const std::uint8_t *>(data);
  while (bytes != 0U) {
    const auto written = ::write(descriptor, cursor, bytes);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      fail(std::string("socket write failed: ") + std::strerror(errno));
    }
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
}

inline void read_all(int descriptor, void *data, std::size_t bytes) {
  auto *cursor = static_cast<std::uint8_t *>(data);
  while (bytes != 0U) {
    const auto received = ::read(descriptor, cursor, bytes);
    if (received < 0) {
      if (errno == EINTR)
        continue;
      fail(std::string("socket read failed: ") + std::strerror(errno));
    }
    if (received == 0)
      fail("socket closed before the message was complete");
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

inline void write_u32(int descriptor, std::uint32_t value) {
  write_all(descriptor, &value, sizeof(value));
}

inline std::uint32_t read_u32(int descriptor) {
  std::uint32_t value = 0;
  read_all(descriptor, &value, sizeof(value));
  return value;
}

inline void write_blob(int descriptor, const std::string &text) {
  if (text.size() > std::numeric_limits<std::uint32_t>::max())
    fail("socket message exceeds the protocol size");
  write_u32(descriptor, static_cast<std::uint32_t>(text.size()));
  write_all(descriptor, text.data(), text.size());
}

inline std::string read_blob(int descriptor) {
  const auto length = read_u32(descriptor);
  if (length > 64U * 1024U * 1024U)
    fail("socket message exceeds 64 MiB");
  std::string text(length, '\0');
  read_all(descriptor, text.data(), length);
  return text;
}

inline int connect_socket(const std::filesystem::path &socket_path,
                          const char *what) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const auto path = socket_path.string();
  if (path.size() >= sizeof(address.sun_path))
    fail("socket path is too long for AF_UNIX: " + path);
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1U);
  const int descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (descriptor < 0)
    fail(std::string("cannot create socket: ") + std::strerror(errno));
  if (::connect(descriptor, reinterpret_cast<const sockaddr *>(&address),
                sizeof(address)) != 0) {
    const auto reason = std::string(std::strerror(errno));
    ::close(descriptor);
    fail(std::string("cannot connect to ") + what + " " + path + ": " + reason);
  }
  return descriptor;
}

// Client side: send argv, print the reply text, return the status.
inline int run_client(const std::filesystem::path &socket_path,
                      const std::vector<std::string> &arguments,
                      const char *what) {
  const int descriptor = connect_socket(socket_path, what);
  write_u32(descriptor, static_cast<std::uint32_t>(arguments.size()));
  for (const auto &argument : arguments)
    write_blob(descriptor, argument);
  std::int32_t status = 0;
  read_all(descriptor, &status, sizeof(status));
  const auto text = read_blob(descriptor);
  ::close(descriptor);
  std::cout << text << std::flush;
  return static_cast<int>(status);
}

// Server side. `tag` prefixes the lifecycle lines (READY/REQUEST/SHUTDOWN/
// STOPPED), `tool` names the process in error text, `ready_suffix` is
// appended to the READY line. handler(arguments) runs one request with
// std::cout captured into the reply and returns its status.
inline int run_server(
    const std::filesystem::path &socket_path, const std::string &tag,
    const std::string &tool, const std::string &ready_suffix,
    const std::function<int(const std::vector<std::string> &)> &handler) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const auto path = socket_path.string();
  if (path.size() >= sizeof(address.sun_path))
    fail("socket path is too long for AF_UNIX: " + path);
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1U);
  ::unlink(path.c_str());
  const int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listener < 0)
    fail(std::string("cannot create socket: ") + std::strerror(errno));
  if (::bind(listener, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 4) != 0) {
    const auto reason = std::string(std::strerror(errno));
    ::close(listener);
    fail("cannot listen on " + path + ": " + reason);
  }
  std::cout << tag << " READY socket=" << path << " pid=" << ::getpid()
            << (ready_suffix.empty() ? "" : " ") << ready_suffix << "\n"
            << std::flush;
  std::size_t served = 0U;
  for (;;) {
    const int connection = ::accept(listener, nullptr, nullptr);
    if (connection < 0) {
      if (errno == EINTR)
        continue;
      ::close(listener);
      fail(std::string("accept failed: ") + std::strerror(errno));
    }
    std::vector<std::string> arguments;
    std::int32_t status = 0;
    std::string reply;
    try {
      const auto count = read_u32(connection);
      if (count > 4096U)
        fail("request carries too many tokens");
      arguments.reserve(count);
      for (std::uint32_t token = 0U; token < count; ++token)
        arguments.push_back(read_blob(connection));
    } catch (const std::exception &error) {
      std::cerr << tool << ": bad request: " << error.what() << "\n";
      ::close(connection);
      continue;
    }
    if (arguments.size() == 1U && arguments.front() == "--shutdown") {
      reply = tag + " SHUTDOWN served=" + std::to_string(served) + "\n";
      write_all(connection, &status, sizeof(status));
      write_blob(connection, reply);
      ::close(connection);
      break;
    }
    const auto request_start = std::chrono::steady_clock::now();
    std::ostringstream captured;
    auto *previous = std::cout.rdbuf(captured.rdbuf());
    try {
      status = handler(arguments);
    } catch (const std::exception &error) {
      captured << tool << ": " << error.what() << "\n";
      status = 1;
    }
    std::cout.rdbuf(previous);
    ++served;
    reply = captured.str();
    try {
      write_all(connection, &status, sizeof(status));
      write_blob(connection, reply);
    } catch (const std::exception &error) {
      std::cerr << tool << ": reply failed: " << error.what() << "\n";
    }
    ::close(connection);
    std::cout << tag << " REQUEST index=" << served << " status=" << status
              << " wall_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - request_start)
                     .count()
              << "\n"
              << std::flush;
  }
  ::close(listener);
  ::unlink(path.c_str());
  std::cout << tag << " STOPPED served=" << served << "\n" << std::flush;
  return 0;
}

} // namespace dif::serve
