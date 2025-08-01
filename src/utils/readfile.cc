#include "utils/readfile.h"

#include <fcntl.h>
#include <unistd.h>

#include <stdexcept>

#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"

namespace lczero {
namespace training {

std::string ReadFile(std::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open file: " + std::string(path));
  }

  auto cleanup = absl::MakeCleanup([fd] { close(fd); });

  absl::InlinedVector<std::string, 4> chunks;
  size_t chunk_size = 32 * 1024;  // Start with 16KB chunks

  while (true) {
    std::string chunk(chunk_size, '\0');
    ssize_t bytes_read = read(fd, chunk.data(), chunk_size);

    if (bytes_read == -1) {
      throw std::runtime_error("Failed to read file: " + std::string(path));
    }

    if (bytes_read == 0) break;  // EOF
    chunk.resize(bytes_read);
    chunks.push_back(std::move(chunk));
    if (bytes_read < static_cast<ssize_t>(chunk_size)) break;
    chunk_size *= 2;  // Double chunk size for next read
  }

  if (chunks.empty()) return "";
  if (chunks.size() == 1) return std::move(chunks[0]);
  return absl::StrJoin(chunks, "");
}

}  // namespace training
}  // namespace lczero