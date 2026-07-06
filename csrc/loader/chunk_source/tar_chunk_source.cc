#include "loader/chunk_source/tar_chunk_source.h"

#include <absl/log/log.h>
#include <absl/strings/str_cat.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

#include "trainingdata/trainingdata_v6.h"
#include "utils/gz.h"
#include "utils/trace.h"

namespace lczero {
namespace training {
namespace {
struct TarHeader {
  std::array<char, 100> name;
  std::array<uint8_t, 8> mode;
  std::array<uint8_t, 8> uid;
  std::array<uint8_t, 8> gid;
  std::array<uint8_t, 12> size;
  std::array<uint8_t, 12> mtime;
  std::array<uint8_t, 8> chksum;
  uint8_t typeflag;
  std::array<uint8_t, 100> linkname;
  std::array<uint8_t, 6> magic;
  std::array<uint8_t, 2> version;
  std::array<uint8_t, 32> uname;
  std::array<uint8_t, 32> gname;
  std::array<uint8_t, 8> devmajor;
  std::array<uint8_t, 8> devminor;
  std::array<uint8_t, 155> prefix;
  std::array<uint8_t, 12> padding;
};
static_assert(sizeof(TarHeader) == 512, "TarHeader must be exactly 512 bytes");

uint64_t ParseOctal(const std::array<uint8_t, 12>& octal) {
  uint64_t value = 0;
  for (uint8_t digit : octal) {
    if (!digit) break;
    value = (value << 3) + (digit - '0');
  }
  return value;
}

bool ReadExact(int fd, off_t offset, void* buffer, size_t size) {
  char* out = static_cast<char*>(buffer);
  size_t read_total = 0;
  while (read_total < size) {
    const ssize_t read_now =
        pread(fd, out + read_total, size - read_total, offset + read_total);
    if (read_now <= 0) return false;
    read_total += static_cast<size_t>(read_now);
  }
  return true;
}

std::optional<std::string> ReadGzipPrefix(int fd, off_t offset, size_t size,
                                          size_t max_bytes) {
  if (max_bytes == 0) return std::string();

  z_stream strm = {};
  if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
    return std::nullopt;
  }

  constexpr size_t kChunkSize = 16384;
  std::array<uint8_t, kChunkSize> input_buffer;
  std::array<char, kChunkSize> output_buffer;

  std::string output;
  output.reserve(std::min<size_t>(max_bytes, kChunkSize));

  size_t remaining = size;
  off_t current_offset = offset;
  bool finished = false;

  while (remaining > 0 && !finished && output.size() < max_bytes) {
    const size_t to_read = std::min(remaining, kChunkSize);
    if (!ReadExact(fd, current_offset, input_buffer.data(), to_read)) {
      inflateEnd(&strm);
      return std::nullopt;
    }
    remaining -= to_read;
    current_offset += static_cast<off_t>(to_read);

    strm.next_in = reinterpret_cast<Bytef*>(input_buffer.data());
    strm.avail_in = static_cast<uInt>(to_read);

    while (strm.avail_in > 0 && output.size() < max_bytes) {
      strm.next_out = reinterpret_cast<Bytef*>(output_buffer.data());
      strm.avail_out = kChunkSize;

      const int ret = inflate(&strm, Z_NO_FLUSH);
      if (ret == Z_STREAM_ERROR || ret == Z_NEED_DICT || ret == Z_DATA_ERROR ||
          ret == Z_MEM_ERROR) {
        inflateEnd(&strm);
        return std::nullopt;
      }

      const size_t produced = kChunkSize - strm.avail_out;
      const size_t to_copy = std::min(produced, max_bytes - output.size());
      output.append(output_buffer.data(), to_copy);

      if (ret == Z_STREAM_END) {
        finished = true;
        break;
      }
    }
  }

  inflateEnd(&strm);
  return output;
}

}  // namespace

TarChunkSource::TarChunkSource(
    const std::filesystem::path& filename,
    ChunkSourceLoaderConfig::FrameFormat frame_format)
    : path_(filename),
      filename_(filename.filename().string()),
      frame_format_(frame_format) {
  fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) {
    throw std::runtime_error(
        absl::StrCat("Failed to open tar file: ", path_.string(), ": ",
                     std::strerror(errno)));
  }
  // Perform indexing during construction.
  Index();
}

TarChunkSource::~TarChunkSource() { Close(); }

void TarChunkSource::Close() {
  if (fd_ >= 0 && close(fd_) != 0) {
    PLOG(WARNING) << "Failed to close tar file descriptor for " << path_;
  }
  fd_ = -1;
}

std::string TarChunkSource::GetChunkSortKey() const { return filename_; }

void TarChunkSource::Index() {
  assert(files_.empty());

  off_t offset = 0;

  while (true) {
    TarHeader header;
    if (!ReadExact(fd_, offset, &header, sizeof(header))) {
      LOG(WARNING) << "Truncated tar file: " << filename_;
      break;
    }
    offset += sizeof(header);

    if (header.name[0] == '\0') break;  // End of file.

    switch (header.typeflag) {
      case '5':  // Directory
        continue;
      case '0':  // Regular file
        break;
      default:
        LOG(WARNING) << "Unsupported tar header type: " << header.typeflag;
        continue;
    }

    std::string_view fname(const_cast<const char*>(header.name.data()));
    const std::filesystem::path filepath = std::filesystem::path(fname);
    const off_t file_offset = offset;
    const size_t size = ParseOctal(header.size);
    const size_t padded_size = ((size + 511) / 512) * 512;
    offset = file_offset + static_cast<off_t>(padded_size);

    if (filepath.filename() == "LICENSE") continue;
    files_.push_back({file_offset, size, filepath.extension() == ".gz"});
  }

  LOG(INFO) << "Read " << files_.size() << " entries from " << filename_;
}

size_t TarChunkSource::GetChunkCount() const { return files_.size(); }

std::optional<std::vector<FrameType>> TarChunkSource::GetChunkData(
    size_t index) {
  LCTRACE_FUNCTION_SCOPE;
  if (index >= files_.size()) {
    throw std::out_of_range("File index out of range");
  }
  const auto& file_entry = files_[index];
  std::string content(file_entry.size, '\0');
  if (!ReadExact(fd_, file_entry.offset, content.data(), file_entry.size)) {
    return std::nullopt;
  }
  if (file_entry.is_gzip) {
    try {
      content = GunzipBuffer(content);
    } catch (const GunzipError& e) {
      return std::nullopt;
    }
  }
  if (content.empty()) return std::nullopt;

  const size_t input_size =
      frame_format_ == ChunkSourceLoaderConfig::V7TrainingData
          ? sizeof(V7TrainingData)
          : sizeof(V6TrainingData);
  if (content.size() % input_size != 0) {
    LOG(WARNING) << "Chunk " << index << " from " << filename_ << " size "
                 << content.size() << " is not a multiple of input frame size "
                 << input_size;
    return std::nullopt;
  }

  const size_t num_frames = content.size() / input_size;
  std::vector<V7TrainingData> result(num_frames);

  if (frame_format_ == ChunkSourceLoaderConfig::V7TrainingData) {
    std::memcpy(result.data(), content.data(), content.size());
  } else {
    const auto* v6_data =
        reinterpret_cast<const V6TrainingData*>(content.data());
    for (size_t i = 0; i < num_frames; ++i) {
      std::memcpy(&result[i], &v6_data[i], sizeof(V6TrainingData));
    }
  }
  return result;
}

std::optional<std::string> TarChunkSource::GetChunkPrefix(size_t index,
                                                          size_t max_bytes) {
  if (index >= files_.size()) {
    throw std::out_of_range("File index out of range");
  }
  const auto& file_entry = files_[index];
  if (file_entry.is_gzip) {
    return ReadGzipPrefix(fd_, file_entry.offset, file_entry.size, max_bytes);
  }

  const size_t to_read = std::min(file_entry.size, max_bytes);
  std::string content(to_read, '\0');
  if (!ReadExact(fd_, file_entry.offset, content.data(), to_read)) {
    return std::nullopt;
  }
  return content;
}

}  // namespace training
}  // namespace lczero
