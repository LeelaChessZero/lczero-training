#include "loader/chunk_feed/tar_chunk_source.h"

#include <absl/log/log.h>
#include <absl/strings/str_cat.h>

#include <stdexcept>

#include "utils/gz.h"

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

}  // namespace

TarChunkSource::TarChunkSource(const std::filesystem::path& filename)
    : file_(fopen(filename.string().c_str(), "rb")), filename_(filename) {
  if (!file_) throw std::runtime_error("Failed to open tar file");
}

TarChunkSource::~TarChunkSource() {
  if (file_) fclose(file_);
}

void TarChunkSource::Index() {
  while (true) {
    TarHeader header;
    if (fread(&header, sizeof(header), 1, file_) != 1) {
      LOG(WARNING) << "Truncated tar file: " << filename_;
      break;
    }

    if (header.name[0] == '\0') break;  // End of file

    switch (header.typeflag) {
      case '5':  // Directory
        continue;
      case '0':  // Regular file
        break;
      default:
        LOG(WARNING) << "Unsupported tar header type: " << header.typeflag;
        continue;
    }

    std::string_view filename(const_cast<const char*>(header.name.data()));
    const std::filesystem::path filepath = std::filesystem::path(filename);
    const long int offset = ftell(file_);
    const long int size = ParseOctal(header.size);
    const long int new_offset = offset + (size + 511) / 512 * 512;
    fseek(file_, new_offset, SEEK_SET);
    if (new_offset != ftell(file_)) {
      LOG(WARNING) << "Truncated tar file at " << filename
                   << ", expected size: " << size
                   << ", actual size: " << (new_offset - offset);
      break;
    }

    if (filepath.filename() == "LICENSE") continue;
    files_.push_back({offset, size, filepath.extension() == ".gz"});
  }

  LOG(INFO) << "Read " << files_.size() << " entries from " << filename_;
}

std::string TarChunkSource::GetChunkSortKey() const { return filename_; }

size_t TarChunkSource::GetChunkCount() const { return files_.size(); }

std::optional<std::string> TarChunkSource::GetChunkData(size_t index) {
  if (index >= files_.size()) {
    throw std::out_of_range("File index out of range");
  }
  const auto& file_entry = files_[index];
  std::string content(file_entry.size, '\0');
  fseek(file_, file_entry.offset, SEEK_SET);
  fread(content.data(), 1, file_entry.size, file_);
  if (file_entry.is_gzip) {
    try {
      return GunzipBuffer(content);
    } catch (const GunzipError& e) {
      return std::nullopt;
    }
  }
  return content;
}

}  // namespace training
}  // namespace lczero