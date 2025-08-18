#include "loader/chunk_feed/tar_chunk_source.h"

#include <absl/log/log.h>
#include <absl/strings/str_cat.h>
#include <archive.h>
#include <archive_entry.h>

#include <stdexcept>

#include "utils/gz.h"

namespace lczero {
namespace training {

TarChunkSource::TarChunkSource(const std::filesystem::path& filename)
    : archive_(archive_read_new()), filename_(filename) {
  if (!archive_) throw std::runtime_error("Failed to create archive reader");
}

TarChunkSource::~TarChunkSource() {
  if (archive_) archive_read_free(archive_);
}

void TarChunkSource::Index() {
  archive_read_support_filter_all(archive_);
  archive_read_support_format_all(archive_);

  int r = archive_read_open_filename(archive_, filename_.data(), 10240);
  if (r != ARCHIVE_OK) {
    archive_read_free(archive_);
    throw std::runtime_error("Failed to open tar file: " +
                             std::string(archive_error_string(archive_)));
  }

  struct archive_entry* entry;
  while (archive_read_next_header(archive_, &entry) == ARCHIVE_OK) {
    const char* pathname = archive_entry_pathname(entry);
    if (!pathname) continue;

    // Skip directories
    if (archive_entry_filetype(entry) == AE_IFDIR) {
      archive_read_data_skip(archive_);
      continue;
    }

    FileEntry file_entry;
    file_entry.offset = archive_read_header_position(archive_);
    file_entry.size = archive_entry_size(entry);

    // Check if file has .gz extension
    std::string_view filename_view(pathname);
    file_entry.is_gzip = filename_view.ends_with(".gz");

    files_.push_back(file_entry);

    // Skip the file data to move to next entry
    archive_read_data_skip(archive_);
  }

  LOG(INFO) << "Read " << files_.size() << " entries from " << filename_;
}

std::string TarChunkSource::GetChunkSortKey() const { return filename_; }

size_t TarChunkSource::GetChunkCount() const { return files_.size(); }

std::string TarChunkSource::GetChunkData(size_t index) {
  if (index >= files_.size())
    throw std::out_of_range("File index out of range");
  const auto& file_entry = files_[index];

  // A filter count > 1 indicates a compressed archive (e.g., tar + gzip).
  // Seeking is not supported on compressed archives.
  if (archive_filter_count(archive_) > 1) {
    throw std::runtime_error("Cannot seek in compressed archive");
  }

  int r = archive_seek_data(archive_, file_entry.offset, SEEK_SET);
  if (r != ARCHIVE_OK) {
    throw std::runtime_error(absl::StrCat(
        "Failed to seek to file offset: ", file_entry.offset,
        " in archive: ", filename_, " - ", archive_error_string(archive_)));
  }

  std::string content(file_entry.size, '\0');
  la_ssize_t bytes_read =
      archive_read_data(archive_, content.data(), file_entry.size);
  if (static_cast<size_t>(bytes_read) != file_entry.size) {
    throw std::runtime_error(absl::StrCat("Failed to read file data: ",
                                          archive_error_string(archive_)));
  }

  // If the file is gzipped, decompress it
  if (file_entry.is_gzip) return GunzipBuffer(content);
  return content;
}

}  // namespace training
}  // namespace lczero