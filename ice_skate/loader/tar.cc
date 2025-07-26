#include "tar.h"

#include <archive.h>
#include <archive_entry.h>

#include <stdexcept>

#include <absl/log/log.h>

namespace lczero {
namespace ice_skate {

TarFile::TarFile(const std::string_view filename)
    : archive_(archive_read_new()) {
  if (!archive_) throw std::runtime_error("Failed to create archive reader");
  ScanTarFile(filename);
}

TarFile::~TarFile() {
  if (archive_) archive_read_free(archive_);
}

void TarFile::ScanTarFile(std::string_view filename) {
  archive_read_support_filter_all(archive_);
  archive_read_support_format_all(archive_);

  int r = archive_read_open_filename(archive_, filename.data(), 10240);
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

    files_.push_back(file_entry);

    // Skip the file data to move to next entry
    archive_read_data_skip(archive_);
  }

  LOG(INFO) << "Read " << files_.size() << " entries from " << filename;
}

}  // namespace ice_skate
}  // namespace lczero