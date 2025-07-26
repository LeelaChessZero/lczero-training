#pragma once

#include <archive.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lczero {
namespace ice_skate {

class TarFile {
 public:
  TarFile(const std::string_view filename);
  ~TarFile();

 private:
  struct FileEntry {
    size_t offset;
  };

  void ScanTarFile(std::string_view filename);

  archive* archive_ = nullptr;
  std::vector<FileEntry> files_;
};

}  // namespace ice_skate
}  // namespace lczero