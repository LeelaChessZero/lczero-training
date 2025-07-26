#pragma once

#include <archive.h>
#include <absl/container/fixed_array.h>

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

  size_t GetFileCount() const;
  absl::FixedArray<char> GetFileContentsByIndex(size_t index);

 private:
  struct FileEntry {
    size_t offset;
    size_t size;
  };

  void ScanTarFile(std::string_view filename);

  archive* archive_ = nullptr;
  std::vector<FileEntry> files_;
  std::string filename_;
};

}  // namespace ice_skate
}  // namespace lczero