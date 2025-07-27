#pragma once

namespace lczero {
namespace ice_skate {

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FileDiscovery {
 public:
  struct File {
    // The directory is the string that was passed to AddDirectory.
    std::string directory;
    // The filename is the path relative to the directory.
    std::string filename;
  };
  using Token = size_t;
  using Observer = std::function<void(std::span<const File>)>;

  Token RegisterObserver(Observer observer);
  void UnregisterObserver(Token token);

  // Starts monitoring the directory. Also returns a list of files that
  // already exist in the directory at the time of starting.
  std::vector<File> AddDirectory(const std::string& directory);
};

}  // namespace ice_skate
}  // namespace lczero