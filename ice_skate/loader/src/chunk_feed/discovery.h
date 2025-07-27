#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>
#include <sys/inotify.h>

#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace lczero {
namespace ice_skate {

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FileDiscovery {
 public:
  enum class FileType {
    kInitial,     // File existed when AddDirectory was called
    kDiscovered   // File was discovered via inotify events
  };

  struct File {
    // Index to the directory (use GetDirectory(idx) to get path)
    size_t directory_idx;
    // The filename is the path relative to the directory.
    std::string filename;
    // Whether this was an initial file or discovered later
    FileType type;
  };
  using Token = size_t;
  using Observer = std::function<void(std::span<const File>)>;

  Token RegisterObserver(Observer observer);
  void UnregisterObserver(Token token);

  FileDiscovery();
  ~FileDiscovery();

  // Starts monitoring the directory. Calls observers with existing files
  // in batches. Returns the directory index for this directory.
  size_t AddDirectory(const std::string& directory);

  // Returns the directory path for the given index.
  const std::string& GetDirectory(size_t idx) const;

 private:
  mutable absl::Mutex mutex_;
  absl::Condition stop_condition_;
  absl::flat_hash_map<Token, Observer> observers_;
  absl::flat_hash_map<int, std::string> watch_descriptors_;  // wd -> directory_path
  absl::flat_hash_map<std::string, int> directory_watches_;
  std::vector<std::string> directories_;  // directory_idx -> directory_path
  std::thread monitor_thread_;
  int inotify_fd_;
  bool should_stop_;
  Token next_token_;

  void MonitorThread();
  void AddWatchRecursive(const std::string& path, size_t directory_idx);
  void RemoveWatchRecursive(const std::string& path);
  std::vector<File> ProcessInotifyEvents();
  void NotifyObservers(std::span<const File> files);
};

}  // namespace ice_skate
}  // namespace lczero