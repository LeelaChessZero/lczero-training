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
  struct File {
    // Full path to the file
    std::string filepath;
  };
  using Token = size_t;
  using Observer = std::function<void(std::span<const File>)>;

  Token RegisterObserver(Observer observer);
  void UnregisterObserver(Token token);

  FileDiscovery();
  ~FileDiscovery();

  // Starts monitoring the directory. Calls initial_observer with existing files
  // in batches. Newly discovered files will be reported to registered observers.
  void AddDirectory(const std::string& directory, Observer initial_observer);


 private:
  mutable absl::Mutex mutex_;
  absl::Condition stop_condition_;
  absl::flat_hash_map<Token, Observer> observers_;
  absl::flat_hash_map<int, std::string> watch_descriptors_;  // wd -> directory_path
  absl::flat_hash_map<std::string, int> directory_watches_;
  std::thread monitor_thread_;
  int inotify_fd_;
  bool should_stop_;
  Token next_token_;

  void MonitorThread();
  void AddWatchRecursive(const std::string& path);
  void RemoveWatchRecursive(const std::string& path);
  std::vector<File> ProcessInotifyEvents();
  void NotifyObservers(std::span<const File> files);
};

}  // namespace ice_skate
}  // namespace lczero