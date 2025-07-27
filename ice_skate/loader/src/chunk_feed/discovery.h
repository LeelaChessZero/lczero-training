#pragma once

#include <absl/base/thread_annotations.h>
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

  Token RegisterObserver(Observer observer) ABSL_LOCKS_EXCLUDED(mutex_);
  void UnregisterObserver(Token token) ABSL_LOCKS_EXCLUDED(mutex_);

  FileDiscovery();
  ~FileDiscovery();

  // Starts monitoring the directory. Calls initial_observer with existing files
  // in batches. Newly discovered files will be reported to registered
  // observers.
  void AddDirectory(const std::string& directory, Observer initial_observer)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::Condition stop_condition_;
  absl::flat_hash_map<Token, Observer> observers_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<int, std::string> watch_descriptors_
      ABSL_GUARDED_BY(mutex_);  // wd -> directory_path
  absl::flat_hash_map<std::string, int> directory_watches_
      ABSL_GUARDED_BY(mutex_);
  std::thread monitor_thread_;
  int inotify_fd_;
  bool should_stop_ ABSL_GUARDED_BY(mutex_);
  Token next_token_ ABSL_GUARDED_BY(mutex_);

  void MonitorThread();
  void AddWatchRecursive(const std::string& path)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void RemoveWatchRecursive(const std::string& path)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  std::vector<File> ProcessInotifyEvents() ABSL_LOCKS_EXCLUDED(mutex_);
  void NotifyObservers(std::span<const File> files) ABSL_LOCKS_EXCLUDED(mutex_);
};

}  // namespace ice_skate
}  // namespace lczero