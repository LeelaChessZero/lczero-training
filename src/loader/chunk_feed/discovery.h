#pragma once

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/notification.h>
#include <sys/inotify.h>

#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace lczero {
namespace training {

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FileDiscovery {
 public:
  using Path = std::filesystem::path;

  struct File {
    Path filepath;
  };
  using ObeserverToken = size_t;
  using Observer = std::function<void(std::span<const File>)>;

  ObeserverToken RegisterObserver(Observer observer);
  void UnregisterObserver(ObeserverToken token);

  FileDiscovery();
  ~FileDiscovery();

  // Starts monitoring the directory. Calls initial_observer with existing files
  // in batches. Newly discovered files will be reported to registered
  // observers.
  void AddDirectory(const Path& directory, Observer initial_observer);

 private:
  void MonitorThread();
  void AddWatchRecursive(const Path& path);
  void RemoveWatchRecursive(const Path& path);
  void PerformInitialScan(const Path& directory, Observer observer);
  void ProcessInotifyEvents();
  std::optional<File> ProcessInotifyEvent(const struct inotify_event& event);
  void NotifyObservers(std::span<const File> files);

  int inotify_fd_;
  ObeserverToken next_token_ = 1;
  // Watch descriptor to directory path.
  absl::flat_hash_map<int, Path> watch_descriptors_;
  absl::flat_hash_map<ObeserverToken, Observer> observers_;

  std::thread monitor_thread_;
  absl::Notification stop_condition_;
};

}  // namespace training
}  // namespace lczero