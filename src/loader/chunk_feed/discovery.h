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

#include "src/utils/queue.h"

namespace lczero {
namespace training {

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FileDiscovery {
 public:
  using Path = std::filesystem::path;

  enum class Phase {
    kInitialScan,  // File found during initial directory scan
    kNewFile       // File discovered via inotify notification
  };

  struct File {
    Path filepath;
    Phase phase;
  };
  explicit FileDiscovery(size_t queue_capacity = 1000);
  ~FileDiscovery();

  // Returns the output queue for this stage
  Queue<File>* output();

  // Closes the output queue, signaling completion
  void Close();

  // Starts monitoring the directory.
  void AddDirectory(const Path& directory);

 private:
  void MonitorThread();
  void AddWatchRecursive(const Path& path);
  void RemoveWatchRecursive(const Path& path);
  void PerformInitialScan(const Path& directory);
  void ProcessInotifyEvents();
  std::optional<File> ProcessInotifyEvent(const struct inotify_event& event);

  int inotify_fd_;
  // Watch descriptor to directory path.
  absl::flat_hash_map<int, Path> watch_descriptors_;

  Queue<File> output_queue_;

  std::thread monitor_thread_;
  absl::Notification stop_condition_;
};

}  // namespace training
}  // namespace lczero