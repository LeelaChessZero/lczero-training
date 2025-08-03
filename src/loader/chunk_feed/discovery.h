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

// Configuration options for FileDiscovery
struct FileDiscoveryOptions {
  size_t queue_capacity = 16;
  std::filesystem::path directory;
};

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FileDiscovery {
 public:
  using Path = std::filesystem::path;

  enum class Phase {
    kInitialScan,          // File found during initial directory scan
    kInitialScanComplete,  // Initial scan is complete (empty filename)
    kNewFile               // File discovered via inotify notification
  };

  struct File {
    Path filepath;
    Phase phase;
  };
  explicit FileDiscovery(const FileDiscoveryOptions& options);
  ~FileDiscovery();

  // Returns the output queue for this stage
  Queue<File>* output();

  // Closes the output queue, signaling completion
  void Close();

 private:
  // Starts monitoring the directory.
  void AddDirectory(const Path& directory);

  void MonitorThread();
  void AddWatchRecursive(const Path& path);
  void RemoveWatchRecursive(const Path& path);
  void PerformInitialScan(const Path& directory);
  void ProcessInotifyEvents(Queue<File>::Producer& producer);
  std::optional<File> ProcessInotifyEvent(const struct inotify_event& event);

  int inotify_fd_;
  // Watch descriptor to directory path.
  absl::flat_hash_map<int, Path> watch_descriptors_;

  Queue<File> output_queue_;
  Queue<File>::Producer producer_;

  std::thread monitor_thread_;
  absl::Notification stop_condition_;
};

}  // namespace training
}  // namespace lczero