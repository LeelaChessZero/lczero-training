#pragma once

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>
#include <absl/synchronization/notification.h>
#include <sys/inotify.h>

#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/utils/metrics/additive_metric.h"
#include "src/utils/metrics/load_metric.h"
#include "src/utils/metrics/statistics_metric.h"
#include "src/utils/queue.h"

namespace lczero {
namespace training {

// Metrics for FilePathProvider performance monitoring.
struct FilePathProviderMetrics {
  AdditiveMetric<size_t> total_files_discovered;
  LoadMetric load;
  StatisticsMetric<size_t, true> queue_size;
};

// Configuration options for FilePathProvider
struct FilePathProviderOptions {
  size_t queue_capacity = 16;
  std::filesystem::path directory;
};

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FilePathProvider {
 public:
  using Path = std::filesystem::path;

  enum class MessageType {
    kFile,                // File discovered (initial scan or inotify)
    kInitialScanComplete  // Initial scan is complete (empty filepath)
  };

  struct File {
    Path filepath;
    MessageType message_type;
  };
  explicit FilePathProvider(const FilePathProviderOptions& options);
  ~FilePathProvider();

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
  void ScanDirectoryWithWatch(const Path& directory);
  void ProcessWatchEventsForNewItems(const std::vector<Path>& known_files);
  void ProcessInotifyEvents(Queue<File>::Producer& producer);
  std::optional<File> ProcessInotifyEvent(const struct inotify_event& event);

  int inotify_fd_;
  // Watch descriptor to directory path.
  absl::flat_hash_map<int, Path> watch_descriptors_;

  Queue<File> output_queue_;
  Path directory_;  // Directory to monitor
  Queue<File>::Producer producer_;

  std::thread monitor_thread_;
  absl::Notification stop_condition_;

  mutable absl::Mutex metrics_mutex_;
  FilePathProviderMetrics metrics_ ABSL_GUARDED_BY(metrics_mutex_);
  LoadMetricUpdater load_metric_updater_ ABSL_GUARDED_BY(metrics_mutex_);
};

}  // namespace training
}  // namespace lczero