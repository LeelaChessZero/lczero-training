#pragma once

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>
#include <absl/synchronization/notification.h>
#include <sys/inotify.h>

#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/metrics/printer.h"
#include "utils/metrics/statistics_metric.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FilePathProvider : public Stage {
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
  explicit FilePathProvider(const FilePathProviderConfig& config,
                            const StageList& existing_stages = {});
  ~FilePathProvider();

  // Returns the output queue for this stage
  Queue<File>* output();

  // Starts monitoring the directory
  void Start() override;

  // Closes the output queue, signaling completion
  void Stop() override;

  // Returns current metrics and clears them.
  StageMetricProto FlushMetrics() override;

  QueueBase* GetOutput(std::string_view name = "") override;

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

  LoadMetricUpdater load_metric_updater_;
};

}  // namespace training
}  // namespace lczero
