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

// Message types for FilePathProvider output.
enum class FilePathProviderMessageType {
  kFile,                // File discovered (initial scan or inotify)
  kInitialScanComplete  // Initial scan is complete (empty filepath)
};

// Output type for FilePathProvider.
struct FilePathProviderFile {
  std::filesystem::path filepath;
  FilePathProviderMessageType message_type;
};

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FilePathProvider : public SingleOutputStage<FilePathProviderFile> {
 public:
  using Path = std::filesystem::path;
  using MessageType = FilePathProviderMessageType;
  using File = FilePathProviderFile;

  explicit FilePathProvider(const FilePathProviderConfig& config);
  ~FilePathProvider();

  // Starts monitoring the directory
  void Start() override;

  // Closes the output queue, signaling completion
  void Stop() override;

  // Returns current metrics and clears them.
  StageMetricProto FlushMetrics() override;

  // FilePathProvider has no inputs.
  void SetStages(absl::Span<QueueBase* const> inputs) override;

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

  Path directory_;  // Directory to monitor
  Queue<File>::Producer producer_;

  std::thread monitor_thread_;
  absl::Notification stop_condition_;

  LoadMetricUpdater load_metric_updater_;
};

}  // namespace training
}  // namespace lczero
