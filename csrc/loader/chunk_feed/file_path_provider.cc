#include "loader/chunk_feed/file_path_provider.h"

#include <absl/cleanup/cleanup.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>
#include <sys/epoll.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "proto/training_config.pb.h"

namespace lczero {
namespace training {

FilePathProvider::FilePathProvider(const FilePathProviderConfig& config)
    : output_queue_(config.queue_capacity()),
      directory_(config.directory()),
      producer_(output_queue_.CreateProducer()),
      load_metric_updater_(metrics_.mutable_load()) {
  LOG(INFO) << "Starting FilePathProvider for directory: "
            << config.directory();
  inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
  CHECK_NE(inotify_fd_, -1)
      << "Failed to initialize inotify: " << strerror(errno);
  monitor_thread_ = std::thread(&FilePathProvider::MonitorThread, this);
}

FilePathProvider::~FilePathProvider() {
  Close();
  if (inotify_fd_ != -1) close(inotify_fd_);
}

Queue<FilePathProvider::File>* FilePathProvider::output() {
  return &output_queue_;
}

void FilePathProvider::Close() {
  // First stop all watches
  for (const auto& [wd, path] : watch_descriptors_) {
    inotify_rm_watch(inotify_fd_, wd);
  }
  watch_descriptors_.clear();

  // Then stop the thread
  stop_condition_.Notify();
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }

  // Finally close the producer to close the queue
  producer_.Close();
}

FilePathProviderMetricsProto FilePathProvider::FlushMetrics() {
  absl::MutexLock lock(&metrics_mutex_);
  load_metric_updater_.Flush();
  FilePathProviderMetricsProto result = std::move(metrics_);
  metrics_.Clear();
  return result;
}

void FilePathProvider::UpdateMetricsForDiscoveredFiles(size_t file_count) {
  absl::MutexLock lock(&metrics_mutex_);
  AddSample(*metrics_.mutable_queue_size(), output_queue_.Size());
  metrics_.set_total_files_discovered(metrics_.total_files_discovered() +
                                      file_count);
}

void FilePathProvider::AddDirectory(const Path& directory) {
  ScanDirectoryWithWatch(directory);

  // Signal that initial scan is complete
  LOG(INFO) << "FilePathProvider initial scan complete";
  UpdateMetricsForDiscoveredFiles(0);
  producer_.Put({{.filepath = Path{},
                  .message_type = MessageType::kInitialScanComplete}});
}

void FilePathProvider::ScanDirectoryWithWatch(const Path& directory) {
  // Step 1: Set up watch first
  int wd = inotify_add_watch(inotify_fd_, directory.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << directory;
  watch_descriptors_[wd] = directory;

  // Step 2: Scan directory non-recursively, remembering files and subdirs
  std::vector<Path> files;
  std::vector<Path> subdirectories;
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(directory, ec);
  CHECK(!ec) << "Failed to iterate directory " << directory << ": "
             << ec.message();

  for (const auto& entry : iterator) {
    if (entry.is_regular_file(ec) && !ec) {
      files.push_back(entry.path());
    } else if (entry.is_directory(ec) && !ec) {
      subdirectories.push_back(entry.path());
    }
  }

  // Send notifications for discovered files
  constexpr size_t kBatchSize = 10000;
  std::vector<File> batch;
  batch.reserve(kBatchSize);

  auto flush_batch = [&]() {
    if (batch.empty()) return;
    UpdateMetricsForDiscoveredFiles(batch.size());
    producer_.Put(batch);
    batch.clear();
  };

  for (const auto& filepath : files) {
    batch.push_back(
        {.filepath = filepath.string(), .message_type = MessageType::kFile});
    if (batch.size() >= kBatchSize) flush_batch();
  }

  // Step 3: Read from watch descriptor, skipping already discovered files
  ProcessWatchEventsForNewItems(files);

  // Step 4: Clean the files vector to save memory
  files.clear();

  if (stop_condition_.HasBeenNotified()) return;

  // Step 5: Recursively call for subdirectories
  for (const auto& subdir : subdirectories) {
    ScanDirectoryWithWatch(subdir);
  }

  // Flush any remaining files
  flush_batch();
}

void FilePathProvider::ProcessWatchEventsForNewItems(
    const std::vector<Path>& known_files) {
  // Create a set for fast lookup of already discovered files
  absl::flat_hash_set<std::string> known_file_set;
  for (const auto& file : known_files) {
    known_file_set.insert(file.string());
  }

  // Process any events that may have occurred during scanning
  std::array<char, 4096> buffer;
  std::vector<File> new_files;

  while (true) {
    ssize_t length = read(inotify_fd_, buffer.data(), buffer.size());
    if (length <= 0) break;  // No more events to process

    ssize_t offset = 0;
    while (offset < length) {
      const struct inotify_event* event =
          reinterpret_cast<const struct inotify_event*>(buffer.data() + offset);

      // Only process file creation/write events, skip already known files
      if ((event->mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) && event->len > 0) {
        const Path directory(watch_descriptors_.at(event->wd));
        Path filepath = directory / event->name;

        // Only add if we haven't seen this file before
        if (!known_file_set.contains(filepath.string())) {
          new_files.push_back({.filepath = filepath.string(),
                               .message_type = MessageType::kFile});
        }
      }

      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  // Send notifications for any new files discovered through watch events
  if (!new_files.empty()) {
    UpdateMetricsForDiscoveredFiles(new_files.size());
    producer_.Put(new_files);
  }
}

void FilePathProvider::AddWatchRecursive(const Path& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(inotify_fd_, path.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << path;
  watch_descriptors_[wd] = path;

  // Recursively add watches for subdirectories
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(path, ec);
  CHECK(!ec) << "Failed to iterate directory " << path << ": " << ec.message();

  for (const auto& entry : iterator) {
    if (!entry.is_directory(ec) || ec) continue;
    AddWatchRecursive(entry.path().string());
  }
}

void FilePathProvider::RemoveWatchRecursive(const Path& base) {
  absl::erase_if(watch_descriptors_, [&](const auto& pair) {
    const auto& [wd, path] = pair;
    const auto mismatch_iter = absl::c_mismatch(base, path).first;
    // If path is not a subdirectory (or equal) of base, skip.
    if (mismatch_iter != base.end()) return false;
    inotify_rm_watch(inotify_fd_, wd);
    return true;
  });
}

void FilePathProvider::MonitorThread() {
  {
    absl::MutexLock lock(&metrics_mutex_);
    load_metric_updater_.LoadStart();
  }
  // Perform directory scanning in background thread
  AddDirectory(directory_);

  int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  CHECK_NE(epoll_fd, -1) << "Failed to create epoll fd";
  absl::Cleanup epoll_cleanup([epoll_fd]() { close(epoll_fd); });

  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = inotify_fd_;
  CHECK_EQ(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd_, &event), 0)
      << "Failed to add inotify fd to epoll";

  while (true) {
    {
      absl::MutexLock lock(&metrics_mutex_);
      load_metric_updater_.LoadStop();
    }
    if (stop_condition_.WaitForNotificationWithTimeout(
            absl::Milliseconds(50))) {
      break;  // Exit if stop condition is notified
    }
    {
      absl::MutexLock lock(&metrics_mutex_);
      load_metric_updater_.LoadStart();
    }

    struct epoll_event event;
    int nfds = epoll_wait(epoll_fd, &event, 1, 0);  // Non-blocking check
    CHECK_NE(nfds, -1) << "epoll_wait failed: " << strerror(errno);
    if (nfds == 0) continue;  // No events.

    do {
      assert(nfds == 1 && event.data.fd == inotify_fd_);
      ProcessInotifyEvents(producer_);
      nfds = epoll_wait(epoll_fd, &event, 1, 0);
    } while (nfds > 0);
  }
}

void FilePathProvider::ProcessInotifyEvents(Queue<File>::Producer& producer) {
  constexpr size_t kNotifyBatchSize = 10000;
  std::vector<File> files;
  std::array<char, 4096> buffer;

  auto flush_batch = [&]() {
    if (files.empty()) return;
    UpdateMetricsForDiscoveredFiles(files.size());
    producer.Put(files);
    files.clear();
  };

  while (true) {
    ssize_t length = read(inotify_fd_, buffer.data(), buffer.size());
    if (length <= 0) break;  // No more events to process

    ssize_t offset = 0;
    while (offset < length) {
      const struct inotify_event* event =
          reinterpret_cast<const struct inotify_event*>(buffer.data() + offset);
      auto file = ProcessInotifyEvent(*event);
      if (file) files.push_back(*file);
      if (files.size() >= kNotifyBatchSize) flush_batch();
      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  flush_batch();  // Flush any remaining files in the batch
}

auto FilePathProvider::ProcessInotifyEvent(const struct inotify_event& event)
    -> std::optional<File> {
  if (event.mask & IN_IGNORED) return std::nullopt;

  const Path directory(watch_descriptors_.at(event.wd));
  // Create full file path
  Path filepath = directory / event.name;

  // Handle different event types
  if (event.mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) {
    // File finished writing or moved into directory
    return File{.filepath = filepath, .message_type = MessageType::kFile};
  }

  constexpr uint32_t kDirCreateMask = IN_CREATE | IN_ISDIR;
  constexpr uint32_t kDirDeleteMask = IN_DELETE | IN_ISDIR;
  if ((event.mask & kDirCreateMask) == kDirCreateMask) {
    ScanDirectoryWithWatch(filepath.string());
  } else if ((event.mask & kDirDeleteMask) == kDirDeleteMask) {
    // Directory deleted - remove all watches for it and subdirectories
    RemoveWatchRecursive(filepath.string());
  } else if (event.mask & IN_DELETE_SELF) {
    RemoveWatchRecursive(directory);
  }

  return std::nullopt;
}

}  // namespace training
}  // namespace lczero