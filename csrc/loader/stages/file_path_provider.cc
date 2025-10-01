#include "loader/stages/file_path_provider.h"

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
#include <string_view>
#include <utility>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

namespace {

bool ShouldSkipName(std::string_view name) {
  return !name.empty() && name.front() == '.';
}

bool ShouldSkipPathEntry(const FilePathProvider::Path& path) {
  return ShouldSkipName(path.filename().string());
}

}  // namespace

FilePathProvider::FilePathProvider(const FilePathProviderConfig& config,
                                   const StageList& existing_stages)
    : output_queue_(config.queue_capacity()),
      directory_(config.directory()),
      producer_(output_queue_.CreateProducer()),
      load_metric_updater_() {
  (void)existing_stages;
  LOG(INFO) << "Initializing FilePathProvider for directory: "
            << config.directory();
  inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
  CHECK_NE(inotify_fd_, -1)
      << "Failed to initialize inotify: " << strerror(errno);
}

FilePathProvider::~FilePathProvider() {
  LOG(INFO) << "FilePathProvider shutting down.";
  Stop();
  if (inotify_fd_ != -1) close(inotify_fd_);
  LOG(INFO) << "FilePathProvider shutdown complete.";
}

Queue<FilePathProvider::File>* FilePathProvider::output() {
  return &output_queue_;
}

QueueBase* FilePathProvider::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

void FilePathProvider::Start() {
  LOG(INFO) << "Starting FilePathProvider monitoring thread.";
  monitor_thread_ = std::thread(&FilePathProvider::MonitorThread, this);
}

void FilePathProvider::Stop() {
  if (stop_condition_.HasBeenNotified()) {
    return;
  }

  LOG(INFO) << "Stopping all watches...";
  for (const auto& [wd, path] : watch_descriptors_) {
    inotify_rm_watch(inotify_fd_, wd);
  }
  watch_descriptors_.clear();

  LOG(INFO) << "Notifying threads to stop.";
  stop_condition_.Notify();
  if (monitor_thread_.joinable()) {
    LOG(INFO) << "Joining monitor thread...";
    monitor_thread_.join();
  }

  LOG(INFO) << "Closing producer to close the queue...";
  producer_.Close();
  LOG(INFO) << "FilePathProvider closed.";
}

StageMetricProto FilePathProvider::FlushMetrics() {
  StageMetricProto stage_metric;
  stage_metric.set_stage_type("file_path_provider");
  auto load_metrics = load_metric_updater_.FlushMetrics();
  load_metrics.set_name("load");
  *stage_metric.add_load_metrics() = std::move(load_metrics);
  *stage_metric.add_queue_metrics() = MetricsFromQueue("output", output_queue_);
  return stage_metric;
}

void FilePathProvider::AddDirectory(const Path& directory) {
  ScanDirectoryWithWatch(directory);

  LOG(INFO) << "FilePathProvider registered " << directory
            << "; active watch descriptors: " << watch_descriptors_.size();

  // Signal that initial scan is complete
  LOG(INFO) << "FilePathProvider initial scan complete";
  producer_.Put({{.filepath = Path{},
                  .message_type = MessageType::kInitialScanComplete}});
}

void FilePathProvider::ScanDirectoryWithWatch(const Path& directory) {
  // Step 1: Set up watch first
  int wd = inotify_add_watch(inotify_fd_, directory.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << directory << ": "
                   << strerror(errno);
  watch_descriptors_[wd] = directory;

  // Step 2: Scan directory non-recursively, remembering files and subdirs
  std::vector<Path> files;
  std::vector<Path> subdirectories;
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(directory, ec);
  CHECK(!ec) << "Failed to iterate directory " << directory << ": "
             << ec.message();

  for (const auto& entry : iterator) {
    const Path entry_path = entry.path();
    if (ShouldSkipPathEntry(entry_path)) continue;

    if (entry.is_regular_file(ec) && !ec) {
      files.push_back(entry_path);
    } else if (entry.is_directory(ec) && !ec) {
      subdirectories.push_back(entry_path);
    }
  }

  const size_t initial_file_count = files.size();
  const size_t subdirectory_count = subdirectories.size();
  LOG(INFO) << "FilePathProvider scanned " << directory << " discovering "
            << initial_file_count << " file(s) and " << subdirectory_count
            << " subdirectory(ies) before watch reconciliation.";

  // Send notifications for discovered files
  constexpr size_t kBatchSize = 10000;
  std::vector<File> batch;
  batch.reserve(kBatchSize);

  auto flush_batch = [&]() {
    if (batch.empty()) return;
    producer_.Put(batch);
    batch.clear();
  };

  for (const auto& filepath : files) {
    batch.push_back(
        {.filepath = filepath.string(), .message_type = MessageType::kFile});
    if (batch.size() >= kBatchSize) flush_batch();
  }

  if (initial_file_count > 0) {
    LOG(INFO) << "FilePathProvider enqueued " << initial_file_count
              << " file(s) from initial scan of " << directory;
  }

  // Step 3: Read from watch descriptor, skipping already discovered files
  ProcessWatchEventsForNewItems(files);

  // Step 4: Clean the files vector to save memory
  files.clear();

  // Step 5: Recursively call for subdirectories
  for (const auto& subdir : subdirectories) {
    if (stop_condition_.HasBeenNotified()) return;
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

      const bool skip_entry = event->len > 0 && ShouldSkipName(event->name);

      // Only process file creation/write events, skip already known files
      if ((event->mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0 &&
          event->len > 0 && !skip_entry) {
        const Path directory(watch_descriptors_.at(event->wd));
        Path filepath = directory / event->name;
        std::string filepath_string = filepath.string();

        // Only add if we haven't seen this file before
        if (!known_file_set.contains(filepath_string)) {
          new_files.push_back({.filepath = std::move(filepath_string),
                               .message_type = MessageType::kFile});
        }
      }

      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  // Send notifications for any new files discovered through watch events
  if (!new_files.empty()) {
    LOG(INFO) << "FilePathProvider observed " << new_files.size()
              << " new file(s) while reconciling race events.";
    producer_.Put(new_files);
  }
}

void FilePathProvider::AddWatchRecursive(const Path& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(inotify_fd_, path.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << path << ": "
                   << strerror(errno);
  watch_descriptors_[wd] = path;

  // Recursively add watches for subdirectories
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(path, ec);
  CHECK(!ec) << "Failed to iterate directory " << path << ": " << ec.message();

  for (const auto& entry : iterator) {
    const Path entry_path = entry.path();
    if (ShouldSkipPathEntry(entry_path)) continue;
    if (!entry.is_directory(ec) || ec) continue;
    AddWatchRecursive(entry_path);
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
  // Perform directory scanning in background thread
  AddDirectory(directory_);

  int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  CHECK_NE(epoll_fd, -1) << "Failed to create epoll fd: " << strerror(errno);
  absl::Cleanup epoll_cleanup([epoll_fd]() { close(epoll_fd); });

  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = inotify_fd_;
  CHECK_EQ(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd_, &event), 0)
      << "Failed to add inotify fd to epoll: " << strerror(errno);

  while (true) {
    {
      LoadMetricPauser pauser(load_metric_updater_);
      if (stop_condition_.WaitForNotificationWithTimeout(
              absl::Milliseconds(50))) {
        pauser.DoNotResume();
        break;  // Exit if stop condition is notified
      }
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
  size_t total_events = 0;
  size_t total_enqueued = 0;
  bool saw_events = false;

  auto flush_batch = [&]() {
    if (files.empty()) return;
    total_enqueued += files.size();
    producer.Put(files);
    files.clear();
  };

  while (true) {
    ssize_t length = read(inotify_fd_, buffer.data(), buffer.size());
    if (length <= 0) break;  // No more events to process
    saw_events = true;

    ssize_t offset = 0;
    while (offset < length) {
      const struct inotify_event* event =
          reinterpret_cast<const struct inotify_event*>(buffer.data() + offset);
      ++total_events;
      auto file = ProcessInotifyEvent(*event);
      if (file) files.push_back(*file);
      if (files.size() >= kNotifyBatchSize) flush_batch();
      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  flush_batch();  // Flush any remaining files in the batch

  if (saw_events) {
    LOG(INFO) << "FilePathProvider processed " << total_events
              << " inotify event(s) and enqueued " << total_enqueued
              << " file notification(s).";
  }
}

auto FilePathProvider::ProcessInotifyEvent(const struct inotify_event& event)
    -> std::optional<File> {
  if (event.mask & IN_IGNORED) return std::nullopt;

  const Path directory(watch_descriptors_.at(event.wd));
  const bool has_name = event.len > 0 && event.name[0] != '\0';
  const bool skip_entry = has_name && ShouldSkipName(event.name);
  const Path filepath = has_name ? directory / event.name : directory;

  // Handle different event types
  if ((event.mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0 && has_name &&
      !skip_entry) {
    // File finished writing or moved into directory
    return File{.filepath = filepath, .message_type = MessageType::kFile};
  }

  constexpr uint32_t kDirCreateMask = IN_CREATE | IN_ISDIR;
  constexpr uint32_t kDirDeleteMask = IN_DELETE | IN_ISDIR;
  if ((event.mask & kDirCreateMask) == kDirCreateMask) {
    if (!has_name || skip_entry) return std::nullopt;
    ScanDirectoryWithWatch(filepath);
  } else if ((event.mask & kDirDeleteMask) == kDirDeleteMask) {
    if (!has_name || skip_entry) return std::nullopt;
    // Directory deleted - remove all watches for it and subdirectories
    RemoveWatchRecursive(filepath);
  } else if (event.mask & IN_DELETE_SELF) {
    RemoveWatchRecursive(directory);
  }

  return std::nullopt;
}

}  // namespace training
}  // namespace lczero
