#include "loader/chunk_feed/discovery.h"

#include <absl/cleanup/cleanup.h>
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

namespace lczero {
namespace training {

FileDiscovery::FileDiscovery() {
  inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
  CHECK_NE(inotify_fd_, -1)
      << "Failed to initialize inotify: " << strerror(errno);
  monitor_thread_ = std::thread(&FileDiscovery::MonitorThread, this);
}

FileDiscovery::~FileDiscovery() {
  stop_condition_.Notify();
  if (monitor_thread_.joinable()) monitor_thread_.join();
  for (const auto& [wd, path] : watch_descriptors_) {
    inotify_rm_watch(inotify_fd_, wd);
  }
  if (inotify_fd_ != -1) close(inotify_fd_);
}

FileDiscovery::ObeserverToken FileDiscovery::RegisterObserver(Observer observer) {
  ObeserverToken token = next_token_++;
  observers_[token] = std::move(observer);
  return token;
}

void FileDiscovery::UnregisterObserver(ObeserverToken token) { observers_.erase(token); }

void FileDiscovery::AddDirectory(const Path& directory,
                                 Observer initial_observer) {
  PerformInitialScan(directory, initial_observer);
  AddWatchRecursive(directory);
}

void FileDiscovery::PerformInitialScan(const Path& directory,
                                       Observer observer) {
  constexpr size_t kBatchSize = 10000;
  std::vector<File> batch;
  batch.reserve(kBatchSize);

  // Scan existing files using recursive directory iterator
  std::error_code ec;
  auto iterator = std::filesystem::recursive_directory_iterator(directory, ec);
  CHECK(!ec) << "Failed to iterate directory " << directory << ": "
             << ec.message();

  auto flush_batch = [&]() {
    if (batch.empty()) return;
    observer(batch);
    batch.clear();
  };

  for (const auto& entry : iterator) {
    if (entry.is_regular_file(ec) && !ec) {
      batch.push_back({entry.path().string()});
      // Flush batch when it reaches the limit
      if (batch.size() >= kBatchSize) flush_batch();
    }
  }

  flush_batch();  // Flush any remaining files in the batch
}

void FileDiscovery::AddWatchRecursive(const Path& path) {
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

void FileDiscovery::RemoveWatchRecursive(const Path& base) {
  absl::erase_if(watch_descriptors_, [&](const auto& pair) {
    const auto& [wd, path] = pair;
    const auto mismatch_iter = absl::c_mismatch(base, path).first;
    // If path is not a subdirectory (or equal) of base, skip.
    if (mismatch_iter != base.end()) return false;
    inotify_rm_watch(inotify_fd_, wd);
    return true;
  });
}

void FileDiscovery::MonitorThread() {
  int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  CHECK_NE(epoll_fd, -1) << "Failed to create epoll fd";
  absl::Cleanup epoll_cleanup([epoll_fd]() { close(epoll_fd); });

  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = inotify_fd_;
  CHECK_EQ(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd_, &event), 0)
      << "Failed to add inotify fd to epoll";

  while (true) {
    if (stop_condition_.WaitForNotificationWithTimeout(
            absl::Milliseconds(50))) {
      break;  // Exit if stop condition is notified
    }

    struct epoll_event event;
    int nfds = epoll_wait(epoll_fd, &event, 1, 0);  // Non-blocking check
    CHECK_NE(nfds, -1) << "epoll_wait failed: " << strerror(errno);
    if (nfds == 0) continue;  // No events.

    do {
      assert(nfds == 1 && event.data.fd == inotify_fd_);
      ProcessInotifyEvents();
      nfds = epoll_wait(epoll_fd, &event, 1, 0);
    } while (nfds > 0);
  }
}

void FileDiscovery::ProcessInotifyEvents() {
  constexpr size_t kNotifyBatchSize = 10000;
  std::vector<File> files;
  std::array<char, 4096> buffer;

  auto flush_batch = [&]() {
    if (files.empty()) return;
    NotifyObservers(files);
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

auto FileDiscovery::ProcessInotifyEvent(const struct inotify_event& event)
    -> std::optional<File> {
  if (event.mask & IN_IGNORED) return std::nullopt;

  const Path directory(watch_descriptors_.at(event.wd));
  // Create full file path
  Path filepath = directory / event.name;

  // Handle different event types
  if (event.mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) {
    // File finished writing or moved into directory
    return File{.filepath = filepath};
  }

  constexpr uint32_t kDirCreateMask = IN_CREATE | IN_ISDIR;
  constexpr uint32_t kDirDeleteMask = IN_DELETE | IN_ISDIR;
  if ((event.mask & kDirCreateMask) == kDirCreateMask) {
    AddWatchRecursive(filepath.string());
  } else if ((event.mask & kDirDeleteMask) == kDirDeleteMask) {
    // Directory deleted - remove all watches for it and subdirectories
    RemoveWatchRecursive(filepath.string());
  } else if (event.mask & IN_DELETE_SELF) {
    RemoveWatchRecursive(directory);
  }

  return std::nullopt;
}

void FileDiscovery::NotifyObservers(std::span<const File> files) {
  if (files.empty()) return;
  absl::c_for_each(observers_, [&](const auto& pair) {
    const auto& [token, observer] = pair;
    observer(files);
  });
}

}  // namespace training
}  // namespace lczero