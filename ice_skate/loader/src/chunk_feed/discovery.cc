#include "chunk_feed/discovery.h"

#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>
#include <sys/epoll.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace lczero {
namespace ice_skate {

FileDiscovery::FileDiscovery()
    : stop_condition_(&should_stop_), should_stop_(false), next_token_(1) {
  inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
  if (inotify_fd_ == -1) {
    throw std::runtime_error("Failed to initialize inotify");
  }

  monitor_thread_ = std::thread(&FileDiscovery::MonitorThread, this);
}

FileDiscovery::~FileDiscovery() {
  {
    absl::MutexLock lock(&mutex_);
    should_stop_ = true;
  }

  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }

  for (const auto& [wd, path] : watch_descriptors_) {
    inotify_rm_watch(inotify_fd_, wd);
  }

  if (inotify_fd_ != -1) {
    close(inotify_fd_);
  }
}

FileDiscovery::Token FileDiscovery::RegisterObserver(Observer observer) {
  absl::MutexLock lock(&mutex_);
  Token token = next_token_++;
  observers_[token] = std::move(observer);
  return token;
}

void FileDiscovery::UnregisterObserver(Token token) {
  absl::MutexLock lock(&mutex_);
  observers_.erase(token);
}

void FileDiscovery::AddDirectory(const std::string& directory,
                                 Observer initial_observer) {
  std::vector<File> existing_files;

  {
    absl::MutexLock lock(&mutex_);
    // Add inotify watches recursively
    AddWatchRecursive(directory);
    // Perform initial scan of existing files
    existing_files = PerformInitialScan(directory);
  }

  // Notify initial observer using the same batching mechanism
  NotifyObserversInBatches(existing_files, initial_observer);
}

void FileDiscovery::AddWatchRecursive(const std::string& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(inotify_fd_, path.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  if (wd == -1) {
    LOG(ERROR) << "Failed to add inotify watch for " << path;
    return;
  }

  // Store watch descriptor mapping
  watch_descriptors_[wd] = path;

  // Recursively add watches for subdirectories
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(path, ec);
  if (ec) {
    LOG(ERROR) << "Failed to iterate directory " << path << ": "
               << ec.message();
    return;
  }

  for (const auto& entry : iterator) {
    if (entry.is_directory(ec) && !ec) {
      AddWatchRecursive(entry.path().string());
    }
  }
}

void FileDiscovery::RemoveWatchRecursive(const std::string& path) {
  // Find and remove watches for this directory and all subdirectories
  std::vector<int> wds_to_remove;
  for (const auto& [wd, dir_path] : watch_descriptors_) {
    if (dir_path == path || dir_path.starts_with(path + "/")) {
      wds_to_remove.push_back(wd);
    }
  }

  for (int wd : wds_to_remove) {
    inotify_rm_watch(inotify_fd_, wd);
    watch_descriptors_.erase(wd);
  }
}

std::vector<FileDiscovery::File> FileDiscovery::ProcessInotifyEvents() {
  std::vector<File> files;
  char buffer[4096];

  ssize_t length = read(inotify_fd_, buffer, sizeof(buffer));
  if (length <= 0) {
    return files;
  }

  size_t offset = 0;
  while (offset < static_cast<size_t>(length)) {
    struct inotify_event* event =
        reinterpret_cast<struct inotify_event*>(buffer + offset);

    if (event->len > 0) {
      std::string directory;
      {
        absl::MutexLock lock(&mutex_);
        auto it = watch_descriptors_.find(event->wd);
        if (it == watch_descriptors_.end()) {
          offset += sizeof(struct inotify_event) + event->len;
          continue;
        }
        directory = it->second;
      }

      // Create full file path
      std::filesystem::path filepath =
          std::filesystem::path(directory) / event->name;

      // Handle different event types
      if (event->mask & IN_CLOSE_WRITE) {
        // File finished writing
        files.push_back({filepath.string()});
      } else if (event->mask & IN_MOVED_TO) {
        // File moved into directory
        files.push_back({filepath.string()});
      } else if ((event->mask & IN_CREATE) && (event->mask & IN_ISDIR)) {
        // New subdirectory created - add watch for it
        absl::MutexLock lock(&mutex_);
        AddWatchRecursive(filepath.string());
      } else if ((event->mask & IN_DELETE) && (event->mask & IN_ISDIR)) {
        // Directory deleted - remove all watches for it and subdirectories
        absl::MutexLock lock(&mutex_);
        RemoveWatchRecursive(filepath.string());
      }
    } else if (event->mask & IN_DELETE_SELF) {
      // The watched directory itself was deleted
      absl::MutexLock lock(&mutex_);
      auto it = watch_descriptors_.find(event->wd);
      if (it != watch_descriptors_.end()) {
        const std::string& directory = it->second;
        RemoveWatchRecursive(directory);
      }
    }

    offset += sizeof(struct inotify_event) + event->len;
  }

  return files;
}

void FileDiscovery::NotifyObservers(std::span<const File> files) {
  if (files.empty()) {
    return;
  }

  // Get snapshot of observers while holding lock
  std::vector<Observer> observer_snapshot;
  {
    absl::MutexLock lock(&mutex_);
    observer_snapshot.reserve(observers_.size());
    for (const auto& [token, observer] : observers_) {
      observer_snapshot.push_back(observer);
    }
  }

  // Call observers without holding lock
  for (const auto& observer : observer_snapshot) {
    try {
      observer(files);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Observer threw exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "Observer threw unknown exception";
    }
  }
}

void FileDiscovery::MonitorThread() {
  int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  if (epoll_fd == -1) {
    LOG(ERROR) << "Failed to create epoll fd";
    return;
  }

  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = inotify_fd_;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd_, &event) == -1) {
    LOG(ERROR) << "Failed to add inotify fd to epoll";
    close(epoll_fd);
    return;
  }

  // Use efficient condition variable waiting instead of polling
  absl::MutexLock lock(&mutex_);
  while (!should_stop_) {
    // Wait for either stop condition or short timeout for event processing
    if (mutex_.AwaitWithTimeout(stop_condition_, absl::Milliseconds(50))) {
      break;
    }

    // Release mutex before doing I/O operations
    mutex_.Unlock();

    struct epoll_event events[1];
    int nfds = epoll_wait(epoll_fd, events, 1, 0);  // Non-blocking check

    if (nfds == -1) {
      if (errno != EINTR) {
        LOG(ERROR) << "epoll_wait failed: " << strerror(errno);
      }
      mutex_.Lock();
      continue;
    }

    if (nfds > 0 && events[0].data.fd == inotify_fd_) {
      // Accumulate multiple event processing cycles to handle high-frequency
      // events
      std::vector<File> all_discovered_files;
      const int max_cycles =
          10;  // Process up to 10 cycles to batch high-frequency events

      for (int cycle = 0; cycle < max_cycles; ++cycle) {
        auto discovered_files = ProcessInotifyEvents();
        if (discovered_files.empty()) {
          break;  // No more events to process
        }

        // Accumulate files from this cycle
        all_discovered_files.insert(all_discovered_files.end(),
                                    discovered_files.begin(),
                                    discovered_files.end());

        // Check if there are more events immediately available
        nfds = epoll_wait(epoll_fd, events, 1, 0);
        if (nfds <= 0) {
          break;
        }
      }

      if (!all_discovered_files.empty()) {
        // Use the generic batching mechanism to notify all observers
        for (const auto& observer : [this]() {
          std::vector<Observer> observer_snapshot;
          {
            absl::MutexLock lock(&mutex_);
            observer_snapshot.reserve(observers_.size());
            for (const auto& [token, observer] : observers_) {
              observer_snapshot.push_back(observer);
            }
          }
          return observer_snapshot;
        }()) {
          NotifyObserversInBatches(all_discovered_files, observer);
        }
      }
    }

    // Re-acquire mutex for next iteration
    mutex_.Lock();
  }

  close(epoll_fd);
}

void FileDiscovery::NotifyObserversInBatches(std::span<const File> files, Observer observer) {
  if (files.empty()) {
    return;
  }

  const size_t batch_size = 10000;
  for (size_t i = 0; i < files.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, files.size());
    std::span<const File> batch(files.data() + i, end - i);
    try {
      observer(batch);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Observer threw exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "Observer threw unknown exception";
    }
  }
}

std::vector<FileDiscovery::File> FileDiscovery::PerformInitialScan(const std::string& directory) {
  std::vector<File> existing_files;
  
  // Scan existing files using recursive directory iterator
  std::error_code ec;
  auto iterator = std::filesystem::recursive_directory_iterator(directory, ec);
  if (ec) {
    LOG(ERROR) << "Failed to scan directory " << directory << ": " << ec.message();
    return existing_files;
  }

  for (const auto& entry : iterator) {
    if (entry.is_regular_file(ec) && !ec) {
      existing_files.push_back({entry.path().string()});
    }
  }
  
  return existing_files;
}

}  // namespace ice_skate
}  // namespace lczero