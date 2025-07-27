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

std::vector<FileDiscovery::File> FileDiscovery::AddDirectory(
    const std::string& directory) {
  absl::MutexLock lock(&mutex_);

  std::vector<File> existing_files;

  // Scan existing files using recursive directory iterator
  std::error_code ec;
  auto iterator = std::filesystem::recursive_directory_iterator(directory, ec);
  if (ec) {
    LOG(ERROR) << "Failed to scan directory " << directory << ": "
               << ec.message();
    return existing_files;
  }

  for (const auto& entry : iterator) {
    if (entry.is_regular_file(ec) && !ec) {
      std::filesystem::path relative_path =
          std::filesystem::relative(entry.path(), directory, ec);
      if (!ec) {
        existing_files.push_back({directory, relative_path.string()});
      }
    }
  }

  // Add inotify watches recursively
  AddWatchRecursive(directory);

  return existing_files;
}

void FileDiscovery::AddWatchRecursive(const std::string& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(
      inotify_fd_, path.c_str(),
      IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_MOVE);
  if (wd == -1) {
    LOG(ERROR) << "Failed to add inotify watch for " << path;
    return;
  }

  // Store watch descriptor mappings
  watch_descriptors_[wd] = path;
  directory_watches_[path] = wd;

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
      absl::MutexLock lock(&mutex_);
      auto it = watch_descriptors_.find(event->wd);
      if (it != watch_descriptors_.end()) {
        const std::string& directory = it->second;
        std::string filename = event->name;
        
        // Handle different event types
        if (event->mask & IN_CLOSE_WRITE) {
          // File finished writing
          files.push_back({directory, filename});
        } else if (event->mask & IN_MOVED_TO) {
          // File moved into directory
          files.push_back({directory, filename});
        } else if ((event->mask & IN_CREATE) && (event->mask & IN_ISDIR)) {
          // New subdirectory created - add watch for it
          std::filesystem::path new_dir = 
              std::filesystem::path(directory) / filename;
          AddWatchRecursive(new_dir.string());
        }
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

  const int timeout_ms = 100; // 100ms timeout for checking stop condition
  
  while (true) {
    // Check stop condition
    {
      absl::MutexLock lock(&mutex_);
      if (should_stop_) {
        break;
      }
    }
    
    struct epoll_event events[1];
    int nfds = epoll_wait(epoll_fd, events, 1, timeout_ms);
    
    if (nfds == -1) {
      if (errno != EINTR) {
        LOG(ERROR) << "epoll_wait failed: " << strerror(errno);
      }
      continue;
    }
    
    if (nfds > 0 && events[0].data.fd == inotify_fd_) {
      // Process inotify events
      auto discovered_files = ProcessInotifyEvents();
      if (!discovered_files.empty()) {
        NotifyObservers(discovered_files);
      }
    }
  }
  
  close(epoll_fd);
}

}  // namespace ice_skate
}  // namespace lczero