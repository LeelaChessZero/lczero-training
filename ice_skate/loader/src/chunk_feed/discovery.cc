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

void FileDiscovery::AddDirectory(const std::string& directory, Observer initial_observer) {
  std::vector<File> existing_files;
  
  {
    absl::MutexLock lock(&mutex_);
    // Add inotify watches recursively
    AddWatchRecursive(directory);
    
    // Scan existing files using recursive directory iterator
    std::error_code ec;
    auto iterator = std::filesystem::recursive_directory_iterator(directory, ec);
    if (ec) {
      LOG(ERROR) << "Failed to scan directory " << directory << ": "
                 << ec.message();
      return;
    }

    for (const auto& entry : iterator) {
      if (entry.is_regular_file(ec) && !ec) {
        existing_files.push_back({entry.path().string()});
      }
    }
  }
  
  // Notify initial observer in batches without holding mutex
  const size_t batch_size = 10000;
  for (size_t i = 0; i < existing_files.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, existing_files.size());
    std::span<const File> batch(existing_files.data() + i, end - i);
    try {
      initial_observer(batch);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Initial observer threw exception: " << e.what();
    } catch (...) {
      LOG(ERROR) << "Initial observer threw unknown exception";
    }
  }
  
}


void FileDiscovery::AddWatchRecursive(const std::string& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(
      inotify_fd_, path.c_str(),
      IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_DELETE_SELF | IN_MOVE);
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

void FileDiscovery::RemoveWatchRecursive(const std::string& path) {
  // Remove watch for this directory
  auto dir_it = directory_watches_.find(path);
  if (dir_it != directory_watches_.end()) {
    int wd = dir_it->second;
    inotify_rm_watch(inotify_fd_, wd);
    
    // Remove from both maps
    directory_watches_.erase(dir_it);
    watch_descriptors_.erase(wd);
  }
  
  // Remove watches for all subdirectories that start with this path
  std::vector<std::string> dirs_to_remove;
  for (const auto& [dir_path, wd] : directory_watches_) {
    if (dir_path.starts_with(path + "/")) {
      dirs_to_remove.push_back(dir_path);
    }
  }
  
  for (const std::string& dir_path : dirs_to_remove) {
    auto it = directory_watches_.find(dir_path);
    if (it != directory_watches_.end()) {
      int wd = it->second;
      inotify_rm_watch(inotify_fd_, wd);
      directory_watches_.erase(it);
      watch_descriptors_.erase(wd);
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
      std::filesystem::path filepath = std::filesystem::path(directory) / event->name;
      
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
        // Notify observers in batches without holding mutex
        const size_t batch_size = 10000;
        for (size_t i = 0; i < discovered_files.size(); i += batch_size) {
          size_t end = std::min(i + batch_size, discovered_files.size());
          std::span<const File> batch(discovered_files.data() + i, end - i);
          NotifyObservers(batch);
        }
      }
    }
  }
  
  close(epoll_fd);
}

}  // namespace ice_skate
}  // namespace lczero