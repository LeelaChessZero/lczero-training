#include "chunk_feed/discovery.h"

#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>
#include <unistd.h>

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

void FileDiscovery::MonitorThread() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(stop_condition_);
}

}  // namespace ice_skate
}  // namespace lczero