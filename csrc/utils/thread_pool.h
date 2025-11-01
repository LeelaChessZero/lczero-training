#pragma once

#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <stop_token>
#include <thread>

#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

namespace lczero {

struct ThreadPoolOptions {
  // If true, starts new thread when task is enqueued and no threads are idle.
  bool grow_automatically = false;
};

class ThreadPool {
 public:
  ThreadPool(size_t initial_threads = 0,
             const ThreadPoolOptions& options = ThreadPoolOptions(),
             std::stop_source stop_source = std::stop_source());

  // Blocks until all tasks are completed and threads are joined.
  ~ThreadPool();

  // Returns the stop_token for this thread pool.
  std::stop_token stop_token() const;

  // Enqueues a task for execution and returns a std::future.
  // If the provided function accepts a std::stop_token as its first argument,
  // one will be passed to it from the thread pool's stop source.
  template <typename F, typename... Args>
  auto Enqueue(F&& f, Args&&... args) {
    // This lambda captures the common queuing logic.
    auto enqueue_common = [&](auto&& task_to_enqueue, auto&& future_to_return) {
      {
        absl::MutexLock lock(&mutex_);
        running_tasks_ += 1;
        while (options_.grow_automatically &&
               running_tasks_ >= threads_.size()) {
          StartWorkerThread();
        }
        pending_tasks_.emplace_back(
            [task = std::move(task_to_enqueue)]() mutable { task(); });
        work_available_.Signal();
      }
      return future_to_return;
    };

    if constexpr (std::is_invocable_v<F, std::stop_token, Args...>) {
      using ReturnType = std::invoke_result_t<F, std::stop_token, Args...>;
      std::packaged_task<ReturnType()> task(
          std::bind(std::forward<F>(f), stop_source_.get_token(),
                    std::forward<Args>(args)...));
      std::future<ReturnType> future = task.get_future();
      return enqueue_common(std::move(task), std::move(future));
    } else {
      using ReturnType = std::invoke_result_t<F, Args...>;
      std::packaged_task<ReturnType()> task(
          std::bind(std::forward<F>(f), std::forward<Args>(args)...));
      std::future<ReturnType> future = task.get_future();
      return enqueue_common(std::move(task), std::move(future));
    }
  }

  // Waits for all tasks to complete.
  void WaitAll();

  // Waits for at least one thread to become available, i.e. no tasks are
  // pending, and number of running tasks is less than the number of threads.
  void WaitForAvailableThread();

  // Waits until the number of queued but not yet started tasks is below
  // the specified threshold.
  void WaitForPendingTasksBelow(size_t threshold);

  // Number of tasks that are not yet started.
  size_t num_pending_tasks() const;

  // Number of tasks that are currently running.
  size_t num_running_tasks() const;

  // Number of worker threads (busy or not).
  size_t num_threads() const;

  // Signal workers to terminate and join all threads.
  void Shutdown();

 private:
  void WorkerLoop();
  void WorkerEntryPoint();
  void StartWorkerThread() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool AllTasksCompletedCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return pending_tasks_.empty() && running_tasks_ == 0;
  }
  bool ThreadAvailableCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return pending_tasks_.empty() && running_tasks_ < threads_.size();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ThreadPoolOptions options_;
  mutable absl::Mutex mutex_;
  absl::CondVar work_available_;
  absl::CondVar work_done_;

  std::stop_source stop_source_;
  std::vector<std::jthread> threads_ ABSL_GUARDED_BY(mutex_);
  std::deque<absl::AnyInvocable<void()>> pending_tasks_ ABSL_GUARDED_BY(mutex_);
  size_t running_tasks_ ABSL_GUARDED_BY(mutex_) = 0;
};

inline ThreadPool::ThreadPool(size_t initial_threads,
                              const ThreadPoolOptions& options,
                              std::stop_source stop_source)
    : options_(options), stop_source_(std::move(stop_source)) {
  absl::MutexLock lock(&mutex_);
  for (size_t i = 0; i < initial_threads; ++i) {
    StartWorkerThread();
  }
}

inline ThreadPool::~ThreadPool() { Shutdown(); }

inline std::stop_token ThreadPool::stop_token() const {
  return stop_source_.get_token();
}

inline void ThreadPool::WorkerLoop() {
  while (true) {
    absl::AnyInvocable<void()> task;
    {
      absl::MutexLock lock(&mutex_);
      while (!stop_source_.stop_requested() && pending_tasks_.empty()) {
        work_available_.Wait(&mutex_);
      }
      if (stop_source_.stop_requested() && pending_tasks_.empty()) return;
      task = std::move(pending_tasks_.front());
      pending_tasks_.pop_front();
    }

    std::move(task)();

    {
      absl::MutexLock lock(&mutex_);
      running_tasks_ -= 1;
      work_done_.SignalAll();
      if (!pending_tasks_.empty()) work_available_.Signal();
    }
  }
}

inline void ThreadPool::WorkerEntryPoint() {
  try {
    WorkerLoop();
  } catch (const std::exception& exception) {
    std::cerr << "ThreadPool worker exited due to uncaught exception: "
              << exception.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "ThreadPool worker exited due to unknown exception."
              << std::endl;
    throw;
  }
}

inline void ThreadPool::WaitAll() {
  absl::MutexLock lock(&mutex_);
  while (!AllTasksCompletedCond()) {
    work_done_.Wait(&mutex_);
  }
}

inline void ThreadPool::WaitForAvailableThread() {
  absl::MutexLock lock(&mutex_);
  while (!ThreadAvailableCond()) {
    work_done_.Wait(&mutex_);
  }
}

inline void ThreadPool::WaitForPendingTasksBelow(size_t threshold) {
  absl::MutexLock lock(&mutex_);
  while (pending_tasks_.size() >= threshold) {
    work_done_.Wait(&mutex_);
  }
}

inline void ThreadPool::StartWorkerThread()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  threads_.emplace_back(&ThreadPool::WorkerEntryPoint, this);
}

inline size_t ThreadPool::num_pending_tasks() const {
  absl::MutexLock lock(&mutex_);
  return pending_tasks_.size();
}

inline size_t ThreadPool::num_running_tasks() const {
  absl::MutexLock lock(&mutex_);
  return std::max(running_tasks_, threads_.size());
}

inline size_t ThreadPool::num_threads() const {
  absl::MutexLock lock(&mutex_);
  return threads_.size();
}

inline void ThreadPool::Shutdown() {
  {
    absl::MutexLock lock(&mutex_);
    if (!stop_source_.stop_requested()) {
      stop_source_.request_stop();
    }
    work_available_.SignalAll();
    work_done_.SignalAll();
  }
  threads_.clear();
}

}  // namespace lczero
