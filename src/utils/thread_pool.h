#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <future>

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
             const ThreadPoolOptions& options = ThreadPoolOptions());

  // Blocks until all tasks are completed.
  ~ThreadPool();

  // Enqueues a task for execution and returns a std::future.
  template <typename F, typename... Args>
  auto Enqueue(F&& f, Args&&... args)
      -> std::future<std::invoke_result_t<F, Args...>>;

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

 private:
  void WorkerLoop();
  void StartWorkerThread() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool TaskAvailableCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return stop_ || !pending_tasks_.empty();
  }
  bool AllTasksCompletedCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return pending_tasks_.empty() && running_tasks_ == 0;
  }
  bool ThreadAvailableCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return pending_tasks_.empty() && running_tasks_ < threads_.size();
  }
  bool TaskCountBelowThreshold(size_t threshold) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return pending_tasks_.size() < threshold;
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ThreadPoolOptions options_;
  mutable absl::Mutex mutex_;

  std::vector<std::thread> threads_ ABSL_GUARDED_BY(mutex_);
  std::deque<absl::AnyInvocable<void()>> pending_tasks_ ABSL_GUARDED_BY(mutex_);
  bool stop_ ABSL_GUARDED_BY(mutex_) = false;
  size_t running_tasks_ ABSL_GUARDED_BY(mutex_) = 0;
};

inline ThreadPool::ThreadPool(size_t initial_threads,
                              const ThreadPoolOptions& options)
    : options_(options) {
  for (size_t i = 0; i < initial_threads; ++i) {
    threads_.emplace_back(&ThreadPool::WorkerLoop, this);
  }
}

inline ThreadPool::~ThreadPool() {
  {
    absl::MutexLock lock(&mutex_);
    stop_ = true;
  }
  for (std::thread& worker : threads_) worker.join();
}

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  std::packaged_task<ReturnType()> task(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<ReturnType> future = task.get_future();

  {
    absl::MutexLock lock(&mutex_);
    running_tasks_ += 1;
    while (options_.grow_automatically && running_tasks_ >= threads_.size()) {
      StartWorkerThread();
    }
    pending_tasks_.emplace_back([task = std::move(task)]() mutable { task(); });
  }

  return future;
}

inline void ThreadPool::WorkerLoop() {
  while (true) {
    absl::AnyInvocable<void()> task;
    {
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &ThreadPool::TaskAvailableCond));
      if (stop_ && pending_tasks_.empty()) return;
      task = std::move(pending_tasks_.front());
      pending_tasks_.pop_front();
    }

    std::move(task)();

    {
      absl::MutexLock lock(&mutex_);
      running_tasks_ -= 1;
    }
  }
}

inline void ThreadPool::WaitAll() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &ThreadPool::AllTasksCompletedCond));
}

inline void ThreadPool::WaitForAvailableThread() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &ThreadPool::ThreadAvailableCond));
}

inline void ThreadPool::WaitForPendingTasksBelow(size_t threshold) {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(
      absl::Condition(this, &ThreadPool::TaskCountBelowThreshold, threshold));
}

inline void ThreadPool::StartWorkerThread()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  threads_.emplace_back(&ThreadPool::WorkerLoop, this);
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

}  // namespace lczero