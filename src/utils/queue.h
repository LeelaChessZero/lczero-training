#pragma once

#include <stdexcept>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace lczero {

// Exception thrown when queue operations are attempted on a closed queue.
class QueueClosedException : public std::runtime_error {
 public:
  QueueClosedException() : std::runtime_error("Queue is closed") {}
};

// Thread-safe fixed-size circular buffer queue with blocking operations.
// Supports both single and batch put/get operations.
// When closed, Put operations throw immediately, but Get operations only throw
// when the queue becomes empty - allowing consumption of remaining elements.
template <typename T>
class Queue {
 public:
  explicit Queue(size_t capacity);

  // Puts a single element into the queue. Blocks if queue is full.
  void Put(const T& item);
  void Put(T&& item);

  // Puts multiple elements into the queue. Blocks if not enough space.
  void Put(absl::Span<const T> items);
  void Put(absl::Span<T> items);

  // Gets a single element from the queue. Blocks if queue is empty.
  T Get();

  // Gets exactly count elements from the queue. Blocks until count elements
  // available.
  absl::FixedArray<T> Get(size_t count);

  // Returns the current size of the queue.
  size_t Size() const;

  // Returns the capacity of the queue.
  size_t Capacity() const;

  // Closes the queue. Put operations immediately throw QueueClosedException.
  // Get operations only throw when attempting to get from an empty closed
  // queue, allowing remaining elements to be consumed.
  void Close();

 private:
  const size_t capacity_;
  absl::FixedArray<T> buffer_ ABSL_GUARDED_BY(mutex_);
  size_t head_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t tail_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t size_ ABSL_GUARDED_BY(mutex_) = 0;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;

  mutable absl::Mutex mutex_;

  // Condition predicates for blocking operations
  bool CanPutOne() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanPutMultiple(size_t count) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanGet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanGetMultiple(size_t count) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

// Implementation

template <typename T>
Queue<T>::Queue(size_t capacity) : capacity_(capacity), buffer_(capacity) {}

template <typename T>
void Queue<T>::Put(const T& item) {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
  if (closed_) throw QueueClosedException();
  buffer_[tail_] = item;
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::Put(T&& item) {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
  if (closed_) throw QueueClosedException();
  buffer_[tail_] = std::move(item);
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::Put(absl::Span<const T> items) {
  if (items.empty()) return;

  struct Args {
    Queue<T>* queue;
    size_t count;
  };
  Args args{this, items.size()};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->CanPutMultiple(args->count);
      },
      &args));
  if (closed_) throw QueueClosedException();

  for (const auto& item : items) {
    buffer_[tail_] = item;
    tail_ = (tail_ + 1) % capacity_;
    ++size_;
  }
}

template <typename T>
void Queue<T>::Put(absl::Span<T> items) {
  if (items.empty()) return;

  struct Args {
    Queue<T>* queue;
    size_t count;
  };
  Args args{this, items.size()};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->CanPutMultiple(args->count);
      },
      &args));
  if (closed_) throw QueueClosedException();

  for (auto& item : items) {
    buffer_[tail_] = std::move(item);
    tail_ = (tail_ + 1) % capacity_;
    ++size_;
  }
}

template <typename T>
T Queue<T>::Get() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanGet));
  if (closed_ && size_ == 0) throw QueueClosedException();

  T item = std::move(buffer_[head_]);
  head_ = (head_ + 1) % capacity_;
  --size_;

  return item;
}

template <typename T>
absl::FixedArray<T> Queue<T>::Get(size_t count) {
  if (count == 0) return absl::FixedArray<T>(0);

  struct Args {
    Queue<T>* queue;
    size_t count;
  };
  Args args{this, count};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->CanGetMultiple(args->count);
      },
      &args));
  if (closed_ && size_ < count) throw QueueClosedException();

  absl::FixedArray<T> result(count);

  for (size_t i = 0; i < count; ++i) {
    result[i] = std::move(buffer_[head_]);
    head_ = (head_ + 1) % capacity_;
    --size_;
  }

  return result;
}

template <typename T>
size_t Queue<T>::Size() const {
  absl::MutexLock lock(&mutex_);
  return size_;
}

template <typename T>
size_t Queue<T>::Capacity() const {
  return capacity_;
}

template <typename T>
void Queue<T>::Close() {
  absl::MutexLock lock(&mutex_);
  closed_ = true;
}

template <typename T>
bool Queue<T>::CanPutOne() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ < capacity_;
}

template <typename T>
bool Queue<T>::CanPutMultiple(size_t count)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ + count <= capacity_;
}

template <typename T>
bool Queue<T>::CanGet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ > 0;
}

template <typename T>
bool Queue<T>::CanGetMultiple(size_t count)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return closed_ || size_ >= count;
}

}  // namespace lczero