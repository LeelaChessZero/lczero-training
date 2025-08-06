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

// Exception thrown when operation would exceed queue capacity.
class QueueCapacityExceededException : public std::invalid_argument {
 public:
  explicit QueueCapacityExceededException(size_t requested, size_t capacity)
      : std::invalid_argument("Requested " + std::to_string(requested) +
                              " elements exceeds queue capacity " +
                              std::to_string(capacity)) {}
};

// Thread-safe fixed-size circular buffer queue with blocking operations.
// Supports both single and batch put/get operations.
// The queue automatically closes when all Producer tokens are destroyed.
// When closed, Put operations throw immediately, but Get operations only throw
// when the queue becomes empty - allowing consumption of remaining elements.
template <typename T>
class Queue {
 public:
  explicit Queue(size_t capacity);

  // RAII token for producers. Queue automatically closes when all producers
  // are destroyed. All Put operations must go through this class.
  class Producer {
   public:
    explicit Producer(Queue<T>& queue);
    ~Producer();

    // Move constructor and assignment
    Producer(Producer&& other) noexcept;
    Producer& operator=(Producer&& other) noexcept;

    // Disable copy to maintain RAII semantics
    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;

    // Puts a single element into the queue. Blocks if queue is full.
    void Put(const T& item);
    void Put(T&& item);

    // Puts multiple elements into the queue. Blocks if not enough space.
    void Put(absl::Span<const T> items);
    void Put(absl::Span<T> items);

    // Explicitly close this producer, decrementing the producer count
    void Close();

   private:
    Queue<T>* queue_;
  };

  // Creates a new producer token for this queue.
  Producer CreateProducer();

  // Gets a single element from the queue. Blocks if queue is empty.
  T Get();

  // Gets exactly count elements from the queue. Blocks until count elements
  // available.
  absl::FixedArray<T> Get(size_t count);

  // Returns the current size of the queue.
  size_t Size() const;

  // Returns the capacity of the queue.
  size_t Capacity() const;

  // Explicitly close the queue, preventing further Put operations.
  void Close();

  // Wait until queue has at least the specified amount of free space.
  void WaitForRoomAtLeast(size_t room);

  // Wait until queue has at most the specified amount of free space.
  void WaitForRoomAtMost(size_t room);

  // Wait until queue has at least the specified number of elements.
  void WaitForSizeAtLeast(size_t size);

  // Wait until queue has at most the specified number of elements.
  void WaitForSizeAtMost(size_t size);

 private:
  friend class Producer;

  const size_t capacity_;
  absl::FixedArray<T> buffer_ ABSL_GUARDED_BY(mutex_);
  size_t head_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t tail_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t size_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t producer_count_ ABSL_GUARDED_BY(mutex_) = 0;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;

  mutable absl::Mutex mutex_;

  // Internal methods for producer management
  void RemoveProducer();

  // Internal Put methods (called by Producer)
  void PutInternal(const T& item);
  void PutInternal(T&& item);
  void PutInternal(absl::Span<const T> items);
  void PutInternal(absl::Span<T> items);

  // Condition predicates for blocking operations
  bool CanPutOne() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanPutMultiple(size_t count) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanGet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanGetMultiple(size_t count) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Additional condition predicates for wait functions
  bool HasRoomAtLeast(size_t room) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasRoomAtMost(size_t room) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasSizeAtLeast(size_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool HasSizeAtMost(size_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

// Implementation

template <typename T>
Queue<T>::Queue(size_t capacity) : capacity_(capacity), buffer_(capacity) {}

// Producer implementation
template <typename T>
Queue<T>::Producer::Producer(Queue<T>& queue) : queue_(&queue) {
  // Producer count is incremented in CreateProducer()
}

template <typename T>
Queue<T>::Producer::~Producer() {
  if (queue_) {
    queue_->RemoveProducer();
  }
}

template <typename T>
Queue<T>::Producer::Producer(Producer&& other) noexcept : queue_(other.queue_) {
  other.queue_ = nullptr;
}

template <typename T>
typename Queue<T>::Producer& Queue<T>::Producer::operator=(
    Producer&& other) noexcept {
  if (this != &other) {
    if (queue_) {
      queue_->RemoveProducer();
    }
    queue_ = other.queue_;
    other.queue_ = nullptr;
  }
  return *this;
}

template <typename T>
void Queue<T>::Producer::Put(const T& item) {
  queue_->PutInternal(item);
}

template <typename T>
void Queue<T>::Producer::Put(T&& item) {
  queue_->PutInternal(std::move(item));
}

template <typename T>
void Queue<T>::Producer::Put(absl::Span<const T> items) {
  queue_->PutInternal(items);
}

template <typename T>
void Queue<T>::Producer::Put(absl::Span<T> items) {
  queue_->PutInternal(items);
}

template <typename T>
void Queue<T>::Producer::Close() {
  if (queue_) {
    queue_->RemoveProducer();
    queue_ = nullptr;
  }
}

// Queue implementation
template <typename T>
typename Queue<T>::Producer Queue<T>::CreateProducer() {
  absl::MutexLock lock(&mutex_);
  if (closed_) throw QueueClosedException();
  ++producer_count_;
  return Producer(*this);
}

template <typename T>
void Queue<T>::RemoveProducer() {
  absl::MutexLock lock(&mutex_);
  --producer_count_;
  if (producer_count_ == 0) closed_ = true;
}

template <typename T>
void Queue<T>::PutInternal(const T& item) {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
  if (closed_) throw QueueClosedException();
  buffer_[tail_] = item;
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::PutInternal(T&& item) {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &Queue<T>::CanPutOne));
  if (closed_) throw QueueClosedException();
  buffer_[tail_] = std::move(item);
  tail_ = (tail_ + 1) % capacity_;
  ++size_;
}

template <typename T>
void Queue<T>::PutInternal(absl::Span<const T> items) {
  if (items.empty()) return;
  if (items.size() > capacity_) {
    throw QueueCapacityExceededException(items.size(), capacity_);
  }

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
void Queue<T>::PutInternal(absl::Span<T> items) {
  if (items.empty()) return;
  if (items.size() > capacity_) {
    throw QueueCapacityExceededException(items.size(), capacity_);
  }

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
  if (count > capacity_) {
    throw QueueCapacityExceededException(count, capacity_);
  }

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
void Queue<T>::WaitForRoomAtLeast(size_t room) {
  struct Args {
    Queue<T>* queue;
    size_t room;
  };
  Args args{this, room};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasRoomAtLeast(args->room);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForRoomAtMost(size_t room) {
  struct Args {
    Queue<T>* queue;
    size_t room;
  };
  Args args{this, room};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasRoomAtMost(args->room);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForSizeAtLeast(size_t size) {
  struct Args {
    Queue<T>* queue;
    size_t size;
  };
  Args args{this, size};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasSizeAtLeast(args->size);
      },
      &args));
}

template <typename T>
void Queue<T>::WaitForSizeAtMost(size_t size) {
  struct Args {
    Queue<T>* queue;
    size_t size;
  };
  Args args{this, size};
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(
      +[](void* data) -> bool {
        auto* args = static_cast<Args*>(data);
        return args->queue->HasSizeAtMost(args->size);
      },
      &args));
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

template <typename T>
bool Queue<T>::HasRoomAtLeast(size_t room)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return capacity_ - size_ >= room;
}

template <typename T>
bool Queue<T>::HasRoomAtMost(size_t room)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return capacity_ - size_ <= room;
}

template <typename T>
bool Queue<T>::HasSizeAtLeast(size_t size)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return size_ >= size;
}

template <typename T>
bool Queue<T>::HasSizeAtMost(size_t size)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  return size_ <= size;
}

}  // namespace lczero