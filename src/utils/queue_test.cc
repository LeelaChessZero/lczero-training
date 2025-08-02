// ABOUTME: Comprehensive unit tests for the Queue template class
// ABOUTME: Tests thread-safe operations, blocking behavior, and edge cases

#include "utils/queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace lczero {

class QueueTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Basic functionality tests

TEST_F(QueueTest, ConstructorCreatesEmptyQueue) {
  Queue<int> queue(5);
  EXPECT_EQ(queue.Size(), 0);
  EXPECT_EQ(queue.Capacity(), 5);
}

TEST_F(QueueTest, SinglePutGet) {
  Queue<int> queue(5);
  queue.Put(42);
  EXPECT_EQ(queue.Size(), 1);

  int value = queue.Get();
  EXPECT_EQ(value, 42);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, MovePutGet) {
  Queue<std::unique_ptr<int>> queue(5);
  auto ptr = std::make_unique<int>(42);
  queue.Put(std::move(ptr));
  EXPECT_EQ(queue.Size(), 1);

  auto result = queue.Get();
  EXPECT_EQ(*result, 42);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, MultiplePutGet) {
  Queue<int> queue(5);

  for (int i = 0; i < 5; ++i) {
    queue.Put(i);
  }
  EXPECT_EQ(queue.Size(), 5);

  for (int i = 0; i < 5; ++i) {
    int value = queue.Get();
    EXPECT_EQ(value, i);
  }
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, CircularBufferBehavior) {
  Queue<int> queue(3);

  // Fill queue
  queue.Put(1);
  queue.Put(2);
  queue.Put(3);

  // Get one item, put another
  EXPECT_EQ(queue.Get(), 1);
  queue.Put(4);

  // Verify remaining items
  EXPECT_EQ(queue.Get(), 2);
  EXPECT_EQ(queue.Get(), 3);
  EXPECT_EQ(queue.Get(), 4);
}

// Batch operations tests

TEST_F(QueueTest, BatchPutConstSpan) {
  Queue<int> queue(5);
  std::vector<int> items = {1, 2, 3};

  queue.Put(absl::Span<const int>(items));
  EXPECT_EQ(queue.Size(), 3);

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(queue.Get(), i + 1);
  }
}

TEST_F(QueueTest, BatchPutMoveSpan) {
  Queue<std::unique_ptr<int>> queue(5);
  std::vector<std::unique_ptr<int>> items;
  items.push_back(std::make_unique<int>(1));
  items.push_back(std::make_unique<int>(2));
  items.push_back(std::make_unique<int>(3));

  queue.Put(absl::Span<std::unique_ptr<int>>(items));
  EXPECT_EQ(queue.Size(), 3);

  for (int i = 0; i < 3; ++i) {
    auto result = queue.Get();
    EXPECT_EQ(*result, i + 1);
  }
}

TEST_F(QueueTest, BatchPutEmptySpan) {
  Queue<int> queue(5);
  std::vector<int> empty_items;

  queue.Put(absl::Span<const int>(empty_items));
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, BatchGet) {
  Queue<int> queue(5);

  for (int i = 0; i < 5; ++i) {
    queue.Put(i);
  }

  auto result = queue.Get(3);
  EXPECT_EQ(result.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], i);
  }
  EXPECT_EQ(queue.Size(), 2);
}

TEST_F(QueueTest, BatchGetZeroCount) {
  Queue<int> queue(5);
  queue.Put(42);

  auto result = queue.Get(0);
  EXPECT_EQ(result.size(), 0);
  EXPECT_EQ(queue.Size(), 1);
}

// Edge cases and error conditions

TEST_F(QueueTest, CapacityOne) {
  Queue<int> queue(1);

  queue.Put(42);
  EXPECT_EQ(queue.Size(), 1);

  EXPECT_EQ(queue.Get(), 42);
  EXPECT_EQ(queue.Size(), 0);
}

// Tests for operations on pre-closed queues

TEST_F(QueueTest, PutOnClosedQueue) {
  Queue<int> queue(5);
  queue.Close();

  EXPECT_THROW(queue.Put(42), QueueClosedException);
}

TEST_F(QueueTest, GetOnClosedQueue) {
  Queue<int> queue(5);
  queue.Close();

  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, BatchPutOnClosedQueue) {
  Queue<int> queue(5);
  queue.Close();

  std::vector<int> items = {1, 2, 3};
  EXPECT_THROW(queue.Put(absl::Span<const int>(items)), QueueClosedException);
}

TEST_F(QueueTest, BatchGetOnClosedQueue) {
  Queue<int> queue(5);
  queue.Close();

  EXPECT_THROW(queue.Get(3), QueueClosedException);
}

// Thread safety tests

TEST_F(QueueTest, SingleProducerSingleConsumer) {
  Queue<int> queue(10);
  std::atomic<bool> producer_done{false};
  std::vector<int> consumed;

  std::thread producer([&queue, &producer_done]() {
    for (int i = 0; i < 100; ++i) {
      queue.Put(i);
    }
    producer_done = true;
  });

  std::thread consumer([&queue, &consumed, &producer_done]() {
    int value;
    while (!producer_done || queue.Size() > 0) {
      try {
        value = queue.Get();
        consumed.push_back(value);
      } catch (const QueueClosedException&) {
        break;
      }
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(consumed.size(), 100);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(consumed[i], i);
  }
}

TEST_F(QueueTest, MultipleProducersMultipleConsumers) {
  Queue<int> queue(10);
  constexpr int num_producers = 2;
  constexpr int items_per_producer = 5;

  std::vector<int> all_produced;
  std::vector<int> all_consumed;
  std::mutex produced_mutex;
  std::vector<std::thread> producers;

  // Start producers
  for (int p = 0; p < num_producers; ++p) {
    producers.emplace_back([&queue, &all_produced, &produced_mutex, p]() {
      for (int i = 0; i < items_per_producer; ++i) {
        int value = p * items_per_producer + i;
        queue.Put(value);
        {
          std::lock_guard<std::mutex> lock(produced_mutex);
          all_produced.push_back(value);
        }
      }
    });
  }

  // Wait for producers to finish
  for (auto& producer : producers) {
    producer.join();
  }

  // Now consume all items
  for (int i = 0; i < num_producers * items_per_producer; ++i) {
    all_consumed.push_back(queue.Get());
  }

  // Verify all items were consumed
  EXPECT_EQ(all_consumed.size(), num_producers * items_per_producer);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, BlockingBehaviorOnFullQueue) {
  Queue<int> queue(2);
  std::atomic<bool> put_blocked{false};
  std::atomic<bool> put_completed{false};

  // Fill the queue
  queue.Put(1);
  queue.Put(2);

  std::thread blocker([&queue, &put_blocked, &put_completed]() {
    put_blocked = true;
    queue.Put(3);  // This should block
    put_completed = true;
  });

  // Give the blocker thread time to block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(put_blocked);
  EXPECT_FALSE(put_completed);

  // Make space in the queue
  EXPECT_EQ(queue.Get(), 1);

  blocker.join();
  EXPECT_TRUE(put_completed);
  EXPECT_EQ(queue.Size(), 2);
}

TEST_F(QueueTest, BlockingBehaviorOnEmptyQueue) {
  Queue<int> queue(5);
  std::atomic<bool> get_blocked{false};
  std::atomic<bool> get_completed{false};
  std::atomic<int> result{-1};

  std::thread blocker([&queue, &get_blocked, &get_completed, &result]() {
    get_blocked = true;
    result = queue.Get();  // This should block
    get_completed = true;
  });

  // Give the blocker thread time to block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(get_blocked);
  EXPECT_FALSE(get_completed);

  // Put an item in the queue
  queue.Put(42);

  blocker.join();
  EXPECT_TRUE(get_completed);
  EXPECT_EQ(result, 42);
}

TEST_F(QueueTest, CloseUnblocksWaitingPut) {
  Queue<int> queue(1);
  queue.Put(1);  // Fill the queue

  std::atomic<bool> put_started{false};
  std::atomic<bool> exception_thrown{false};

  std::thread blocker([&queue, &put_started, &exception_thrown]() {
    put_started = true;
    try {
      queue.Put(2);  // This should block
    } catch (const QueueClosedException&) {
      exception_thrown = true;
    }
  });

  // Give the blocker thread time to block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(put_started);
  EXPECT_FALSE(exception_thrown);

  // Close the queue - this should unblock the waiting Put()
  queue.Close();

  blocker.join();
  EXPECT_TRUE(exception_thrown);
}

TEST_F(QueueTest, CloseUnblocksWaitingGet) {
  Queue<int> queue(5);  // Empty queue

  std::atomic<bool> get_started{false};
  std::atomic<bool> exception_thrown{false};

  std::thread blocker([&queue, &get_started, &exception_thrown]() {
    get_started = true;
    try {
      queue.Get();  // This should block
    } catch (const QueueClosedException&) {
      exception_thrown = true;
    }
  });

  // Give the blocker thread time to block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(get_started);
  EXPECT_FALSE(exception_thrown);

  // Close the queue - this should unblock the waiting Get()
  queue.Close();

  blocker.join();
  EXPECT_TRUE(exception_thrown);
}

TEST_F(QueueTest, CloseUnblocksBatchOperations) {
  Queue<int> queue(2);
  queue.Put(1);
  queue.Put(2);  // Fill the queue

  std::atomic<bool> batch_put_started{false};
  std::atomic<bool> batch_put_exception{false};
  std::atomic<bool> batch_get_started{false};
  std::atomic<bool> batch_get_exception{false};

  std::thread batch_put_blocker(
      [&queue, &batch_put_started, &batch_put_exception]() {
        batch_put_started = true;
        try {
          std::vector<int> items = {3, 4, 5};
          queue.Put(absl::Span<const int>(items));  // Should block
        } catch (const QueueClosedException&) {
          batch_put_exception = true;
        }
      });

  Queue<int> empty_queue(5);
  std::thread batch_get_blocker(
      [&empty_queue, &batch_get_started, &batch_get_exception]() {
        batch_get_started = true;
        try {
          empty_queue.Get(3);  // Should block
        } catch (const QueueClosedException&) {
          batch_get_exception = true;
        }
      });

  // Give the blocker threads time to block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(batch_put_started);
  EXPECT_TRUE(batch_get_started);
  EXPECT_FALSE(batch_put_exception);
  EXPECT_FALSE(batch_get_exception);

  // Close both queues
  queue.Close();
  empty_queue.Close();

  batch_put_blocker.join();
  batch_get_blocker.join();
  EXPECT_TRUE(batch_put_exception);
  EXPECT_TRUE(batch_get_exception);
}

}  // namespace lczero