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
  {
    auto producer = queue.CreateProducer();
    producer.Put(42);
    EXPECT_EQ(queue.Size(), 1);
  }  // Producer destroyed here, queue closes

  int value = queue.Get();
  EXPECT_EQ(value, 42);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, MovePutGet) {
  Queue<std::unique_ptr<int>> queue(5);
  {
    auto producer = queue.CreateProducer();
    auto ptr = std::make_unique<int>(42);
    producer.Put(std::move(ptr));
    EXPECT_EQ(queue.Size(), 1);
  }  // Producer destroyed here, queue closes

  auto result = queue.Get();
  EXPECT_EQ(*result, 42);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, MultiplePutGet) {
  Queue<int> queue(5);

  {
    auto producer = queue.CreateProducer();
    for (int i = 0; i < 5; ++i) {
      producer.Put(i);
    }
    EXPECT_EQ(queue.Size(), 5);
  }  // Producer destroyed here, queue closes

  for (int i = 0; i < 5; ++i) {
    int value = queue.Get();
    EXPECT_EQ(value, i);
  }
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, CircularBufferBehavior) {
  Queue<int> queue(3);
  auto producer = queue.CreateProducer();

  // Fill queue
  producer.Put(1);
  producer.Put(2);
  producer.Put(3);

  // Get one item, put another
  EXPECT_EQ(queue.Get(), 1);
  producer.Put(4);

  // Verify remaining items
  EXPECT_EQ(queue.Get(), 2);
  EXPECT_EQ(queue.Get(), 3);
  EXPECT_EQ(queue.Get(), 4);
}

// Batch operations tests

TEST_F(QueueTest, BatchPutConstSpan) {
  Queue<int> queue(5);
  std::vector<int> items = {1, 2, 3};

  {
    auto producer = queue.CreateProducer();
    producer.Put(absl::Span<const int>(items));
    EXPECT_EQ(queue.Size(), 3);
  }  // Producer destroyed here, queue closes

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

  {
    auto producer = queue.CreateProducer();
    producer.Put(absl::Span<std::unique_ptr<int>>(items));
    EXPECT_EQ(queue.Size(), 3);
  }  // Producer destroyed here, queue closes

  for (int i = 0; i < 3; ++i) {
    auto result = queue.Get();
    EXPECT_EQ(*result, i + 1);
  }
}

TEST_F(QueueTest, BatchPutEmptySpan) {
  Queue<int> queue(5);
  std::vector<int> empty_items;

  {
    auto producer = queue.CreateProducer();
    producer.Put(absl::Span<const int>(empty_items));
    EXPECT_EQ(queue.Size(), 0);
  }  // Producer destroyed here, queue closes
}

TEST_F(QueueTest, BatchGet) {
  Queue<int> queue(5);

  {
    auto producer = queue.CreateProducer();
    for (int i = 0; i < 5; ++i) {
      producer.Put(i);
    }
  }  // Producer destroyed here, queue closes

  auto result = queue.Get(3);
  EXPECT_EQ(result.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], i);
  }
  EXPECT_EQ(queue.Size(), 2);
}

TEST_F(QueueTest, BatchGetZeroCount) {
  Queue<int> queue(5);
  {
    auto producer = queue.CreateProducer();
    producer.Put(42);
  }  // Producer destroyed here, queue closes

  auto result = queue.Get(0);
  EXPECT_EQ(result.size(), 0);
  EXPECT_EQ(queue.Size(), 1);
}

// Edge cases and error conditions

TEST_F(QueueTest, CapacityOne) {
  Queue<int> queue(1);

  {
    auto producer = queue.CreateProducer();
    producer.Put(42);
    EXPECT_EQ(queue.Size(), 1);
  }  // Producer destroyed here, queue closes

  EXPECT_EQ(queue.Get(), 42);
  EXPECT_EQ(queue.Size(), 0);
}

// Tests for operations when all producer tokens are destroyed

TEST_F(QueueTest, CreateProducerOnClosedQueue) {
  Queue<int> queue(5);
  // Create and immediately destroy producer to close queue
  {
    auto producer = queue.CreateProducer();
  }

  // Trying to create a new producer after queue is closed results in an
  // exception.
  EXPECT_THROW(queue.CreateProducer(), QueueClosedException);
}

TEST_F(QueueTest, GetOnClosedQueue) {
  Queue<int> queue(5);
  // Create and immediately destroy producer to close queue
  {
    auto producer = queue.CreateProducer();
  }

  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, BatchGetOnClosedQueue) {
  Queue<int> queue(5);
  // Create and immediately destroy producer to close queue
  {
    auto producer = queue.CreateProducer();
  }

  EXPECT_THROW(queue.Get(3), QueueClosedException);
}

// Thread safety tests

TEST_F(QueueTest, SingleProducerSingleConsumer) {
  Queue<int> queue(10);
  std::atomic<bool> producer_done{false};
  std::vector<int> consumed;

  std::thread producer([&queue, &producer_done]() {
    auto prod = queue.CreateProducer();
    for (int i = 0; i < 100; ++i) {
      prod.Put(i);
    }
    producer_done = true;
    // Producer destroyed here, closing the queue
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
  constexpr int total_items = num_producers * items_per_producer;

  std::vector<int> all_consumed;
  std::vector<std::thread> producers;

  // Use a single producer token that we control explicitly
  auto producer_token = queue.CreateProducer();

  // Start producers - they all share the same producer token via reference
  for (int p = 0; p < num_producers; ++p) {
    producers.emplace_back([&producer_token, p]() {
      for (int i = 0; i < items_per_producer; ++i) {
        int value = p * items_per_producer + i;
        producer_token.Put(value);
      }
    });
  }

  // Wait for all producers to finish
  for (auto& producer : producers) {
    producer.join();
  }

  // Now explicitly close the queue by destroying the producer token
  {
    auto temp = std::move(producer_token);
  }  // Queue is now closed

  // Now consume all items from the closed queue
  for (int i = 0; i < total_items; ++i) {
    all_consumed.push_back(queue.Get());
  }

  // Verify all items were consumed
  EXPECT_EQ(all_consumed.size(), total_items);
  EXPECT_EQ(queue.Size(), 0);

  // Trying to get one more should throw
  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, BlockingBehaviorOnFullQueue) {
  Queue<int> queue(2);
  std::atomic<bool> put_blocked{false};
  std::atomic<bool> put_completed{false};
  auto producer = queue.CreateProducer();

  // Fill the queue
  producer.Put(1);
  producer.Put(2);

  std::thread blocker([&producer, &put_blocked, &put_completed]() {
    put_blocked = true;
    producer.Put(3);  // This should block
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
  auto producer = queue.CreateProducer();

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
  producer.Put(42);

  blocker.join();
  EXPECT_TRUE(get_completed);
  EXPECT_EQ(result, 42);
}

TEST_F(QueueTest, ProducerDestructionUnblocksWaitingGet) {
  Queue<int> queue(5);  // Empty queue

  std::atomic<bool> get_started{false};
  std::atomic<bool> exception_thrown{false};

  // Create a producer to keep queue open initially
  std::unique_ptr<Queue<int>::Producer> producer =
      std::make_unique<Queue<int>::Producer>(queue.CreateProducer());

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

  // Destroy the producer - this should close queue and unblock the waiting
  // Get()
  producer.reset();

  blocker.join();
  EXPECT_TRUE(exception_thrown);
}

// Test: Get() should not throw when queue is closed but has elements
TEST_F(QueueTest, GetFromClosedQueueWithElements) {
  Queue<int> queue(5);

  // Put some elements in the queue, then destroy producer to close it
  {
    auto producer = queue.CreateProducer();
    producer.Put(1);
    producer.Put(2);
    producer.Put(3);
    EXPECT_EQ(queue.Size(), 3);
  }  // Producer destroyed here, queue closes

  // Should be able to get elements that were already in the queue
  EXPECT_EQ(queue.Get(), 1);
  EXPECT_EQ(queue.Get(), 2);
  EXPECT_EQ(queue.Get(), 3);
  EXPECT_EQ(queue.Size(), 0);

  // Only now should Get() throw when queue is empty and closed
  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, BatchGetFromClosedQueueWithElements) {
  Queue<int> queue(5);

  // Put some elements in the queue, then destroy producer to close it
  {
    auto producer = queue.CreateProducer();
    producer.Put(1);
    producer.Put(2);
    producer.Put(3);
    EXPECT_EQ(queue.Size(), 3);
  }  // Producer destroyed here, queue closes

  // Should be able to get elements that were already in the queue
  auto result = queue.Get(2);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(queue.Size(), 1);

  // Get remaining element
  EXPECT_EQ(queue.Get(), 3);
  EXPECT_EQ(queue.Size(), 0);

  // Only now should Get() throw when queue is empty and closed
  EXPECT_THROW(queue.Get(1), QueueClosedException);
}

// Test producer token mechanism specifically
TEST_F(QueueTest, ProducerTokenMechanism) {
  Queue<int> queue(5);

  // Create multiple producers
  auto producer1 = queue.CreateProducer();
  auto producer2 = queue.CreateProducer();

  // Both should be able to put items
  producer1.Put(1);
  producer2.Put(2);
  EXPECT_EQ(queue.Size(), 2);

  // Destroy one producer - queue should still be open
  {
    auto temp = std::move(producer1);
  }  // producer1 is destroyed here
  producer2.Put(3);
  EXPECT_EQ(queue.Size(), 3);

  // Destroy last producer - queue should close
  {
    auto temp = std::move(producer2);
  }  // producer2 is destroyed here

  // Should still be able to get existing items
  EXPECT_EQ(queue.Get(), 1);
  EXPECT_EQ(queue.Get(), 2);
  EXPECT_EQ(queue.Get(), 3);

  // But trying to get more should throw
  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, ProducerMoveSemantics) {
  Queue<int> queue(5);

  auto producer1 = queue.CreateProducer();
  producer1.Put(42);

  // Move constructor
  auto producer2 = std::move(producer1);
  producer2.Put(43);
  EXPECT_EQ(queue.Size(), 2);

  // Create another producer and use move assignment
  auto producer3 = queue.CreateProducer();
  producer3 = std::move(producer2);
  producer3.Put(44);
  EXPECT_EQ(queue.Size(), 3);

  // Destroy the last producer
  {
    auto temp = std::move(producer3);
  }  // producer3 is destroyed here

  // Should be able to get all items
  EXPECT_EQ(queue.Get(), 42);
  EXPECT_EQ(queue.Get(), 43);
  EXPECT_EQ(queue.Get(), 44);
  EXPECT_THROW(queue.Get(), QueueClosedException);
}

}  // namespace lczero