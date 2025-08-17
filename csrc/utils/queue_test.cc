// ABOUTME: Comprehensive unit tests for the Queue template class
// ABOUTME: Tests thread-safe operations, blocking behavior, and edge cases

#include "utils/queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
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
  std::promise<void> about_to_block;
  std::future<void> about_to_block_future = about_to_block.get_future();
  std::atomic<bool> put_completed{false};
  auto producer = queue.CreateProducer();

  // Fill the queue
  producer.Put(1);
  producer.Put(2);

  std::thread blocker([&producer, &about_to_block, &put_completed]() {
    about_to_block.set_value();  // Signal we're about to block
    producer.Put(3);             // This should block
    put_completed = true;
  });

  // Wait for thread to signal it's about to block
  about_to_block_future.wait();
  EXPECT_FALSE(put_completed);

  // Make space in the queue
  EXPECT_EQ(queue.Get(), 1);

  blocker.join();
  EXPECT_TRUE(put_completed);
  EXPECT_EQ(queue.Size(), 2);
}

TEST_F(QueueTest, BlockingBehaviorOnEmptyQueue) {
  Queue<int> queue(5);
  std::promise<void> about_to_block;
  std::future<void> about_to_block_future = about_to_block.get_future();
  std::atomic<bool> get_completed{false};
  std::atomic<int> result{-1};
  auto producer = queue.CreateProducer();

  std::thread blocker([&queue, &about_to_block, &get_completed, &result]() {
    about_to_block.set_value();  // Signal we're about to block
    result = queue.Get();        // This should block
    get_completed = true;
  });

  // Wait for thread to signal it's about to block
  about_to_block_future.wait();
  EXPECT_FALSE(get_completed);

  // Put an item in the queue
  producer.Put(42);

  blocker.join();
  EXPECT_TRUE(get_completed);
  EXPECT_EQ(result, 42);
}

TEST_F(QueueTest, ProducerDestructionUnblocksWaitingGet) {
  Queue<int> queue(5);  // Empty queue

  std::promise<void> about_to_block;
  std::future<void> about_to_block_future = about_to_block.get_future();
  std::atomic<bool> exception_thrown{false};

  // Create a producer to keep queue open initially
  std::unique_ptr<Queue<int>::Producer> producer =
      std::make_unique<Queue<int>::Producer>(queue.CreateProducer());

  std::thread blocker([&queue, &about_to_block, &exception_thrown]() {
    about_to_block.set_value();  // Signal we're about to block
    try {
      queue.Get();  // This should block
    } catch (const QueueClosedException&) {
      exception_thrown = true;
    }
  });

  // Wait for thread to signal it's about to block
  about_to_block_future.wait();
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

// Tests for Put operations on closed queue
TEST_F(QueueTest, PutOnClosedQueueThrowsException) {
  Queue<int> queue(5);

  // Create producer and close it
  auto producer = queue.CreateProducer();
  queue.Close();

  // All Put operations should throw on closed queue
  EXPECT_THROW(producer.Put(42), QueueClosedException);
  EXPECT_THROW(producer.Put(std::move(42)), QueueClosedException);

  std::vector<int> items = {1, 2, 3};
  EXPECT_THROW(producer.Put(absl::Span<const int>(items)),
               QueueClosedException);
  EXPECT_THROW(producer.Put(absl::Span<int>(items)), QueueClosedException);
}

TEST_F(QueueTest, PutOnClosedQueueAfterProducerDestruction) {
  Queue<int> queue(5);

  // Create producer, add item, then close by destroying all producers
  auto producer = queue.CreateProducer();
  producer.Put(1);
  {
    auto temp_producer = std::move(producer);
  }  // All producers destroyed, queue closed

  // Try to create new producer after close
  EXPECT_THROW(queue.CreateProducer(), QueueClosedException);
}

TEST_F(QueueTest, BatchPutOnClosedQueueThrowsException) {
  Queue<int> queue(10);

  auto producer = queue.CreateProducer();
  queue.Close();

  // Batch put operations should throw on closed queue
  std::vector<int> items = {1, 2, 3, 4, 5};
  EXPECT_THROW(producer.Put(absl::Span<const int>(items)),
               QueueClosedException);

  std::vector<int> mutable_items = {6, 7, 8};
  EXPECT_THROW(producer.Put(absl::Span<int>(mutable_items)),
               QueueClosedException);
}

TEST_F(QueueTest, PublicCloseMethod) {
  Queue<int> queue(5);

  auto producer = queue.CreateProducer();
  producer.Put(1);
  producer.Put(2);

  // Explicitly close the queue using public Close() method
  queue.Close();

  // Put operations should now throw
  EXPECT_THROW(producer.Put(3), QueueClosedException);

  // But Get operations should still work for existing items
  EXPECT_EQ(queue.Get(), 1);
  EXPECT_EQ(queue.Get(), 2);

  // Get should throw when queue is empty and closed
  EXPECT_THROW(queue.Get(), QueueClosedException);
}

TEST_F(QueueTest, CloseUnblocksWaitingSinglePut) {
  Queue<int> queue(2);  // Small capacity
  auto producer = queue.CreateProducer();

  // Fill the queue
  producer.Put(1);
  producer.Put(2);

  std::promise<void> about_to_block;
  std::future<void> about_to_block_future = about_to_block.get_future();
  std::atomic<bool> exception_thrown{false};

  std::thread blocker([&producer, &about_to_block, &exception_thrown]() {
    about_to_block.set_value();  // Signal we're about to block
    try {
      producer.Put(3);  // This should block since queue is full
    } catch (const QueueClosedException&) {
      exception_thrown = true;
    }
  });

  // Wait for thread to signal it's about to block
  about_to_block_future.wait();
  EXPECT_FALSE(exception_thrown);

  // Close the queue - this should unblock the waiting Put()
  queue.Close();

  blocker.join();
  EXPECT_TRUE(exception_thrown);
}

TEST_F(QueueTest, CloseUnblocksWaitingBatchPut) {
  Queue<int> queue(3);  // Small capacity
  auto producer = queue.CreateProducer();

  // Fill the queue partially
  producer.Put(1);
  producer.Put(2);

  std::promise<void> about_to_block;
  std::future<void> about_to_block_future = about_to_block.get_future();
  std::atomic<bool> exception_thrown{false};

  std::thread blocker([&producer, &about_to_block, &exception_thrown]() {
    about_to_block.set_value();  // Signal we're about to block
    try {
      std::vector<int> items = {3, 4, 5};  // Need 3 slots but only 1 available
      producer.Put(absl::Span<const int>(items));  // This should block
    } catch (const QueueClosedException&) {
      exception_thrown = true;
    }
  });

  // Wait for thread to signal it's about to block
  about_to_block_future.wait();
  EXPECT_FALSE(exception_thrown);

  // Close the queue - this should unblock the waiting batch Put()
  queue.Close();

  blocker.join();
  EXPECT_TRUE(exception_thrown);
}

// Tests for new wait functions
TEST_F(QueueTest, WaitForRoomAtLeast) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  // Initially empty queue should have room >= 5
  queue.WaitForRoomAtLeast(5);
  EXPECT_EQ(queue.Size(), 0);

  // Fill queue partially
  producer.Put(1);
  producer.Put(2);

  // Should have room >= 3
  queue.WaitForRoomAtLeast(3);
  EXPECT_EQ(queue.Size(), 2);

  // Fill more
  producer.Put(3);
  producer.Put(4);

  // Should have room >= 1
  queue.WaitForRoomAtLeast(1);
  EXPECT_EQ(queue.Size(), 4);

  // Test blocking behavior
  producer.Put(5);  // Queue is now full

  std::promise<void> wait_started;
  std::future<void> wait_started_future = wait_started.get_future();
  std::atomic<bool> wait_completed{false};

  std::thread waiter([&queue, &wait_started, &wait_completed]() {
    wait_started.set_value();
    queue.WaitForRoomAtLeast(2);  // Should block until 2 slots are free
    wait_completed = true;
  });

  wait_started_future.wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  // Free up space
  queue.Get();
  queue.Get();

  waiter.join();
  EXPECT_TRUE(wait_completed);
}

TEST_F(QueueTest, WaitForRoomAtMost) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  // Fill queue partially
  producer.Put(1);
  producer.Put(2);
  producer.Put(3);

  // Should wait until room <= 2 (currently room = 2)
  queue.WaitForRoomAtMost(2);
  EXPECT_EQ(queue.Size(), 3);

  // Test blocking behavior
  std::promise<void> wait_started;
  std::future<void> wait_started_future = wait_started.get_future();
  std::atomic<bool> wait_completed{false};

  std::thread waiter([&queue, &wait_started, &wait_completed]() {
    wait_started.set_value();
    queue.WaitForRoomAtMost(1);  // Should block until room <= 1
    wait_completed = true;
  });

  wait_started_future.wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  // Add one more item to make room = 1
  producer.Put(4);

  waiter.join();
  EXPECT_TRUE(wait_completed);
  EXPECT_EQ(queue.Size(), 4);
}

TEST_F(QueueTest, WaitForSizeAtLeast) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  // Test blocking behavior on empty queue
  std::promise<void> wait_started;
  std::future<void> wait_started_future = wait_started.get_future();
  std::atomic<bool> wait_completed{false};

  std::thread waiter([&queue, &wait_started, &wait_completed]() {
    wait_started.set_value();
    queue.WaitForSizeAtLeast(3);  // Should block until size >= 3
    wait_completed = true;
  });

  wait_started_future.wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  // Add items
  producer.Put(1);
  producer.Put(2);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  producer.Put(3);  // Now size = 3

  waiter.join();
  EXPECT_TRUE(wait_completed);
  EXPECT_EQ(queue.Size(), 3);
}

TEST_F(QueueTest, WaitForSizeAtMost) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  // Initially empty, size <= 3
  queue.WaitForSizeAtMost(3);
  EXPECT_EQ(queue.Size(), 0);

  // Fill queue
  producer.Put(1);
  producer.Put(2);
  producer.Put(3);
  producer.Put(4);
  producer.Put(5);

  // Test blocking behavior
  std::promise<void> wait_started;
  std::future<void> wait_started_future = wait_started.get_future();
  std::atomic<bool> wait_completed{false};

  std::thread waiter([&queue, &wait_started, &wait_completed]() {
    wait_started.set_value();
    queue.WaitForSizeAtMost(2);  // Should block until size <= 2
    wait_completed = true;
  });

  wait_started_future.wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  // Remove items
  queue.Get();
  queue.Get();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(wait_completed);

  queue.Get();  // Now size = 2

  waiter.join();
  EXPECT_TRUE(wait_completed);
  EXPECT_EQ(queue.Size(), 2);
}

TEST_F(QueueTest, WaitFunctionsEdgeCases) {
  Queue<int> queue(3);
  auto producer = queue.CreateProducer();

  // Wait for room = 0 should work when queue is full
  producer.Put(1);
  producer.Put(2);
  producer.Put(3);
  queue.WaitForRoomAtMost(0);
  EXPECT_EQ(queue.Size(), 3);

  // Wait for size = 0 should work when queue is empty
  queue.Get();
  queue.Get();
  queue.Get();
  queue.WaitForSizeAtMost(0);
  EXPECT_EQ(queue.Size(), 0);

  // Wait for room >= capacity should always succeed
  queue.WaitForRoomAtLeast(3);
  EXPECT_EQ(queue.Size(), 0);

  // Wait for size >= 0 should always succeed
  queue.WaitForSizeAtLeast(0);
  EXPECT_EQ(queue.Size(), 0);
}

// Tests for gradual large range operations

TEST_F(QueueTest, BatchPutAtCapacityWorks) {
  Queue<int> queue(3);
  auto producer = queue.CreateProducer();

  // Putting exactly capacity worth of items should work
  std::vector<int> items = {1, 2, 3};
  producer.Put(absl::Span<const int>(items));
  EXPECT_EQ(queue.Size(), 3);
}

TEST_F(QueueTest, BatchGetAtCapacityWorks) {
  Queue<int> queue(3);
  auto producer = queue.CreateProducer();

  producer.Put(1);
  producer.Put(2);
  producer.Put(3);

  // Getting exactly capacity worth of items should work
  auto result = queue.Get(3);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
}

TEST_F(QueueTest, LargeRangePutGetGradual) {
  Queue<int> queue(3);  // Small capacity
  auto producer = queue.CreateProducer();

  // Put more items than capacity - should work gradually
  std::vector<int> large_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::thread put_thread([&producer, &large_items]() {
    producer.Put(absl::Span<const int>(large_items));
  });

  // Consume items as they become available
  std::vector<int> consumed;
  for (int i = 0; i < 10; ++i) {
    consumed.push_back(queue.Get());
  }

  put_thread.join();

  // Verify all items were transferred correctly
  EXPECT_EQ(consumed.size(), 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(consumed[i], i + 1);
  }
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, LargeRangePutMove) {
  Queue<std::unique_ptr<int>> queue(2);  // Very small capacity
  auto producer = queue.CreateProducer();

  // Create large batch of move-only items
  std::vector<std::unique_ptr<int>> large_items;
  for (int i = 1; i <= 5; ++i) {
    large_items.push_back(std::make_unique<int>(i));
  }

  std::thread put_thread([&producer, &large_items]() {
    producer.Put(absl::Span<std::unique_ptr<int>>(large_items));
  });

  // Consume items as they become available
  std::vector<std::unique_ptr<int>> consumed;
  for (int i = 0; i < 5; ++i) {
    consumed.push_back(queue.Get());
  }

  put_thread.join();

  // Verify all items were transferred correctly
  EXPECT_EQ(consumed.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(*consumed[i], i + 1);
  }
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, LargeRangeGetGradual) {
  Queue<int> queue(3);  // Small capacity
  auto producer = queue.CreateProducer();

  // Start a thread that will gradually produce items
  std::thread producer_thread([&producer]() {
    for (int i = 1; i <= 10; ++i) {
      producer.Put(i);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  // Get more items than capacity - should work gradually
  auto result = queue.Get(10);

  producer_thread.join();

  // Verify all items were retrieved correctly
  EXPECT_EQ(result.size(), 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], i + 1);
  }
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, LargeRangePutGetConcurrent) {
  Queue<int> queue(5);  // Medium capacity

  constexpr int total_items = 100;
  constexpr int batch_size = 25;

  auto producer1 = queue.CreateProducer();
  auto producer2 = queue.CreateProducer();

  std::vector<int> batch1, batch2;
  for (int i = 0; i < batch_size; ++i) {
    batch1.push_back(i);
    batch2.push_back(i + batch_size);
  }

  std::atomic<int> items_consumed{0};
  std::vector<int> all_consumed;
  all_consumed.reserve(total_items);

  // Multiple producers
  std::thread producer_thread1([&producer1, &batch1]() {
    producer1.Put(absl::Span<const int>(batch1));
    producer1.Put(absl::Span<const int>(batch1));  // Put twice
  });

  std::thread producer_thread2([&producer2, &batch2]() {
    producer2.Put(absl::Span<const int>(batch2));
    producer2.Put(absl::Span<const int>(batch2));  // Put twice
  });

  // Consumer getting in large batches
  std::thread consumer_thread([&queue, &all_consumed, &items_consumed]() {
    while (items_consumed < total_items) {
      try {
        auto batch = queue.Get(std::min(15, total_items - items_consumed));
        for (const auto& item : batch) {
          all_consumed.push_back(item);
        }
        items_consumed += batch.size();
      } catch (const QueueClosedException&) {
        break;
      }
    }
  });

  producer_thread1.join();
  producer_thread2.join();

  // Close the queue by destroying producers
  producer1.Close();
  producer2.Close();

  consumer_thread.join();

  EXPECT_EQ(all_consumed.size(), total_items);
  EXPECT_EQ(queue.Size(), 0);
}

TEST_F(QueueTest, GradualOperationsWithQueueClosure) {
  Queue<int> queue(2);  // Very small capacity
  auto producer = queue.CreateProducer();

  std::vector<int> large_batch = {1, 2, 3, 4, 5};
  std::atomic<bool> exception_caught{false};

  std::thread producer_thread([&producer, &large_batch, &exception_caught]() {
    try {
      producer.Put(absl::Span<const int>(large_batch));
    } catch (const QueueClosedException&) {
      exception_caught = true;
    }
  });

  // Let producer start putting items
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Consume a couple items
  queue.Get();  // Should get 1
  queue.Get();  // Should get 2

  // Close the queue while producer is still trying to put items
  queue.Close();

  producer_thread.join();

  EXPECT_TRUE(exception_caught);
  // Queue might have some items that were put before closure
  // but we can't predict exactly how many due to timing
}

// Tests for total put count functionality

TEST_F(QueueTest, GetTotalPutCountBasic) {
  Queue<int> queue(5);
  EXPECT_EQ(queue.GetTotalPutCount(), 0);

  {
    auto producer = queue.CreateProducer();
    producer.Put(1);
    EXPECT_EQ(queue.GetTotalPutCount(), 1);

    producer.Put(2);
    producer.Put(3);
    EXPECT_EQ(queue.GetTotalPutCount(), 3);
  }

  // Count should persist after producer destruction
  EXPECT_EQ(queue.GetTotalPutCount(), 3);

  // Count should persist after getting items
  queue.Get();
  queue.Get();
  EXPECT_EQ(queue.GetTotalPutCount(), 3);
}

TEST_F(QueueTest, GetTotalPutCountBatch) {
  Queue<int> queue(10);
  auto producer = queue.CreateProducer();

  std::vector<int> batch1 = {1, 2, 3};
  std::vector<int> batch2 = {4, 5};

  producer.Put(absl::Span<const int>(batch1));
  EXPECT_EQ(queue.GetTotalPutCount(), 3);

  producer.Put(absl::Span<const int>(batch2));
  EXPECT_EQ(queue.GetTotalPutCount(), 5);

  // Single put after batch
  producer.Put(6);
  EXPECT_EQ(queue.GetTotalPutCount(), 6);
}

TEST_F(QueueTest, GetTotalPutCountReset) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  producer.Put(1);
  producer.Put(2);
  producer.Put(3);
  EXPECT_EQ(queue.GetTotalPutCount(), 3);

  // Reset and verify return value
  EXPECT_EQ(queue.GetTotalPutCount(true), 3);
  EXPECT_EQ(queue.GetTotalPutCount(), 0);

  // Add more items
  producer.Put(4);
  EXPECT_EQ(queue.GetTotalPutCount(), 1);

  // Non-reset call should not affect counter
  EXPECT_EQ(queue.GetTotalPutCount(false), 1);
  EXPECT_EQ(queue.GetTotalPutCount(), 1);
}

TEST_F(QueueTest, GetTotalPutCountThreadSafe) {
  Queue<int> queue(50);  // Large capacity to avoid blocking
  constexpr int items_per_thread = 10;
  constexpr int num_threads = 2;

  std::vector<std::thread> threads;
  std::vector<Queue<int>::Producer> producers;

  // Create producers for each thread
  for (int t = 0; t < num_threads; ++t) {
    producers.push_back(queue.CreateProducer());
  }

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&producers, t]() {
      for (int i = 0; i < items_per_thread; ++i) {
        producers[t].Put(t * items_per_thread + i);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(queue.GetTotalPutCount(), items_per_thread * num_threads);
}

TEST_F(QueueTest, GetTotalPutCountBatchThreadSafe) {
  Queue<int> queue(100);  // Large capacity to avoid blocking

  std::vector<std::thread> threads;
  std::vector<Queue<int>::Producer> producers;

  // Create producers for each thread
  for (int t = 0; t < 2; ++t) {
    producers.push_back(queue.CreateProducer());
  }

  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&producers, t]() {
      std::vector<int> batch;
      int batch_size = (t + 1) * 5;  // 5, 10 items
      for (int i = 0; i < batch_size; ++i) {
        batch.push_back(t * 100 + i);
      }
      producers[t].Put(absl::Span<const int>(batch));
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(queue.GetTotalPutCount(), 15);  // 5 + 10
}

TEST_F(QueueTest, GetTotalPutCountWithMoveSemantics) {
  Queue<std::unique_ptr<int>> queue(5);
  auto producer = queue.CreateProducer();

  // Single move put
  auto ptr1 = std::make_unique<int>(42);
  producer.Put(std::move(ptr1));
  EXPECT_EQ(queue.GetTotalPutCount(), 1);

  // Batch move put
  std::vector<std::unique_ptr<int>> batch;
  for (int i = 0; i < 3; ++i) {
    batch.push_back(std::make_unique<int>(i));
  }
  producer.Put(absl::Span<std::unique_ptr<int>>(batch));
  EXPECT_EQ(queue.GetTotalPutCount(), 4);
}

TEST_F(QueueTest, GetTotalPutCountEmptyBatch) {
  Queue<int> queue(5);
  auto producer = queue.CreateProducer();

  std::vector<int> empty_batch;
  producer.Put(absl::Span<const int>(empty_batch));
  EXPECT_EQ(queue.GetTotalPutCount(), 0);

  producer.Put(1);
  EXPECT_EQ(queue.GetTotalPutCount(), 1);

  producer.Put(absl::Span<const int>(empty_batch));
  EXPECT_EQ(queue.GetTotalPutCount(), 1);
}

}  // namespace lczero