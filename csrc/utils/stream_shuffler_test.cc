#include "utils/stream_shuffler.h"

#include <absl/container/flat_hash_set.h>
#include <absl/random/random.h>
#include <gtest/gtest.h>

#include <set>
#include <vector>

namespace lczero {
namespace training {

class StreamShufflerTest : public ::testing::Test {
 protected:
  void SetUp() override { shuffler_.SetBucketSize(4); }

  StreamShuffler shuffler_;
};

TEST_F(StreamShufflerTest, EmptyRangeReturnsNullopt) {
  shuffler_.SetUpperBound(10);
  shuffler_.SetLowerBound(10);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, SingleItemRange) {
  shuffler_.SetUpperBound(1);
  shuffler_.SetLowerBound(0);

  auto item = shuffler_.GetNextItem();
  ASSERT_TRUE(item.has_value());
  EXPECT_EQ(item.value(), 0);

  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, BasicRangeGeneration) {
  shuffler_.SetUpperBound(5);
  shuffler_.SetLowerBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 5; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 5);
    EXPECT_TRUE(received.insert(item.value()).second);
  }

  EXPECT_EQ(received.size(), 5);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, HeadAdvancesByBucketMultiples) {
  shuffler_.SetUpperBound(4);
  shuffler_.SetLowerBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    received.insert(item.value());
  }
  EXPECT_EQ(received.size(), 4);

  shuffler_.SetUpperBound(8);
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 8);
    EXPECT_TRUE(received.insert(item.value()).second);
  }
  EXPECT_EQ(received.size(), 8);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, HeadAdvancesByNonMultiples) {
  shuffler_.SetUpperBound(3);
  shuffler_.SetLowerBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    received.insert(item.value());
  }

  shuffler_.SetUpperBound(7);
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 7);
    EXPECT_TRUE(received.insert(item.value()).second);
  }
  EXPECT_EQ(received.size(), 7);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, TailAdvancesByBucketMultiples) {
  shuffler_.SetUpperBound(12);
  shuffler_.SetLowerBound(0);

  std::set<size_t> all_received;
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 12);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  shuffler_.SetLowerBound(4);
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 4);
    EXPECT_LT(item.value(), 12);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  // Verify all items in range [4, 12) were eventually fetched
  for (size_t i = 4; i < 12; ++i) {
    EXPECT_TRUE(all_received.count(i) > 0)
        << "Item " << i << " was never fetched";
  }
}

TEST_F(StreamShufflerTest, TailAdvancesByNonMultiples) {
  shuffler_.SetUpperBound(10);
  shuffler_.SetLowerBound(0);

  std::set<size_t> all_received;
  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 10);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  shuffler_.SetLowerBound(3);
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 3);
    EXPECT_LT(item.value(), 10);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  // Verify all items in range [3, 10) were eventually fetched
  for (size_t i = 3; i < 10; ++i) {
    EXPECT_TRUE(all_received.count(i) > 0)
        << "Item " << i << " was never fetched";
  }
}

TEST_F(StreamShufflerTest, BothBoundsSlideSimultaneously) {
  shuffler_.SetUpperBound(10);
  shuffler_.SetLowerBound(0);

  std::set<size_t> all_received;
  for (int i = 0; i < 5; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 10);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  shuffler_.SetUpperBound(15);
  shuffler_.SetLowerBound(5);

  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 5);
    EXPECT_LT(item.value(), 15);
    EXPECT_TRUE(all_received.insert(item.value()).second);
  }

  // Verify all items in range [5, 15) were eventually fetched
  for (size_t i = 5; i < 15; ++i) {
    EXPECT_TRUE(all_received.count(i) > 0)
        << "Item " << i << " was never fetched";
  }
}

TEST_F(StreamShufflerTest, ComplexSlidingWindow) {
  std::set<size_t> all_received;

  shuffler_.SetUpperBound(6);
  shuffler_.SetLowerBound(0);

  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    all_received.insert(item.value());
  }

  shuffler_.SetUpperBound(11);
  for (int i = 0; i < 2; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    all_received.insert(item.value());
  }

  shuffler_.SetLowerBound(2);
  shuffler_.SetUpperBound(14);

  std::set<size_t> final_received;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 2);
    EXPECT_LT(item.value(), 14);
    EXPECT_TRUE(final_received.insert(item.value()).second);
  }

  for (const auto& val : final_received) {
    EXPECT_GE(val, 2);
    EXPECT_LT(val, 14);
  }
}

TEST_F(StreamShufflerTest, UniquenessAcrossMultipleBuckets) {
  shuffler_.SetUpperBound(20);
  shuffler_.SetLowerBound(0);

  std::set<size_t> received;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 0);
    EXPECT_LT(item.value(), 20);
    EXPECT_TRUE(received.insert(item.value()).second);
  }

  EXPECT_EQ(received.size(), 20);
}

TEST_F(StreamShufflerTest, TailCatchesUpToHead) {
  shuffler_.SetUpperBound(8);
  shuffler_.SetLowerBound(0);

  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
  }

  shuffler_.SetLowerBound(8);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, ResetAllowsIterationRestart) {
  shuffler_.SetUpperBound(5);
  shuffler_.SetLowerBound(0);

  // Exhaust all items
  absl::flat_hash_set<size_t> first_round;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    first_round.insert(item.value());
  }

  // Should have gotten all 5 items
  EXPECT_EQ(first_round.size(), 5);

  // Shuffler should be exhausted
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);

  // Reset the shuffler
  shuffler_.Reset(0, 5);

  // Should be able to get items again
  absl::flat_hash_set<size_t> second_round;
  int count = 0;
  while ((item = shuffler_.GetNextItem()).has_value() && count < 10) {
    second_round.insert(item.value());
    count++;
  }

  // Should get all items again
  EXPECT_EQ(second_round.size(), 5);

  // Both rounds should contain the same set of items
  EXPECT_EQ(first_round, second_round);
}

}  // namespace training
}  // namespace lczero