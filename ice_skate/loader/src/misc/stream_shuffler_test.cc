#include "misc/stream_shuffler.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>

#include <set>
#include <vector>

namespace lczero {
namespace ice_skate {

class StreamShufflerTest : public ::testing::Test {
 protected:
  void SetUp() override { shuffler_.SetBucketSize(4); }

  StreamShuffler shuffler_;
};

TEST_F(StreamShufflerTest, EmptyRangeReturnsNullopt) {
  shuffler_.SetHeadBound(10);
  shuffler_.SetTailBound(10);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, SingleItemRange) {
  shuffler_.SetHeadBound(1);
  shuffler_.SetTailBound(0);

  auto item = shuffler_.GetNextItem();
  ASSERT_TRUE(item.has_value());
  EXPECT_EQ(item.value(), 0);

  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

TEST_F(StreamShufflerTest, BasicRangeGeneration) {
  shuffler_.SetHeadBound(5);
  shuffler_.SetTailBound(0);

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
  shuffler_.SetHeadBound(4);
  shuffler_.SetTailBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    received.insert(item.value());
  }
  EXPECT_EQ(received.size(), 4);

  shuffler_.SetHeadBound(8);
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
  shuffler_.SetHeadBound(3);
  shuffler_.SetTailBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    received.insert(item.value());
  }

  shuffler_.SetHeadBound(7);
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
  shuffler_.SetHeadBound(12);
  shuffler_.SetTailBound(0);

  std::set<size_t> received;
  for (int i = 0; i < 4; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    received.insert(item.value());
  }

  shuffler_.SetTailBound(4);
  std::set<size_t> remaining_received;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 4);
    EXPECT_LT(item.value(), 12);
    EXPECT_TRUE(remaining_received.insert(item.value()).second);
  }
  EXPECT_EQ(remaining_received.size(), 8);
}

TEST_F(StreamShufflerTest, TailAdvancesByNonMultiples) {
  shuffler_.SetHeadBound(10);
  shuffler_.SetTailBound(0);

  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
  }

  shuffler_.SetTailBound(3);
  std::set<size_t> remaining_received;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 3);
    EXPECT_LT(item.value(), 10);
    EXPECT_TRUE(remaining_received.insert(item.value()).second);
  }
  EXPECT_EQ(remaining_received.size(), 7);
}

TEST_F(StreamShufflerTest, BothBoundsSlideSimultaneously) {
  shuffler_.SetHeadBound(10);
  shuffler_.SetTailBound(0);

  for (int i = 0; i < 5; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
  }

  shuffler_.SetHeadBound(15);
  shuffler_.SetTailBound(5);

  std::set<size_t> remaining_received;
  std::optional<size_t> item;
  while ((item = shuffler_.GetNextItem()).has_value()) {
    EXPECT_GE(item.value(), 5);
    EXPECT_LT(item.value(), 15);
    EXPECT_TRUE(remaining_received.insert(item.value()).second);
  }
  EXPECT_EQ(remaining_received.size(), 10);
}

TEST_F(StreamShufflerTest, ComplexSlidingWindow) {
  std::set<size_t> all_received;

  shuffler_.SetHeadBound(6);
  shuffler_.SetTailBound(0);

  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    all_received.insert(item.value());
  }

  shuffler_.SetHeadBound(11);
  for (int i = 0; i < 2; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
    all_received.insert(item.value());
  }

  shuffler_.SetTailBound(2);
  shuffler_.SetHeadBound(14);

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
  shuffler_.SetHeadBound(20);
  shuffler_.SetTailBound(0);

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
  shuffler_.SetHeadBound(8);
  shuffler_.SetTailBound(0);

  for (int i = 0; i < 3; ++i) {
    auto item = shuffler_.GetNextItem();
    ASSERT_TRUE(item.has_value());
  }

  shuffler_.SetTailBound(8);
  EXPECT_EQ(shuffler_.GetNextItem(), std::nullopt);
}

}  // namespace ice_skate
}  // namespace lczero