#pragma once

#include <absl/container/fixed_array.h>
#include <absl/random/random.h>

#include <cstddef>
#include <deque>
#include <numeric>
#include <optional>

namespace lczero {
namespace training {

// Returns a number between [lower_bound, upper_bound) in shuffled order.
// Both bounds can be changed at any time, and the stream will adapt
// accordingly. Not thread-safe.
class StreamShuffler {
 public:
  // Sets the upper bound (exclusive). Can only be increased.
  void SetUpperBound(size_t upper_bound);

  // Sets the lower bound (inclusive). Can only be increased.
  void SetLowerBound(size_t lower_bound);

  // Sets the bucket size for internal storage optimization.
  void SetBucketSize(size_t bucket_size) { bucket_size_ = bucket_size; }

  // Returns the next item in shuffled order, or nullopt if exhausted.
  std::optional<size_t> GetNextItem();

  // Resets the shuffler to restart iteration with specified bounds.
  void Reset(size_t lower_bound, size_t upper_bound);

 private:
  class Bucket {
   public:
    Bucket(size_t lower_bound, size_t capacity);
    size_t GetRemainingCapacity() const;
    void Extend(size_t new_upper_bound);
    size_t Fetch(size_t item_idx);
    void DeclareLowerBound(size_t new_lower_bound);

    size_t upper_bound() const { return upper_bound_; }
    size_t size() const { return items_count_; }

   private:
    size_t upper_bound_ = 0;
    size_t items_count_ = 0;
    absl::FixedArray<size_t> items_;
  };

  absl::BitGen gen_;
  std::deque<Bucket> buckets_;
  size_t stream_size_ = 0;
  size_t upper_bound_ = 0;
  size_t lower_bound_ = 0;
  size_t bucket_size_ = 524288;
};

}  // namespace training
}  // namespace lczero