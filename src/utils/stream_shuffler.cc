#include "utils/stream_shuffler.h"

namespace lczero {
namespace training {

void StreamShuffler::SetUpperBound(size_t upper_bound) {
  assert(upper_bound >= upper_bound_);
  stream_size_ += upper_bound - upper_bound_;
  while (upper_bound_ < upper_bound) {
    if (buckets_.empty() || buckets_.back().GetRemainingCapacity() == 0) {
      buckets_.emplace_back(upper_bound_, bucket_size_);
    }
    upper_bound_ = std::min(
        upper_bound, upper_bound_ + buckets_.back().GetRemainingCapacity());
    buckets_.back().Extend(upper_bound_);
  }
}

void StreamShuffler::SetLowerBound(size_t lower_bound) {
  assert(lower_bound >= lower_bound_);
  lower_bound_ = lower_bound;
  if (lower_bound >= upper_bound_) {
    upper_bound_ = lower_bound;
    stream_size_ = 0;
    buckets_.clear();
    return;
  }
  while (!buckets_.empty() && buckets_.front().upper_bound() <= lower_bound_) {
    stream_size_ -= buckets_.front().size();
    buckets_.pop_front();
  }
  if (!buckets_.empty()) {
    auto old_size = buckets_.front().size();
    buckets_.front().DeclareLowerBound(lower_bound_);
    stream_size_ -= old_size - buckets_.front().size();
  }
}

std::optional<size_t> StreamShuffler::GetNextItem() {
  auto try_fetch = [&]() -> size_t {
    size_t item_idx = absl::Uniform(gen_, size_t{0}, stream_size_);
    --stream_size_;
    for (auto& bucket : buckets_) {
      if (item_idx < bucket.size()) return bucket.Fetch(item_idx);
      item_idx -= bucket.size();
    }
    throw std::logic_error("StreamShuffler: item index out of bounds");
  };

  while (stream_size_ > 0) {
    if (auto item = try_fetch(); item >= lower_bound_) return item;
  }

  return std::nullopt;
}

void StreamShuffler::Reset(size_t lower_bound, size_t upper_bound) {
  // Reset all internal state
  buckets_.clear();
  stream_size_ = 0;
  upper_bound_ = lower_bound;
  lower_bound_ = lower_bound;

  // Establish the bounds, which will build the buckets with fresh data
  if (upper_bound > lower_bound) {
    SetUpperBound(upper_bound);
  }
}

StreamShuffler::Bucket::Bucket(size_t lower_bound, size_t capacity)
    : upper_bound_(lower_bound), items_(capacity) {}

size_t StreamShuffler::Bucket::GetRemainingCapacity() const {
  return items_.size() - items_count_;
}

void StreamShuffler::Bucket::Extend(size_t new_upper_bound) {
  assert(new_upper_bound >= upper_bound_);
  const size_t increase = new_upper_bound - upper_bound_;
  assert(increase <= GetRemainingCapacity());
  std::iota(items_.begin() + items_count_,
            items_.begin() + items_count_ + increase, upper_bound_);
  items_count_ += increase;
  upper_bound_ = new_upper_bound;
}

size_t StreamShuffler::Bucket::Fetch(size_t item_idx) {
  assert(item_idx < items_count_);
  size_t item = items_[item_idx];
  std::swap(items_[item_idx], items_[--items_count_]);
  return item;
}

void DeclareLowerBound(size_t new_lower_bound);
void StreamShuffler::Bucket::DeclareLowerBound(size_t new_lower_bound) {
  if (upper_bound_ - new_lower_bound < 2 * items_count_) return;

  // If the bucket has much more items that the allowed range, there are many
  // items out of the range. It makes sense to sort and remove them.
  std::sort(items_.begin(), items_.begin() + items_count_,
            std::greater<size_t>());
  // Find the first item that is under the new lower bound.
  auto it = std::upper_bound(items_.begin(), items_.begin() + items_count_,
                             new_lower_bound, std::greater<size_t>());
  items_count_ = it - items_.begin();
}

}  // namespace training
}  // namespace lczero