#include "misc/stream_shuffler.h"

namespace lczero {
namespace ice_skate {

void StreamShuffler::SetHeadBound(size_t head_bound) {
  assert(head_bound >= head_bound_);
  stream_size_ += head_bound - head_bound_;
  while (head_bound_ < head_bound) {
    if (buckets_.empty() || buckets_.back().GetRemainingCapacity() == 0) {
      buckets_.emplace_back(head_bound_, bucket_size_);
    }
    head_bound_ = std::min(
        head_bound, head_bound_ + buckets_.back().GetRemainingCapacity());
    buckets_.back().Extend(head_bound_);
  }
}

void StreamShuffler::SetTailBound(size_t tail_bound) {
  assert(tail_bound >= tail_bound_);
  tail_bound_ = tail_bound;
  if (tail_bound >= head_bound_) {
    head_bound_ = tail_bound;
    stream_size_ = 0;
    buckets_.clear();
    return;
  }
  while (!buckets_.empty() && buckets_.front().upper_bound() <= tail_bound_) {
    stream_size_ -= buckets_.front().size();
    buckets_.pop_front();
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
    if (auto item = try_fetch(); item >= tail_bound_) return item;
  }

  return std::nullopt;
}

StreamShuffler::Bucket::Bucket(size_t lower_bound, size_t capacity)
    : lower_bound_(lower_bound), upper_bound_(lower_bound), items_(capacity) {}

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

}  // namespace ice_skate
}  // namespace lczero