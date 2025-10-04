#include "loader/chunk_source/debug_chunk_source.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <random>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"

namespace lczero {
namespace training {

DebugChunkSource::DebugChunkSource(uint64_t id, double mean_chunk_count)
    : id_(id), mean_chunk_count_(mean_chunk_count) {}

std::string DebugChunkSource::GetChunkSortKey() const {
  return absl::StrFormat("%08" PRIu64, id_);
}

void DebugChunkSource::Index() {}

size_t DebugChunkSource::GetChunkCount() const {
  if (!cached_chunk_count_.has_value()) {
    std::mt19937_64 rng(id_);
    const double stddev = std::max(1.0, mean_chunk_count_ / 4.0);
    std::normal_distribution<double> distribution(mean_chunk_count_, stddev);
    const double sampled = distribution(rng);
    const auto rounded =
        static_cast<long long>(std::llround(std::max(sampled, 1.0)));
    cached_chunk_count_ = static_cast<size_t>(rounded);
  }
  return *cached_chunk_count_;
}

std::optional<std::string> DebugChunkSource::GetChunkData(size_t index) {
  const auto seed_pair = std::make_pair(id_, index);
  const uint64_t seed = static_cast<uint64_t>(
      absl::Hash<std::pair<uint64_t, size_t>>{}(seed_pair));
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> frame_count_distribution(1, 200);
  const int frame_count = frame_count_distribution(rng);

  const size_t bytes_per_frame = sizeof(FrameType);
  std::string chunk(static_cast<size_t>(frame_count) * bytes_per_frame, '\0');
  char* chunk_data = chunk.data();
  FrameType frame{};
  frame.planes[0] = static_cast<uint64_t>(id_);
  frame.planes[1] = static_cast<uint64_t>(index);

  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    frame.planes[2] = static_cast<uint64_t>(frame_index);
    std::memcpy(chunk_data + static_cast<size_t>(frame_index) * bytes_per_frame,
                &frame, bytes_per_frame);
  }

  return chunk;
}

}  // namespace training
}  // namespace lczero
