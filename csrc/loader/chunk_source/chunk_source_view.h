#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "loader/chunk_source/chunk_source.h"

namespace lczero {
namespace training {

// ChunkSourceView provides a view over another ChunkSource.
// It exposes a remapped subset/order of chunks defined by indices into the
// underlying source. It does not own or copy the data; it forwards calls to
// the wrapped source.
class ChunkSourceView : public ChunkSource {
 public:
  // Constructs a view over an existing chunk source. The indices vector maps
  // local indices in the view to indices of the underlying source.
  ChunkSourceView(std::shared_ptr<ChunkSource> source,
                  std::vector<uint32_t> indices)
      : source_(std::move(source)), indices_(std::move(indices)) {}

  ~ChunkSourceView() override = default;

 private:
  std::string GetChunkSortKey() const override {
    return source_->GetChunkSortKey();
  }

  size_t GetChunkCount() const override { return indices_.size(); }

  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override {
    if (index >= indices_.size()) return std::nullopt;
    const size_t src_index = static_cast<size_t>(indices_[index]);
    return source_->GetChunkData(src_index);
  }

  std::shared_ptr<ChunkSource> source_;
  std::vector<uint32_t> indices_;
};

}  // namespace training
}  // namespace lczero
