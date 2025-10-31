#include "loader/stages/stage_factory.h"

#include <stdexcept>
#include <string>

#include "loader/stages/chunk_rescorer.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/chunk_source_splitter.h"
#include "loader/stages/chunk_unpacker.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/shuffling_chunk_pool.h"
#include "loader/stages/shuffling_frame_sampler.h"
#include "loader/stages/simple_chunk_extractor.h"
#include "loader/stages/tensor_generator.h"

namespace lczero {
namespace training {

namespace {

int CountStageConfigs(const StageConfig& config) {
  return static_cast<int>(config.has_file_path_provider()) +
         static_cast<int>(config.has_chunk_source_loader()) +
         static_cast<int>(config.has_shuffling_chunk_pool()) +
         static_cast<int>(config.has_chunk_rescorer()) +
         static_cast<int>(config.has_chunk_unpacker()) +
         static_cast<int>(config.has_shuffling_frame_sampler()) +
         static_cast<int>(config.has_tensor_generator()) +
         static_cast<int>(config.has_chunk_source_splitter()) +
         static_cast<int>(config.has_simple_chunk_extractor());
}

}  // namespace

std::unique_ptr<Stage> CreateStage(const StageConfig& config) {
  if (CountStageConfigs(config) != 1) {
    throw std::runtime_error(
        "StageConfig must have exactly one stage-specific config set.");
  }

  if (config.has_file_path_provider()) {
    return std::make_unique<FilePathProvider>(config.file_path_provider());
  }
  if (config.has_chunk_source_loader()) {
    return std::make_unique<ChunkSourceLoader>(config.chunk_source_loader());
  }
  if (config.has_shuffling_chunk_pool()) {
    return std::make_unique<ShufflingChunkPool>(config.shuffling_chunk_pool());
  }
  if (config.has_chunk_rescorer()) {
    return std::make_unique<ChunkRescorer>(config.chunk_rescorer());
  }
  if (config.has_chunk_unpacker()) {
    return std::make_unique<ChunkUnpacker>(config.chunk_unpacker());
  }
  if (config.has_shuffling_frame_sampler()) {
    return std::make_unique<ShufflingFrameSampler>(
        config.shuffling_frame_sampler());
  }
  if (config.has_tensor_generator()) {
    return std::make_unique<TensorGenerator>(config.tensor_generator());
  }
  if (config.has_chunk_source_splitter()) {
    return std::make_unique<ChunkSourceSplitter>(
        config.chunk_source_splitter());
  }
  if (config.has_simple_chunk_extractor()) {
    return std::make_unique<SimpleChunkExtractor>(
        config.simple_chunk_extractor());
  }

  throw std::runtime_error(
      "StageConfig did not contain a recognized stage configuration.");
}

}  // namespace training
}  // namespace lczero
