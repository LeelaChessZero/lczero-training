#include "loader/stages/stage_factory.h"

#include <stdexcept>
#include <string>

#include "loader/stages/chunk_rescorer.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/chunk_unpacker.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/shuffling_chunk_pool.h"
#include "loader/stages/shuffling_frame_sampler.h"
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
         static_cast<int>(config.has_tensor_generator());
}

}  // namespace

std::unique_ptr<Stage> CreateStage(const StageConfig& config,
                                   const Stage::StageList& existing_stages) {
  if (CountStageConfigs(config) != 1) {
    throw std::runtime_error(
        "StageConfig must have exactly one stage-specific config set.");
  }

  if (config.has_file_path_provider()) {
    return std::make_unique<FilePathProvider>(config.file_path_provider(),
                                              existing_stages);
  }
  if (config.has_chunk_source_loader()) {
    return std::make_unique<ChunkSourceLoader>(config.chunk_source_loader(),
                                               existing_stages);
  }
  if (config.has_shuffling_chunk_pool()) {
    return std::make_unique<ShufflingChunkPool>(config.shuffling_chunk_pool(),
                                                existing_stages);
  }
  if (config.has_chunk_rescorer()) {
    return std::make_unique<ChunkRescorer>(config.chunk_rescorer(),
                                           existing_stages);
  }
  if (config.has_chunk_unpacker()) {
    return std::make_unique<ChunkUnpacker>(config.chunk_unpacker(),
                                           existing_stages);
  }
  if (config.has_shuffling_frame_sampler()) {
    return std::make_unique<ShufflingFrameSampler>(
        config.shuffling_frame_sampler(), existing_stages);
  }
  if (config.has_tensor_generator()) {
    return std::make_unique<TensorGenerator>(config.tensor_generator(),
                                             existing_stages);
  }

  throw std::runtime_error(
      "StageConfig did not contain a recognized stage configuration.");
}

}  // namespace training
}  // namespace lczero
