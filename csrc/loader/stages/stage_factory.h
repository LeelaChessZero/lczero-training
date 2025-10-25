#pragma once

#include <memory>

#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

std::unique_ptr<Stage> CreateStage(const StageConfig& config,
                                   const StageRegistry& existing_stages);

}  // namespace training
}  // namespace lczero
