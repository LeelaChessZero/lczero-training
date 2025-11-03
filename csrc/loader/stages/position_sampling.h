#pragma once

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

float ComputePositionSamplingWeight(const FrameType& frame,
                                    const PositionSamplingConfig& config);

}  // namespace training
}  // namespace lczero