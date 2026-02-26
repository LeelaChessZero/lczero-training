#pragma once

#include "loader/frame_type.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

float ComputePositionSamplingWeight(const FrameType& frame,
                                    const PositionSamplingConfig& config);

}  // namespace training
}  // namespace lczero