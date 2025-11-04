#include "loader/stages/position_sampling.h"

#include <cmath>

namespace lczero {
namespace training {

float ComputePositionSamplingWeight(const FrameType& frame,
                                    const PositionSamplingConfig& config) {
  if (!config.has_diff_focus_q_weight() && !config.has_diff_focus_pol_scale()) {
    return 1.0;
  }
  const float diff_q = std::abs(frame.best_q - frame.orig_q);
  const float q_weight = config.diff_focus_q_weight();
  const float pol_scale = config.diff_focus_pol_scale();
  const float total =
      (q_weight * diff_q + frame.policy_kld) / (q_weight + pol_scale);
  return std::min(
      std::pow(total * config.diff_focus_alpha() + config.diff_focus_beta(),
               config.diff_focus_gamma()),
      config.diff_focus_tau());
}

}  // namespace training
}  // namespace lczero