#include "loader/stages/stage.h"

#include <absl/algorithm/container.h>

namespace lczero {
namespace training {

void StageRegistry::AddStage(std::string_view stage_name,
                             std::unique_ptr<Stage> stage) {
  if (absl::c_find_if(stages_, [&](const auto& pair) {
        return pair.first == stage_name;
      }) != stages_.end()) {
    throw std::runtime_error(
        absl::StrCat("Duplicate stage name detected: ", stage_name));
  }

  stages_.emplace_back(stage_name, std::move(stage));
}

QueueBase* StageRegistry::GetStageOutput(std::string_view stage_name) const {
  auto [actual_stage_name, output_name] = [&stage_name]() {
    size_t dot_pos = stage_name.find('.');
    return dot_pos == std::string_view::npos
               ? std::pair{stage_name, std::string_view{}}
               : std::pair{stage_name.substr(0, dot_pos),
                           stage_name.substr(dot_pos + 1)};
  }();

  auto it = absl::c_find_if(stages_, [&](const auto& pair) {
    return pair.first == actual_stage_name;
  });
  return it != stages_.end() ? it->second->GetOutput(output_name) : nullptr;
}

}  // namespace training
}  // namespace lczero