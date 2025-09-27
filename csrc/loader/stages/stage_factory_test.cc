#include "loader/stages/stage_factory.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace lczero {
namespace training {

TEST(StageFactoryTest, CreatesFilePathProviderStage) {
  StageConfig config;
  config.mutable_file_path_provider()->set_directory(".");

  auto stage = CreateStage(config, {});

  ASSERT_NE(stage, nullptr);
  EXPECT_NE(stage->GetOutput(), nullptr);
}

TEST(StageFactoryTest, ThrowsWhenNoStageConfigSet) {
  StageConfig config;

  EXPECT_THROW(CreateStage(config, {}), std::runtime_error);
}

TEST(StageFactoryTest, ThrowsWhenMultipleStageConfigsSet) {
  StageConfig config;
  config.mutable_file_path_provider()->set_directory(".");
  config.mutable_tensor_generator();

  EXPECT_THROW(CreateStage(config, {}), std::runtime_error);
}

}  // namespace training
}  // namespace lczero
