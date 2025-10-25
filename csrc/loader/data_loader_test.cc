#include "loader/data_loader.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace lczero {
namespace training {

TEST(DataLoaderTest, AllowsNoOutputsConfigured) {
  DataLoaderConfig config;
  auto* file_stage = config.add_stage();
  file_stage->set_name("file_path_provider");
  file_stage->mutable_file_path_provider()->set_directory(".");

  EXPECT_NO_THROW(DataLoader(config.OutputAsString()));
}

TEST(DataLoaderTest, ThrowsOnDuplicateStageName) {
  DataLoaderConfig config;
  auto* first_stage = config.add_stage();
  first_stage->set_name("duplicate");
  first_stage->mutable_file_path_provider()->set_directory(".");

  auto* second_stage = config.add_stage();
  second_stage->set_name("duplicate");
  second_stage->mutable_file_path_provider()->set_directory(".");

  EXPECT_THROW(DataLoader(config.OutputAsString()), std::runtime_error);
}

}  // namespace training
}  // namespace lczero
