#include <absl/log/initialize.h>
#include <absl/log/globals.h>

#include <iostream>

#include "data_loader.h"

namespace lczero {
namespace ice_skate {

void Run() {
  DataLoaderConfig config{.training_data_path =
                              "/home/crem/tmp/2025-07/lczero-training/"};
  DataLoader loader(config);
}

}  // namespace ice_skate
}  // namespace lczero

int main(int, char*[]) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  lczero::ice_skate::Run();
  return 0;
}
