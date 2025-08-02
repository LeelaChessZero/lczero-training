#include <absl/log/globals.h>
#include <absl/log/initialize.h>

#include <iostream>

#include "data_loader.h"

namespace lczero {
namespace training {

void Run() {
  DataLoaderConfig config{
      .training_data_path = "/home/crem/tmp/2025-07/lczero-training/",
      .num_chunks_window = 1000};
  DataLoader loader(config);
}

}  // namespace training
}  // namespace lczero

int main(int, char*[]) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  lczero::training::Run();
  return 0;
}
