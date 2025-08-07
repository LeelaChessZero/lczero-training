#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <iostream>

#include "data_loader.h"

ABSL_FLAG(std::string, directory, "/home/crem/tmp/2025-07/lczero-training/",
          "Directory to watch for training data files");
ABSL_FLAG(size_t, chunk_pool_size, 1000000, "Size of the chunk shuffle buffer");
ABSL_FLAG(size_t, reservoir_size_per_thread, 1000000,
          "Size of the reservoir for frame sampling per thread");

namespace lczero {
namespace training {

void Run() {
  DataLoaderConfig config{
      .file_path_provider = {.directory = absl::GetFlag(FLAGS_directory)},
      .chunk_source_loader = {},
      .shuffling_chunk_pool = {.chunk_pool_size =
                                   absl::GetFlag(FLAGS_chunk_pool_size)},
      .chunk_unpacker = {},
      .shuffling_frame_sampler = {.reservoir_size_per_thread = absl::GetFlag(
                                      FLAGS_reservoir_size_per_thread)},
      .tensor_generator = {}};
  DataLoader loader(config);

  size_t batch_count = 0;
  auto start_time = absl::Now();
  auto last_report_time = start_time;

  while (true) {
    TensorTuple batch = loader.GetNext();
    ++batch_count;

    auto current_time = absl::Now();
    auto elapsed_since_report = current_time - last_report_time;

    // Report every 10 seconds
    if (elapsed_since_report >= absl::Seconds(10)) {
      auto total_elapsed = current_time - start_time;
      double rate = batch_count / absl::ToDoubleSeconds(total_elapsed);

      std::cout << "Processed " << batch_count << " batches in "
                << absl::ToDoubleSeconds(total_elapsed) << " seconds. "
                << "Rate: " << rate << " batches/sec" << std::endl;

      last_report_time = current_time;
    }
  }
}

}  // namespace training
}  // namespace lczero

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  lczero::training::Run();
  return 0;
}
