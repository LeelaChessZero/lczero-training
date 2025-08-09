#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <chrono>
#include <iostream>

#include "data_loader.h"
#include "utils/metrics/printer.h"

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

  while (true) {
    TensorTuple batch = loader.GetNext();
    ++batch_count;

    auto current_time = absl::Now();
    auto total_elapsed = current_time - start_time;
    double rate = batch_count / absl::ToDoubleSeconds(total_elapsed);

    // Log metrics every second
    LOG_EVERY_N_SEC(INFO, 1) << [&]() {
      auto [metrics, duration] =
          loader.GetMetricsAggregator().GetAggregateEndingNow(
              std::chrono::seconds(1));
      std::string metrics_str;
      lczero::StringMetricPrinter printer(&metrics_str);
      metrics.Print(printer);

      return absl::StrCat("Processed ", batch_count, " batches in ",
                          absl::ToDoubleSeconds(total_elapsed),
                          "s. Rate: ", absl::StrFormat("%.2f", rate),
                          " batches/sec. ", "Metrics: ", metrics_str);
    }();
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
