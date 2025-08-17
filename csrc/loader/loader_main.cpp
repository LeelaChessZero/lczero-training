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
#include "proto/training_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/printer.h"

ABSL_FLAG(std::string, directory, "/home/crem/tmp/2025-07/lczero-training/",
          "Directory to watch for training data files");
ABSL_FLAG(size_t, chunk_pool_size, 1000000, "Size of the chunk shuffle buffer");
ABSL_FLAG(size_t, reservoir_size_per_thread, 1000000,
          "Size of the reservoir for frame sampling per thread");

namespace lczero {
namespace training {

void Run() {
  DataLoaderConfig config;

  // Configure file path provider
  auto* file_path_provider = config.mutable_file_path_provider();
  file_path_provider->set_directory(absl::GetFlag(FLAGS_directory));

  // Configure shuffling chunk pool
  auto* shuffling_chunk_pool = config.mutable_shuffling_chunk_pool();
  shuffling_chunk_pool->set_chunk_pool_size(
      absl::GetFlag(FLAGS_chunk_pool_size));

  // Configure shuffling frame sampler
  auto* shuffling_frame_sampler = config.mutable_shuffling_frame_sampler();
  shuffling_frame_sampler->set_reservoir_size_per_thread(
      absl::GetFlag(FLAGS_reservoir_size_per_thread));

  // Serialize config and create loader
  std::string config_string = config.OutputAsString();
  DataLoader loader(config_string);

  return;

  std::atomic<size_t> batch_count = 0;
  auto start_time = absl::Now();

  // Start logging thread
  std::atomic<bool> should_stop{false};
  std::thread logging_thread([&]() {
    while (!should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      auto current_time = absl::Now();
      auto total_elapsed = current_time - start_time;
      double rate = batch_count / absl::ToDoubleSeconds(total_elapsed);

      auto [stats_string, duration] =
          loader.GetBucketMetrics(0, false);  // k1Second = 0
      DataLoaderMetricsProto metrics;
      metrics.ParseFromString(stats_string);
      std::string metrics_json = metrics.OutputAsJson();

      LOG(INFO) << absl::StrCat("Processed ", batch_count.load(),
                                " batches in ",
                                absl::ToDoubleSeconds(total_elapsed),
                                "s. Rate: ", absl::StrFormat("%.2f", rate),
                                " batches/sec. ", "Metrics: ", metrics_json);
    }
  });

  while (true) {
    TensorTuple batch = loader.GetNext();
    ++batch_count;
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
