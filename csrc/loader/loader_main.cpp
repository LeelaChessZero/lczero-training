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
#include "proto/data_loader_config.pb.h"
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

  // Configure file path provider stage.
  auto* file_stage = config.add_stage();
  file_stage->set_name("file_path_provider");
  auto* file_path_provider = file_stage->mutable_file_path_provider();
  file_path_provider->set_directory(absl::GetFlag(FLAGS_directory));

  // Configure chunk source loader stage.
  auto* chunk_loader_stage = config.add_stage();
  chunk_loader_stage->set_name("chunk_source_loader");
  auto* chunk_source_loader = chunk_loader_stage->mutable_chunk_source_loader();
  chunk_source_loader->set_input(file_stage->name());

  // Configure shuffling chunk pool stage.
  auto* chunk_pool_stage = config.add_stage();
  chunk_pool_stage->set_name("shuffling_chunk_pool");
  auto* shuffling_chunk_pool = chunk_pool_stage->mutable_shuffling_chunk_pool();
  shuffling_chunk_pool->set_input(chunk_loader_stage->name());
  shuffling_chunk_pool->set_chunk_pool_size(
      absl::GetFlag(FLAGS_chunk_pool_size));

  // Configure chunk unpacker stage.
  auto* unpacker_stage = config.add_stage();
  unpacker_stage->set_name("chunk_unpacker");
  auto* chunk_unpacker = unpacker_stage->mutable_chunk_unpacker();
  chunk_unpacker->set_input(chunk_pool_stage->name());

  // Configure shuffling frame sampler stage.
  auto* sampler_stage = config.add_stage();
  sampler_stage->set_name("shuffling_frame_sampler");
  auto* shuffling_frame_sampler =
      sampler_stage->mutable_shuffling_frame_sampler();
  shuffling_frame_sampler->set_input(unpacker_stage->name());
  shuffling_frame_sampler->set_reservoir_size_per_thread(
      absl::GetFlag(FLAGS_reservoir_size_per_thread));

  // Configure tensor generator stage.
  auto* tensor_stage = config.add_stage();
  tensor_stage->set_name("tensor_generator");
  auto* tensor_generator = tensor_stage->mutable_tensor_generator();
  tensor_generator->set_input(sampler_stage->name());

  // Serialize config and create loader
  config.add_output("tensor_generator");
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
    TensorTuple batch = loader.GetNext("train");
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
