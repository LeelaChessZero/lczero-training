#include <absl/algorithm/container.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "loader/chunk_source/chunk_source.h"
#include "loader/chunk_source/tar_chunk_source.h"
#include "loader/stages/position_sampling.h"
#include "proto/data_loader_config.pb.h"
#include "trainingdata/trainingdata_v6.h"
#include "utils/training_data_printer.h"

ABSL_FLAG(std::string, input_dir, "", "Directory to scan for .tar files.");
ABSL_FLAG(float, q_weight, 6.0, "Value for diff_focus_q_weight.");
ABSL_FLAG(float, pol_scale, 3.5, "Value for diff_focus_pol_scale.");

namespace {

namespace fs = std::filesystem;

using ::lczero::training::ChunkSource;
using ::lczero::training::ChunkSourceLoaderConfig;
using ::lczero::training::ComputePositionSamplingWeight;
using ::lczero::training::FrameType;
using ::lczero::training::PositionSamplingConfig;
using ::lczero::training::PrintTrainingDataEntry;
using ::lczero::training::TarChunkSource;

struct WeightedPosition {
  FrameType data;
  float weight;
};

std::vector<fs::path> CollectTarFiles(const fs::path& directory) {
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(directory)) {
    const fs::path& path = entry.path();
    if (entry.is_regular_file() && path.extension() == ".tar") {
      files.push_back(path);
    }
  }
  absl::c_sort(files, [](const fs::path& lhs, const fs::path& rhs) {
    return lhs.filename() < rhs.filename();
  });
  return files;
}

std::vector<float> CollectWeights(const fs::path& tar_path,
                                  const PositionSamplingConfig& config,
                                  WeightedPosition* max_weighted) {
  std::vector<float> weights;
  std::unique_ptr<ChunkSource> source = std::make_unique<TarChunkSource>(
      tar_path, ChunkSourceLoaderConfig::V6TrainingData);

  const size_t total = source->GetChunkCount();
  for (size_t index = 0; index < total; ++index) {
    if (index % 1000 == 0) {
      LOG(INFO) << absl::StreamFormat("  Progress: %zu/%zu chunks (%.1f%%)",
                                      index, total, 100.0 * index / total);
    }

    const std::optional<std::vector<FrameType>> chunk =
        source->GetChunkData(index);
    if (!chunk) {
      LOG(WARNING) << "Skipping unreadable chunk " << index << " in "
                   << tar_path.string();
      continue;
    }

    if (chunk->empty()) continue;

    for (const auto& entry : *chunk) {
      const float weight = ComputePositionSamplingWeight(entry, config);
      weights.push_back(weight);
      if (max_weighted && weight > max_weighted->weight) {
        max_weighted->data = entry;
        max_weighted->weight = weight;
      }
    }
  }

  return weights;
}

void PrintHistogram(const std::vector<float>& sorted_weights) {
  if (sorted_weights.empty()) return;

  constexpr int kBuckets = 50;
  constexpr int kMaxWidth = 60;
  const float min_val = sorted_weights.front();
  const float max_val = sorted_weights.back();
  const float range = max_val - min_val;

  if (range == 0.0f) {
    LOG(INFO) << "\nHistogram: All weights are identical (" << min_val << ")";
    return;
  }

  std::vector<int> buckets(kBuckets, 0);
  for (float weight : sorted_weights) {
    int bucket = static_cast<int>((weight - min_val) / range * (kBuckets - 1));
    bucket = std::clamp(bucket, 0, kBuckets - 1);
    ++buckets[bucket];
  }

  const int max_count = *absl::c_max_element(buckets);
  if (max_count == 0) return;

  LOG(INFO) << "\nHistogram:";
  for (int bucket = 0; bucket < kBuckets; ++bucket) {
    if (buckets[bucket] == 0) continue;

    const float bucket_start = min_val + range * bucket / kBuckets;
    const float bucket_end = min_val + range * (bucket + 1) / kBuckets;
    const int width = (buckets[bucket] * kMaxWidth + max_count / 2) / max_count;

    std::string bar;
    for (int i = 0; i < width; ++i) bar += "█";

    LOG(INFO) << absl::StreamFormat("[%.4f-%.4f) │%s (%d)", bucket_start,
                                    bucket_end, bar, buckets[bucket]);
  }
}

void PrintPercentiles(const std::vector<float>& sorted_weights) {
  if (sorted_weights.empty()) return;

  LOG(INFO) << "\nPercentiles:";
  for (int p = 0; p <= 100; ++p) {
    const size_t idx = (sorted_weights.size() - 1) * p / 100;
    LOG(INFO) << absl::StreamFormat("  %3d%%: %.6f", p, sorted_weights[idx]);
  }
}

void PrintStatistics(const std::vector<float>& weights) {
  if (weights.empty()) {
    LOG(INFO) << "No weights collected.";
    return;
  }

  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  const double mean = sum / weights.size();

  LOG(INFO) << "\nStatistics:";
  LOG(INFO) << "  Total positions: " << weights.size();
  LOG(INFO) << absl::StreamFormat("  Mean: %.6f", mean);
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string input_dir_str = absl::GetFlag(FLAGS_input_dir);
  const float q_weight = absl::GetFlag(FLAGS_q_weight);
  const float pol_scale = absl::GetFlag(FLAGS_pol_scale);

  if (input_dir_str.empty()) {
    LOG(FATAL) << "--input_dir must be specified.";
  }

  const fs::path input_dir(input_dir_str);
  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    LOG(FATAL) << "Input directory does not exist: " << input_dir.string();
  }

  PositionSamplingConfig config;
  config.set_diff_focus_q_weight(q_weight);
  config.set_diff_focus_pol_scale(pol_scale);

  const std::vector<fs::path> tar_files = CollectTarFiles(input_dir);
  LOG(INFO) << "Found " << tar_files.size() << " tar file(s).";

  WeightedPosition max_weighted = {{}, 0.0f};
  std::vector<float> all_weights;
  for (const auto& tar_path : tar_files) {
    LOG(INFO) << "Processing: " << tar_path.string();
    std::vector<float> weights =
        CollectWeights(tar_path, config, &max_weighted);
    all_weights.insert(all_weights.end(), weights.begin(), weights.end());
    LOG(INFO) << "  Collected " << weights.size() << " position(s).";
  }

  absl::c_sort(all_weights);

  PrintStatistics(all_weights);
  PrintPercentiles(all_weights);
  PrintHistogram(all_weights);

  if (max_weighted.weight > 0.0f) {
    const std::string header = absl::StrFormat(
        "\nPosition with highest weight (%.6f):", max_weighted.weight);
    PrintTrainingDataEntry(max_weighted.data, header, 8, 4);
  }

  return 0;
}
