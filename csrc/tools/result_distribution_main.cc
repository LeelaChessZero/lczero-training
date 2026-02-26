#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/string_view.h>
#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "loader/chunk_source/tar_chunk_source.h"
#include "trainingdata/trainingdata_v6.h"

ABSL_FLAG(std::string, output_csv, "",
          "Destination CSV file. Writes to stdout if empty.");
ABSL_FLAG(int, num_threads, 0,
          "Number of worker threads. Defaults to hardware concurrency.");

namespace {

namespace fs = std::filesystem;

using ::lczero::training::ChunkSourceLoaderConfig;
using ::lczero::training::FrameType;
using ::lczero::training::TarChunkSource;

enum class ChunkResult { kWin, kDraw, kLoss };

struct ResultCounts {
  uint64_t wins = 0;
  uint64_t draws = 0;
  uint64_t losses = 0;
};

class CsvWriter {
 public:
  CsvWriter(std::ostream* output, absl::Mutex* mutex)
      : output_(output), mutex_(mutex) {}

  void Write(absl::string_view basename, const ResultCounts& counts) const {
    absl::MutexLock lock(mutex_);
    *output_ << basename << ',' << counts.wins << ',' << counts.draws << ','
             << counts.losses << '\n';
    output_->flush();
  }

 private:
  std::ostream* output_;
  absl::Mutex* mutex_;
};

std::ostream& SelectOutput(const fs::path& output_path,
                           std::ofstream& file_stream) {
  if (output_path.empty()) return std::cout;

  file_stream.open(output_path, std::ios::out | std::ios::trunc);
  if (!file_stream) {
    LOG(FATAL) << "Failed to open output file: " << output_path.string();
  }
  return file_stream;
}

constexpr float kFloatTolerance = 1e-6f;

std::optional<ChunkResult> DetermineChunkResult(absl::string_view chunk_payload,
                                                size_t chunk_index,
                                                const fs::path& tar_path) {
  if (chunk_payload.size() < sizeof(FrameType)) {
    LOG(WARNING) << "Chunk " << chunk_index << " in " << tar_path.string()
                 << " is too small.";
    return std::nullopt;
  }

  FrameType frame;
  std::memcpy(&frame, chunk_payload.data(), sizeof(frame));

  if (std::fabs(frame.result_d - 1.0f) <= kFloatTolerance) {
    return ChunkResult::kDraw;
  }
  if (std::fabs(frame.result_d) > kFloatTolerance) {
    LOG(WARNING) << "Chunk " << chunk_index << " in " << tar_path.string()
                 << " has unexpected result_d=" << frame.result_d << '.';
    return std::nullopt;
  }

  const bool side_to_move = frame.side_to_move_or_enpassant != 0;
  if (std::fabs(frame.result_q - 1.0f) <= kFloatTolerance) {
    return side_to_move ? ChunkResult::kLoss : ChunkResult::kWin;
  }
  if (std::fabs(frame.result_q + 1.0f) <= kFloatTolerance) {
    return side_to_move ? ChunkResult::kWin : ChunkResult::kLoss;
  }

  LOG(WARNING) << "Chunk " << chunk_index << " in " << tar_path.string()
               << " has unexpected result_q=" << frame.result_q << '.';
  return std::nullopt;
}

ResultCounts CountResultsInTar(const fs::path& tar_path) {
  ResultCounts counts;

  TarChunkSource source(tar_path, ChunkSourceLoaderConfig::V6TrainingData);

  const size_t chunk_count = source.GetChunkCount();
  for (size_t index = 0; index < chunk_count; ++index) {
    const std::optional<std::string> chunk =
        source.GetChunkPrefix(index, sizeof(FrameType));
    if (!chunk) {
      LOG(WARNING) << "Skipping unreadable chunk " << index << " in "
                   << tar_path.string();
      continue;
    }

    const std::optional<ChunkResult> result =
        DetermineChunkResult(*chunk, index, tar_path);
    if (!result) continue;

    switch (*result) {
      case ChunkResult::kWin:
        ++counts.wins;
        break;
      case ChunkResult::kDraw:
        ++counts.draws;
        break;
      case ChunkResult::kLoss:
        ++counts.losses;
        break;
    }
  }

  return counts;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  if (positional.size() <= 1) {
    LOG(FATAL) << "Provide at least one .tar file as a positional argument.";
  }

  const fs::path output_path(absl::GetFlag(FLAGS_output_csv));
  std::ofstream file_stream;
  std::ostream& output = SelectOutput(output_path, file_stream);
  absl::Mutex output_mutex;
  const CsvWriter writer(&output, &output_mutex);

  std::vector<fs::path> tar_files;
  tar_files.reserve(positional.size() - 1);
  for (size_t i = 1; i < positional.size(); ++i) {
    tar_files.emplace_back(positional[i]);
  }

  const int num_threads_flag = absl::GetFlag(FLAGS_num_threads);
  size_t worker_count = 0;
  if (num_threads_flag > 0) {
    worker_count = static_cast<size_t>(num_threads_flag);
  } else {
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    worker_count = hw_threads > 0 ? static_cast<size_t>(hw_threads) : 1;
  }
  worker_count = std::max<size_t>(1, std::min(worker_count, tar_files.size()));

  std::atomic<size_t> next_index(0);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
    workers.emplace_back([&tar_files, &next_index, &writer]() {
      while (true) {
        const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= tar_files.size()) break;
        const fs::path& tar_path = tar_files[index];
        LOG(INFO) << "Processing tar file: " << tar_path.string();
        try {
          const ResultCounts counts = CountResultsInTar(tar_path);
          writer.Write(tar_path.filename().string(), counts);
        } catch (const std::exception& exception) {
          LOG(WARNING) << "Failed to process tar file " << tar_path.string()
                       << ": " << exception.what();
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  return 0;
}
