#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "chess/board.h"
#include "proto/data_loader_config.pb.h"
#include "syzygy/syzygy.h"
#include "trainingdata/reader.h"
#include "trainingdata/rescorer.h"
#include "trainingdata/trainingdata_v6.h"
#include "trainingdata/writer.h"
#include "utils/exception.h"

ABSL_FLAG(std::string, chunk_path, "",
          "Path to the chunk file (.gz) that should be rescored.");
ABSL_FLAG(std::string, syzygy_paths, "",
          "Comma-separated list of Syzygy tablebase directories.");
ABSL_FLAG(double, dist_temp, 1.0,
          "Policy temperature applied during rescoring.");
ABSL_FLAG(double, dist_offset, 0.0, "Policy offset applied during rescoring.");
ABSL_FLAG(double, dtz_boost, 0.0,
          "DTZ boost applied during policy adjustments.");
ABSL_FLAG(int, new_input_format, -1,
          "Optional conversion target for input format (-1 keeps original).");
ABSL_FLAG(
    double, deblunder_threshold, -1.0,
    "Threshold for policy deblundering adjustments (negative to disable).");
ABSL_FLAG(
    double, deblunder_width, -1.0,
    "Width controlling smoothing around threshold (negative to disable).");

namespace {

namespace fs = std::filesystem;
using ::lczero::training::ChunkRescorerConfig;

std::vector<lczero::V6TrainingData> ReadChunkFrames(const fs::path& path) {
  std::vector<lczero::V6TrainingData> frames;
  lczero::TrainingDataReader reader(path.string());
  lczero::V6TrainingData frame;
  while (reader.ReadChunk(&frame)) {
    frames.push_back(frame);
  }
  return frames;
}

void WriteChunkFrames(const fs::path& path,
                      const std::vector<lczero::V6TrainingData>& frames) {
  lczero::TrainingDataWriter writer(path.string());
  for (const auto& frame : frames) {
    writer.WriteChunk(frame);
  }
  writer.Finalize();
}

fs::path BuildOutputPath(const fs::path& input_path) {
  fs::path directory = input_path.parent_path();
  fs::path stem = input_path.stem();
  fs::path filename = stem;
  filename += "_rescored.gz";
  return directory / filename;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string chunk_path_flag = absl::GetFlag(FLAGS_chunk_path);
  if (chunk_path_flag.empty()) {
    LOG(FATAL) << "--chunk_path flag is required.";
  }
  const fs::path chunk_path(chunk_path_flag);

  ChunkRescorerConfig config;

  const std::string syzygy_paths_flag = absl::GetFlag(FLAGS_syzygy_paths);
  if (!syzygy_paths_flag.empty()) {
    config.set_syzygy_paths(syzygy_paths_flag);
  }
  config.set_dist_temp(static_cast<float>(absl::GetFlag(FLAGS_dist_temp)));
  config.set_dist_offset(static_cast<float>(absl::GetFlag(FLAGS_dist_offset)));
  config.set_dtz_boost(static_cast<float>(absl::GetFlag(FLAGS_dtz_boost)));
  config.set_new_input_format(
      static_cast<int32_t>(absl::GetFlag(FLAGS_new_input_format)));

  const double deblunder_threshold_flag =
      absl::GetFlag(FLAGS_deblunder_threshold);
  const double deblunder_width_flag = absl::GetFlag(FLAGS_deblunder_width);
  if (deblunder_threshold_flag >= 0.0 && deblunder_width_flag >= 0.0) {
    config.set_deblunder_threshold(
        static_cast<float>(deblunder_threshold_flag));
    config.set_deblunder_width(static_cast<float>(deblunder_width_flag));
  } else if (deblunder_threshold_flag >= 0.0 || deblunder_width_flag >= 0.0) {
    LOG(FATAL) << "Both --deblunder_threshold and --deblunder_width must be "
               << "set to non-negative values together.";
  }

  LOG(INFO) << "Reading chunk from " << chunk_path.string();
  std::vector<lczero::V6TrainingData> frames;
  try {
    frames = ReadChunkFrames(chunk_path);
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to read chunk: " << exception.what();
  }
  LOG(INFO) << "Loaded " << frames.size() << " frame(s) from chunk.";
  if (frames.empty()) {
    LOG(WARNING) << "Chunk contains no frames; writing empty output.";
    try {
      WriteChunkFrames(BuildOutputPath(chunk_path), frames);
    } catch (const lczero::Exception& exception) {
      LOG(FATAL) << "Failed to write rescored chunk: " << exception.what();
    }
    return 0;
  }

  lczero::InitializeMagicBitboards();

  if (config.has_deblunder_threshold() && config.has_deblunder_width()) {
    lczero::RescorerDeblunderSetup(config.deblunder_threshold(),
                                   config.deblunder_width());
  }

  lczero::SyzygyTablebase tablebase;
  if (!config.syzygy_paths().empty()) {
    LOG(INFO) << "Initializing Syzygy tablebases from '"
              << config.syzygy_paths() << "'.";
    const std::string syzygy_paths(config.syzygy_paths());
    if (!tablebase.init(syzygy_paths)) {
      LOG(WARNING) << "Failed to initialize Syzygy tablebases.";
    }
  }

  LOG(INFO) << "Rescoring chunk with dist_temp=" << config.dist_temp()
            << ", dist_offset=" << config.dist_offset()
            << ", dtz_boost=" << config.dtz_boost()
            << ", new_input_format=" << config.new_input_format() << ".";

  try {
    frames = lczero::RescoreTrainingData(
        std::move(frames), &tablebase, config.dist_temp(), config.dist_offset(),
        config.dtz_boost(), config.new_input_format());
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to rescore chunk: " << exception.what();
  }

  const fs::path output_path = BuildOutputPath(chunk_path);
  LOG(INFO) << "Writing rescored chunk to " << output_path.string();
  try {
    WriteChunkFrames(output_path, frames);
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to write rescored chunk: " << exception.what();
  }
  LOG(INFO) << "Completed rescoring of chunk.";

  return 0;
}
