#include <absl/algorithm/container.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/string_view.h>
#include <absl/types/span.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "loader/chunk_source/chunk_source.h"
#include "loader/chunk_source/tar_chunk_source.h"
#include "trainingdata/trainingdata_v6.h"

ABSL_FLAG(std::string, input_dir, ".", "Directory to scan for .tar files.");
ABSL_FLAG(std::string, output_csv, "",
          "Destination CSV file. Writes to stdout if empty.");

namespace {

namespace fs = std::filesystem;

using ::lczero::training::ChunkSource;
using ::lczero::training::ChunkSourceLoaderConfig;
using ::lczero::training::FrameType;
using ::lczero::training::TarChunkSource;

constexpr std::array<uint64_t, 16> kStartPositionPlanes = {
    0x000000000000ff00ull, 0x0000000000000042ull, 0x0000000000000024ull,
    0x0000000000000081ull, 0x0000000000000010ull, 0x0000000000000008ull,
    0x00ff000000000000ull, 0x4200000000000000ull, 0x2400000000000000ull,
    0x8100000000000000ull, 0x1000000000000000ull, 0x0800000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull};

using PolicyProbe = std::pair<int, absl::string_view>;

constexpr std::array<PolicyProbe, 20> kPolicyProbes = {
    {{378, "g2g4"}, {346, "f2f3"}, {34, "b1a3"},  {161, "g1h3"},
     {403, "h2h4"}, {351, "f2f4"}, {234, "b2b4"}, {207, "a2a4"},
     {288, "d2d3"}, {204, "a2a3"}, {259, "c2c3"}, {36, "b1c3"},
     {400, "h2h3"}, {230, "b2b3"}, {322, "e2e4"}, {317, "e2e3"},
     {374, "g2g3"}, {264, "c2c4"}, {159, "g1f3"}, {293, "d2d4"}}};

bool MatchesStartPosition(const FrameType& data) {
  return absl::c_equal(
      kStartPositionPlanes,
      absl::Span<const uint64_t>(data.planes, kStartPositionPlanes.size()));
}

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

void WriteHeader(std::ostream& output) {
  output << "file,index";
  for (const auto& probe : kPolicyProbes) output << ',' << probe.second;
  output << '\n';
}

void WriteRow(std::ostream& output, absl::string_view sort_key, size_t index,
              const FrameType& data) {
  output << sort_key << ',' << index;
  for (const auto& probe : kPolicyProbes) {
    output << ',' << data.probabilities[probe.first];
  }
  output << '\n';
}

void ProcessTarFile(const fs::path& tar_path, std::ostream& output) {
  std::unique_ptr<ChunkSource> source = std::make_unique<TarChunkSource>(
      tar_path, ChunkSourceLoaderConfig::V6TrainingData);
  const std::string sort_key = source->GetChunkSortKey();

  for (size_t i = 0, count = source->GetChunkCount(); i < count; ++i) {
    const std::optional<std::vector<FrameType>> chunk = source->GetChunkData(i);
    if (!chunk || chunk->empty()) continue;

    const FrameType& entry = chunk->front();
    if (!MatchesStartPosition(entry)) continue;

    WriteRow(output, sort_key, i, entry);
  }
}

std::ostream& SelectOutput(const fs::path& output_path,
                           std::ofstream& file_stream) {
  if (output_path.empty()) return std::cout;
  file_stream.open(output_path, std::ios::out | std::ios::trunc);
  if (!file_stream) {
    LOG(FATAL) << "Failed to open output file: " << output_path.string();
  }
  return file_stream;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  const fs::path input_dir(absl::GetFlag(FLAGS_input_dir));
  const fs::path output_path(absl::GetFlag(FLAGS_output_csv));

  if (!fs::is_directory(input_dir)) {
    LOG(FATAL) << "Input directory does not exist: " << input_dir.string();
  }

  std::ofstream file_stream;
  std::ostream& output = SelectOutput(output_path, file_stream);

  WriteHeader(output);

  for (const auto& tar_path : CollectTarFiles(input_dir)) {
    LOG(INFO) << "Processing tar file: " << tar_path.string();
    try {
      ProcessTarFile(tar_path, output);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to process tar file " << tar_path << ": "
                   << e.what();
    }
  }

  return 0;
}
