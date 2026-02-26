#include <absl/algorithm/container.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>
#include <absl/types/span.h>
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "loader/chunk_source/chunk_source.h"
#include "loader/chunk_source/tar_chunk_source.h"
#include "trainingdata/trainingdata_v6.h"

ABSL_FLAG(std::string, input_dir, ".", "Directory to scan for .tar files.");
ABSL_FLAG(std::string, output_dir, ".",
          "Directory where matching chunks will be written.");
ABSL_FLAG(std::string, plane_values, "",
          "Comma separated list of plane values (decimal or hex).");

namespace {

namespace fs = std::filesystem;

using ::lczero::training::ChunkSource;
using ::lczero::training::ChunkSourceLoaderConfig;
using ::lczero::training::FrameType;
using ::lczero::training::TarChunkSource;

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

std::vector<uint64_t> ParsePlaneValues(absl::string_view value_list) {
  std::vector<uint64_t> result;
  if (value_list.empty()) {
    LOG(FATAL) << "--plane_values flag must not be empty.";
  }
  for (absl::string_view token :
       absl::StrSplit(value_list, ',', absl::SkipWhitespace())) {
    token = absl::StripAsciiWhitespace(token);
    if (token.empty()) continue;

    uint64_t value = 0;
    if (absl::StartsWithIgnoreCase(token, "0x")) {
      const absl::string_view hex_part = token.substr(2);
      if (hex_part.empty() || !absl::SimpleHexAtoi(hex_part, &value)) {
        LOG(FATAL) << "Invalid hex plane value: " << token;
      }
    } else if (!absl::SimpleAtoi(token, &value)) {
      LOG(FATAL) << "Invalid decimal plane value: " << token;
    }
    result.push_back(value);
  }
  if (result.empty()) {
    LOG(FATAL) << "No plane values were parsed.";
  }
  return result;
}

bool PlanesMatch(const FrameType& entry, absl::Span<const uint64_t> expected) {
  if (expected.size() > std::size(entry.planes)) return false;
  const size_t bytes = expected.size() * sizeof(uint64_t);
  return std::memcmp(entry.planes, expected.data(), bytes) == 0;
}

std::optional<size_t> FindMatchingFrameIndex(
    const std::vector<FrameType>& chunk, absl::Span<const uint64_t> expected) {
  for (size_t frame = 0; frame < chunk.size(); ++frame) {
    if (PlanesMatch(chunk[frame], expected)) return frame;
  }
  return std::nullopt;
}

void WriteChunk(const fs::path& output_dir, absl::string_view base_name,
                size_t index, size_t frame_index,
                const std::vector<FrameType>& chunk) {
  fs::create_directories(output_dir);
  const fs::path output_path =
      output_dir / absl::StrCat(base_name, "_", index, "_", frame_index, ".gz");

  gzFile file = gzopen(output_path.string().c_str(), "wb");
  if (file == nullptr) {
    LOG(FATAL) << "Failed to open output file: " << output_path.string();
  }

  size_t remaining = chunk.size() * sizeof(FrameType);
  const char* data = reinterpret_cast<const char*>(chunk.data());
  while (remaining > 0) {
    const unsigned int to_write = static_cast<unsigned int>(
        std::min<size_t>(remaining, std::numeric_limits<unsigned int>::max()));
    const int written = gzwrite(file, data, to_write);
    if (written == 0) {
      int errnum = 0;
      const char* error_message = gzerror(file, &errnum);
      gzclose(file);
      LOG(FATAL) << "Failed to write chunk: " << error_message;
    }
    data += written;
    remaining -= static_cast<size_t>(written);
  }

  if (gzclose(file) != Z_OK) {
    LOG(FATAL) << "Failed to close output file: " << output_path.string();
  }

  LOG(INFO) << "Wrote matching chunk to " << output_path.string();
}

void ProcessTar(const fs::path& tar_path, const fs::path& output_dir,
                absl::Span<const uint64_t> expected_planes) {
  std::unique_ptr<ChunkSource> source = std::make_unique<TarChunkSource>(
      tar_path, ChunkSourceLoaderConfig::V6TrainingData);

  const std::string base_name = tar_path.stem().string();
  size_t written_count = 0;

  for (size_t index = 0, total = source->GetChunkCount(); index < total;
       ++index) {
    const std::optional<std::vector<FrameType>> chunk =
        source->GetChunkData(index);
    if (!chunk) {
      LOG(WARNING) << "Skipping unreadable chunk " << index << " in "
                   << tar_path.string();
      continue;
    }

    const std::optional<size_t> match =
        FindMatchingFrameIndex(*chunk, expected_planes);
    if (!match) continue;

    WriteChunk(output_dir, base_name, index, *match, *chunk);
    ++written_count;
  }

  LOG(INFO) << "Finished processing " << tar_path.string() << ": wrote "
            << written_count << " chunk(s).";
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const fs::path input_dir(absl::GetFlag(FLAGS_input_dir));
  const fs::path output_dir(absl::GetFlag(FLAGS_output_dir));
  const std::string plane_values = absl::GetFlag(FLAGS_plane_values);

  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    LOG(FATAL) << "Input directory does not exist: " << input_dir.string();
  }

  const std::vector<uint64_t> expected_planes = ParsePlaneValues(plane_values);

  fs::create_directories(output_dir);

  const std::vector<fs::path> tar_files = CollectTarFiles(input_dir);
  const absl::Span<const uint64_t> expected_span(expected_planes);

  std::vector<std::thread> workers;
  workers.reserve(tar_files.size());

  for (const auto& tar_path : tar_files) {
    workers.emplace_back([tar_path, output_dir, expected_span]() {
      LOG(INFO) << "Processing tar file: " << tar_path.string();
      try {
        ProcessTar(tar_path, output_dir, expected_span);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to process tar file " << tar_path << ": "
                     << e.what();
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  return 0;
}
