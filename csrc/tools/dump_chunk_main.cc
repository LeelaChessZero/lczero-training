#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/string_view.h>
#include <zlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>

#include "trainingdata/trainingdata_v6.h"
#include "utils/training_data_printer.h"

ABSL_FLAG(std::string, chunk_path, "", "Path to the chunk file (.gz) to dump.");
ABSL_FLAG(int64_t, max_entries, -1,
          "Maximum number of entries to print. -1 prints all entries.");
ABSL_FLAG(int64_t, float_values_per_line, 8,
          "Number of floating point values per output line.");
ABSL_FLAG(int64_t, plane_values_per_line, 4,
          "Number of plane values per output line.");

namespace lczero {
namespace training {

namespace {

using ::lczero::training::FrameType;
using ::lczero::training::PrintTrainingDataEntry;

void DumpChunk(const std::string& path, int64_t max_entries,
               int64_t float_per_line, int64_t plane_per_line) {
  gzFile file = gzopen(path.c_str(), "rb");
  if (file == nullptr) {
    LOG(FATAL) << "Failed to open chunk file: " << path;
  }

  size_t index = 0;
  while (true) {
    FrameType entry;
    const int bytes_read = gzread(file, &entry, sizeof(entry));
    if (bytes_read == 0) {
      break;
    }
    if (bytes_read < 0) {
      int errnum = 0;
      const char* error_message = gzerror(file, &errnum);
      gzclose(file);
      LOG(FATAL) << "Error while reading chunk: " << error_message;
    }
    if (bytes_read != sizeof(entry)) {
      gzclose(file);
      LOG(FATAL) << "Unexpected chunk size. Expected " << sizeof(entry)
                 << " bytes, got " << bytes_read << ".";
    }

    const std::string header = absl::StrFormat("Entry %zu:", index);
    PrintTrainingDataEntry(entry, header, float_per_line, plane_per_line);
    ++index;

    if (max_entries >= 0 && static_cast<int64_t>(index) >= max_entries) {
      break;
    }
  }

  gzclose(file);
  LOG(INFO) << "Printed " << index << " entries.";
}

}  // namespace

}  // namespace training
}  // namespace lczero

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  if (chunk_path.empty()) {
    LOG(FATAL) << "--chunk_path flag is required.";
  }

  const int64_t max_entries = absl::GetFlag(FLAGS_max_entries);
  const int64_t float_per_line = absl::GetFlag(FLAGS_float_values_per_line);
  const int64_t plane_per_line = absl::GetFlag(FLAGS_plane_values_per_line);

  lczero::training::DumpChunk(chunk_path, max_entries, float_per_line,
                              plane_per_line);
  return 0;
}
