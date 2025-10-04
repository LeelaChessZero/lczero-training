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

using ::lczero::V6TrainingData;

void PrintFloatArray(const float* data, size_t size, absl::string_view name,
                     int64_t per_line) {
  per_line = std::max<int64_t>(1, per_line);
  std::cout << "  " << name << ":\n";
  for (size_t i = 0; i < size; ++i) {
    if (i % per_line == 0) {
      std::cout << "    [" << absl::StrFormat("%4zu", i) << "]: ";
    }
    std::cout << absl::StrFormat("% .6g", data[i]);
    if ((i + 1) % per_line == 0 || i + 1 == size) {
      std::cout << "\n";
    } else {
      std::cout << ", ";
    }
  }
}

void PrintUint64Array(const uint64_t* data, size_t size, absl::string_view name,
                      int64_t per_line) {
  per_line = std::max<int64_t>(1, per_line);
  std::cout << "  " << name << ":\n";
  for (size_t i = 0; i < size; ++i) {
    if (i % per_line == 0) {
      std::cout << "    [" << absl::StrFormat("%3zu", i) << "]: ";
    }
    std::cout << absl::StrFormat("0x%016x", data[i]);
    if ((i + 1) % per_line == 0 || i + 1 == size) {
      std::cout << "\n";
    } else {
      std::cout << ", ";
    }
  }
}

std::string DecodeInvarianceInfo(uint8_t invariance_info) {
  return absl::StrFormat(
      "flip=%d, mirror=%d, transpose=%d, best_move_proven=%d, "
      "max_length=%d, adjudicated=%d, rescorer_deleted=%d, side_to_move=%d",
      invariance_info & 0x1, (invariance_info >> 1) & 0x1,
      (invariance_info >> 2) & 0x1, (invariance_info >> 3) & 0x1,
      (invariance_info >> 4) & 0x1, (invariance_info >> 5) & 0x1,
      (invariance_info >> 6) & 0x1, (invariance_info >> 7) & 0x1);
}

void PrintEntry(const V6TrainingData& entry, size_t index,
                int64_t float_per_line, int64_t plane_per_line) {
  std::cout << "Entry " << index << ":\n";
  std::cout << "  version: " << entry.version << "\n";
  std::cout << "  input_format: " << entry.input_format << "\n";
  std::cout << "  castling_us_ooo: " << static_cast<int>(entry.castling_us_ooo)
            << "\n";
  std::cout << "  castling_us_oo: " << static_cast<int>(entry.castling_us_oo)
            << "\n";
  std::cout << "  castling_them_ooo: "
            << static_cast<int>(entry.castling_them_ooo) << "\n";
  std::cout << "  castling_them_oo: "
            << static_cast<int>(entry.castling_them_oo) << "\n";
  std::cout << "  side_to_move_or_enpassant: "
            << static_cast<int>(entry.side_to_move_or_enpassant) << "\n";
  std::cout << "  rule50_count: " << static_cast<int>(entry.rule50_count)
            << "\n";
  std::cout << "  invariance_info: " << static_cast<int>(entry.invariance_info)
            << " (" << DecodeInvarianceInfo(entry.invariance_info) << ")\n";
  std::cout << "  dummy: " << static_cast<int>(entry.dummy) << "\n";
  std::cout << "  root_q: " << entry.root_q << "\n";
  std::cout << "  best_q: " << entry.best_q << "\n";
  std::cout << "  root_d: " << entry.root_d << "\n";
  std::cout << "  best_d: " << entry.best_d << "\n";
  std::cout << "  root_m: " << entry.root_m << "\n";
  std::cout << "  best_m: " << entry.best_m << "\n";
  std::cout << "  plies_left: " << entry.plies_left << "\n";
  std::cout << "  result_q: " << entry.result_q << "\n";
  std::cout << "  result_d: " << entry.result_d << "\n";
  std::cout << "  played_q: " << entry.played_q << "\n";
  std::cout << "  played_d: " << entry.played_d << "\n";
  std::cout << "  played_m: " << entry.played_m << "\n";
  std::cout << "  orig_q: " << entry.orig_q << "\n";
  std::cout << "  orig_d: " << entry.orig_d << "\n";
  std::cout << "  orig_m: " << entry.orig_m << "\n";
  std::cout << "  visits: " << entry.visits << "\n";
  std::cout << "  played_idx: " << entry.played_idx << "\n";
  std::cout << "  best_idx: " << entry.best_idx << "\n";
  std::cout << "  policy_kld: " << entry.policy_kld << "\n";
  std::cout << "  reserved: " << entry.reserved << "\n";
  PrintFloatArray(entry.probabilities, std::size(entry.probabilities),
                  "probabilities", float_per_line);
  PrintUint64Array(entry.planes, std::size(entry.planes), "planes",
                   plane_per_line);
  std::cout << std::flush;
}

void DumpChunk(const std::string& path, int64_t max_entries,
               int64_t float_per_line, int64_t plane_per_line) {
  gzFile file = gzopen(path.c_str(), "rb");
  if (file == nullptr) {
    LOG(FATAL) << "Failed to open chunk file: " << path;
  }

  size_t index = 0;
  while (true) {
    V6TrainingData entry;
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

    PrintEntry(entry, index, float_per_line, plane_per_line);
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
