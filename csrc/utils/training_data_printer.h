#ifndef LCZERO_TRAINING_UTILS_TRAINING_DATA_PRINTER_H_
#define LCZERO_TRAINING_UTILS_TRAINING_DATA_PRINTER_H_

#include <absl/strings/string_view.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "loader/frame_type.h"
#include "trainingdata/trainingdata_v6.h"

namespace lczero {
namespace training {

// Prints a float array with configurable number of values per line.
void PrintFloatArray(const float* data, size_t size, absl::string_view name,
                     int64_t per_line);

// Prints a uint64 array with configurable number of values per line.
void PrintUint64Array(const uint64_t* data, size_t size, absl::string_view name,
                      int64_t per_line);

// Decodes the invariance_info byte into a human-readable string.
std::string DecodeInvarianceInfo(uint8_t invariance_info);

// Converts a V6TrainingData entry to FEN (Forsyth-Edwards Notation).
std::string TrainingDataToFen(const V6TrainingData& entry);

// Prints a V6TrainingData entry with a custom header and formatting options.
void PrintTrainingDataEntry(const FrameType& entry,
                            absl::string_view header_text,
                            int64_t float_per_line, int64_t plane_per_line);

}  // namespace training
}  // namespace lczero

#endif  // LCZERO_TRAINING_UTILS_TRAINING_DATA_PRINTER_H_
