#include "utils/training_data_printer.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <iostream>

#include "chess/board.h"
#include "neural/decoder.h"
#include "trainingdata/reader.h"

namespace lczero {
namespace training {

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

std::string TrainingDataToFen(const FrameType& entry) {
  InputPlanes planes = PlanesFromTrainingData(entry);
  ChessBoard board;
  int rule50 = 0;
  int gameply = 0;
  PopulateBoard(
      static_cast<pblczero::NetworkFormat::InputFormat>(entry.input_format),
      planes, &board, &rule50, &gameply);
  std::string fen = BoardToFen(board);
  fen += " " + std::to_string(rule50);
  fen += " " + std::to_string((gameply / 2) + 1);
  return fen;
}

void PrintTrainingDataEntry(const FrameType& entry,
                            absl::string_view header_text,
                            int64_t float_per_line, int64_t plane_per_line) {
  std::cout << header_text << "\n";
  std::cout << "  FEN: " << TrainingDataToFen(entry) << "\n";
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
  PrintFloatArray(entry.probabilities, std::size(entry.probabilities),
                  "probabilities", float_per_line);
  PrintUint64Array(entry.planes, std::size(entry.planes), "planes",
                   plane_per_line);
  std::cout << std::flush;
}

}  // namespace training
}  // namespace lczero
