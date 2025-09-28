// Minimal stub of Gaviota tablebase probing API to allow builds without the
// optional dependency. The real implementation is provided by the gaviotatb
// project and offers full functionality. These definitions satisfy the
// interface expected by rescorer.cc but do not perform any probing work.

#pragma once

#include <cstddef>
#include <cstdint>

using TB_squares = unsigned int;

constexpr unsigned int tb_WHITE_TO_MOVE = 0;
constexpr unsigned int tb_BLACK_TO_MOVE = 1;
constexpr unsigned int tb_NOSQUARE = 0;
constexpr unsigned int tb_NOCASTLE = 0;
constexpr unsigned int tb_CP4 = 0;
constexpr unsigned int tb_NOPIECE = 0;
constexpr unsigned int tb_KING = 1;
constexpr unsigned int tb_QUEEN = 2;
constexpr unsigned int tb_ROOK = 3;
constexpr unsigned int tb_BISHOP = 4;
constexpr unsigned int tb_KNIGHT = 5;
constexpr unsigned int tb_PAWN = 6;
constexpr unsigned int tb_WMATE = 1;
constexpr unsigned int tb_BMATE = 2;
constexpr char SEP_CHAR = ';';

inline void* tbpaths_init() { return nullptr; }

inline void* tbpaths_add(void* paths, const char*) { return paths; }

inline void tb_init(int, unsigned int, void*) {}

inline void tbcache_init(std::size_t, int) {}

inline unsigned int tb_availability() { return 0; }

inline void tb_probe_hard(unsigned int, unsigned int, unsigned int,
                          unsigned int*, unsigned int*, unsigned char*,
                          unsigned char*, unsigned int* info,
                          unsigned int* dtm) {
  if (info != nullptr) {
    *info = tb_NOPIECE;
  }
  if (dtm != nullptr) {
    *dtm = 0;
  }
}
