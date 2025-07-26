#include "gz.h"

#include <absl/log/log.h>
#include <zlib.h>

#include <array>
#include <stdexcept>

namespace lczero {
namespace ice_skate {

std::string GunzipBuffer(std::span<char> buffer) {
  z_stream strm = {};
  int ret = inflateInit2(&strm, 16 + MAX_WBITS);
  if (ret != Z_OK) {
    throw std::runtime_error("Failed to initialize zlib inflate");
  }

  strm.avail_in = buffer.size();
  strm.next_in = reinterpret_cast<Bytef*>(buffer.data());

  constexpr size_t kChunkSize = 16384;
  std::string output;
  std::array<char, kChunkSize> temp_buffer;

  do {
    strm.avail_out = kChunkSize;
    strm.next_out = reinterpret_cast<Bytef*>(temp_buffer.data());

    ret = inflate(&strm, Z_NO_FLUSH);
    if (ret == Z_STREAM_ERROR || ret == Z_NEED_DICT || ret == Z_DATA_ERROR ||
        ret == Z_MEM_ERROR) {
      inflateEnd(&strm);
      throw std::runtime_error("zlib inflate error");
    }

    size_t bytes_written = kChunkSize - strm.avail_out;
    output.append(temp_buffer.begin(), temp_buffer.begin() + bytes_written);
  } while (strm.avail_out == 0);

  inflateEnd(&strm);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error("Incomplete gzip decompression");
  }

  return output;
}

}  // namespace ice_skate
}  // namespace lczero