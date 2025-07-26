#pragma once

#include <span>
#include <vector>

namespace lczero {
namespace ice_skate {

std::vector<char> GunzipBuffer(std::span<char> buffer);

}  // namespace ice_skate
}  // namespace lczero