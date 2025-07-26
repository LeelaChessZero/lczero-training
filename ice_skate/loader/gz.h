#pragma once

#include <span>
#include <string>

namespace lczero {
namespace ice_skate {

std::string GunzipBuffer(std::span<char> buffer);

}  // namespace ice_skate
}  // namespace lczero