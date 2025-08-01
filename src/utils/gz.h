#pragma once

#include <span>
#include <string>

namespace lczero {
namespace training {

std::string GunzipBuffer(std::span<const char> buffer);

}  // namespace training
}  // namespace lczero