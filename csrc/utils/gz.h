#pragma once

#include <span>
#include <stdexcept>
#include <string>

namespace lczero {
namespace training {

class GunzipError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string GunzipBuffer(std::string_view buffer);

}  // namespace training
}  // namespace lczero