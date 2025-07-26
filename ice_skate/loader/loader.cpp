#include <iostream>

#include "tar.h"

namespace lczero {
namespace ice_skate {

void Run() {
  TarFile tar(
      "/home/crem/tmp/2025-07/lczero-training/data/"
      "training-run1-test80-20250722-0617.tar");
}

}  // namespace ice_skate
}  // namespace lczero

int main() {
  lczero::ice_skate::Run();
  return 0;
}
