#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <iostream>
#include <string>

#include "discovery.h"

using namespace lczero::ice_skate;

int main(int argc, char* argv[]) {
  absl::InitializeLog();

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
    return 1;
  }

  std::string directory = argv[1];

  FileDiscovery discovery;

  auto token = discovery.RegisterObserver(
      [](std::span<const FileDiscovery::File> files) {
        for (const auto& file : files) {
          std::cout << "File Discovered: " << file.filepath << std::endl;
        }
      });

  std::cout << "Starting to monitor directory: " << directory << std::endl;
  std::cout << "Scanning for existing files..." << std::endl;

  discovery.AddDirectory(
      directory, [](std::span<const FileDiscovery::File> files) {
        for (const auto& file : files) {
          std::cout << "File Initial: " << file.filepath << std::endl;
        }
      });

  std::cout << "Scan completed." << std::endl;
  std::cout << "Initial files will be reported via observer callback above."
            << std::endl;

  std::cout << "Monitoring for new files... Press Enter to exit." << std::endl;

  std::cin.get();

  discovery.UnregisterObserver(token);

  return 0;
}