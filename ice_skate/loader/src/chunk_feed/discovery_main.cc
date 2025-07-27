#include "discovery.h"

#include <absl/log/log.h>
#include <absl/log/initialize.h>

#include <iostream>
#include <string>

using namespace lczero::ice_skate;

int main(int argc, char* argv[]) {
  absl::InitializeLog();
  
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
    return 1;
  }
  
  std::string directory = argv[1];
  
  FileDiscovery discovery;
  
  auto token = discovery.RegisterObserver([&discovery](std::span<const FileDiscovery::File> files) {
    for (const auto& file : files) {
      const std::string& directory = discovery.GetDirectory(file.directory_idx);
      const char* type_str = (file.type == FileDiscovery::FileType::kInitial) ? "Initial" : "Discovered";
      std::cout << "File " << type_str << ": " << directory << "/" << file.filename << std::endl;
    }
  });
  
  std::cout << "Starting to monitor directory: " << directory << std::endl;
  std::cout << "Scanning for existing files..." << std::endl;
  
  size_t directory_idx = discovery.AddDirectory(directory);
  
  std::cout << "Scan completed. Directory index: " << directory_idx << std::endl;
  std::cout << "Initial files will be reported via observer callback above." << std::endl;
  
  std::cout << "Monitoring for new files... Press Enter to exit." << std::endl;
  
  std::cin.get();
  
  discovery.UnregisterObserver(token);
  
  return 0;
}