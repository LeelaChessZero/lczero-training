#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <iostream>
#include <string>

#include "discovery.h"

ABSL_FLAG(std::string, directory, "", "Directory to monitor for files");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::string directory = absl::GetFlag(FLAGS_directory);
  if (directory.empty()) {
    std::cerr << "Usage: " << argv[0] << " --directory=<directory>"
              << std::endl;
    return 1;
  }

  lczero::ice_skate::FileDiscovery discovery;

  auto token = discovery.RegisterObserver(
      [](std::span<const lczero::ice_skate::FileDiscovery::File> files) {
        for (const auto& file : files) {
          LOG(INFO) << "File Discovered: " << file.filepath;
        }
      });

  LOG(INFO) << "Starting to monitor directory: " << directory;
  LOG(INFO) << "Scanning for existing files...";

  discovery.AddDirectory(
      directory,
      [](std::span<const lczero::ice_skate::FileDiscovery::File> files) {
        for (const auto& file : files) {
          LOG(INFO) << "File Initial: " << file.filepath;
        }
      });

  LOG(INFO) << "Scan completed.";
  LOG(INFO) << "Initial files will be reported via observer callback above.";

  LOG(INFO) << "Monitoring for new files... Press Enter to exit.";

  std::cin.get();

  discovery.UnregisterObserver(token);

  return 0;
}