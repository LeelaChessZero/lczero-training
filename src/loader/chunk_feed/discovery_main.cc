#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <iostream>
#include <string>
#include <thread>

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

  LOG(INFO) << "Starting to monitor directory: " << directory;
  lczero::training::FileDiscovery discovery(
      lczero::training::FileDiscoveryOptions{.queue_capacity = 16,
                                             .directory = directory});

  // Consumer thread to read from the queue
  std::thread consumer_thread([&discovery]() {
    auto* queue = discovery.output();
    try {
      while (true) {
        auto file = queue->Get();
        const char* type_str =
            (file.message_type ==
             lczero::training::FileDiscovery::MessageType::kFile)
                ? "File"
                : "Initial scan complete";
        LOG(INFO) << "File " << type_str << ": " << file.filepath;
      }
    } catch (const lczero::QueueClosedException&) {
      LOG(INFO) << "Queue closed, consumer thread exiting";
    }
  });

  LOG(INFO) << "Monitoring for files... Press Enter to exit.";
  std::cin.get();

  // Close the queue and wait for consumer to finish
  discovery.Close();
  consumer_thread.join();

  return 0;
}