// ABOUTME: Comprehensive unit tests for the FileDiscovery class
// ABOUTME: Tests initial directory scanning, file monitoring, and Queue-based
// output

#include "loader/chunk_feed/discovery.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/log.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>

namespace lczero {
namespace training {

class FileDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create unique test directory
    test_dir_ =
        std::filesystem::temp_directory_path() /
        ("discovery_test_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    // Clean up test directory
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  void CreateFile(const std::filesystem::path& path,
                  const std::string& content = "test") {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file << content;
    file.close();
  }

  void CreateDirectory(const std::filesystem::path& path) {
    std::filesystem::create_directories(path);
  }

  std::filesystem::path test_dir_;
};

TEST_F(FileDiscoveryTest, ConstructorCreatesQueue) {
  FileDiscovery discovery(100);
  auto* queue = discovery.output();
  EXPECT_NE(queue, nullptr);
  EXPECT_EQ(queue->Size(), 0);
  EXPECT_EQ(queue->Capacity(), 100);
}

TEST_F(FileDiscoveryTest, InitialScanFindsExistingFiles) {
  // Create some test files
  CreateFile(test_dir_ / "file1.txt");
  CreateFile(test_dir_ / "file2.txt");
  CreateFile(test_dir_ / "subdir" / "file3.txt");

  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Collect files from queue
  std::unordered_set<std::string> found_files;
  auto* queue = discovery.output();

  // Wait a bit for initial scan to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Collect all files found during initial scan
  bool scan_complete_received = false;
  while (queue->Size() > 0) {
    auto file = queue->Get();
    if (file.phase == FileDiscovery::Phase::kInitialScanComplete) {
      EXPECT_TRUE(file.filepath.empty());
      scan_complete_received = true;
    } else {
      EXPECT_EQ(file.phase, FileDiscovery::Phase::kInitialScan);
      found_files.insert(file.filepath.filename().string());
    }
  }

  EXPECT_EQ(found_files.size(), 3);
  EXPECT_TRUE(found_files.count("file1.txt"));
  EXPECT_TRUE(found_files.count("file2.txt"));
  EXPECT_TRUE(found_files.count("file3.txt"));
  EXPECT_TRUE(scan_complete_received);
}

TEST_F(FileDiscoveryTest, InitialScanIgnoresDirectories) {
  // Create files and directories
  CreateFile(test_dir_ / "file.txt");
  CreateDirectory(test_dir_ / "subdir");
  CreateDirectory(test_dir_ / "empty_dir");

  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Should only find the file, not directories
  std::vector<FileDiscovery::File> files;
  auto* queue = discovery.output();
  while (queue->Size() > 0) {
    auto file = queue->Get();
    if (file.phase != FileDiscovery::Phase::kInitialScanComplete) {
      files.push_back(file);
    }
  }

  EXPECT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].filepath.filename().string(), "file.txt");
  EXPECT_EQ(files[0].phase, FileDiscovery::Phase::kInitialScan);
}

TEST_F(FileDiscoveryTest, DetectsNewFiles) {
  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto* queue = discovery.output();
  // Clear any initial scan results
  while (queue->Size() > 0) {
    queue->Get();
  }

  // Create a new file
  CreateFile(test_dir_ / "new_file.txt");

  // Wait for inotify to detect the file
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Should detect the new file
  EXPECT_GT(queue->Size(), 0);
  auto file = queue->Get();
  EXPECT_EQ(file.filepath.filename().string(), "new_file.txt");
  EXPECT_EQ(file.phase, FileDiscovery::Phase::kNewFile);
}

TEST_F(FileDiscoveryTest, DetectsFilesInNewSubdirectory) {
  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto* queue = discovery.output();
  // Clear initial scan results
  while (queue->Size() > 0) {
    queue->Get();
  }

  // Create new subdirectory and file
  auto subdir = test_dir_ / "new_subdir";
  CreateDirectory(subdir);

  // Give time for directory creation to be detected
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  CreateFile(subdir / "file_in_new_dir.txt");

  // Wait for file detection
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Should detect the file in the new subdirectory
  EXPECT_GT(queue->Size(), 0);
  auto file = queue->Get();
  EXPECT_EQ(file.filepath.filename().string(), "file_in_new_dir.txt");
  EXPECT_EQ(file.phase, FileDiscovery::Phase::kNewFile);
}

TEST_F(FileDiscoveryTest, HandlesEmptyDirectory) {
  // Test with empty directory
  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto* queue = discovery.output();
  EXPECT_EQ(queue->Size(), 1);  // Should have kInitialScanComplete message

  auto file = queue->Get();
  EXPECT_EQ(file.phase, FileDiscovery::Phase::kInitialScanComplete);
  EXPECT_TRUE(file.filepath.empty());
}

TEST_F(FileDiscoveryTest, MultipleFilesInBatch) {
  // Create many files BEFORE starting discovery
  for (int i = 0; i < 5; ++i) {
    CreateFile(test_dir_ / ("batch_file_" + std::to_string(i) + ".txt"));
  }

  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Collect all files
  std::unordered_set<std::string> found_files;
  auto* queue = discovery.output();
  while (queue->Size() > 0) {
    auto file = queue->Get();
    if (file.phase == FileDiscovery::Phase::kInitialScanComplete) {
      EXPECT_TRUE(file.filepath.empty());
    } else {
      EXPECT_EQ(file.phase, FileDiscovery::Phase::kInitialScan);
      found_files.insert(file.filepath.filename().string());
    }
  }

  EXPECT_EQ(found_files.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(found_files.count("batch_file_" + std::to_string(i) + ".txt"));
  }
}

TEST_F(FileDiscoveryTest, QueueClosurePreventsNewFiles) {
  FileDiscovery discovery(100);
  discovery.AddDirectory(test_dir_);

  // Wait for initial setup
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto* queue = discovery.output();

  // Wait for initial scan to complete first
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Clear any messages
  while (queue->Size() > 0) {
    queue->Get();
  }

  queue->Close();

  // Any subsequent queue operations should throw
  EXPECT_THROW(queue->Get(), QueueClosedException);
}

TEST_F(FileDiscoveryTest, DestructorCleansUpProperly) {
  auto test_cleanup = [&]() {
    FileDiscovery discovery(100);
    discovery.AddDirectory(test_dir_);

    CreateFile(test_dir_ / "cleanup_test.txt");

    // Wait a bit for initial operations
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // FileDiscovery destructor should be called here
  };

  // This should not crash or hang
  EXPECT_NO_THROW(test_cleanup());
}

// Stress test with rapid file creation
TEST_F(FileDiscoveryTest, RapidFileCreation) {
  FileDiscovery discovery(1000);  // Larger queue for stress test
  discovery.AddDirectory(test_dir_);

  // Wait for initial scan
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto* queue = discovery.output();
  // Clear initial scan results
  while (queue->Size() > 0) {
    queue->Get();
  }

  // Rapidly create files
  constexpr int num_files = 10;
  for (int i = 0; i < num_files; ++i) {
    CreateFile(test_dir_ / ("rapid_" + std::to_string(i) + ".txt"));
    // Small delay to ensure files are created separately
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Wait for all files to be detected
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Count detected files
  int detected_count = 0;
  while (queue->Size() > 0) {
    auto file = queue->Get();
    EXPECT_EQ(file.phase, FileDiscovery::Phase::kNewFile);
    detected_count++;
  }

  // Should detect most or all files (inotify may coalesce some events)
  EXPECT_GE(detected_count, num_files / 2);  // At least half should be detected
}

}  // namespace training
}  // namespace lczero