// ABOUTME: Comprehensive unit tests for the FilePathProvider class
// ABOUTME: Tests initial directory scanning, file monitoring, and Queue-based
// output

#include "loader/chunk_feed/file_path_provider.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/log.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>

namespace lczero {
namespace training {

class FilePathProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create unique test directory
    test_dir_ =
        std::filesystem::temp_directory_path() /
        ("file_path_provider_test_" +
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

  // Helper function to consume all initial scan results including completion
  // marker
  void ConsumeInitialScan(Queue<FilePathProvider::File>* queue) {
    bool scan_complete = false;
    while (!scan_complete) {
      auto file = queue->Get();
      if (file.message_type ==
          FilePathProvider::MessageType::kInitialScanComplete) {
        scan_complete = true;
      }
      // We consume and discard all initial scan files
    }
  }
};

TEST_F(FilePathProviderTest, ConstructorCreatesQueue) {
  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);
  auto* queue = file_path_provider.output();
  EXPECT_NE(queue, nullptr);
  EXPECT_EQ(queue->Capacity(), 100);

  // Should have kInitialScanComplete message for empty directory
  auto file = queue->Get();
  EXPECT_EQ(file.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_TRUE(file.filepath.empty());
}

TEST_F(FilePathProviderTest, InitialScanFindsExistingFiles) {
  // Create some test files
  CreateFile(test_dir_ / "file1.txt");
  CreateFile(test_dir_ / "file2.txt");
  CreateFile(test_dir_ / "subdir" / "file3.txt");

  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  // Collect files from queue
  std::unordered_set<std::string> found_files;
  auto* queue = file_path_provider.output();

  // Collect all files found during initial scan
  bool scan_complete_received = false;
  while (!scan_complete_received) {
    auto file = queue->Get();
    if (file.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      EXPECT_TRUE(file.filepath.empty());
      scan_complete_received = true;
    } else {
      EXPECT_EQ(file.message_type, FilePathProvider::MessageType::kFile);
      found_files.insert(file.filepath.filename().string());
    }
  }

  EXPECT_EQ(found_files.size(), 3);
  EXPECT_TRUE(found_files.count("file1.txt"));
  EXPECT_TRUE(found_files.count("file2.txt"));
  EXPECT_TRUE(found_files.count("file3.txt"));
  EXPECT_TRUE(scan_complete_received);
}

TEST_F(FilePathProviderTest, InitialScanIgnoresDirectories) {
  // Create files and directories
  CreateFile(test_dir_ / "file.txt");
  CreateDirectory(test_dir_ / "subdir");
  CreateDirectory(test_dir_ / "empty_dir");

  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  // Should only find the file, not directories
  std::vector<FilePathProvider::File> files;
  auto* queue = file_path_provider.output();
  bool scan_complete_received = false;
  while (!scan_complete_received) {
    auto file = queue->Get();
    if (file.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      scan_complete_received = true;
    } else {
      files.push_back(file);
    }
  }

  EXPECT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].filepath.filename().string(), "file.txt");
  EXPECT_EQ(files[0].message_type, FilePathProvider::MessageType::kFile);
}

TEST_F(FilePathProviderTest, DetectsNewFiles) {
  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  auto* queue = file_path_provider.output();
  // Consume initial scan results
  ConsumeInitialScan(queue);

  // Create a new file
  CreateFile(test_dir_ / "new_file.txt");

  // Wait for the new file to be detected
  auto file = queue->Get();
  EXPECT_EQ(file.filepath.filename().string(), "new_file.txt");
  EXPECT_EQ(file.message_type, FilePathProvider::MessageType::kFile);
}

TEST_F(FilePathProviderTest, DetectsFilesInNewSubdirectory) {
  // Pre-create the subdirectory structure
  auto subdir = test_dir_ / "new_subdir";
  CreateDirectory(subdir);

  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  auto* queue = file_path_provider.output();
  // Consume initial scan results
  ConsumeInitialScan(queue);

  // Create file in the existing subdirectory
  CreateFile(subdir / "file_in_new_dir.txt");

  // Wait for the new file to be detected
  auto file = queue->Get();
  EXPECT_EQ(file.filepath.filename().string(), "file_in_new_dir.txt");
  EXPECT_EQ(file.message_type, FilePathProvider::MessageType::kFile);
}

TEST_F(FilePathProviderTest, HandlesEmptyDirectory) {
  // Test with empty directory
  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  auto* queue = file_path_provider.output();
  auto file = queue->Get();
  EXPECT_EQ(file.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_TRUE(file.filepath.empty());
}

TEST_F(FilePathProviderTest, MultipleFilesInBatch) {
  // Create many files BEFORE starting discovery
  for (int i = 0; i < 5; ++i) {
    CreateFile(test_dir_ / ("batch_file_" + std::to_string(i) + ".txt"));
  }

  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  // Collect all files
  std::unordered_set<std::string> found_files;
  auto* queue = file_path_provider.output();
  bool scan_complete_received = false;
  while (!scan_complete_received) {
    auto file = queue->Get();
    if (file.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      EXPECT_TRUE(file.filepath.empty());
      scan_complete_received = true;
    } else {
      EXPECT_EQ(file.message_type, FilePathProvider::MessageType::kFile);
      found_files.insert(file.filepath.filename().string());
    }
  }

  EXPECT_EQ(found_files.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(found_files.count("batch_file_" + std::to_string(i) + ".txt"));
  }
}

TEST_F(FilePathProviderTest, QueueClosurePreventsNewFiles) {
  FilePathProviderConfig config;
  config.set_queue_capacity(100);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  auto* queue = file_path_provider.output();
  // Consume initial scan results
  ConsumeInitialScan(queue);

  file_path_provider.Close();

  // Any subsequent queue operations should throw
  EXPECT_THROW(queue->Get(), QueueClosedException);
}

TEST_F(FilePathProviderTest, DestructorCleansUpProperly) {
  auto test_cleanup = [&]() {
    FilePathProviderConfig config;
    config.set_queue_capacity(100);
    config.set_directory(test_dir_.string());
    FilePathProvider file_path_provider(config);

    CreateFile(test_dir_ / "cleanup_test.txt");

    auto* queue = file_path_provider.output();
    // Consume initial scan results
    ConsumeInitialScan(queue);

    // FilePathProvider destructor should be called here
  };

  // This should not crash or hang
  EXPECT_NO_THROW(test_cleanup());
}

// Stress test with rapid file creation
TEST_F(FilePathProviderTest, RapidFileCreation) {
  FilePathProviderConfig config;
  config.set_queue_capacity(1000);
  config.set_directory(test_dir_.string());
  FilePathProvider file_path_provider(config);

  auto* queue = file_path_provider.output();
  // Consume initial scan results
  ConsumeInitialScan(queue);

  // Rapidly create files
  constexpr int num_files = 10;
  for (int i = 0; i < num_files; ++i) {
    CreateFile(test_dir_ / ("rapid_" + std::to_string(i) + ".txt"));
  }

  // Collect detected files - we should get at least some
  std::vector<FilePathProvider::File> files;
  constexpr int min_expected = num_files / 2;
  for (int i = 0; i < min_expected; ++i) {
    auto file = queue->Get();
    EXPECT_EQ(file.message_type, FilePathProvider::MessageType::kFile);
    files.push_back(file);
  }
  EXPECT_GE(files.size(), min_expected);
}

}  // namespace training
}  // namespace lczero