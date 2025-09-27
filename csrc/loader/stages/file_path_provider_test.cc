#include "loader/stages/file_path_provider.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace lczero {
namespace training {

namespace {

FilePathProviderConfig MakeConfig(const std::filesystem::path& directory) {
  FilePathProviderConfig config;
  config.set_queue_capacity(128);
  config.set_directory(directory.string());
  return config;
}

std::string RelativeTo(const std::filesystem::path& base,
                       const std::filesystem::path& target) {
  return target.lexically_relative(base).generic_string();
}

}  // namespace

class FilePathProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ =
        std::filesystem::temp_directory_path() /
        ("file_path_provider_test_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  void CreateFile(const std::filesystem::path& path,
                  const std::string& content = "payload") {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file << content;
  }

  void CreateDirectory(const std::filesystem::path& path) {
    std::filesystem::create_directories(path);
  }

  std::vector<std::filesystem::path> DrainInitialScan(
      Queue<FilePathProvider::File>* queue) {
    std::vector<std::filesystem::path> files;
    while (true) {
      auto message = queue->Get();
      if (message.message_type ==
          FilePathProvider::MessageType::kInitialScanComplete) {
        EXPECT_TRUE(message.filepath.empty());
        break;
      }
      if (message.message_type != FilePathProvider::MessageType::kFile) {
        ADD_FAILURE() << "Unexpected message type in initial scan.";
        continue;
      }
      files.push_back(message.filepath);
    }
    return files;
  }

  FilePathProvider::File AwaitNextFile(Queue<FilePathProvider::File>* queue) {
    while (true) {
      auto message = queue->Get();
      if (message.message_type == FilePathProvider::MessageType::kFile) {
        return message;
      }
      if (message.message_type !=
          FilePathProvider::MessageType::kInitialScanComplete) {
        ADD_FAILURE()
            << "Unexpected message type while waiting for file notification.";
      }
    }
  }

  FilePathProviderConfig Config() const { return MakeConfig(test_dir_); }

  std::filesystem::path test_dir_;
};

TEST_F(FilePathProviderTest, ConstructorCreatesQueue) {
  FilePathProvider provider(Config());
  provider.Start();

  auto* queue = provider.output();
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->Capacity(), 128);

  auto message = queue->Get();
  EXPECT_EQ(message.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_TRUE(message.filepath.empty());

  provider.Stop();
}

TEST_F(FilePathProviderTest, InitialScanFindsVisibleFiles) {
  CreateFile(test_dir_ / "file1.txt");
  CreateFile(test_dir_ / "file2.txt");
  CreateFile(test_dir_ / "sub" / "nested.txt");

  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();

  auto discovered = DrainInitialScan(queue);
  std::unordered_set<std::string> relative_paths;
  for (const auto& path : discovered) {
    relative_paths.insert(RelativeTo(test_dir_, path));
  }

  EXPECT_EQ(relative_paths.size(), 3u);
  EXPECT_TRUE(relative_paths.count("file1.txt"));
  EXPECT_TRUE(relative_paths.count("file2.txt"));
  EXPECT_TRUE(relative_paths.count("sub/nested.txt"));

  provider.Stop();
}

TEST_F(FilePathProviderTest, InitialScanSkipsHiddenEntries) {
  CreateFile(test_dir_ / "visible.txt");
  CreateFile(test_dir_ / ".hidden_file");
  CreateFile(test_dir_ / ".hidden_dir" / "nested.txt");
  CreateFile(test_dir_ / "visible_dir" / "child.txt");

  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();

  auto discovered = DrainInitialScan(queue);
  std::unordered_set<std::string> relative_paths;
  for (const auto& path : discovered) {
    relative_paths.insert(RelativeTo(test_dir_, path));
  }

  EXPECT_TRUE(relative_paths.count("visible.txt"));
  EXPECT_TRUE(relative_paths.count("visible_dir/child.txt"));
  EXPECT_FALSE(relative_paths.count(".hidden_file"));
  EXPECT_FALSE(relative_paths.count(".hidden_dir/nested.txt"));

  provider.Stop();
}

TEST_F(FilePathProviderTest, DetectsNewVisibleFile) {
  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();
  DrainInitialScan(queue);

  CreateFile(test_dir_ / "new_file.txt");

  auto message = AwaitNextFile(queue);
  EXPECT_EQ(RelativeTo(test_dir_, message.filepath), "new_file.txt");

  provider.Stop();
}

TEST_F(FilePathProviderTest, DetectsFilesInPreExistingSubdirectory) {
  auto subdir = test_dir_ / "subdir";
  CreateDirectory(subdir);

  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();
  DrainInitialScan(queue);

  CreateFile(subdir / "from_subdir.txt");

  auto message = AwaitNextFile(queue);
  EXPECT_EQ(RelativeTo(test_dir_, message.filepath), "subdir/from_subdir.txt");

  provider.Stop();
}

TEST_F(FilePathProviderTest, IgnoresHiddenFileEvents) {
  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();
  DrainInitialScan(queue);

  CreateFile(test_dir_ / ".hidden_event.txt");
  CreateFile(test_dir_ / "visible_after_hidden.txt");

  auto message = AwaitNextFile(queue);
  EXPECT_EQ(RelativeTo(test_dir_, message.filepath),
            "visible_after_hidden.txt");

  provider.Stop();
}

TEST_F(FilePathProviderTest, SkipsHiddenDirectoryRecursion) {
  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();
  DrainInitialScan(queue);

  CreateDirectory(test_dir_ / ".hidden_dir");
  CreateFile(test_dir_ / ".hidden_dir" / "inner.txt");
  CreateFile(test_dir_ / "outer.txt");

  auto message = AwaitNextFile(queue);
  EXPECT_EQ(RelativeTo(test_dir_, message.filepath), "outer.txt");

  provider.Stop();
}

TEST_F(FilePathProviderTest, HandlesEmptyDirectory) {
  FilePathProvider provider(Config());
  provider.Start();
  auto* queue = provider.output();

  auto discovered = DrainInitialScan(queue);
  EXPECT_TRUE(discovered.empty());

  provider.Stop();
}

}  // namespace training
}  // namespace lczero
