# FileDiscovery Implementation Plan

## Overview
The `FileDiscovery` class monitors directories recursively for new files and notifies observers when files are closed after writing or renamed into the directory. It uses a background thread and Linux inotify for efficient file system monitoring.

## Architecture

### Core Components
1. **Observer Management**: Token-based registration system using `absl::flat_hash_map`
2. **Directory Monitoring**: inotify-based recursive directory watching
3. **Background Thread**: Event processing loop with proper shutdown
4. **File Discovery Logic**: Handle `IN_CLOSE_WRITE` and `IN_MOVED_TO` events
5. **Thread Safety**: Abseil synchronization primitives

### Dependencies
- `absl::Mutex` and `absl::CondVar` for thread synchronization
- `absl::flat_hash_map` for efficient observer and watch descriptor management
- `std::filesystem` for directory traversal and path operations
- `std::thread` for background monitoring
- Linux `inotify` API for file system events

## Implementation TODO

### Phase 1: Basic Structure
- [ ] Add required includes (`sys/inotify.h`, `filesystem`, `thread`, `absl/synchronization/mutex.h`, etc.)
- [ ] Define private member variables:
  - [ ] `absl::Mutex mutex_` for thread safety
  - [ ] `absl::CondVar stop_condition_` for thread coordination
  - [ ] `absl::flat_hash_map<Token, Observer> observers_` for observer management
  - [ ] `absl::flat_hash_map<int, size_t> watch_descriptors_` to map inotify wd to directory index
  - [ ] `absl::flat_hash_map<std::string, int> directory_watches_` to map directory paths to watch descriptors
  - [ ] `std::vector<std::string> directories_` to map directory index to directory path
  - [ ] `std::thread monitor_thread_` for background monitoring
  - [ ] `int inotify_fd_` for inotify file descriptor
  - [ ] `bool should_stop_` for shutdown signaling
  - [ ] `Token next_token_` for unique token generation

### Phase 2: Constructor/Destructor
- [ ] Constructor:
  - [ ] Initialize inotify with `inotify_init1(IN_CLOEXEC | IN_NONBLOCK)`
  - [ ] Start background monitoring thread
  - [ ] Initialize member variables
- [ ] Destructor:
  - [ ] Signal shutdown (`should_stop_ = true`)
  - [ ] Notify condition variable
  - [ ] Join monitor thread
  - [ ] Close inotify file descriptor
  - [ ] Clean up watch descriptors

### Phase 3: Observer Management
- [ ] `RegisterObserver(Observer observer)`:
  - [ ] Lock mutex
  - [ ] Generate unique token
  - [ ] Store observer in map
  - [ ] Return token
- [ ] `UnregisterObserver(Token token)`:
  - [ ] Lock mutex
  - [ ] Remove observer from map

### Phase 4: Directory Monitoring
- [ ] `AddDirectory(const std::string& directory)`:
  - [ ] Lock mutex to add directory to index and set up watches
  - [ ] Scan existing files using `std::filesystem::recursive_directory_iterator`
  - [ ] Add inotify watch with `inotify_add_watch(fd, path, IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_MOVE)`
  - [ ] Store watch descriptor mapping to directory index
  - [ ] Call observers with existing files in batches (10000 files) without holding mutex
  - [ ] Return directory index
- [ ] `GetDirectory(size_t idx)`: Return directory path for given index
- [ ] Helper method `AddWatchRecursive(const std::string& path, size_t directory_idx)`:
  - [ ] Add watch for current directory
  - [ ] Recursively add watches for subdirectories

### Phase 5: Background Thread Implementation
- [ ] `MonitorThread()` method:
  - [ ] Event loop with `epoll` or `select` on inotify fd
  - [ ] Handle `IN_CLOSE_WRITE` events (file finished writing)
  - [ ] Handle `IN_MOVED_TO` events (file moved into directory)
  - [ ] Handle `IN_CREATE` + `IN_ISDIR` events (new subdirectory created)
  - [ ] Batch events and notify observers
  - [ ] Handle shutdown signal
- [ ] `ProcessInotifyEvents()` helper:
  - [ ] Read events from inotify fd
  - [ ] Parse event structure
  - [ ] Convert events to `File` structures
  - [ ] Filter for relevant events

### Phase 6: Event Processing
- [ ] `NotifyObservers(std::span<const File> files)`:
  - [ ] Lock mutex to get observer list snapshot
  - [ ] Call each observer with file list
  - [ ] Handle observer exceptions gracefully
- [ ] File path resolution:
  - [ ] Map watch descriptor to directory index
  - [ ] Get directory path from index using directories_ vector
  - [ ] Create `File` structure with directory_idx, relative filename, and FileType::kDiscovered

### Phase 7: Error Handling & Edge Cases
- [ ] Handle directory deletion (remove watches)
- [ ] Proper cleanup on destruction

### Phase 8: Thread Safety & Performance
- [ ] Minimize mutex lock duration (observers called without holding mutex)
- [ ] Use conditions for efficient waiting
- [ ] Batch file notifications for performance (10000 files per batch)
- [ ] Handle high-frequency file events efficiently
- [ ] Directory index system for efficient path lookups

## Key Implementation Details

### Inotify Events to Monitor
- `IN_CLOSE_WRITE`: File closed after being opened for writing
- `IN_MOVED_TO`: File moved into watched directory
- `IN_CREATE | IN_ISDIR`: New subdirectory created (need to add watch)

### Thread Communication
- Main thread adds directories and manages observers
- Background thread processes inotify events
- Use `absl::CondVar` for efficient shutdown signaling
- Mutex protects shared state (observers, watch descriptors)

### File Structure
```cpp
enum class FileType {
    kInitial,     // File existed when AddDirectory was called
    kDiscovered   // File was discovered via inotify events
};

struct File {
    size_t directory_idx;   // Index to the directory (use GetDirectory(idx) to get path)
    std::string filename;   // Relative path from directory
    FileType type;          // Whether this was an initial file or discovered later
};
```

### Error Handling Strategy
- Log errors using `absl::log`
- Continue operation when possible (don't crash on individual file errors)
- Gracefully handle system limits (max inotify watches)
- Clean up resources properly on errors