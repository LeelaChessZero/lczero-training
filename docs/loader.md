# Data Loader

The Data Loader is a C++ module (exposed to Python via pybind11) that handles
loading, preprocessing, shuffling, and feeding training data for the Leela Chess
Zero training process.

## High-Level Overview

The Data Loader consists of the following stages connected through a
[Queue](../src/utils/queue.h):

* [Discovery](../src/loader/chunk_feed/discovery.h) — Training data discovery
  worker (watches a directory and provides feed of filenames)
* [ChunkSource Feed](../src/loader/chunk_feed/chunk_source_feed.h) — Reads
  chunks from files, providing a stream of chunks.
* [Chunk Set](../src/loader/chunk_feed/chunk_set.h) — Keeps a set of chunks,
  managing the last `num_chunks` available and removing old ones, and outputting
  them in shuffled order.
* [Chunk Filter](../src/loader/chunk_feed/chunk_filter.h) — Filters the chunk
  stream, filtering out invalid chunks.
* [Chunk Rescorer](../src/loader/chunk_feed/chunk_rescorer.h) — Rescores chunks
  based on tablebase or intentional blunders.
* [Frame Shuffler](../src/loader/frame_shuffler.h) — Takes a stream of chunks
  and provides shuffled batches of frames for training, using reservoir
  sampling.
* [Tensor Generator](../src/loader/tensor_generator.h) — Takes frames and
  provides tensor buffers for the training process.

All stages are quite sloppy in closing the output queues, the idea that we don't
need clean shutdown. (TODO fix this.)

## Stage interface

All stages implement the similar API and structure, although not sharing any
base class.

All stages/workers (except pure producers) wait for the input queue to Close(),
then Close() output Queue.

```cpp
class Stage {
 public:
    using InputType = ...;  // Type of input data for this stage
    using OutputType = ...; // Type of output data from this stage
    // input_queue is omitted in the producer stages like Discovery.
    Stage(Queue<InputType>* input_queue, /* other params */);
    Queue<OutputType>* output();

private:
    ThreadPool thread_pool_;
    Queue<InputType>* input_queue_;
    Queue<OutputType> output_queue_;
};
```

## Chunk Set

The Chunk Set takes the feed of chunk sources, indexes them, and assigns the
chunk range (base; base+num_chunks) to each chunk source. It aims to keep the
newest chunk sources that cover the last `chunks_window_` chunks, and removes
old chunk sources when new ones are added.

On the output side, it returns a stream of chunks within
(`last - chunk_window_`, `last`) range, without repetitions. To do that, it
utilizes a [StreamShuffler](../src/loader/stream_shuffler.h) to which provides
the shuffled stream of numbers within the (dynamic) range.

The Chunk Set gets the stream of chunks (initial chunks are read in the
constructor), and then starts:

* Input indexing worker pool. Input indexing worker pool calls `Index()` on each
  chunk source, and then appends the chunk source to the `chunk_sources_` deque
  (under mutex).

* Chunk output worker pool. Fetches the next number from `stream_shuffler_`
  under mutex, then reads the chunk from the chunk source using per-source mutex.

If `stream_shuffler_` runs out of numbers, it's reset to the range
(`last - chunk_window_`, `last`) (and warning message is logged).

## ChunkSet Implementation Plan

**Current State Analysis:**
- ✅ Initial scan handling and ChunkSource creation
- ✅ Sorting and basic indexing logic  
- ❌ Missing output queue and continuous operation
- ❌ Missing chunk output workers with StreamShuffler
- ❌ Missing dynamic window management for old chunks

**Implementation Tasks:**

1. **Add output queue and stage interface**
   - Add `Queue<std::string> output_queue_` member
   - Add public `Queue<std::string>* output()` method
   - Initialize output queue in constructor

2. **Implement continuous file processing**
   - Move initial scan logic to separate method
   - Add input worker that continuously processes new files from queue
   - Handle queue close signal properly

3. **Add chunk output worker pool**
   - Implement workers that fetch numbers from `stream_shuffler_`
   - Add chunk reading logic with per-source mutexes
   - Handle shuffler reset when range is exhausted

4. **Implement dynamic window management**
   - Add logic to remove old chunk sources when total exceeds `chunks_window_`
   - Update `stream_shuffler_` bounds when chunks are added/removed
   - Maintain proper start_chunk_index tracking

5. **Add proper lifecycle management**
   - Handle input queue closure
   - Graceful shutdown of worker threads
   - Close output queue when done

**Files to modify:**
- `src/loader/chunk_feed/chunk_set.h` - Add output queue, missing includes
- `src/loader/chunk_feed/chunk_set.cc` - Complete implementation per above tasks
  