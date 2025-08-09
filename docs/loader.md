# Data Loader

The Data Loader is a C++ module (exposed to Python via pybind11) that handles
loading, preprocessing, shuffling, and feeding training data for the Leela Chess
Zero training process.

## Python Integration

The loader has been exposed to Python through pybind11, allowing direct use of the C++ DataLoader from Python code. Key aspects:

* **Configuration**: Python dataclasses mirror C++ configuration structures (e.g., `FilePathProviderConfig`, `DataLoaderConfig`)
* **Memory Management**: Uses `unique_ptr::release()` with `py::return_value_policy::take_ownership` for efficient tensor ownership transfer
* **Output Format**: Returns tuple of numpy arrays compatible with JAX through the buffer protocol
* **Usage**: Import via `from lczero_training import DataLoader, DataLoaderConfig`

## High-Level Overview

The Data Loader consists of the following stages connected through a
[Queue](../csrc/utils/queue.h):

* [FilePathProvider](../csrc/loader/chunk_feed/file_path_provider.h) — Training
  data discovery worker (watches a directory and provides feed of filenames)
* [ChunkSourceLoader](../csrc/loader/chunk_feed/chunk_source_loader.h) — Reads
  chunks from files, providing a stream of chunks.
* [ShufflingChunkPool](../csrc/loader/chunk_feed/shuffling_chunk_pool.h) — Keeps
  a set of chunks, managing the last `num_chunks` available and removing old
  ones, and outputting them in shuffled order.
* (skip for now) [ChunkValidator](../csrc/loader/chunk_feed/chunk_validator.h) —
  Filters the chunk stream, filtering out invalid chunks.
* (skip for now) [ChunkRescorer](../csrc/loader/chunk_feed/chunk_rescorer.h) —
  Rescores chunks based on tablebase or intentional blunders.
* [ChunkUnpacker](../csrc/loader/chunk_feed/chunk_unpacker.h) — Unpacks
  chunks into frames, which are then processed by the next stages.
* [ShufflingFrameSampler](../csrc/loader/shuffling_frame_sampler.h) — Takes a
  stream of frames and provides shuffled batches of frames for training, using
  reservoir sampling.
* [TensorGenerator](../csrc/loader/tensor_generator.h) — Takes frames and
  provides tensor buffers for the training process.

## TensorGenerator

Batch size is configurable in the stage options.

The `TensorGenerator` stage takes frames from the input queue and produces tuple
of tensors for tensor returned as [TensorTuple](../csrc/utils/tensor.h).
The first dimension of every tensor in the tuple is the batch size, and the
rest are described in the [Training Tuple Format](training_tuple.md).

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
    // input_queue is omitted in the producer stages like FilePathProvider.
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
utilizes a [StreamShuffler](../csrc/loader/stream_shuffler.h) to which provides
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

## ShufflingFrameSampler

The sampler uses reservoir sampling:

* It has a reservoir of predefined size (1000000 is quite typical)
* Initially it just fills the reservoir with frames from the input queue until
  it's full.
* After that, it picks random frames from the reservoir and outputs them,
  refilling the used spot from the input queue.
* It closes the output queue when either explicit Close() is called or the input
  queue is closed.
* `using FrameType = V6TrainingData;`, use `absl::FixedArray<FrameType>` for
  the reservoir.