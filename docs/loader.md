# Data Loader

The Data Loader is a C++ module (exposed to Python via pybind11) that handles
loading, preprocessing, shuffling, and feeding training data for the Leela Chess
Zero training process.

## High-Level Overview

The Data Loader consists of the following stages connected through a
[Queue](../src/utils/queue.h):

* [Discovery](../src/loader/chunk_feed/discovery.h) — Training data discovery
  worker (watches a directory and provides feed of filenames)
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

## Stage interface

All stages implement the similar API and structure, although not sharing any
base class.

```cpp
class Stage {
 public:
    using InputType = ...;  // Type of input data for this stage
    using OutputType = ...; // Type of output data from this stage
    // input_queue is omitted in the producer stages like Discovery.
    Stage(Queue<InputType>* input_queue, /* other params */);
    Queue<OutputType>* output() const;

private:
    ThreadPool thread_pool_;
    Queue<InputType>* input_queue_;
    Queue<OutputType> output_queue_;
};
```
