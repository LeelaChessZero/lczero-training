# Data Loader

The Data Loader is a C++ module (exposed to Python via pybind11) that handles
loading, preprocessing, shuffling, and feeding training data for the Leela Chess
Zero training process.

## High-Level Overview

The Data Loader consists of the following parts:

* [Chunk Feed](../src/loader/chunk_feed) — Provides a stream of chunks.
  * [Discovery](../src/loader/chunk_feed/discovery.h) — Watches a directory for
    new chunk files. Also performs initial chunk discovery.
  * [Chunk Set](../src/loader/chunk_feed/chunk_set.h) — Manages a set of chunks,
    keeping last `num_chunks` available and removing old ones. Typical values
    for `num_chunks` is around 30'000'000.
  * TODO Describe further.
* [Frame Shuffler](../src/loader/frame_shuffler) — Takes a stream of chunks and
  provides shuffled batches of frames for training.
  * TODO Describe further.
