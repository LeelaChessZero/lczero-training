# Implement single position sampling in Shuffling Pool

This document defines a new way of sampling in
[Shuffling Pool](../csrc/loader/stages/shuffling_chunk_pool.h) (and .cc).

## reshuffle_count -> use_count

In `TrainingChunk` in
[training_chunk.h](../csrc/loader/stages/training_chunk.h) the `reshuffle_count`
should be renamed to `use_count`. As with the new sampling method, usage is not
necessarily tied to reshuffling. Also the code that
[uses it](../csrs/loader/stages/chunk_unpacker.cc) will need to be updated.

In `ChunkSourceItem`, instead of one `reshuffle_count` per entire chunk source,
we'll have a `std::vector<uint16_t> use_counts`, initially filled with zeros.
Instead of updating it on reshuffling, we will update it for individual chunks
when they are returned. In `GetNextChunkData()` we return old value (i.e. 0 for
the first time).

Local struct `ShufflingChunkPool::ChunkData` in
../csrc/loader/stages/shuffling_chunk_pool.cc should also have `use_count`
instead of `reshuffle_count`.

## Configuration changes

In the [config](../proto/data_loader_config.proto), in `ShufflingChunkPoolConfig`,
we add the following fields:

```proto
message ShufflingChunkPoolConfig {
  // existing fields...
  optional uint64 hanse_sampling_threshold = 6;  // by default, do not use new sampling.
  optional double hanse_sampling_gamma = 7 [default = 1.0];
}
```

## Algorithm changes

In addition to `use_counts`, `ChunkSourceItem` will have a new field:
`std::vector<uint16_t> num_records;`. It will contain the number of records in
each chunk, and will act as a cache. Initially, it's filled with zeros.

When `record_bound` is not set, the sampling method is the same as before.

When `record_bound` is set, we will use the new sampling method. It goes like
this:

`GetNextChunkData()` doesn't call `LoadChunkData()` right away.
Instead, it first checks if `num_records[chunk_index]` is zero. If so, it
calls `LoadChunkData()` to load the chunk, and counts the number of records in
it (by dividing its size to `sizeof<FrameType>`).

Then, we decide whether to return this chunk or to sample again. We do this by
drawing a random number `u` uniformly from `[0, 1)`, and comparing it to
`p = min(1.0, num_records/hthreshold) ^ gamma`. If `u < p`, we return this chunk
(we need to call `LoadChunkData()` if we didn't already) and increment
use_count. Otherwise, we sample again (i.e. pick a new `chunk_index` and repeat
the process) — without incrementing `use_count`.