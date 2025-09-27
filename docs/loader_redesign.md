# Specification: Dynamic Data Loading Pipeline

This document captures the end-to-end plan for refactoring the data loader into
an extensible, configuration-driven pipeline. The implementation will proceed in
explicit phases so we can land the work incrementally without losing track of
context.

## Project Goal

Refactor the current hardcoded C++ data loading pipeline into a dynamically
constructible system that:

- Builds stage graphs from protobuf configuration.
- Wires stages through type-erased queues.
- Surfaces uniform metrics and control-plane hooks.
- Keeps Python bindings thin while enabling experimentation.

## Phased Implementation Overview

**Important:** Once Phase 1 work starts and until Phase 4 is complete, expect
the project _not_ to be in a buildable state. Plan local work accordingly and
avoid landing intermediate commits on `main`.

### Phase 1 – Scaffolding and Protocols

Lay the groundwork that everything else builds on.

1. **Type-erased queues**
   - Introduce `QueueBase` in `utils/queue.h` with virtual accessors for size,
     capacity, close state, and counters.
   - Make `Queue<T>` inherit from `QueueBase` without changing existing
     semantics.
2. **Generic stage interface**
   - Create `loader/stages/stage.h` defining:
     - `class Stage` with `Start()`, `Stop()`, `FlushMetrics()`, `GetOutput()`,
       and `Control()`.
     - Helper templates (e.g., `SingleInputStage`) that parse input bindings and
       perform `dynamic_cast` validation.
3. **Protobuf overhaul**
   - Update `proto/data_loader_config.proto` to a repeated `StageConfig` list
     with stage-specific submessages that optionally include `input` fields.
   - Rewrite `proto/training_metrics.proto` so
     `DataLoaderMetricsProto.stage_metrics` collects `StageMetricProto`, each
     with stage-specific metrics plus `repeated QueueMetricProto` where each
     entry has a `name` field.
   - Add `proto/stage_control.proto` containing generic request/response
     envelopes (initially only `ShufflingChunkPool` commands).
4. **Metrics helpers**
   - Adjust `loader/data_loader_metrics.h/cc` to aggregate the new metrics
     layout: queue metrics keyed by name, stage metrics grouped per stage.
5. **Build plumbing**
   - Update Meson, pybind, and Python packaging to build the new protos.

Deliverables: new headers, updated protos, regenerated bindings, no behavioural
changes yet. All existing stages continue to compile against the old
constructors.

### Phase 2 – Stage Migration

Port each stage to the generic interface while keeping the old `DataLoader`
implementation functioning.

1. Update stage headers (`file_path_provider`, `chunk_source_loader`,
   `shuffling_chunk_pool`, `chunk_unpacker`, `shuffling_frame_sampler`,
   `tensor_generator`) to inherit from `Stage`/`SingleInputStage`.
2. Replace direct member wiring (`Queue<T>* input_queue_`) with calls to the
   helper templates to resolve inputs.
3. Update `FlushMetrics()` implementations to return `StageMetricProto` and emit
   queue metrics via `MetricsFromQueue(name, queue)` helper.
4. Implement `GetOutput()` and (where applicable) `Control()` to serve
   stage-specific control requests.
5. Ensure each stage’s `Stop()` obeys the new lifecycle (closing output queues,
   stopping threads) without relying on legacy orchestrator behaviour.

Deliverables: all stages compile against the new interface. The legacy
`DataLoader` still owns concrete members but stages have constructors accepting
(a) their config message and (b) the vector of existing stages.

### Phase 3 – Orchestrator and Factory

Replace the monolithic `DataLoader` wiring with dynamic assembly.

1. **Factory**
   - Implement `loader/stages/stage_factory.cc` with
     `std::unique_ptr<Stage> CreateStage(const StageConfig&, const StageList&)`.
   - Validate the “exactly one sub-config set” rule and perform type-specific
     construction.
2. **DataLoader rewrite**
   - Store `std::vector<std::pair<std::string, std::unique_ptr<Stage>>>`.
   - Provide `AddStage()` / `AddStages()` to parse serialized protos, call the
     factory, and keep a typed pointer to the final stage’s `Queue<TensorTuple>`.
   - Rewrite `Start()`, `Stop()`, `GetNext()` to iterate over `stages_`.
   - Rework metrics thread to iterate stages and aggregate
     `StageMetricProto` into `DataLoaderMetricsProto`.
   - Add `SendControlMessage()` to fan out control requests to each stage and
     collect responses.
3. **Backward-compatibility shim**
   - Introduce a temporary translator that maps the old `DataLoaderConfig`
     layout to the staged representation for Python callers until Phase 4 is
     done.

Deliverables: new `DataLoader` orchestrator in place, tests updated to cover the
factory and failure cases. Legacy chunk-anchor helpers now wrap the generic
control plane.

### Phase 4 – Python Integration and Cleanup

Finalize the migration and restore build health.

1. Update Python dataclasses/serialization helpers to emit the staged proto
   format directly (removing the compatibility shim).
2. Adjust pybind bindings to expose new methods (`AddStages`,
   `SendControlMessage`, etc.) or adopt the finalized constructor semantics.
3. Refresh TUI and daemon code to consume staged metrics (iterate over
   `stage_metrics`, use queue names).
4. Remove legacy code paths, delete unused proto fields, and clean up build
   glue.
5. Run `just pre-commit` to verify formatting, lint, and tests.

**Important:** After Phase 4 completes and verification passes, the project is
back to a buildable state.

## Architectural Reference

The following sections provide detailed design notes referenced by the phases.

### Queue Abstraction (`utils/queue.h`)

```cpp
class QueueBase {
 public:
  virtual ~QueueBase() = default;
  virtual size_t Size() const = 0;
  virtual size_t Capacity() const = 0;
  virtual bool IsClosed() const = 0;
  virtual void Close() = 0;
  virtual size_t GetTotalPutCount(bool reset = false) = 0;
  virtual size_t GetTotalGetCount(bool reset = false) = 0;
  virtual size_t GetTotalDropCount(bool reset = false) = 0;
};

template <typename T>
class Queue : public QueueBase {
  // Existing implementation remains unchanged beyond inheriting QueueBase.
};
```

### Stage Interface (`loader/stages/stage.h`)

```cpp
class Stage {
 public:
  virtual ~Stage() = default;
  virtual void Start() = 0;
  virtual void Stop() = 0;  // Graceful drain no longer supported.
  virtual StageMetricProto FlushMetrics() = 0;
  virtual QueueBase* GetOutput(std::string_view name) = 0;
  virtual std::optional<StageControlResponse> Control(
      const StageControlRequest& request) = 0;
};

template <typename ConfigT, typename InputT>
class SingleInputStage : public Stage {
 protected:
  explicit SingleInputStage(
      std::string_view input_name,
      const std::vector<std::pair<std::string, Stage*>>& existing_stages);
  Queue<InputT>* input_queue();
};
```

`SingleInputStage` will parse `config.input()`, locate the producing stage by
name, and `dynamic_cast` the output queue to `Queue<InputT>`. Failures raise
`std::runtime_error` to surface misconfiguration early.

### Protobuf Schema Highlights

`proto/data_loader_config.proto`:

```proto
message DataLoaderConfig {
  repeated StageConfig stage = 1;
}

message StageConfig {
  optional string name = 1;
  optional FilePathProviderConfig file_path_provider = 2;
  optional ChunkSourceLoaderConfig chunk_source_loader = 3;
  optional ShufflingChunkPoolConfig shuffling_chunk_pool = 4;
  optional ChunkUnpackerConfig chunk_unpacker = 5;
  optional ShufflingFrameSamplerConfig shuffling_frame_sampler = 6;
  optional TensorGeneratorConfig tensor_generator = 7;
}

message ChunkUnpackerConfig {
  optional string input = 1;
  optional uint64 threads = 2 [default = 1];
  optional uint64 queue_capacity = 3 [default = 16];
}
// Other stage configs gain similar `input` fields where applicable.
```

`proto/training_metrics.proto`:

```proto
message QueueMetricProto {
  optional string name = 1;
  optional uint64 put_count = 2 [default = 0];
  optional uint64 get_count = 3 [default = 0];
  optional uint64 drop_count = 4 [default = 0];
  optional StatisticsProtoInt64 queue_fullness = 5;
  optional uint64 queue_capacity = 6 [default = 0];
}

message StageMetricProto {
  optional string name = 1;
  optional FilePathProviderMetricsProto file_path_provider = 2;
  optional ChunkSourceLoaderMetricsProto chunk_source_loader = 3;
  optional ShufflingChunkPoolMetricsProto shuffling_chunk_pool = 4;
  optional ChunkUnpackerMetricsProto chunk_unpacker = 5;
  optional ShufflingFrameSamplerMetricsProto shuffling_frame_sampler = 6;
  optional TensorGeneratorMetricsProto tensor_generator = 7;
  repeated QueueMetricProto output_queue_metrics = 10;
}

message DataLoaderMetricsProto {
  repeated StageMetricProto stage_metrics = 1;
}
```

`proto/stage_control.proto`:

```proto
message ShufflingChunkPoolControlRequest {
  optional bool reset_chunk_anchor = 1 [default = false];
  optional string set_chunk_anchor = 2;
}

message StageControlRequest {
  optional ShufflingChunkPoolControlRequest chunk_pool_request = 1;
}

message ShufflingChunkPoolControlResponse {
  optional string chunk_anchor = 1;
  optional int32 chunks_since_anchor = 2;
}

message StageControlResponse {
  optional ShufflingChunkPoolControlResponse chunk_pool_response = 1;
}
```

### DataLoader Orchestrator (Phase 3 Target)

Key responsibilities:

- Maintain ordered stage list and final output queue pointer.
- Validate stage uniqueness and output types during `AddStage`.
- Start/stop stages in order and manage metrics aggregation thread.
- Provide `SendControlMessage()` returning serialized responses.
- Preserve chunk-anchor helpers by translating them into control messages.

### Stage Factory Pattern

`CreateStage(const StageConfig&, const StageList&)` will:

1. Ensure exactly one sub-config is populated.
2. Use `if/else` on `has_...()` to dispatch to concrete constructors.
3. Pass both the specific config and the existing stage list for input
   resolution.
4. Return `std::unique_ptr<Stage>`.

### Testing Strategy

- Extend stage unit tests to cover constructor failures and control handling.
- Add factory tests for missing/duplicate config validation.
- Provide integration tests that build minimal pipelines from serialized protos
  and assert correct data flow.
- Exercise metrics aggregation via deterministic hooks to avoid timing flakes.

## Useful Commands

- `just build` — Rebuilds the C++ components (including regenerating protobufs).
- `just build-proto` — Regenerates the Python protobuf stubs.
- `just pre-commit` — Runs formatting, lint, build, and test checks locally.
