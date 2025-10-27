# Writing a New Data Loader Stage

This guide walks through the lifecycle of adding another stage to the dynamic
data loader pipeline. It assumes you are working in the C++ orchestrator (under
`csrc/loader/stages/`) and that the Python bindings already consume staged
configurations.

## 1. Design the Stage Surface

- **Purpose and data flow**: Decide whether the stage produces new data (no
  upstream input) or transforms items from an existing queue.
- **Configuration shape**: Determine which knobs are required during
  construction (thread counts, capacities, etc.). These become fields on the
  stage-specific protobuf message.
- **Outputs and control hooks**: Clarify what queue type the stage emits and
  whether it needs control-plane messages.

## 2. Extend the Protobufs

- Add a new `message <YourStage>Config` to
  `proto/data_loader_config.proto`, including an `optional string input` field
  if the stage consumes upstream data.
- For single-output stages, add `optional QueueConfig output = N` to configure
  the output queue. `QueueConfig` provides `queue_capacity` (default 4),
  `overflow_behavior` (BLOCK, DROP_NEW, KEEP_NEWEST), and optional `name`.
- For multi-output stages, use `repeated QueueConfig output` with parallel
  configuration arrays (see `ChunkSourceSplitterConfig` for reference).
- Update `StageConfig` with an `optional <YourStage>Config` entry so the stage
  can be referenced from the `repeated stage` list.
- If the stage emits custom metrics, extend `StageMetricProto` in
  `proto/training_metrics.proto`. Prefer the existing `load_metrics`,
  `queue_metrics`, and `count_metrics` collections when possible.
- When the stage needs control requests or responses, extend
  `proto/stage_control.proto` so they can be carried through
  `StageControlRequest`/`StageControlResponse`.
- Regenerate protobufs (`meson compile -C builddir` or `just build-proto`).

## 3. Choose a Base Class

- **Use `SingleInputStage<ConfigT, InputT>`** when the stage consumes exactly
  one upstream queue. The helper resolves the input binding, performs the
  `dynamic_cast`, and surfaces the typed `Queue<InputT>*` via `input_queue()`.
- **Use `SingleOutputStage<OutputT>`** when the stage produces exactly one
  output queue. The helper manages the output queue, implements `GetOutput()`
  with name validation, and surfaces the typed `Queue<OutputT>*` via
  `output_queue()`.
- **Most stages inherit from both** `SingleInputStage` and `SingleOutputStage`
  using virtual inheritance (both base classes virtually inherit from `Stage`
  to avoid the diamond problem). Example:
  ```cpp
  class MyStage : public SingleInputStage<MyStageConfig, InputType>,
                  public SingleOutputStage<OutputType> {
   public:
    MyStage(const MyStageConfig& config, const StageRegistry& existing_stages)
        : SingleInputStage<MyStageConfig, InputType>(config, existing_stages),
          SingleOutputStage<OutputType>(config.output()) {}
  };
  ```
- **Inherit `Stage` directly** when the stage has multiple inputs, multiple
  outputs, or manages more complex wiring. In that case you must implement
  input/output discovery and `GetOutput()` yourself.
- Place declarations in `csrc/loader/stages/<stage_name>.h` and definitions in
  the matching `.cc` file.

## 4. Implement the Stage API

- **Constructor**: Initialize base classes with config and `config.output()` (or
  `config.input()` for `SingleInputStage`). Store additional config fields and
  initialize worker pools. Avoid starting threads here.
- **`Start()`**: Launch background work. Acquire `Queue::Producer` instances
  from `output_queue()->CreateProducer()` for emitting data and honour
  `stop_requested_` flags so shutdown is cooperative.
- **`Stop()`**: Close queues via `output_queue()->Close()`, signal workers to
  exit, and join threads. Remember that downstream stages expect
  `Queue::Close()` to signal completion.
- **`GetOutput(std::string_view name)`**: Only implement if the stage has
  multiple outputs. `SingleOutputStage` provides this automatically for
  single-output stages, including name validation.
- **`Control()`**: Handle relevant `StageControlRequest` sub-messages and return
  a populated `StageControlResponse` wrapped in `std::optional`. Return
  `std::nullopt` for requests the stage does not recognise.

## 5. Report Metrics

- **Accumulate state** while workers run (e.g., load metrics, counters,
  queue statistics).
- **`FlushMetrics()`** should snapshot the current values, reset internal
  counters as needed, and populate `StageMetricProto`. Set the
  `stage_type` field so the UI can identify the stage kind. Use helpers like
  `MetricsFromQueue("output", *output_queue())` to expose queue utilisation
  under `queue_metrics`, and append load information via `load_metrics`.
- For multiple queues or distinct metric groups, add additional entries with
  meaningful names (`"output"`, `"prefetch"`, etc.) so downstream tooling can
  pick the right series.
- If you rename or split a metric, document the change and update dashboards.
  For example, `ShufflingChunkPool` now emits `chunks_current` (window size)
  and `chunks_total` (total indexed chunks) instead of a single `chunks`
  series; the Grafana panels consuming the old series were repointed to
  `chunks_current` so the graphs remain accurate.

## 6. Register the Stage

- Update `CreateStage` in `csrc/loader/stages/stage_factory.cc` to construct
  the new class when its config is present. Enforce the “exactly one sub-config”
  rule by keeping the existing `CountStageConfigs()` logic in sync.
- Ensure `meson.build` lists the new source files so the static library rebuilds.

## 7. Wire Up Tests

- Add focused unit tests under `csrc/loader/stages/` validating constructor
  errors, thread lifecycle, metric flushing, and (if applicable) control-plane
  behaviour.
- Provide integration coverage where the stage participates in a small pipeline
  built from serialized `DataLoaderConfig` messages.
- If Python bindings surface stage-specific behaviour, extend the relevant
  `pytest` suites too.

## 8. Update Documentation and Examples

- Document new config fields in `docs/` (for example, augment `docs/loader.md`
  or create stage-specific notes).
- Add sample snippets or textproto fragments showing how to reference the
  stage in a pipeline.
- Mention any new control commands so the daemon/TUI maintainers know how to
  surface them.

Following these steps keeps the stage ecosystem consistent: configurations are
validated at construction time, queues remain type-safe, metrics feed the UI,
and Python clients continue to operate through the generic factory and control
plane.
