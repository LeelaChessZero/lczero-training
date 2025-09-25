# **Specification: Dynamic Data Loading Pipeline**

## **1. Project Goal**

To refactor the existing hardcoded C++ data loading pipeline into a dynamically constructible system. The new design will be driven by a flexible protobuf configuration, allowing for arbitrary pipeline stage composition. It will use a factory pattern for stage creation, type erasure for connecting queues, and a refactored metrics and control system.

---

## **2. Core C++ Interface Changes**

### **2.1. `QueueBase` Virtual Interface**

To enable type-erased handling of queues, a non-templated base class `QueueBase` will be created. The existing `Queue<T>` class will inherit from it.

```cpp
// in utils/queue.h
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
    // ... existing implementation ...
};
```

### **2.2. `Stage` Abstract Base Class**

All pipeline stages will inherit from a new `Stage` abstract base class.

```cpp
// in loader/stages/stage.h
class Stage {
public:
    virtual ~Stage() = default;
    virtual void Start() = 0;
    virtual void Stop() = 0; // No graceful_drain parameter
    virtual StageMetricProto FlushMetrics() = 0;
    virtual QueueBase* GetOutput(std::string_view name) = 0;
    virtual std::optional<StageControlResponse> Control(
        const StageControlRequest& request) = 0;
};
```

### **2.3. Templated Base Classes for Stages**

To reduce boilerplate for common patterns (e.g., single-input stages), helper templates will be provided.

```cpp
// in loader/stages/stage.h
template <typename InputT>
class SingleInputStage : public Stage {
protected:
    Queue<InputT>* input_queue_;

public:
    // Constructor will take the stage's specific config proto and the vector
    // of existing stages. It is responsible for parsing the input name from the
    // config, looking up the predecessor stage and its output queue, and
    // performing a dynamic_cast to the expected Queue<InputT>*.
    // Throws std::runtime_error on lookup or type mismatch.
    explicit SingleInputStage(const google::protobuf::Message& config,
                   const std::vector<std::pair<std::string, Stage*>>& existing_stages);
};
```

---

## **3. Protobuf Schema Refactoring**

### **3.1. `data_loader_config.proto`**

The main configuration will be changed from fixed fields to a repeated list of generic stage configurations.

```proto
// Top-level configuration
message DataLoaderConfig {
  repeated StageConfig stages = 1;
}

// Generic container for any stage configuration
message StageConfig {
  // A unique name for this stage instance, e.g., "unpacker_1"
  optional string name = 1;

  // Exactly ONE of the following must be set.
  optional FilePathProviderConfig file_path_provider = 2;
  optional ChunkSourceLoaderConfig chunk_source_loader = 3;
  optional ShufflingChunkPoolConfig shuffling_chunk_pool = 4;
  optional ChunkUnpackerConfig chunk_unpacker = 5;
  optional ShufflingFrameSamplerConfig shuffling_frame_sampler = 6;
  optional TensorGeneratorConfig tensor_generator = 7;
}

// Stage-specific configs that take one input must have an 'input' field.
// Source stages (e.g., FilePathProviderConfig) will not have this field.
message ChunkUnpackerConfig {
  // Name of the input queue, e.g., "shuffling_pool.output"
  optional string input = 1;

  optional uint64 threads = 2 [default = 1];
  optional uint64 queue_capacity = 3 [default = 16];
}
// Other stage configs (ChunkSourceLoader, ShufflingChunkPool, etc.)
// that require an input must be modified similarly.
```

**3.2. `data_loader_metrics.proto`**

The metrics proto will be refactored to support a dynamic list of stages.

```proto
// Top-level metrics
message DataLoaderMetricsProto {
  repeated StageMetricProto stage_metrics = 1;
}

// A generic container for a single stage's metrics
message StageMetricProto {
  // The instance name of the stage, matching StageConfig.name
  optional string name = 1;

  // Stage-type-specific metrics (without queue info)
  // Exactly ONE of the following will be set.
  optional FilePathProviderMetricsProto file_path_provider = 2;
  optional ChunkSourceLoaderMetricsProto chunk_source_loader = 3;
  // ... etc. for all other stage types

  // Metrics for ALL of this stage's output queues
  repeated QueueMetricProto output_queue_metrics = 10;
}

// QueueMetricProto is modified to include a name
message QueueMetricProto {
  optional string name = 1; // e.g., "output"
  optional uint64 put_count = 2 [default = 0];
  // ... other fields shifted down ...
}

// All stage-specific metrics protos (e.g., ChunkUnpackerMetricsProto)
// must have their hardcoded 'optional QueueMetricProto queue' field REMOVED.
// This information is now captured in StageMetricProto.
```

**3.3. New `stage_control.proto`**

A new file will be created to handle the generic control plane.

```proto
syntax = "proto2";
package lczero.training;

// --- Request Messages ---
message ShufflingChunkPoolControlRequest {
  optional bool reset_chunk_anchor = 1 [default = false];
  optional string set_chunk_anchor = 2;
}

message StageControlRequest {
  optional ShufflingChunkPoolControlRequest chunk_pool_request = 1;
}

// --- Response Messages ---
message ShufflingChunkPoolControlResponse {
  optional string chunk_anchor = 1;
  optional int32 chunks_since_anchor = 2;
}

message StageControlResponse {
  optional ShufflingChunkPoolControlResponse chunk_pool_response = 1;
}
```

---

## **4. `DataLoader` Class Refactoring**

The `DataLoader` class will be refactored to be a generic pipeline orchestrator.

*   **Members:**
    *   `std::vector<std::pair<std::string, std::unique_ptr<Stage>>> stages_`: Stores all stage instances in topological order.
    *   `Queue<TensorTuple>* final_output_queue_`: A raw pointer to the typed output queue of the final stage.
*   **Constructor:** `DataLoader()` will take no arguments.
*   **Configuration:** New methods `void AddStage(const std::string& serialized_stage_config)` and `void AddStages(const std::string& serialized_dataloader_config)` will be added. These will use the `CreateStage` factory to build the pipeline. The final stage added *must* have an output named "output" of type `Queue<TensorTuple>`; a `dynamic_cast` will verify this, throwing an exception on failure.
*   **`GetNext()`:** Will call `Get()` on the cached `final_output_queue_`.
*   **`Start()` / `Stop()`:** Will iterate through `stages_` in order and call `Start()` / `Stop()` on each stage.
*   **Metrics:** The `MetricsThread` will iterate through `stages_`, call `stage->FlushMetrics()`, and aggregate the returned `StageMetricProto` messages into a `DataLoaderMetricsProto`.
*   **Control Plane:** A new method `std::vector<std::string> SendControlMessage(const std::string& serialized_request)` will be added. It will deserialize the request, pass it to the `Control()` method of every stage, collect all non-empty responses, and return them as a vector of serialized strings.

---

## **5. Factory Implementation**

A free function `std::unique_ptr<Stage> CreateStage(const StageConfig& config, const std::vector<std::pair<std::string, Stage*>>& existing_stages)` will be implemented.

*   **Logic:**
    1.  Validate that exactly one stage-specific config field is set in `config`. Throw an exception otherwise.
    2.  Use a hardcoded `if/else if` chain on the `has_...()` methods to determine the stage type.
    3.  Within each block, call the appropriate constructor, passing the specific config message and the `existing_stages` vector.
    4.  Example: `return std::make_unique<ChunkUnpacker>(config.chunk_unpacker_config(), existing_stages);`
*   This factory will reside in a new file, e.g., `loader/stages/stage_factory.cc`.