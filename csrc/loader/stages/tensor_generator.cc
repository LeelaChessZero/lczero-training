// ABOUTME: Implementation of TensorGenerator stage for training pipeline.
// ABOUTME: Converts V6TrainingData frames to batched tensors for training.

#include "loader/stages/tensor_generator.h"

#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

TensorGenerator::TensorGenerator(const TensorGeneratorConfig& config)
    : SingleInputStage<TensorGeneratorConfig, InputType>(config),
      SingleOutputStage<OutputType>(config.output()),
      batch_size_(config.batch_size()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Initializing TensorGenerator with " << config.threads()
            << " threads, batch size " << config.batch_size();

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

TensorGenerator::~TensorGenerator() { Stop(); }

void TensorGenerator::Start() {
  LOG(INFO) << "Starting TensorGenerator worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i](std::stop_token stop_token) {
      Worker(stop_token, thread_contexts_[i].get());
    });
  }
}

void TensorGenerator::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;

  LOG(INFO) << "Stopping TensorGenerator.";
  thread_pool_.Shutdown();
  output_queue()->Close();
  LOG(INFO) << "TensorGenerator stopped.";
}

void TensorGenerator::Worker(std::stop_token stop_token,
                             ThreadContext* context) {
  auto producer = output_queue()->CreateProducer();
  std::vector<FrameType> batch;
  batch.reserve(batch_size_);

  try {
    while (true) {
      // Collect frames for a batch.
      batch.clear();
      for (size_t i = 0; i < batch_size_; ++i) {
        LoadMetricPauser pauser(context->load_metric_updater);
        batch.push_back(input_queue()->Get(stop_token));
      }

      // Convert batch to tensors.
      TensorTuple tensors = ConvertFramesToTensors(batch);
      {
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(tensors), stop_token);
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "TensorGenerator worker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "TensorGenerator worker stopping, request cancelled.";
  }
}

TensorTuple TensorGenerator::ConvertFramesToTensors(
    const std::vector<FrameType>& frames) {
  const size_t batch_size = frames.size();
  constexpr size_t kNumPlanes = 112;
  constexpr size_t kNumPolicyMoves = 1858;
  constexpr size_t kNumValueTypes = 6;
  constexpr size_t kValuesPerType = 3;

  TensorTuple result;
  result.reserve(3);

  // Index 0: Input planes (batch_size, 112, 8, 8)
  auto planes_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, kNumPlanes, 8, 8});
  ProcessPlanes(frames, *planes_tensor);
  result.push_back(std::move(planes_tensor));

  // Index 1: Probabilities (batch_size, 1858)
  auto probs_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, kNumPolicyMoves});
  for (size_t i = 0; i < batch_size; ++i) {
    auto probs_slice = probs_tensor->slice({static_cast<ssize_t>(i)});
    std::memcpy(probs_slice.data(), frames[i].probabilities,
                kNumPolicyMoves * sizeof(float));
  }
  result.push_back(std::move(probs_tensor));

  // Index 2: Values (batch_size, 6, 3) with [q, d, m] for each type.
  // [0]: result, [1]: best, [2]: played, [3]: orig, [4]: root, [5]: st
  auto values_tensor =
      std::make_unique<TypedTensor<float>>(std::initializer_list<size_t>{
          batch_size, kNumValueTypes, kValuesPerType});
  for (size_t i = 0; i < batch_size; ++i) {
    const auto& frame = frames[i];
    auto batch_slice = values_tensor->slice({static_cast<ssize_t>(i)});

    // Index 0: result [result_q, result_d, plies_left]
    auto result_slice = batch_slice.subspan(0 * kValuesPerType, kValuesPerType);
    result_slice[0] = frame.result_q;
    result_slice[1] = frame.result_d;
    result_slice[2] = frame.plies_left;

    // Index 1: best [best_q, best_d, best_m]
    auto best_slice = batch_slice.subspan(1 * kValuesPerType, kValuesPerType);
    best_slice[0] = frame.best_q;
    best_slice[1] = frame.best_d;
    best_slice[2] = frame.best_m;

    // Index 2: played [played_q, played_d, played_m]
    auto played_slice = batch_slice.subspan(2 * kValuesPerType, kValuesPerType);
    played_slice[0] = frame.played_q;
    played_slice[1] = frame.played_d;
    played_slice[2] = frame.played_m;

    // Index 3: orig [orig_q, orig_d, orig_m] (may be NaN)
    auto orig_slice = batch_slice.subspan(3 * kValuesPerType, kValuesPerType);
    orig_slice[0] = frame.orig_q;
    orig_slice[1] = frame.orig_d;
    orig_slice[2] = frame.orig_m;

    // Index 4: root [root_q, root_d, root_m]
    auto root_slice = batch_slice.subspan(4 * kValuesPerType, kValuesPerType);
    root_slice[0] = frame.root_q;
    root_slice[1] = frame.root_d;
    root_slice[2] = frame.root_m;

    // Index 5: st [q_st, d_st, NaN]
    auto st_slice = batch_slice.subspan(5 * kValuesPerType, kValuesPerType);
    st_slice[0] = frame.q_st;
    st_slice[1] = frame.d_st;
    st_slice[2] = std::numeric_limits<float>::quiet_NaN();
  }
  result.push_back(std::move(values_tensor));

  return result;
}

void TensorGenerator::ProcessPlanes(const std::vector<FrameType>& frames,
                                    TypedTensor<float>& planes_tensor) {
  const size_t batch_size = frames.size();

  for (size_t i = 0; i < batch_size; ++i) {
    const auto& frame = frames[i];
    auto batch_slice = planes_tensor.slice({static_cast<ssize_t>(i)});

    // Process first 104 planes from frame.planes (each uint64_t represents 64
    // bits).
    for (ssize_t plane = 0; plane < 104; ++plane) {
      auto plane_slice = batch_slice.subspan(plane * 64, 64);
      uint64_t plane_bits = frame.planes[plane];

      for (ssize_t square = 0; square < 64; ++square) {
        // XOR with 7 remaps the index within each byte from 0..7 to 7..0.
        plane_slice[square] =
            static_cast<float>((plane_bits >> (square ^ 7)) & 1);
      }
    }

    // Add 8 additional planes for metadata (planes 104-111).
    const std::pair<ssize_t, float> meta_planes[] = {
        {104, static_cast<float>(frame.castling_us_ooo)},
        {105, static_cast<float>(frame.castling_us_oo)},
        {106, static_cast<float>(frame.castling_them_ooo)},
        {107, static_cast<float>(frame.castling_them_oo)},
        {108, static_cast<float>(frame.side_to_move_or_enpassant)},
        {109, static_cast<float>(frame.rule50_count) / 99.0f},
        {110, 0.0f},  // All zeros (constant plane).
        {111, 1.0f},  // All ones (constant plane).
    };

    for (const auto& [plane_num, value] : meta_planes) {
      auto plane_slice = batch_slice.subspan(plane_num * 64, 64);
      absl::c_fill(plane_slice, value);
    }
  }
}

StageMetricProto TensorGenerator::FlushMetrics() {
  StageMetricProto stage_metric;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);
  *stage_metric.add_queue_metrics() =
      MetricsFromQueue("output", *output_queue());
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
