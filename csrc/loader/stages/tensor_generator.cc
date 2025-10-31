// ABOUTME: Implementation of TensorGenerator stage for training pipeline.
// ABOUTME: Converts V6TrainingData frames to batched tensors for training.

#include "loader/stages/tensor_generator.h"

#include <cstring>
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
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

void TensorGenerator::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }

  LOG(INFO) << "Stopping TensorGenerator.";
  input_queue()->Close();
  output_queue()->Close();
  thread_pool_.WaitAll();
  thread_pool_.Shutdown();
  LOG(INFO) << "TensorGenerator stopped.";
}

void TensorGenerator::Worker(ThreadContext* context) {
  auto producer = output_queue()->CreateProducer();
  std::vector<FrameType> batch;
  batch.reserve(batch_size_);

  try {
    while (true) {
      // Collect frames for a batch.
      batch.clear();
      for (size_t i = 0; i < batch_size_; ++i) {
        LoadMetricPauser pauser(context->load_metric_updater);
        batch.push_back(input_queue()->Get());
      }

      // Convert batch to tensors.
      TensorTuple tensors;
      ConvertFramesToTensors(batch, tensors);
      {
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(tensors));
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "TensorGenerator worker stopping, input queue closed.";
    // Input queue is closed.
  }
}

void TensorGenerator::ConvertFramesToTensors(
    const std::vector<FrameType>& frames, TensorTuple& tensors) {
  const size_t batch_size = frames.size();
  constexpr size_t kNumPlanes = 112;
  constexpr size_t kNumPolicyMoves = 1858;

  // Create tensors according to training tuple format:
  // 1. planes: (batch_size, 112, 64) as float32
  // 2. probs: (batch_size, 1858) as float32
  // 3. winner: (batch_size, 3) as float32
  // 4. best_q: (batch_size, 3) as float32
  // 5. plies_left: (batch_size,) as float32

  tensors.clear();
  tensors.reserve(5);

  // 1. Planes tensor: (batch_size, 112, 8, 8)
  auto planes_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, kNumPlanes, 8, 8});
  ProcessPlanes(frames, *planes_tensor);
  tensors.push_back(std::move(planes_tensor));

  // 2. Probabilities tensor: (batch_size, 1858)
  auto probs_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, kNumPolicyMoves});
  for (size_t i = 0; i < batch_size; ++i) {
    auto probs_slice = probs_tensor->slice({static_cast<ssize_t>(i)});
    std::memcpy(probs_slice.data(), frames[i].probabilities,
                kNumPolicyMoves * sizeof(float));
  }
  tensors.push_back(std::move(probs_tensor));

  // 3. Winner tensor: (batch_size, 3)
  auto winner_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, 3});
  for (size_t i = 0; i < batch_size; ++i) {
    auto winner_slice = winner_tensor->slice({static_cast<ssize_t>(i)});
    // Convert result_q, result_d to win/draw/loss probabilities.
    const float q = frames[i].result_q;
    const float d = frames[i].result_d;
    const float win = (1.0f + q - d) / 2.0f;
    const float loss = (1.0f - q - d) / 2.0f;
    winner_slice[0] = win;
    winner_slice[1] = d;
    winner_slice[2] = loss;
  }
  tensors.push_back(std::move(winner_tensor));

  // 4. Best Q tensor: (batch_size, 3)
  auto best_q_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, 3});
  for (size_t i = 0; i < batch_size; ++i) {
    auto best_q_slice = best_q_tensor->slice({static_cast<ssize_t>(i)});
    // Convert best_q, best_d to win/draw/loss probabilities.
    const float q = frames[i].best_q;
    const float d = frames[i].best_d;
    const float win = (1.0f + q - d) / 2.0f;
    const float loss = (1.0f - q - d) / 2.0f;
    best_q_slice[0] = win;
    best_q_slice[1] = d;
    best_q_slice[2] = loss;
  }
  tensors.push_back(std::move(best_q_tensor));

  // 5. Plies left tensor: (batch_size,)
  auto plies_left_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size});
  for (size_t i = 0; i < batch_size; ++i) {
    auto plies_left_slice = plies_left_tensor->slice({static_cast<ssize_t>(i)});
    plies_left_slice[0] = frames[i].plies_left;
  }
  tensors.push_back(std::move(plies_left_tensor));
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
