// ABOUTME: Implementation of TensorGenerator stage for training pipeline.
// ABOUTME: Converts V6TrainingData frames to batched tensors for training.

#include "loader/tensor_generator.h"

#include <cstring>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"

namespace lczero {
namespace training {

TensorGenerator::TensorGenerator(Queue<InputType>* input_queue,
                                 const TensorGeneratorOptions& options)
    : input_queue_(input_queue),
      output_queue_(options.output_queue_size),
      thread_pool_(options.worker_threads, ThreadPoolOptions{}),
      batch_size_(options.batch_size) {
  for (size_t i = 0; i < options.worker_threads; ++i) {
    thread_pool_.Enqueue([this]() { Worker(); });
  }
}

Queue<TensorGenerator::OutputType>* TensorGenerator::output() {
  return &output_queue_;
}

void TensorGenerator::Worker() {
  auto producer = output_queue_.CreateProducer();
  std::vector<FrameType> batch;
  batch.reserve(batch_size_);

  try {
    while (true) {
      // Collect frames for a batch.
      batch.clear();
      for (size_t i = 0; i < batch_size_; ++i) {
        batch.push_back(input_queue_->Get());
      }

      // Convert batch to tensors.
      TensorTuple tensors;
      ConvertFramesToTensors(batch, tensors);
      producer.Put(std::move(tensors));
    }
  } catch (const QueueClosedException&) {
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

  // 1. Planes tensor: (batch_size, 112, 64)
  auto planes_tensor = std::make_unique<TypedTensor<float>>(
      std::initializer_list<size_t>{batch_size, kNumPlanes, 64});
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
        plane_slice[square] = static_cast<float>((plane_bits >> square) & 1);
      }
    }

    // Add 8 additional planes for metadata (planes 104-111).
    const std::pair<ssize_t, float> meta_planes[] = {
        {104, static_cast<float>(frame.castling_us_ooo)},
        {105, static_cast<float>(frame.castling_us_oo)},
        {106, static_cast<float>(frame.castling_them_ooo)},
        {107, static_cast<float>(frame.castling_them_oo)},
        {108, static_cast<float>(frame.side_to_move_or_enpassant)},
        {109, static_cast<float>(frame.rule50_count) / 100.0f},
        {110, 1.0f},  // All ones (constant plane).
        {111, 0.0f},  // All zeros (constant plane).
    };

    for (const auto& [plane_num, value] : meta_planes) {
      auto plane_slice = batch_slice.subspan(plane_num * 64, 64);
      absl::c_fill(plane_slice, value);
    }
  }
}

}  // namespace training
}  // namespace lczero