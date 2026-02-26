#include "loader/stages/shuffling_chunk_pool.h"

#include <absl/algorithm/container.h>
#include <absl/base/thread_annotations.h>
#include <absl/log/log.h>
#include <absl/random/random.h>
#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "loader/chunk_source/chunk_source.h"
#include "loader/data_loader_metrics.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/position_sampling.h"
#include "proto/data_loader_config.pb.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

thread_local absl::BitGen ShufflingChunkPool::bitgen_{absl::MakeSeedSeq()};

ShufflingChunkPool::ShufflingChunkPool(const ShufflingChunkPoolConfig& config)
    : primary_output_name_(config.output().name()),
      primary_output_queue_(
          config.output().queue_capacity(),
          ToOverflowBehavior(config.output().overflow_behavior())),
      chunk_pool_size_(config.chunk_pool_size()),
      config_(config),
      source_ingestion_pool_(config.source_ingestion_threads(),
                             ThreadPoolOptions{}, stop_source_),
      chunk_loading_pool_(config.chunk_loading_threads(), ThreadPoolOptions{},
                          stop_source_),
      caching_pool_(config.has_cachehit_output() ? config.caching_threads() : 0,
                    ThreadPoolOptions{}, stop_source_) {
  if (config.has_cachehit_output()) {
    cachehit_output_name_ = config.cachehit_output().name();
    cachehit_output_queue_.emplace(
        config.cachehit_output().queue_capacity(),
        ToOverflowBehavior(config.cachehit_output().overflow_behavior()));
    if (primary_output_name_ == *cachehit_output_name_) {
      throw std::runtime_error(absl::StrCat(
          "ShufflingChunkPool output names must be different, got: '",
          primary_output_name_, "'"));
    }
  }
  LOG(INFO) << "Initializing ShufflingChunkPool with pool size "
            << config.chunk_pool_size();
}

ShufflingChunkPool::~ShufflingChunkPool() { Stop(); }

void ShufflingChunkPool::SetInputs(absl::Span<QueueBase* const> inputs) {
  if (inputs.size() != 1 && inputs.size() != 2) {
    throw std::runtime_error(absl::StrCat(
        "ShufflingChunkPool expects 1 or 2 inputs, got ", inputs.size()));
  }
  if (inputs.size() == 2 && !cachehit_output_queue_.has_value()) {
    throw std::runtime_error(
        "ShufflingChunkPool received 2 inputs but cachehit_output is not "
        "configured");
  }
  if (inputs.size() == 1 && cachehit_output_queue_.has_value()) {
    throw std::runtime_error(
        "ShufflingChunkPool has cachehit_output configured but received only "
        "1 input");
  }
  primary_input_queue_ = dynamic_cast<Queue<ChunkSourceWithPhase>*>(inputs[0]);
  if (!primary_input_queue_) {
    throw std::runtime_error("ShufflingChunkPool primary input type mismatch");
  }
  if (inputs.size() == 2) {
    cache_request_queue_ = dynamic_cast<Queue<CacheRequest>*>(inputs[1]);
    if (!cache_request_queue_) {
      throw std::runtime_error(
          "ShufflingChunkPool cache request input type mismatch");
    }
  }
}

QueueBase* ShufflingChunkPool::GetOutput(std::string_view name) {
  if (name == primary_output_name_) return &primary_output_queue_;
  if (cachehit_output_name_.has_value() && name == *cachehit_output_name_) {
    return &*cachehit_output_queue_;
  }
  std::string available = absl::StrCat("'", primary_output_name_, "'");
  if (cachehit_output_name_.has_value()) {
    absl::StrAppend(&available, ", '", *cachehit_output_name_, "'");
  }
  throw std::runtime_error(absl::StrCat("ShufflingChunkPool unknown output '",
                                        name,
                                        "'. Available outputs: ", available));
}

void ShufflingChunkPool::Start() {
  LOG(INFO) << "Starting ShufflingChunkPool initialization thread.";
  initialization_thread_ = std::jthread([this]() {
    try {
      LOG(INFO) << "Starting ShufflingChunkPool with pool size "
                << config_.chunk_pool_size();
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources =
          InitializeChunkSources();
      ProcessInputFiles(std::move(uninitialized_sources));

      // Start input processing worker that continuously processes new files.
      for (size_t i = 0; i < source_ingestion_pool_.num_threads(); ++i) {
        auto* context =
            source_ingestion_thread_contexts_
                .emplace_back(std::make_unique<SourceIngestionThreadContext>())
                .get();
        source_ingestion_pool_.Enqueue(
            [this, context](std::stop_token stop_token) {
              SourceIngestionWorker(stop_token, context);
            });
      }

      // Start output workers after everything is fully initialized.
      LOG(INFO) << "ShufflingChunkPool initialization done, starting workers";
      for (size_t i = 0; i < chunk_loading_pool_.num_threads(); ++i) {
        auto* context =
            chunk_loading_thread_contexts_
                .emplace_back(std::make_unique<ChunkLoadingThreadContext>())
                .get();
        chunk_loading_pool_.Enqueue(
            [this, context](std::stop_token stop_token) {
              OutputWorker(stop_token, context);
            });
      }

      // Start caching workers if configured.
      if (cachehit_output_queue_.has_value()) {
        for (size_t i = 0; i < caching_pool_.num_threads(); ++i) {
          auto* context =
              caching_thread_contexts_
                  .emplace_back(std::make_unique<CachingThreadContext>())
                  .get();
          caching_pool_.Enqueue([this, context](std::stop_token stop_token) {
            CachingWorker(stop_token, context);
          });
        }
      }
    } catch (const QueueClosedException&) {
      LOG(INFO) << "ShufflingChunkPool initialization interrupted, input "
                   "queue closed.";
      output_queue()->Close();
    } catch (const std::exception& e) {
      LOG(ERROR) << "ShufflingChunkPool initialization failed: " << e.what();
      output_queue()->Close();
    }
  });
}

void ShufflingChunkPool::Stop() {
  if (stop_source_.stop_requested()) return;

  LOG(INFO) << "Stopping ShufflingChunkPool.";
  stop_source_.request_stop();
  if (initialization_thread_.joinable()) {
    initialization_thread_.request_stop();
    initialization_thread_.join();
  }

  source_ingestion_pool_.Shutdown();
  chunk_loading_pool_.Shutdown();
  if (cachehit_output_queue_) caching_pool_.Shutdown();
  output_queue()->Close();
  if (cachehit_output_queue_) cachehit_output_queue_->Close();
  LOG(INFO) << "ShufflingChunkPool stopped.";
}

std::vector<std::unique_ptr<ChunkSource>>
ShufflingChunkPool::InitializeChunkSources() {
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources;

  // Read from input queue until kInitialScanComplete.
  while (true) {
    auto chunk_source_with_phase = input_queue()->Get();

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      LOG(INFO)
          << "ShufflingChunkPool received initial scan completion marker.";
      break;
    }

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kFile) {
      // Add ChunkSource to uninitialized sources.
      uninitialized_sources.push_back(
          std::move(chunk_source_with_phase.source));
    }
  }

  LOG(INFO) << "ShufflingChunkPool initial directory walk produced "
            << uninitialized_sources.size() << " chunk source candidate(s).";

  // Sort in descending order (newest first).
  std::sort(uninitialized_sources.begin(), uninitialized_sources.end(),
            [](const auto& a, const auto& b) {
              return a->GetChunkSortKey() > b->GetChunkSortKey();
            });
  std::atomic<size_t> total_chunks = 0;
  size_t sources_to_keep = 0;

  // Process sources sequentially until we have enough chunks.
  std::string current_anchor;
  {
    absl::MutexLock lock(&anchor_mutex_);
    current_anchor = anchor_;
  }

  for (auto& source : uninitialized_sources) {
    if (output_queue()->IsClosed()) {
      LOG(INFO) << "Output queue closed, stopping source ingestion.";
      break;
    }
    if (total_chunks >= chunk_pool_size_) break;

    // Count chunks immediately; constructors have already prepared metadata.
    const size_t chunk_count = source->GetChunkCount();
    total_chunks += chunk_count;

    // Count chunks since anchor during initial load.
    if (source->GetChunkSortKey() > current_anchor) {
      chunks_since_anchor_ += chunk_count;
    }

    LOG_EVERY_N_SEC(INFO, 4) << "Loaded so far: " << total_chunks.load()
                             << "; new: " << chunks_since_anchor_;
    ++sources_to_keep;
  }

  LOG(INFO) << "ShufflingChunkPool indexed " << total_chunks.load()
            << " chunk(s) across " << sources_to_keep
            << " source(s) during startup.";

  if (total_chunks < chunk_pool_size_ && !output_queue()->IsClosed()) {
    LOG(ERROR) << "ShufflingChunkPool startup chunk requirement not met: "
               << total_chunks.load() << " < " << chunk_pool_size_;
  }

  // Trim the vector to only keep the sources we need.
  uninitialized_sources.resize(sources_to_keep);
  return uninitialized_sources;
}

void ShufflingChunkPool::ProcessInputFiles(
    std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources) {
  // Initialize chunk sources from the initial scan.
  size_t initial_window_sources = 0;
  size_t initial_total_chunks = 0;
  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    size_t start_chunk_index = 0;
    // Newest sources first, so we add in reverse order.
    std::for_each(uninitialized_sources.rbegin(), uninitialized_sources.rend(),
                  [this, &start_chunk_index](auto& source) {
                    const size_t count = source->GetChunkCount();
                    chunk_sources_.push_back(
                        {.start_chunk_index = start_chunk_index,
                         .source = std::move(source),
                         .dropped_chunks = {},
                         .use_counts = std::vector<uint16_t>(count, 0),
                         .weight = std::vector<float>(count, -1.0f),
                         .cache = std::vector<std::unique_ptr<CacheNode>>(
                             cachehit_output_queue_.has_value() ? count : 0)});
                    start_chunk_index +=
                        chunk_sources_.back().source->GetChunkCount();
                  });

    // Initialize stream shuffler with the initial bounds.
    if (!chunk_sources_.empty()) {
      size_t total_chunks = chunk_sources_.back().start_chunk_index +
                            chunk_sources_.back().source->GetChunkCount();
      // Set bounds to provide the last chunk_pool_size_ chunks.
      size_t lower_bound =
          total_chunks > chunk_pool_size_ ? total_chunks - chunk_pool_size_ : 0;
      stream_shuffler_.SetLowerBound(lower_bound);
      stream_shuffler_.SetUpperBound(total_chunks);
      initial_total_chunks = total_chunks;
    }
    initial_window_sources = chunk_sources_.size();
  }

  LOG(INFO) << "ShufflingChunkPool initial window ready with "
            << initial_window_sources << " source(s) totaling "
            << initial_total_chunks << " chunk(s).";

  // Log anchor and sources after initial scan completion.
  {
    absl::MutexLock anchor_lock(&anchor_mutex_);
    LOG(INFO) << "Current anchor: '" << anchor_ << "'";

    absl::MutexLock sources_lock(&chunk_sources_mutex_);
    std::vector<const ChunkSourceItem*> sources_after_anchor;
    for (const auto& item : chunk_sources_) {
      if (item.source->GetChunkSortKey() > anchor_) {
        sources_after_anchor.push_back(&item);
      }
    }

    LOG(INFO) << sources_after_anchor.size()
              << " chunk source(s) after anchor, " << chunks_since_anchor_
              << " total chunks since anchor";

    const size_t to_log = std::min(sources_after_anchor.size(), size_t(20));
    for (size_t i = 0; i < to_log; ++i) {
      LOG(INFO) << "  Source [" << (i + 1) << "/" << sources_after_anchor.size()
                << "]: key='"
                << sources_after_anchor[i]->source->GetChunkSortKey()
                << "', chunks="
                << sources_after_anchor[i]->source->GetChunkCount();
    }
  }

  if (initial_total_chunks == 0) {
    throw std::runtime_error(
        "ShufflingChunkPool requires at least one chunk during startup.");
  }
}

void ShufflingChunkPool::SourceIngestionWorker(
    std::stop_token stop_token, SourceIngestionThreadContext* context) {
  try {
    while (true) {
      auto chunk_source_with_phase = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get(stop_token);
      }();

      if (chunk_source_with_phase.message_type ==
          FilePathProvider::MessageType::kFile) {
        // Ingest the new chunk source.
        auto source = std::move(chunk_source_with_phase.source);
        size_t chunk_count = source->GetChunkCount();
        absl::MutexLock lock(&chunk_sources_mutex_);
        chunks_since_anchor_ += chunk_count;
        AddNewChunkSource(std::move(source));
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "SourceIngestionWorker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "SourceIngestionWorker stopping, request cancelled.";
  }
}

void ShufflingChunkPool::OutputWorker(std::stop_token stop_token,
                                      ChunkLoadingThreadContext* context) {
  // Create a local producer for this worker
  auto primary_producer = output_queue()->CreateProducer();
  std::optional<decltype(cachehit_output_queue_->CreateProducer())>
      cachehit_producer;
  if (cachehit_output_queue_.has_value()) {
    cachehit_producer.emplace(cachehit_output_queue_->CreateProducer());
  }

  try {
    while (true) {
      auto result = GetNextChunkData();
      if (!result) {
        if (output_queue()->IsClosed()) break;
        continue;
      }
      LoadMetricPauser pauser(context->load_metric_updater);
      if (std::holds_alternative<TrainingChunk>(*result)) {
        primary_producer.Put(std::move(std::get<TrainingChunk>(*result)),
                             stop_token);
      } else {
        cachehit_producer->Put(std::move(std::get<FrameType>(*result)),
                               stop_token);
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "OutputWorker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "OutputWorker stopping, request cancelled.";
  } catch (const std::exception& e) {
    LOG(FATAL) << "OutputWorker encountered an error: " << e.what();
  }
}

void ShufflingChunkPool::CachingWorker(std::stop_token stop_token,
                                       CachingThreadContext* context) {
  constexpr double kTheta = 0.99;
  double reminder = 0.0;
  double exponential_avg_probability = 1.0;
  try {
    while (true) {
      auto cache_request = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return cache_request_queue_->Get(stop_token);
      }();

      absl::MutexLock lock(&chunk_sources_mutex_);

      // Find the chunk source containing this global index.
      auto it = absl::c_lower_bound(
          chunk_sources_, cache_request.global_index,
          [](const auto& source_item, size_t chunk_idx) {
            return source_item.start_chunk_index +
                       source_item.source->GetChunkCount() <=
                   chunk_idx;
          });

      if (it == chunk_sources_.end() ||
          cache_request.global_index < it->start_chunk_index) {
        chunk_source_not_found_.fetch_add(1, std::memory_order_acq_rel);
        continue;
      }

      const size_t local_index =
          cache_request.global_index - it->start_chunk_index;
      assert(local_index < it->use_counts.size());

      // Check use_count match.
      if (it->use_counts[local_index] != cache_request.next_use) {
        mismatched_use_counts_.fetch_add(1, std::memory_order_acq_rel);
        continue;
      }

      // Compute how many positions to cache.
      const float weight = it->weight[local_index];
      assert(weight >= 0.0f);
      const double probability = ComputeHanseProbability(weight);
      exponential_avg_probability =
          exponential_avg_probability * (1.0 - kTheta) + probability * kTheta;
      const double n = (probability * config_.position_cache_size() /
                        chunk_pool_size_ / exponential_avg_probability) +
                       reminder;
      reminder = n - std::floor(n);
      const size_t positions_to_cache = static_cast<size_t>(std::floor(n));
      // Traverse and extend the cache chain.
      std::unique_ptr<CacheNode>* current = &it->cache[local_index];
      for (size_t i = 0; i < positions_to_cache; ++i) {
        if (*current) {
          current = &(*current)->next;
          continue;
        }
        if (i >= cache_request.items.size()) break;
        auto node = std::make_unique<CacheNode>();
        node->frame = cache_request.items[i];
        *current = std::move(node);
        current = &(*current)->next;
        newly_cached_.fetch_add(1, std::memory_order_acq_rel);
        cached_positions_.fetch_add(1, std::memory_order_acq_rel);
      }

      const size_t dropped =
          cache_request.items.size() > positions_to_cache
              ? cache_request.items.size() - positions_to_cache
              : 0;
      dropped_cache_positions_.fetch_add(dropped, std::memory_order_acq_rel);
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "CachingWorker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "CachingWorker stopping, request cancelled.";
  }
}

struct ShufflingChunkPool::ChunkData {
  std::vector<FrameType> data;
  std::string sort_key;
  size_t local_index = 0;
  size_t global_index = 0;
  uint32_t use_count = 0;
  ChunkSourceItem* source_item = nullptr;
};

std::optional<std::variant<TrainingChunk, FrameType>>
ShufflingChunkPool::GetNextChunkData() {
  while (true) {
    ChunkData chunk_data;
    ChunkStatus status;
    {
      absl::MutexLock lock(&chunk_sources_mutex_);
      status = GetChunkInfo(chunk_data);
      if (status == ChunkStatus::kEnd) return std::nullopt;
      if (status == ChunkStatus::kRetry) continue;

      const bool hanse_enabled = config_.hanse_sampling_threshold() > 0;
      if (hanse_enabled && !HanseAccept(chunk_data)) continue;

      // Increment use_count for this chunk.
      assert(chunk_data.source_item->use_counts.size() >
             chunk_data.local_index);
      chunk_data.use_count =
          chunk_data.source_item->use_counts[chunk_data.local_index]++;

      // Check cache if configured.
      if (cachehit_output_queue_.has_value()) {
        auto& cache_chain =
            chunk_data.source_item->cache[chunk_data.local_index];
        if (cache_chain) {
          cache_hits_.fetch_add(1, std::memory_order_acq_rel);
          cached_positions_.fetch_sub(1, std::memory_order_acq_rel);
          FrameType cached_frame = cache_chain->frame;
          cache_chain = std::move(cache_chain->next);
          return cached_frame;
        }
        cache_misses_.fetch_add(1, std::memory_order_acq_rel);
      }

      if (chunk_data.data.empty()) {
        if (!LoadChunkData(chunk_data)) continue;
      }
    }

    TrainingChunk chunk;
    chunk.sort_key = std::move(chunk_data.sort_key);
    chunk.index_within_sort_key = chunk_data.local_index;
    chunk.use_count = chunk_data.use_count;
    chunk.global_index = chunk_data.global_index;
    chunk.frames = std::move(chunk_data.data);

    return chunk;
  }
}

bool ShufflingChunkPool::LoadChunkData(ChunkData& chunk_data) {
  std::optional<std::vector<FrameType>> data =
      chunk_data.source_item->source->GetChunkData(chunk_data.local_index);

  if (!data || data->empty()) {
    chunk_data.source_item->dropped_chunks.insert(chunk_data.local_index);
    dropped_chunks_metric_.fetch_add(1, std::memory_order_acq_rel);
    return false;
  }

  chunk_data.data = std::move(*data);
  return true;
}

ShufflingChunkPool::ChunkStatus ShufflingChunkPool::GetChunkInfo(
    ChunkData& out_chunk_data) {
  std::optional<size_t> chunk_index = stream_shuffler_.GetNextItem();

  if (!chunk_index && !chunk_sources_.empty()) {
    size_t total_chunks = chunk_sources_.back().start_chunk_index +
                          chunk_sources_.back().source->GetChunkCount();
    size_t lower_bound = total_chunks > chunk_pool_size_
                             ? total_chunks - chunk_pool_size_
                             : chunk_sources_.front().start_chunk_index;
    stream_shuffler_.Reset(lower_bound, total_chunks);
    reshuffles_.fetch_add(1, std::memory_order_acq_rel);
    chunk_index = stream_shuffler_.GetNextItem();
  }

  if (!chunk_index) return ChunkStatus::kEnd;

  auto it =
      absl::c_lower_bound(chunk_sources_, *chunk_index,
                          [](const auto& source_item, size_t chunk_idx) {
                            return source_item.start_chunk_index +
                                       source_item.source->GetChunkCount() <=
                                   chunk_idx;
                          });

  if (ABSL_PREDICT_FALSE(it == chunk_sources_.end() ||
                         *chunk_index < it->start_chunk_index)) {
    LOG(WARNING) << "Chunk index " << *chunk_index
                 << " out of range for available chunk sources.";
    return ChunkStatus::kRetry;
  }

  out_chunk_data.local_index = *chunk_index - it->start_chunk_index;
  if (it->dropped_chunks.contains(out_chunk_data.local_index)) {
    return ChunkStatus::kRetry;
  }

  out_chunk_data.source_item = &(*it);
  out_chunk_data.sort_key = it->source->GetChunkSortKey();
  out_chunk_data.global_index = *chunk_index;

  return ChunkStatus::kOk;
}

double ShufflingChunkPool::ComputeHanseProbability(float weight) {
  if (max_weight_ <= 0.0f) return 1.0;
  return std::pow(weight / max_weight_, config_.hanse_sampling_gamma());
}

float ShufflingChunkPool::ComputeChunkWeight(
    absl::Span<const FrameType> frames) {
  return absl::c_accumulate(frames, 0.0f, [this](float sum, const auto& frame) {
    return sum +
           ComputePositionSamplingWeight(frame, config_.position_sampling());
  });
}

bool ShufflingChunkPool::HanseAccept(ChunkData& chunk_data) {
  assert(chunk_data.source_item);
  assert(chunk_data.source_item->weight.size() > chunk_data.local_index);

  if (chunk_data.source_item->weight[chunk_data.local_index] < 0.0f) {
    hanse_cache_misses_.fetch_add(1, std::memory_order_acq_rel);
    if (!LoadChunkData(chunk_data)) return false;
    const float weight = ComputeChunkWeight(chunk_data.data);
    chunk_data.source_item->weight[chunk_data.local_index] = weight;
    max_weight_ = std::max(max_weight_, weight);
    AddSample(chunk_weight_stats_, static_cast<double>(weight));
  } else {
    hanse_cache_hits_.fetch_add(1, std::memory_order_acq_rel);
  }

  const double p = ComputeHanseProbability(
      chunk_data.source_item->weight[chunk_data.local_index]);
  const double u = absl::Uniform<double>(bitgen_, 0.0, 1.0);
  if (u >= p) {
    hanse_rejected_.fetch_add(1, std::memory_order_acq_rel);
    return false;
  }
  return true;
}

void ShufflingChunkPool::AddNewChunkSource(std::unique_ptr<ChunkSource> source)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_) {
  // Add new chunk source to the end of the deque.
  size_t old_upper_bound = 0;
  if (!chunk_sources_.empty()) {
    const auto& last_source = chunk_sources_.back();
    old_upper_bound =
        last_source.start_chunk_index + last_source.source->GetChunkCount();
  }

  size_t count = source->GetChunkCount();
  chunk_sources_.push_back(
      {.start_chunk_index = old_upper_bound,
       .source = std::move(source),
       .dropped_chunks = {},
       .use_counts = std::vector<uint16_t>(count, 0),
       .weight = std::vector<float>(count, -1.0f),
       .cache = std::vector<std::unique_ptr<CacheNode>>(
           cachehit_output_queue_.has_value() ? count : 0)});

  // Calculate current window bounds.
  size_t new_upper_bound = chunk_sources_.back().start_chunk_index +
                           chunk_sources_.back().source->GetChunkCount();

  // Remove old chunks if window exceeds chunk_pool_size_.
  while (!chunk_sources_.empty() && chunk_sources_.size() > 1) {
    size_t window_start = chunk_sources_.front().start_chunk_index +
                          chunk_sources_.front().source->GetChunkCount();
    size_t window_size = new_upper_bound - window_start;

    if (window_size < chunk_pool_size_) break;

    // Count cached positions in the evicted source.
    if (cachehit_output_queue_.has_value()) {
      size_t evicted_cached = 0;
      for (const auto& cache_chain : chunk_sources_.front().cache) {
        const CacheNode* node = cache_chain.get();
        while (node) {
          ++evicted_cached;
          node = node->next.get();
        }
      }
      cached_positions_.fetch_sub(evicted_cached, std::memory_order_acq_rel);
    }

    // Remove the oldest chunk source (front of deque).
    chunk_sources_.pop_front();
  }

  // Update stream shuffler bounds with the sliding window.
  size_t window_start = chunk_sources_.front().start_chunk_index;
  size_t new_lower_bound = new_upper_bound > chunk_pool_size_
                               ? new_upper_bound - chunk_pool_size_
                               : window_start;
  stream_shuffler_.SetUpperBound(new_upper_bound);
  stream_shuffler_.SetLowerBound(new_lower_bound);
}

StageMetricProto ShufflingChunkPool::FlushMetrics() {
  StageMetricProto stage_metric;
  // Aggregate source ingestion load metrics from all ingestion threads.
  LoadMetricProto ingestion_load;
  ingestion_load.set_name("source_ingestion");
  for (const auto& context : source_ingestion_thread_contexts_) {
    UpdateFrom(ingestion_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(ingestion_load);

  // Aggregate chunk loading load metrics from all chunk loading threads.
  LoadMetricProto chunk_loading_load;
  chunk_loading_load.set_name("chunk_loading");
  for (const auto& context : chunk_loading_thread_contexts_) {
    UpdateFrom(chunk_loading_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(chunk_loading_load);

  // Get chunk sources statistics and pool state.
  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    auto* chunk_sources_metric = stage_metric.add_gauge_metrics();
    chunk_sources_metric->set_name("chunk_sources");
    chunk_sources_metric->set_value(
        static_cast<uint64_t>(chunk_sources_.size()));

    size_t upper = 0;
    size_t current = 0;
    if (!chunk_sources_.empty()) {
      const auto& first = chunk_sources_.front();
      const auto& last = chunk_sources_.back();
      upper = last.start_chunk_index + last.source->GetChunkCount();
      current = upper - first.start_chunk_index;
    }

    auto* current_chunks_metric = stage_metric.add_gauge_metrics();
    current_chunks_metric->set_name("chunks_current");
    current_chunks_metric->set_value(static_cast<uint64_t>(current));
    current_chunks_metric->set_capacity(
        static_cast<uint64_t>(chunk_pool_size_));

    auto* total_chunks_metric = stage_metric.add_gauge_metrics();
    total_chunks_metric->set_name("chunks_total");
    total_chunks_metric->set_value(static_cast<uint64_t>(upper));
  }

  // Get anchor-related metrics.
  {
    absl::MutexLock lock(&anchor_mutex_);
    auto* chunks_since_anchor_metric = stage_metric.add_gauge_metrics();
    chunks_since_anchor_metric->set_name("chunks_since_anchor");
    chunks_since_anchor_metric->set_value(chunks_since_anchor_);
    stage_metric.set_anchor(anchor_);
  }

  auto* dropped_metric = stage_metric.add_count_metrics();
  dropped_metric->set_name("dropped");
  dropped_metric->set_count(
      dropped_chunks_metric_.exchange(0, std::memory_order_acq_rel));

  // Hanse sampling and shuffler metrics.
  {
    auto* hits = stage_metric.add_count_metrics();
    hits->set_name("hanse_cache_hits");
    hits->set_count(hanse_cache_hits_.exchange(0, std::memory_order_acq_rel));

    auto* misses = stage_metric.add_count_metrics();
    misses->set_name("hanse_cache_misses");
    misses->set_count(
        hanse_cache_misses_.exchange(0, std::memory_order_acq_rel));

    auto* rejected = stage_metric.add_count_metrics();
    rejected->set_name("hanse_rejected");
    rejected->set_count(hanse_rejected_.exchange(0, std::memory_order_acq_rel));

    auto* resh = stage_metric.add_count_metrics();
    resh->set_name("reshuffles");
    resh->set_count(reshuffles_.exchange(0, std::memory_order_acq_rel));
  }

  // Position cache metrics.
  if (cachehit_output_queue_.has_value()) {
    LoadMetricProto caching_load;
    caching_load.set_name("caching");
    for (const auto& context : caching_thread_contexts_) {
      UpdateFrom(caching_load, context->load_metric_updater.FlushMetrics());
    }
    *stage_metric.add_load_metrics() = std::move(caching_load);

    auto* cache_hits = stage_metric.add_count_metrics();
    cache_hits->set_name("cache_hits");
    cache_hits->set_count(cache_hits_.exchange(0, std::memory_order_acq_rel));

    auto* cache_misses = stage_metric.add_count_metrics();
    cache_misses->set_name("cache_misses");
    cache_misses->set_count(
        cache_misses_.exchange(0, std::memory_order_acq_rel));

    auto* mismatched = stage_metric.add_count_metrics();
    mismatched->set_name("mismatched_use_counts");
    mismatched->set_count(
        mismatched_use_counts_.exchange(0, std::memory_order_acq_rel));

    auto* newly_cached = stage_metric.add_count_metrics();
    newly_cached->set_name("newly_cached");
    newly_cached->set_count(
        newly_cached_.exchange(0, std::memory_order_acq_rel));

    auto* dropped = stage_metric.add_count_metrics();
    dropped->set_name("dropped_cache_positions");
    dropped->set_count(
        dropped_cache_positions_.exchange(0, std::memory_order_acq_rel));

    auto* not_found = stage_metric.add_count_metrics();
    not_found->set_name("chunk_source_not_found");
    not_found->set_count(
        chunk_source_not_found_.exchange(0, std::memory_order_acq_rel));

    auto* cached = stage_metric.add_gauge_metrics();
    cached->set_name("cached_positions");
    cached->set_value(cached_positions_.load(std::memory_order_acquire));
    cached->set_capacity(config_.position_cache_size());
  }

  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    chunk_weight_stats_.set_name("chunk_weight");
    UpdateFrom(*stage_metric.add_statistics_metrics(), chunk_weight_stats_);
    chunk_weight_stats_.Clear();
  }

  *stage_metric.add_queue_metrics() =
      MetricsFromQueue(primary_output_name_, *output_queue());
  if (cachehit_output_queue_.has_value()) {
    *stage_metric.add_queue_metrics() =
        MetricsFromQueue(*cachehit_output_name_, *cachehit_output_queue_);
  }
  return stage_metric;
}

std::pair<std::string, int> ShufflingChunkPool::ResetAnchor() {
  absl::MutexLock anchor_lock(&anchor_mutex_);
  absl::MutexLock sources_lock(&chunk_sources_mutex_);

  if (chunk_sources_.empty()) {
    int previous_count = chunks_since_anchor_.exchange(0);
    return {anchor_, previous_count};
  }

  anchor_ = chunk_sources_.back().source->GetChunkSortKey();
  int previous_count = chunks_since_anchor_.exchange(0);
  return {anchor_, previous_count};
}

int ShufflingChunkPool::ChunksSinceAnchor() { return chunks_since_anchor_; }

std::string ShufflingChunkPool::CurrentAnchor() {
  absl::MutexLock lock(&anchor_mutex_);
  return anchor_;
}

void ShufflingChunkPool::SetAnchor(std::string_view anchor) {
  absl::MutexLock lock(&anchor_mutex_);
  anchor_ = anchor;
}

std::optional<StageControlResponse> ShufflingChunkPool::Control(
    const StageControlRequest& request) {
  if (!request.has_chunk_pool_request()) {
    return std::nullopt;
  }

  const auto& chunk_request = request.chunk_pool_request();
  StageControlResponse response;
  auto* chunk_response = response.mutable_chunk_pool_response();

  if (chunk_request.reset_chunk_anchor()) {
    auto [anchor, chunks] = ResetAnchor();
    chunk_response->set_chunk_anchor(anchor);
    chunk_response->set_chunks_since_anchor(chunks);
    return response;
  }

  if (chunk_request.has_set_chunk_anchor()) {
    SetAnchor(chunk_request.set_chunk_anchor());
    chunk_response->set_chunk_anchor(chunk_request.set_chunk_anchor());
    chunk_response->set_chunks_since_anchor(ChunksSinceAnchor());
    return response;
  }

  chunk_response->set_chunk_anchor(CurrentAnchor());
  chunk_response->set_chunks_since_anchor(ChunksSinceAnchor());
  return response;
}

}  // namespace training
}  // namespace lczero
