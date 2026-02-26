#include "loader/stages/chunk_source_loader.h"

#include <filesystem>
#include <utility>

#include "absl/log/log.h"
#include "loader/chunk_source/rawfile_chunk_source.h"
#include "loader/chunk_source/tar_chunk_source.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath,
    ChunkSourceLoaderConfig::FrameFormat frame_format) {
  auto extension = filepath.extension();
  try {
    if (extension == ".gz") {
      return std::make_unique<RawFileChunkSource>(filepath, frame_format);
    }
    if (extension == ".tar") {
      return std::make_unique<TarChunkSource>(filepath, frame_format);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to create chunk source for " << filepath << ": "
               << e.what();
    return nullptr;
  }
  return nullptr;
}

ChunkSourceLoader::ChunkSourceLoader(const ChunkSourceLoaderConfig& config)
    : SingleInputStage<ChunkSourceLoaderConfig, InputType>(config),
      SingleOutputStage<OutputType>(config.output()),
      thread_pool_(config.threads(), ThreadPoolOptions{}),
      frame_format_(config.frame_format()) {
  LOG(INFO) << "Initializing ChunkSourceLoader with " << config.threads()
            << " worker threads";

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkSourceLoader::~ChunkSourceLoader() { Stop(); }

void ChunkSourceLoader::Start() {
  LOG(INFO) << "Starting ChunkSourceLoader worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i](std::stop_token stop_token) {
      Worker(stop_token, thread_contexts_[i].get());
    });
  }
}

void ChunkSourceLoader::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;

  LOG(INFO) << "Stopping ChunkSourceLoader.";
  thread_pool_.Shutdown();
  output_queue()->Close();
  LOG(INFO) << "ChunkSourceLoader stopped.";
}

void ChunkSourceLoader::Worker(std::stop_token stop_token,
                               ThreadContext* context) {
  auto producer = output_queue()->CreateProducer();
  LOG(INFO) << "ChunkSourceLoader worker@" << static_cast<const void*>(context)
            << " started.";

  try {
    while (true) {
      auto file = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get(stop_token);
      }();

      if (file.message_type ==
          FilePathProvider::MessageType::kInitialScanComplete) {
        LOG(INFO)
            << "ChunkSourceLoader received initial scan completion marker.";

        bool should_forward;
        {
          absl::MutexLock lock(&phase_mutex_);
          sentinel_received_ = true;
          should_forward = (pre_sentinel_work_count_ == 0);
        }

        if (should_forward) {
          LOG(INFO) << "ChunkSourceLoader forwarding initial scan completion "
                       "marker.";
          producer.Put({.source = nullptr, .message_type = file.message_type},
                       stop_token);
        }
        continue;
      }

      // Track pre-sentinel work.
      bool is_pre_sentinel;
      {
        absl::MutexLock lock(&phase_mutex_);
        is_pre_sentinel = !sentinel_received_;
        if (is_pre_sentinel) pre_sentinel_work_count_++;
      }

      // Create ChunkSource from the file.
      LOG_EVERY_N(INFO, 1000)
          << "ChunkSourceLoader preparing chunk source for " << file.filepath;
      auto source = CreateChunkSourceFromFile(file.filepath, frame_format_);
      if (source) {
        {
          absl::MutexLock lock(&last_chunk_key_mutex_);
          last_chunk_key_ = source->GetChunkSortKey();
        }
        ChunkSourceWithPhase output{.source = std::move(source),
                                    .message_type = file.message_type};
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(output), stop_token);
      } else {
        LOG_EVERY_N(INFO, 100)
            << "ChunkSourceLoader skipping unsupported file: " << file.filepath;
        skipped_files_count_++;
      }

      // Complete pre-sentinel work tracking.
      if (is_pre_sentinel) {
        absl::MutexLock lock(&phase_mutex_);
        if (--pre_sentinel_work_count_ == 0 && sentinel_received_) {
          LOG(INFO) << "ChunkSourceLoader forwarding initial scan completion "
                       "marker after all pre-sentinel work completed.";
          producer.Put(
              {.source = nullptr,
               .message_type =
                   FilePathProvider::MessageType::kInitialScanComplete},
              stop_token);
        }
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ChunkSourceLoader worker@"
              << static_cast<const void*>(context)
              << " stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "ChunkSourceLoader worker@"
              << static_cast<const void*>(context)
              << " stopping, request cancelled.";
  } catch (const std::exception& e) {
    LOG(ERROR) << "ChunkSourceLoader worker@"
               << static_cast<const void*>(context)
               << " exiting due to exception: " << e.what();
    throw;
  }

  LOG(INFO) << "ChunkSourceLoader worker@" << static_cast<const void*>(context)
            << " exiting loop.";
}

StageMetricProto ChunkSourceLoader::FlushMetrics() {
  StageMetricProto stage_metric;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);

  auto* skipped_metric = stage_metric.add_count_metrics();
  skipped_metric->set_name("skipped_files");
  skipped_metric->set_count(skipped_files_count_.exchange(0));

  // Get the last chunk key.
  {
    absl::MutexLock lock(&last_chunk_key_mutex_);
    if (!last_chunk_key_.empty()) {
      stage_metric.set_last_chunk_key(last_chunk_key_);
    }
  }

  *stage_metric.add_queue_metrics() =
      MetricsFromQueue("output", *output_queue());
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
