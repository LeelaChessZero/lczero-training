// ABOUTME: PyBind11 binding module exposing C++ DataLoader to Python.
// ABOUTME: Handles configuration conversion and tensor memory management for
// numpy arrays.

#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "loader/data_loader.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/chunk_unpacker.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/shuffling_chunk_pool.h"
#include "loader/stages/shuffling_frame_sampler.h"
#include "loader/stages/tensor_generator.h"
#include "utils/tensor.h"

namespace py = pybind11;

namespace lczero {
namespace training {

// Helper function to convert TensorBase to numpy array using buffer protocol.
py::array tensor_to_numpy(std::unique_ptr<TensorBase> tensor) {
  // Extract raw pointer and release ownership from unique_ptr.
  TensorBase* raw_tensor = tensor.release();

  // Create numpy array with take_ownership policy.
  // This transfers memory ownership to Python/numpy.
  return py::array(
      py::dtype(raw_tensor->py_format()), raw_tensor->shape(),
      raw_tensor->strides(), raw_tensor->data(),
      py::cast(raw_tensor, py::return_value_policy::take_ownership));
}

// Convert TensorTuple to tuple of numpy arrays.
py::tuple tensor_tuple_to_numpy_tuple(TensorTuple tensor_tuple) {
  py::tuple result(tensor_tuple.size());
  for (size_t i = 0; i < tensor_tuple.size(); ++i) {
    result[i] = tensor_to_numpy(std::move(tensor_tuple[i]));
  }
  return result;
}

PYBIND11_MODULE(_lczero_training, m) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  m.doc() = "Leela Chess Zero training data loader";

  // Configuration is now handled via protobuf serialized strings

  // Expose the main DataLoader class.
  py::class_<DataLoader>(m, "DataLoader")
      .def(py::init([](py::bytes config) {
             std::string config_string = config;
             return new DataLoader(config_string);
           }),
           py::arg("config"),
           "Create DataLoader with serialized protobuf configuration bytes")
      .def(
          "get_next",
          [](DataLoader& self) {
            return tensor_tuple_to_numpy_tuple([&] {
              py::gil_scoped_release release;
              return self.GetNext();
            }());
          },
          "Get next batch of tensors as tuple of numpy arrays")
      .def(
          "get_bucket_metrics",
          [](const DataLoader& self, int time_period, bool include_pending) {
            auto [metrics, duration] = [&] {
              py::gil_scoped_release release;
              return self.GetBucketMetrics(time_period, include_pending);
            }();
            return py::make_tuple(py::bytes(metrics), duration);
          },
          "Get serialized metrics for bucket and duration as (bytes, float)")
      .def(
          "get_aggregate_ending_now",
          [](const DataLoader& self, float duration_seconds,
             bool include_pending) {
            auto [metrics, duration] = [&] {
              py::gil_scoped_release release;
              return self.GetAggregateEndingNow(duration_seconds,
                                                include_pending);
            }();
            return py::make_tuple(py::bytes(metrics), duration);
          },
          "Get serialized metrics for aggregate duration and actual duration "
          "as (bytes, float)")
      .def("start", &DataLoader::Start, "Start the data loader processing")
      .def("stop", &DataLoader::Stop, py::arg("graceful_drain") = false,
           "Stop the data loader")
      .def("reset_chunk_anchor", &DataLoader::ResetChunkAnchor,
           "Reset chunk anchor to current position and return anchor key")
      .def("chunks_since_anchor", &DataLoader::ChunksSinceAnchor,
           "Get number of chunks processed since anchor")
      .def("current_chunk_anchor", &DataLoader::CurrentChunkAnchor,
           "Get current chunk anchor key")
      .def("set_chunk_anchor", &DataLoader::SetChunkAnchor, py::arg("anchor"),
           "Set chunk anchor to specific key");

  // Expose TensorBase for potential advanced usage.
  py::class_<TensorBase>(m, "TensorBase")
      .def("shape", &TensorBase::shape, py::return_value_policy::reference)
      .def("strides", &TensorBase::strides, py::return_value_policy::reference)
      .def("element_size", &TensorBase::element_size)
      .def("py_format", &TensorBase::py_format);
}

}  // namespace training
}  // namespace lczero