// ABOUTME: PyBind11 binding module exposing C++ DataLoader to Python.
// ABOUTME: Handles configuration conversion and tensor memory management for
// numpy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/chunk_feed/chunk_unpacker.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "loader/chunk_feed/shuffling_chunk_pool.h"
#include "loader/data_loader.h"
#include "loader/shuffling_frame_sampler.h"
#include "loader/tensor_generator.h"
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
  m.doc() = "Leela Chess Zero training data loader";

  // Expose configuration structures.
  py::class_<FilePathProviderOptions>(m, "FilePathProviderOptions")
      .def(py::init<>())
      .def_readwrite("queue_capacity", &FilePathProviderOptions::queue_capacity)
      .def_readwrite("directory", &FilePathProviderOptions::directory);

  py::class_<ChunkSourceLoaderOptions>(m, "ChunkSourceLoaderOptions")
      .def(py::init<>())
      .def_readwrite("worker_threads",
                     &ChunkSourceLoaderOptions::worker_threads)
      .def_readwrite("output_queue_size",
                     &ChunkSourceLoaderOptions::output_queue_size);

  py::class_<ShufflingChunkPoolOptions>(m, "ShufflingChunkPoolOptions")
      .def(py::init<>())
      .def_readwrite("chunk_pool_size",
                     &ShufflingChunkPoolOptions::chunk_pool_size)
      .def_readwrite("num_startup_indexing_threads",
                     &ShufflingChunkPoolOptions::num_startup_indexing_threads)
      .def_readwrite("num_indexing_threads",
                     &ShufflingChunkPoolOptions::num_indexing_threads)
      .def_readwrite("num_chunk_loading_threads",
                     &ShufflingChunkPoolOptions::num_chunk_loading_threads)
      .def_readwrite("output_queue_size",
                     &ShufflingChunkPoolOptions::output_queue_size);

  py::class_<ChunkUnpackerOptions>(m, "ChunkUnpackerOptions")
      .def(py::init<>())
      .def_readwrite("worker_threads", &ChunkUnpackerOptions::worker_threads)
      .def_readwrite("output_queue_size",
                     &ChunkUnpackerOptions::output_queue_size);

  py::class_<ShufflingFrameSamplerOptions>(m, "ShufflingFrameSamplerOptions")
      .def(py::init<>())
      .def_readwrite("num_worker_threads",
                     &ShufflingFrameSamplerOptions::num_worker_threads)
      .def_readwrite("reservoir_size_per_thread",
                     &ShufflingFrameSamplerOptions::reservoir_size_per_thread)
      .def_readwrite("output_queue_size",
                     &ShufflingFrameSamplerOptions::output_queue_size);

  py::class_<TensorGeneratorOptions>(m, "TensorGeneratorOptions")
      .def(py::init<>())
      .def_readwrite("worker_threads", &TensorGeneratorOptions::worker_threads)
      .def_readwrite("batch_size", &TensorGeneratorOptions::batch_size)
      .def_readwrite("output_queue_size",
                     &TensorGeneratorOptions::output_queue_size);

  py::class_<DataLoaderConfig>(m, "DataLoaderConfig")
      .def(py::init<>())
      .def_readwrite("file_path_provider",
                     &DataLoaderConfig::file_path_provider)
      .def_readwrite("chunk_source_loader",
                     &DataLoaderConfig::chunk_source_loader)
      .def_readwrite("shuffling_chunk_pool",
                     &DataLoaderConfig::shuffling_chunk_pool)
      .def_readwrite("chunk_unpacker", &DataLoaderConfig::chunk_unpacker)
      .def_readwrite("shuffling_frame_sampler",
                     &DataLoaderConfig::shuffling_frame_sampler)
      .def_readwrite("tensor_generator", &DataLoaderConfig::tensor_generator);

  // Expose the main DataLoader class.
  py::class_<DataLoader>(m, "DataLoader")
      .def(py::init<const DataLoaderConfig&>(),
           "Create DataLoader with the given configuration")
      .def(
          "get_next",
          [](DataLoader& self) {
            TensorTuple tensors = self.GetNext();
            return tensor_tuple_to_numpy_tuple(std::move(tensors));
          },
          "Get next batch of tensors as tuple of numpy arrays");

  // Expose TensorBase for potential advanced usage.
  py::class_<TensorBase>(m, "TensorBase")
      .def("shape", &TensorBase::shape, py::return_value_policy::reference)
      .def("strides", &TensorBase::strides, py::return_value_policy::reference)
      .def("element_size", &TensorBase::element_size)
      .def("py_format", &TensorBase::py_format);
}

}  // namespace training
}  // namespace lczero