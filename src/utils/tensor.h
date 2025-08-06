#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/types/span.h"

namespace lczero {

// Class that holds tensor which will be exposed through pybind11.
class TensorBase {
 public:
  virtual ~TensorBase() = default;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual const std::vector<ssize_t>& shape() const = 0;
  virtual const std::vector<ssize_t>& strides() const = 0;
  virtual size_t element_size() const = 0;
  virtual std::string py_format() const = 0;
};

template <typename T>
class TypedTensor : public TensorBase {
 private:
  static size_t CalculateTotalSize(std::initializer_list<size_t> shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
      total_size *= dim;
    }
    return total_size;
  }

 public:
  TypedTensor(std::initializer_list<size_t> shape)
      : data_(CalculateTotalSize(shape)), shape_(shape.begin(), shape.end()) {
    // Calculate strides in row-major order (in bytes).
    strides_.resize(shape_.size());
    size_t total_size = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      strides_[i] = total_size * sizeof(T);
      total_size *= shape_[i];
    }
  }

  void* data() override { return data_.data(); }

  const void* data() const override { return data_.data(); }

  const std::vector<ssize_t>& shape() const override { return shape_; }

  const std::vector<ssize_t>& strides() const override { return strides_; }

  size_t element_size() const override { return sizeof(T); }

  std::string py_format() const override {
    if constexpr (std::is_same_v<T, float>) {
      return "f";
    } else if constexpr (std::is_same_v<T, double>) {
      return "d";
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return "i";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return "q";
    } else {
      static_assert(std::is_same_v<T, void>, "Unsupported tensor type");
    }
  }

  T& operator[](absl::Span<const ssize_t> dims) {
    if (dims.size() != shape_.size()) {
      throw std::invalid_argument(
          "Number of dimensions must match tensor rank");
    }
    size_t offset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      offset += dims[i] * strides_[i] / sizeof(T);
    }
    return data_[offset];
  }

  const T& operator[](absl::Span<const ssize_t> dims) const {
    if (dims.size() != shape_.size()) {
      throw std::invalid_argument(
          "Number of dimensions must match tensor rank");
    }
    size_t offset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      offset += dims[i] * strides_[i] / sizeof(T);
    }
    return data_[offset];
  }

  absl::Span<T> slice(absl::Span<const ssize_t> dims) {
    if (dims.size() > shape_.size()) {
      throw std::invalid_argument(
          "Number of dimensions cannot exceed tensor rank");
    }
    size_t offset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      offset += dims[i] * strides_[i] / sizeof(T);
    }
    size_t slice_size = 1;
    for (size_t i = dims.size(); i < shape_.size(); ++i) {
      slice_size *= shape_[i];
    }
    return absl::Span<T>(data_.data() + offset, slice_size);
  }

  absl::Span<const T> slice(absl::Span<const ssize_t> dims) const {
    if (dims.size() > shape_.size()) {
      throw std::invalid_argument(
          "Number of dimensions cannot exceed tensor rank");
    }
    size_t offset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      offset += dims[i] * strides_[i] / sizeof(T);
    }
    size_t slice_size = 1;
    for (size_t i = dims.size(); i < shape_.size(); ++i) {
      slice_size *= shape_[i];
    }
    return absl::Span<const T>(data_.data() + offset, slice_size);
  }

 private:
  absl::FixedArray<T> data_;
  std::vector<ssize_t> shape_;
  std::vector<ssize_t> strides_;
};

using TensorTuple = std::vector<std::unique_ptr<TensorBase>>;

}  // namespace lczero