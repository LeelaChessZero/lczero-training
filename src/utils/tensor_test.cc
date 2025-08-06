// ABOUTME: Unit tests for tensor classes and their data access methods.
// ABOUTME: Tests construction, element access, slicing, and error conditions.

#include "src/utils/tensor.h"

#include <gtest/gtest.h>

namespace lczero {
namespace {

TEST(TypedTensorTest, ConstructorAndBasicProperties) {
  TypedTensor<float> tensor({2, 3, 4});

  // Check shape
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.shape()[0], 2);
  EXPECT_EQ(tensor.shape()[1], 3);
  EXPECT_EQ(tensor.shape()[2], 4);

  // Check strides (in bytes, row-major order)
  EXPECT_EQ(tensor.strides().size(), 3);
  EXPECT_EQ(tensor.strides()[0], 12 * sizeof(float));  // 3 * 4 elements
  EXPECT_EQ(tensor.strides()[1], 4 * sizeof(float));   // 4 elements
  EXPECT_EQ(tensor.strides()[2], 1 * sizeof(float));   // 1 element

  // Check element size
  EXPECT_EQ(tensor.element_size(), sizeof(float));

  // Check py_format
  EXPECT_EQ(tensor.py_format(), "f");

  // Check data pointer is valid
  EXPECT_NE(tensor.data(), nullptr);
}

TEST(TypedTensorTest, PyFormatForDifferentTypes) {
  TypedTensor<float> float_tensor({2});
  EXPECT_EQ(float_tensor.py_format(), "f");

  TypedTensor<double> double_tensor({2});
  EXPECT_EQ(double_tensor.py_format(), "d");

  TypedTensor<int32_t> int32_tensor({2});
  EXPECT_EQ(int32_tensor.py_format(), "i");

  TypedTensor<int64_t> int64_tensor({2});
  EXPECT_EQ(int64_tensor.py_format(), "q");
}

TEST(TypedTensorTest, ElementAccess) {
  TypedTensor<int> tensor({2, 3});

  // Set some values
  tensor[{0, 0}] = 10;
  tensor[{0, 1}] = 11;
  tensor[{0, 2}] = 12;
  tensor[{1, 0}] = 20;
  tensor[{1, 1}] = 21;
  tensor[{1, 2}] = 22;

  // Check values
  EXPECT_EQ((tensor[{0, 0}]), 10);
  EXPECT_EQ((tensor[{0, 1}]), 11);
  EXPECT_EQ((tensor[{0, 2}]), 12);
  EXPECT_EQ((tensor[{1, 0}]), 20);
  EXPECT_EQ((tensor[{1, 1}]), 21);
  EXPECT_EQ((tensor[{1, 2}]), 22);
}

TEST(TypedTensorTest, ConstElementAccess) {
  TypedTensor<int> tensor({2, 2});
  tensor[{0, 0}] = 1;
  tensor[{0, 1}] = 2;
  tensor[{1, 0}] = 3;
  tensor[{1, 1}] = 4;

  const auto& const_tensor = tensor;
  EXPECT_EQ((const_tensor[{0, 0}]), 1);
  EXPECT_EQ((const_tensor[{0, 1}]), 2);
  EXPECT_EQ((const_tensor[{1, 0}]), 3);
  EXPECT_EQ((const_tensor[{1, 1}]), 4);
}

TEST(TypedTensorTest, SliceAccess) {
  TypedTensor<int> tensor({2, 3, 4});

  // Fill with test data
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        tensor[{i, j, k}] = i * 100 + j * 10 + k;
      }
    }
  }

  // Test 1D slice (fix first dimension)
  auto slice1d = tensor.slice({1});
  EXPECT_EQ(slice1d.size(), 12);  // 3 * 4 elements
  EXPECT_EQ(slice1d[0], 100);     // tensor[{1, 0, 0}]
  EXPECT_EQ(slice1d[4], 110);     // tensor[{1, 1, 0}]

  // Test 2D slice (fix first two dimensions)
  auto slice2d = tensor.slice({0, 1});
  EXPECT_EQ(slice2d.size(), 4);  // 4 elements
  EXPECT_EQ(slice2d[0], 10);     // tensor[{0, 1, 0}]
  EXPECT_EQ(slice2d[1], 11);     // tensor[{0, 1, 1}]
  EXPECT_EQ(slice2d[2], 12);     // tensor[{0, 1, 2}]
  EXPECT_EQ(slice2d[3], 13);     // tensor[{0, 1, 3}]

  // Test full tensor slice (no dimensions fixed)
  auto full_slice = tensor.slice({});
  EXPECT_EQ(full_slice.size(), 24);  // 2 * 3 * 4 elements
}

TEST(TypedTensorTest, ConstSliceAccess) {
  TypedTensor<int> tensor({2, 2});
  tensor[{0, 0}] = 1;
  tensor[{0, 1}] = 2;
  tensor[{1, 0}] = 3;
  tensor[{1, 1}] = 4;

  const auto& const_tensor = tensor;
  auto slice = const_tensor.slice({0});
  EXPECT_EQ(slice.size(), 2);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 2);
}

TEST(TypedTensorTest, ElementAccessWrongDimensions) {
  TypedTensor<int> tensor({2, 3});

  EXPECT_THROW((tensor[{0}]), std::invalid_argument);
  EXPECT_THROW((tensor[{0, 1, 2}]), std::invalid_argument);
}

TEST(TypedTensorTest, SliceAccessTooManyDimensions) {
  TypedTensor<int> tensor({2, 3});

  EXPECT_THROW((tensor.slice({0, 1, 2})), std::invalid_argument);
}

TEST(TypedTensorTest, OneDimensionalTensor) {
  TypedTensor<float> tensor({5});

  EXPECT_EQ(tensor.shape().size(), 1);
  EXPECT_EQ(tensor.shape()[0], 5);
  EXPECT_EQ(tensor.strides()[0], sizeof(float));

  tensor[{0}] = 1.0f;
  tensor[{4}] = 5.0f;

  EXPECT_EQ((tensor[{0}]), 1.0f);
  EXPECT_EQ((tensor[{4}]), 5.0f);

  auto slice = tensor.slice({});
  EXPECT_EQ(slice.size(), 5);
}

}  // namespace
}  // namespace lczero