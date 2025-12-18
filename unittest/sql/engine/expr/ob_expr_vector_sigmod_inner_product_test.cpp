/**
 * Copyright (c) 2025 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#include <gtest/gtest.h>
#include "share/vector_type/ob_vector_sigmod_inner_product.h"
#include "common/ob_target_specific.h"
#include <cmath>

using namespace oceanbase;
using namespace oceanbase::common;

class TestVectorSigmodInnerProduct : public ::testing::Test
{
public:
  TestVectorSigmodInnerProduct() {}
  ~TestVectorSigmodInnerProduct() {}

private:
  // disallow copy
  DISALLOW_COPY_AND_ASSIGN(TestVectorSigmodInnerProduct);

protected:
  // Helper function to calculate expected sigmod inner product score
  static double calculate_expected_score(const float *a, const float *b, const float *w, int64_t len)
  {
    double score = 0.0;
    for (int64_t i = 0; i < len; ++i) {
      const double product = static_cast<double>(a[i]) * static_cast<double>(b[i]);
      const double sigmoid = 1.0 / (1.0 + std::exp(-product));
      score += sigmoid * static_cast<double>(w[i]);
    }
    return score;
  }
};

TEST_F(TestVectorSigmodInnerProduct, basic_test)
{
  // Test case 1: Simple 3-dimensional vectors
  float vec_a[3] = {1.0f, 2.0f, 3.0f};
  float vec_b[3] = {4.0f, 5.0f, 6.0f};
  float vec_w[3] = {0.5f, 0.5f, 0.5f};

  double expected_score = calculate_expected_score(vec_a, vec_b, vec_w, 3);
  double actual_score = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, 3);

  // Allow small floating point differences
  EXPECT_NEAR(expected_score, actual_score, 1e-6);
}

TEST_F(TestVectorSigmodInnerProduct, zero_vectors_test)
{
  // Test case 2: Zero vectors
  float vec_a[3] = {0.0f, 0.0f, 0.0f};
  float vec_b[3] = {0.0f, 0.0f, 0.0f};
  float vec_w[3] = {1.0f, 1.0f, 1.0f};

  double expected_score = calculate_expected_score(vec_a, vec_b, vec_w, 3);
  double actual_score = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, 3);

  // When both vectors are zero, sigmoid(0) = 0.5, so score = 0.5 * 1.0 * 3 = 1.5
  EXPECT_NEAR(1.5, actual_score, 1e-6);
  EXPECT_NEAR(expected_score, actual_score, 1e-6);
}

TEST_F(TestVectorSigmodInnerProduct, negative_values_test)
{
  // Test case 3: Negative values
  float vec_a[3] = {-1.0f, -2.0f, -3.0f};
  float vec_b[3] = {1.0f, 2.0f, 3.0f};
  float vec_w[3] = {1.0f, 1.0f, 1.0f};

  double expected_score = calculate_expected_score(vec_a, vec_b, vec_w, 3);
  double actual_score = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, 3);

  EXPECT_NEAR(expected_score, actual_score, 1e-6);
}

TEST_F(TestVectorSigmodInnerProduct, large_values_test)
{
  // Test case 4: Large values
  float vec_a[3] = {10.0f, 20.0f, 30.0f};
  float vec_b[3] = {1.0f, 2.0f, 3.0f};
  float vec_w[3] = {0.1f, 0.2f, 0.3f};

  double expected_score = calculate_expected_score(vec_a, vec_b, vec_w, 3);
  double actual_score = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, 3);

  EXPECT_NEAR(expected_score, actual_score, 1e-5);
}

TEST_F(TestVectorSigmodInnerProduct, different_dimensions_test)
{
  // Test case 5: Different dimensions
  const int64_t dims[] = {1, 8, 16, 32, 64, 128};
  ObArenaAllocator allocator(ObModIds::TEST);

  for (int64_t i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i) {
    int64_t dim = dims[i];
    float *vec_a = static_cast<float*>(allocator.alloc(dim * sizeof(float)));
    float *vec_b = static_cast<float*>(allocator.alloc(dim * sizeof(float)));
    float *vec_w = static_cast<float*>(allocator.alloc(dim * sizeof(float)));

    ASSERT_TRUE(vec_a != nullptr && vec_b != nullptr && vec_w != nullptr);

    // Initialize with simple values
    for (int64_t j = 0; j < dim; ++j) {
      vec_a[j] = 1.0f;
      vec_b[j] = 1.0f;
      vec_w[j] = 1.0f / dim;  // Normalize weights
    }

    double expected_score = calculate_expected_score(vec_a, vec_b, vec_w, dim);
    double actual_score = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, dim);

    EXPECT_NEAR(expected_score, actual_score, 1e-5) << "Failed for dimension " << dim;
  }
}

TEST_F(TestVectorSigmodInnerProduct, edge_cases_test)
{
  // Test case 6: Edge cases
  // Very small values
  float vec_a_small[3] = {1e-6f, 1e-7f, 1e-8f};
  float vec_b_small[3] = {1e-6f, 1e-7f, 1e-8f};
  float vec_w_small[3] = {1.0f, 1.0f, 1.0f};

  double expected_small = calculate_expected_score(vec_a_small, vec_b_small, vec_w_small, 3);
  double actual_small = specific::normal::sigmod_inner_product_score(vec_a_small, vec_b_small, vec_w_small, 3);
  EXPECT_NEAR(expected_small, actual_small, 1e-6);

  // Very large values (but not overflow)
  float vec_a_large[3] = {100.0f, 200.0f, 300.0f};
  float vec_b_large[3] = {1.0f, 1.0f, 1.0f};
  float vec_w_large[3] = {0.01f, 0.01f, 0.01f};

  double expected_large = calculate_expected_score(vec_a_large, vec_b_large, vec_w_large, 3);
  double actual_large = specific::normal::sigmod_inner_product_score(vec_a_large, vec_b_large, vec_w_large, 3);
  EXPECT_NEAR(expected_large, actual_large, 1e-4);
}

TEST_F(TestVectorSigmodInnerProduct, null_pointer_test)
{
  // Test case 7: Null pointer handling
  float vec_a[3] = {1.0f, 2.0f, 3.0f};

  // Test with null pointers
  double score_null = specific::normal::sigmod_inner_product_score(nullptr, vec_a, vec_a, 3);
  EXPECT_EQ(0.0, score_null);

  score_null = specific::normal::sigmod_inner_product_score(vec_a, nullptr, vec_a, 3);
  EXPECT_EQ(0.0, score_null);

  score_null = specific::normal::sigmod_inner_product_score(vec_a, vec_a, nullptr, 3);
  EXPECT_EQ(0.0, score_null);
}

TEST_F(TestVectorSigmodInnerProduct, zero_length_test)
{
  // Test case 8: Zero length
  float vec_a[3] = {1.0f, 2.0f, 3.0f};
  float vec_b[3] = {4.0f, 5.0f, 6.0f};
  float vec_w[3] = {0.5f, 0.5f, 0.5f};

  double score_zero = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, 0);
  EXPECT_EQ(0.0, score_zero);

  score_zero = specific::normal::sigmod_inner_product_score(vec_a, vec_b, vec_w, -1);
  EXPECT_EQ(0.0, score_zero);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  init_arches();
  return RUN_ALL_TESTS();
}

