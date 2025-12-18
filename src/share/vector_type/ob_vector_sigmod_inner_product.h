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

#ifndef OCEANBASE_LIB_OB_VECTOR_SIGMOD_INNER_PRODUCT_H_
#define OCEANBASE_LIB_OB_VECTOR_SIGMOD_INNER_PRODUCT_H_

#include "ob_vector_op_common.h"
#include <cmath>

namespace oceanbase
{
namespace common
{

OB_INLINE static double sigmoid_normal(const double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

OB_DECLARE_DEFAULT_CODE(
  inline static double sigmod_inner_product_score(const float *a, const float *b, const float *w, const int64_t len)
  {
    double score = 0.0;
    if (nullptr == a || nullptr == b || nullptr == w || len <= 0) {
      return score;
    }
    for (int64_t i = 0; i < len; ++i) {
      const double product = static_cast<double>(a[i]) * static_cast<double>(b[i]);
      score += sigmoid_normal(product) * static_cast<double>(w[i]);
    }
    return score;
  }
)

#if OB_USE_MULTITARGET_CODE
// AVX2 fast sigmoid (throughput-oriented): exp approximation + reciprocal refinement.
OB_DECLARE_AVX2_SPECIFIC_CODE(
  inline static __m256 avx2_exp_ps(__m256 x)
  {
    // Clamp to avoid overflow.
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    // Compute n = floor(x * log2(e) + 0.5)
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256 fx = _mm256_fmadd_ps(x, log2ef, half);
    fx = _mm256_floor_ps(fx);

    // x = x - n * ln2
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 tmp = _mm256_mul_ps(fx, ln2_hi);
    const __m256 z = _mm256_mul_ps(fx, ln2_lo);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    // Polynomial approximation: exp(x) ~= 1 + x + x^2 * P(x)
    const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 y = c0;
    y = _mm256_fmadd_ps(y, x, c1);
    y = _mm256_fmadd_ps(y, x, c2);
    y = _mm256_fmadd_ps(y, x, c3);
    y = _mm256_fmadd_ps(y, x, c4);
    y = _mm256_fmadd_ps(y, x, c5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_fmadd_ps(y, x, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));

    // 2^n
    const __m256i i = _mm256_cvtps_epi32(fx);
    const __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(127)), 23);
    const __m256 pow2nf = _mm256_castsi256_ps(pow2n);
    return _mm256_mul_ps(y, pow2nf);
  }

  // 1 / (1 + exp(-x))
  inline static __m256 avx2_fast_sigmoid(const __m256 x)
  {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 e = avx2_exp_ps(neg);
    const __m256 denom = _mm256_add_ps(one, e);
    // Use rcp + one Newton iteration instead of div.
    __m256 r = _mm256_rcp_ps(denom);
    r = _mm256_mul_ps(r, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(denom, r)));
    return r;
  }

  inline static float avx2_reduce_add_ps(const __m256 v)
  {
    __m256 vsum = _mm256_hadd_ps(v, v);
    vsum = _mm256_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(_mm256_castps256_ps128(vsum))
           + _mm_cvtss_f32(_mm256_extractf128_ps(vsum, 1));
  }

  inline static __m256i avx2_tail_mask_i(const int32_t tail)
  {
    // tail in [0, 8]
    static const int32_t k_mask_tbl[9][8] __attribute__((aligned(32))) = {
      { 0, 0, 0, 0, 0, 0, 0, 0 },                       // 0
      { -1, 0, 0, 0, 0, 0, 0, 0 },                      // 1
      { -1, -1, 0, 0, 0, 0, 0, 0 },                     // 2
      { -1, -1, -1, 0, 0, 0, 0, 0 },                    // 3
      { -1, -1, -1, -1, 0, 0, 0, 0 },                   // 4
      { -1, -1, -1, -1, -1, 0, 0, 0 },                  // 5
      { -1, -1, -1, -1, -1, -1, 0, 0 },                 // 6
      { -1, -1, -1, -1, -1, -1, -1, 0 },                // 7
      { -1, -1, -1, -1, -1, -1, -1, -1 },               // 8
    };
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(k_mask_tbl[tail]));
  }

  // score = sum(sigmoid(a[i] * b[i]) * w[i])
  inline static double sigmod_inner_product_score(const float *a, const float *b, const float *w, const int64_t len)
  {
    if (nullptr == a || nullptr == b || nullptr == w || len <= 0) {
      return 0.0;
    }
    const float *pa = a;
    const float *pb = b;
    const float *pw = w;
    int64_t i = 0;
    const int64_t dim = (len / 64) * 64; // 8*8 unroll

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    for (; i < dim; i += 64) {
      const __m256 a0 = _mm256_loadu_ps(pa + i);
      const __m256 b0 = _mm256_loadu_ps(pb + i);
      const __m256 w0 = _mm256_loadu_ps(pw + i);
      const __m256 a1 = _mm256_loadu_ps(pa + i + 8);
      const __m256 b1 = _mm256_loadu_ps(pb + i + 8);
      const __m256 w1 = _mm256_loadu_ps(pw + i + 8);
      const __m256 a2 = _mm256_loadu_ps(pa + i + 16);
      const __m256 b2 = _mm256_loadu_ps(pb + i + 16);
      const __m256 w2 = _mm256_loadu_ps(pw + i + 16);
      const __m256 a3 = _mm256_loadu_ps(pa + i + 24);
      const __m256 b3 = _mm256_loadu_ps(pb + i + 24);
      const __m256 w3 = _mm256_loadu_ps(pw + i + 24);
      const __m256 a4 = _mm256_loadu_ps(pa + i + 32);
      const __m256 b4 = _mm256_loadu_ps(pb + i + 32);
      const __m256 w4 = _mm256_loadu_ps(pw + i + 32);
      const __m256 a5 = _mm256_loadu_ps(pa + i + 40);
      const __m256 b5 = _mm256_loadu_ps(pb + i + 40);
      const __m256 w5 = _mm256_loadu_ps(pw + i + 40);
      const __m256 a6 = _mm256_loadu_ps(pa + i + 48);
      const __m256 b6 = _mm256_loadu_ps(pb + i + 48);
      const __m256 w6 = _mm256_loadu_ps(pw + i + 48);
      const __m256 a7 = _mm256_loadu_ps(pa + i + 56);
      const __m256 b7 = _mm256_loadu_ps(pb + i + 56);
      const __m256 w7 = _mm256_loadu_ps(pw + i + 56);

      const __m256 p0 = _mm256_mul_ps(a0, b0);
      const __m256 p1 = _mm256_mul_ps(a1, b1);
      const __m256 p2 = _mm256_mul_ps(a2, b2);
      const __m256 p3 = _mm256_mul_ps(a3, b3);
      const __m256 p4 = _mm256_mul_ps(a4, b4);
      const __m256 p5 = _mm256_mul_ps(a5, b5);
      const __m256 p6 = _mm256_mul_ps(a6, b6);
      const __m256 p7 = _mm256_mul_ps(a7, b7);

      acc0 = _mm256_fmadd_ps(avx2_fast_sigmoid(p0), w0, acc0);
      acc1 = _mm256_fmadd_ps(avx2_fast_sigmoid(p1), w1, acc1);
      acc2 = _mm256_fmadd_ps(avx2_fast_sigmoid(p2), w2, acc2);
      acc3 = _mm256_fmadd_ps(avx2_fast_sigmoid(p3), w3, acc3);
      acc4 = _mm256_fmadd_ps(avx2_fast_sigmoid(p4), w4, acc4);
      acc5 = _mm256_fmadd_ps(avx2_fast_sigmoid(p5), w5, acc5);
      acc6 = _mm256_fmadd_ps(avx2_fast_sigmoid(p6), w6, acc6);
      acc7 = _mm256_fmadd_ps(avx2_fast_sigmoid(p7), w7, acc7);
    }

    __m256 acc01 = _mm256_add_ps(acc0, acc1);
    __m256 acc23 = _mm256_add_ps(acc2, acc3);
    __m256 acc45 = _mm256_add_ps(acc4, acc5);
    __m256 acc67 = _mm256_add_ps(acc6, acc7);
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc01, acc23), _mm256_add_ps(acc45, acc67));

    for (; i + 8 <= len; i += 8) {
      const __m256 va = _mm256_loadu_ps(pa + i);
      const __m256 vb = _mm256_loadu_ps(pb + i);
      const __m256 vw = _mm256_loadu_ps(pw + i);
      const __m256 prod = _mm256_mul_ps(va, vb);
      acc = _mm256_fmadd_ps(avx2_fast_sigmoid(prod), vw, acc);
    }

    const int64_t tail = len - i;
    if (tail > 0) {
      // maskload sets invalid elements to 0, so sigmoid(prod) * 0 = 0 for invalid lanes
      // No need for additional mask operation
      const __m256i mask_i = avx2_tail_mask_i(static_cast<int32_t>(tail));
      const __m256 va = _mm256_maskload_ps(pa + i, mask_i);
      const __m256 vb = _mm256_maskload_ps(pb + i, mask_i);
      const __m256 vw = _mm256_maskload_ps(pw + i, mask_i);
      const __m256 prod = _mm256_mul_ps(va, vb);
      acc = _mm256_fmadd_ps(avx2_fast_sigmoid(prod), vw, acc);
    }

    return static_cast<double>(avx2_reduce_add_ps(acc));
  }
)

// for avx512f,avx512bw,avx512vl
OB_DECLARE_AVX512_SPECIFIC_CODE(
  inline static __m512 avx512_exp_ps(__m512 x)
  {
    // Clamp to avoid overflow.
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    // Compute n = floor(x * log2(e) + 0.5)
    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half = _mm512_set1_ps(0.5f);
    __m512 fx = _mm512_fmadd_ps(x, log2ef, half);
    fx = _mm512_floor_ps(fx);

    // x = x - n * ln2
    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 tmp = _mm512_mul_ps(fx, ln2_hi);
    const __m512 z = _mm512_mul_ps(fx, ln2_lo);
    x = _mm512_sub_ps(x, tmp);
    x = _mm512_sub_ps(x, z);

    // Polynomial approximation: exp(x) ~= 1 + x + x^2 * P(x)
    const __m512 c0 = _mm512_set1_ps(1.9875691500E-4f);
    const __m512 c1 = _mm512_set1_ps(1.3981999507E-3f);
    const __m512 c2 = _mm512_set1_ps(8.3334519073E-3f);
    const __m512 c3 = _mm512_set1_ps(4.1665795894E-2f);
    const __m512 c4 = _mm512_set1_ps(1.6666665459E-1f);
    const __m512 c5 = _mm512_set1_ps(5.0000001201E-1f);

    __m512 y = c0;
    y = _mm512_fmadd_ps(y, x, c1);
    y = _mm512_fmadd_ps(y, x, c2);
    y = _mm512_fmadd_ps(y, x, c3);
    y = _mm512_fmadd_ps(y, x, c4);
    y = _mm512_fmadd_ps(y, x, c5);
    y = _mm512_mul_ps(y, x);
    y = _mm512_fmadd_ps(y, x, _mm512_add_ps(x, _mm512_set1_ps(1.0f)));

    // 2^n
    const __m512i i = _mm512_cvtps_epi32(fx);
    const __m512i pow2n = _mm512_slli_epi32(_mm512_add_epi32(i, _mm512_set1_epi32(127)), 23);
    const __m512 pow2nf = _mm512_castsi512_ps(pow2n);
    return _mm512_mul_ps(y, pow2nf);
  }

  // 1 / (1 + exp(-x)) using vrcp14ps (higher precision than AVX2 rcp)
  inline static __m512 avx512_fast_sigmoid(const __m512 x)
  {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
    const __m512 e = avx512_exp_ps(neg);
    const __m512 denom = _mm512_add_ps(one, e);
    // Use vrcp14ps: ~14 bits precision (better than AVX2 rcp which is ~12 bits)
    // vrcp14ps is accurate enough for sigmoid, no Newton iteration needed
    return _mm512_rcp14_ps(denom);
  }

  inline static float avx512_reduce_add_ps(const __m512 v)
  {
    // Horizontal reduction for __m512: split to two __m256, then reduce
    __m256 v_low = _mm512_castps512_ps256(v);
    __m256 v_high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 vsum = _mm256_add_ps(v_low, v_high);
    vsum = _mm256_hadd_ps(vsum, vsum);
    vsum = _mm256_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(_mm256_castps256_ps128(vsum))
           + _mm_cvtss_f32(_mm256_extractf128_ps(vsum, 1));
  }

  // score = sum(sigmoid(a[i] * b[i]) * w[i])
  inline static double sigmod_inner_product_score(const float *a, const float *b, const float *w, const int64_t len)
  {
    if (nullptr == a || nullptr == b || nullptr == w || len <= 0) {
      return 0.0;
    }
    const float *pa = a;
    const float *pb = b;
    const float *pw = w;
    int64_t i = 0;
    const int64_t dim = (len / 128) * 128; // 8*16 unroll

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();

    for (; i < dim; i += 128) {
      const __m512 a0 = _mm512_loadu_ps(pa + i);
      const __m512 b0 = _mm512_loadu_ps(pb + i);
      const __m512 w0 = _mm512_loadu_ps(pw + i);
      const __m512 a1 = _mm512_loadu_ps(pa + i + 16);
      const __m512 b1 = _mm512_loadu_ps(pb + i + 16);
      const __m512 w1 = _mm512_loadu_ps(pw + i + 16);
      const __m512 a2 = _mm512_loadu_ps(pa + i + 32);
      const __m512 b2 = _mm512_loadu_ps(pb + i + 32);
      const __m512 w2 = _mm512_loadu_ps(pw + i + 32);
      const __m512 a3 = _mm512_loadu_ps(pa + i + 48);
      const __m512 b3 = _mm512_loadu_ps(pb + i + 48);
      const __m512 w3 = _mm512_loadu_ps(pw + i + 48);
      const __m512 a4 = _mm512_loadu_ps(pa + i + 64);
      const __m512 b4 = _mm512_loadu_ps(pb + i + 64);
      const __m512 w4 = _mm512_loadu_ps(pw + i + 64);
      const __m512 a5 = _mm512_loadu_ps(pa + i + 80);
      const __m512 b5 = _mm512_loadu_ps(pb + i + 80);
      const __m512 w5 = _mm512_loadu_ps(pw + i + 80);
      const __m512 a6 = _mm512_loadu_ps(pa + i + 96);
      const __m512 b6 = _mm512_loadu_ps(pb + i + 96);
      const __m512 w6 = _mm512_loadu_ps(pw + i + 96);
      const __m512 a7 = _mm512_loadu_ps(pa + i + 112);
      const __m512 b7 = _mm512_loadu_ps(pb + i + 112);
      const __m512 w7 = _mm512_loadu_ps(pw + i + 112);

      const __m512 p0 = _mm512_mul_ps(a0, b0);
      const __m512 p1 = _mm512_mul_ps(a1, b1);
      const __m512 p2 = _mm512_mul_ps(a2, b2);
      const __m512 p3 = _mm512_mul_ps(a3, b3);
      const __m512 p4 = _mm512_mul_ps(a4, b4);
      const __m512 p5 = _mm512_mul_ps(a5, b5);
      const __m512 p6 = _mm512_mul_ps(a6, b6);
      const __m512 p7 = _mm512_mul_ps(a7, b7);

      acc0 = _mm512_fmadd_ps(avx512_fast_sigmoid(p0), w0, acc0);
      acc1 = _mm512_fmadd_ps(avx512_fast_sigmoid(p1), w1, acc1);
      acc2 = _mm512_fmadd_ps(avx512_fast_sigmoid(p2), w2, acc2);
      acc3 = _mm512_fmadd_ps(avx512_fast_sigmoid(p3), w3, acc3);
      acc4 = _mm512_fmadd_ps(avx512_fast_sigmoid(p4), w4, acc4);
      acc5 = _mm512_fmadd_ps(avx512_fast_sigmoid(p5), w5, acc5);
      acc6 = _mm512_fmadd_ps(avx512_fast_sigmoid(p6), w6, acc6);
      acc7 = _mm512_fmadd_ps(avx512_fast_sigmoid(p7), w7, acc7);
    }

    __m512 acc01 = _mm512_add_ps(acc0, acc1);
    __m512 acc23 = _mm512_add_ps(acc2, acc3);
    __m512 acc45 = _mm512_add_ps(acc4, acc5);
    __m512 acc67 = _mm512_add_ps(acc6, acc7);
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc01, acc23), _mm512_add_ps(acc45, acc67));

    for (; i + 16 <= len; i += 16) {
      const __m512 va = _mm512_loadu_ps(pa + i);
      const __m512 vb = _mm512_loadu_ps(pb + i);
      const __m512 vw = _mm512_loadu_ps(pw + i);
      const __m512 prod = _mm512_mul_ps(va, vb);
      acc = _mm512_fmadd_ps(avx512_fast_sigmoid(prod), vw, acc);
    }

    const int64_t tail = len - i;
    if (tail > 0) {
      // Use AVX-512 mask register for efficient tail processing
      const __mmask16 mask = (1ULL << tail) - 1;
      const __m512 va = _mm512_maskz_loadu_ps(mask, pa + i);
      const __m512 vb = _mm512_maskz_loadu_ps(mask, pb + i);
      const __m512 vw = _mm512_maskz_loadu_ps(mask, pw + i);
      const __m512 prod = _mm512_mul_ps(va, vb);
      __m512 add = _mm512_mul_ps(avx512_fast_sigmoid(prod), vw);
      acc = _mm512_mask_add_ps(acc, mask, acc, add);
    }

    return static_cast<double>(avx512_reduce_add_ps(acc));
  }
)
#endif

} // namespace common
} // namespace oceanbase

#endif // OCEANBASE_LIB_OB_VECTOR_SIGMOD_INNER_PRODUCT_H_

