/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

extern "C" __global__ void test_pe_info(int *out_my_pe, int *out_n_pes) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out_my_pe = rocshmem_my_pe();
    *out_n_pes = rocshmem_n_pes();
  }
}

extern "C" __global__ void test_put(int *symmetric_buf, int my_pe, int n_pes) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    rocshmem_int_p(symmetric_buf, my_pe * 100 + 42, next_pe);
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_putmem_getmem(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_putmem(sym_dst, sym_src, count * sizeof(int), next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_getmem(local_result, sym_src, count * sizeof(int), prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_int_put_get(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_int_put(sym_dst, sym_src, count, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_int_get(local_result, sym_src, count, prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_float_put_get(
    float *sym_src, float *sym_dst, float *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_float_put(sym_dst, sym_src, count, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_float_get(local_result, sym_src, count, prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_atomic(
    int64_t *sym_counter, int64_t *out_old, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int64_t old = rocshmem_int64_atomic_fetch_add(sym_counter, 10, my_pe);
    out_old[0] = old;

    rocshmem_barrier_all();

    int64_t swapped = rocshmem_int64_atomic_compare_swap(
        sym_counter, 10LL, 999LL, my_pe);
    out_old[1] = swapped;
  }
}

extern "C" __global__ void test_typed_put_signal(
    int *sym_src, int *sym_dst, uint64_t *sym_sig,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    rocshmem_int_put_signal(
        sym_dst, sym_src, count,
        sym_sig, 1, ROCSHMEM_SIGNAL_SET, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_typed_wait_until(
    int *sym_flag, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    rocshmem_int_p(sym_flag, my_pe + 1, next_pe);
    rocshmem_int_wait_until(sym_flag, ROCSHMEM_CMP_NE, 0);
  }
}

extern "C" __global__ void test_typed_int_put_get_wave(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (blockIdx.x == 0) {
    int wf_id = threadIdx.x / warpSize;
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    if (wf_id == 0) {
      rocshmem_int_put_wave(sym_dst, sym_src, count, next_pe);
      rocshmem_barrier_all_wave();
    }
    __syncthreads();

    if (wf_id == 0) {
      rocshmem_int_get_wave(local_result, sym_src, count, prev_pe);
    }
  }
}

extern "C" __global__ void test_typed_amo_extended(
    int64_t *sym_val, int64_t *out_results, int my_pe)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem_int64_atomic_set(sym_val, 42, my_pe);
    int64_t fetched = rocshmem_int64_atomic_fetch(sym_val, my_pe);
    out_results[0] = fetched;

    int64_t old = rocshmem_int64_atomic_swap(sym_val, 99LL, my_pe);
    out_results[1] = old;

    int64_t final_val = rocshmem_int64_atomic_fetch(sym_val, my_pe);
    out_results[2] = final_val;
  }
}

extern "C" __global__ void test_typed_amo_bitwise(
    uint64_t *sym_val, uint64_t *out_results, int my_pe)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem_int64_atomic_set(reinterpret_cast<int64_t*>(sym_val), 0xFF, my_pe);
    uint64_t old = rocshmem_uint64_atomic_fetch_and(sym_val, 0x0FUL, my_pe);
    out_results[0] = old;

    uint64_t after_and = rocshmem_int64_atomic_fetch(
        reinterpret_cast<int64_t*>(sym_val), my_pe);
    out_results[1] = after_and;

    rocshmem_uint64_atomic_and(sym_val, 0x03UL, my_pe);
    uint64_t final_val = rocshmem_int64_atomic_fetch(
        reinterpret_cast<int64_t*>(sym_val), my_pe);
    out_results[2] = final_val;
  }
}

extern "C" __global__ void test_typed_int64_p(
    int64_t *sym_buf, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    rocshmem_int64_p(sym_buf, (int64_t)my_pe * 1000000LL + 12345LL, next_pe);
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_typed_wait_vector(
    int *sym_arr, int *out_idx, int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    for (int i = 0; i < count; i++)
      rocshmem_int_p(sym_arr + i, (my_pe + 1) * 10 + i, next_pe);
    rocshmem_quiet();

    int vals[4] = {0, 0, 0, 0};
    rocshmem_int_wait_until_any_vector(sym_arr, count, nullptr, ROCSHMEM_CMP_NE, vals);
    rocshmem_int_wait_until_all_vector(sym_arr, count, nullptr, ROCSHMEM_CMP_NE, vals);

    *out_idx = 1;
  }
}

extern "C" __global__ void test_int_sum_reduce(
    int *sym_dest, int *sym_src, int nreduce)
{
  rocshmem_ctx_int_sum_reduce_wg(ROCSHMEM_CTX_DEFAULT, ROCSHMEM_TEAM_WORLD,
                                 sym_dest, sym_src, nreduce);
}
