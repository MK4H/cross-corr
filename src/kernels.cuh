#pragma once

#include "types.cuh"

namespace cross {

template<typename T>
void run_hadamard_original(
    const T* __restrict__ ref,
    T* __restrict__ deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t num_threads
);

template<typename T>
void run_hadamard_n_to_m_over_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template<typename T>
void run_hadamard_n_to_m_over_output(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template<typename T, typename RES>
void run_cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    cudaStream_t cudaStream = nullptr
);

template<typename T>
void run_scatter(
    const T* __restrict__ src,
    T* __restrict__ dst,
    dsize2_t src_matrix_size,
    dsize_t src_num_matrices,
    dsize2_t dst_matrix_size,
    dsize2_t dst_pos,
    dsize_t threads_per_block,
    dsize_t items_per_threads
);

template<typename T, typename RES>
void run_ccn_warp_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t right_matrices_per_thread
);

template<typename T, typename RES>
void run_ccn_shift_per_warp(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template<typename T, typename RES>
void run_ccn_shift_per_warp_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template<typename DIST, typename T, typename RES>
void run_ccn_shift_per_warp_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template<typename DIST, typename T, typename RES>
void run_ccn_warp_shuffle_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread,
    cudaStream_t cudaStream = nullptr
);

template<typename T, typename RES>
void run_ccn_shift_per_warp_shared_mem(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t shifts_per_cuda_block,
    dsize_t shared_mem_row_size,
    dsize_t shared_mem_rows,
    dsize_t right_matrices_per_block,
    bool strided_load,
    bool column_group_per_block
);

template<typename DIST, typename T, typename RES>
void run_ccn_warp_shuffle_n_to_m_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template<typename T, typename RES>
void run_ccn_multirow_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_rows_per_block,
    dsize_t max_right_rows
);

template<typename T, typename RES>
void run_ccn_multileft_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_rows_per_block,
    dsize_t max_shifts_per_thread,
    dsize_t max_left_rows
);

template<typename T, typename RES>
void run_ccn_shift_per_block(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
);

template<typename T, typename RES>
void run_ccn_multirow_multiright_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t max_right_rows,
    dsize_t right_matrices_per_thread
);

template<typename T, typename RES>
void run_ccn_n_to_mn_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t right_matrices_per_thread,
    dsize_t left_rows_per_iteration
);

template<typename T, typename RES>
void run_ccn_n_to_m_shuffle_multirow(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t warps_per_thread_block,
    dsize_t shifts_per_thread_right_matrix,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t left_rows_per_iteration
);

namespace orig {

template<typename DIST, typename T, typename RES>
void run_ccn_warp_shuffle_n_to_m_work_distribution(
    const T *__restrict__ left,
    const T *__restrict__ right,
    RES *__restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template<typename T, typename RES>
void run_ccn_multileft_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_rows_per_block,
    dsize_t max_shifts_per_thread,
    dsize_t max_left_rows
);

}

}
