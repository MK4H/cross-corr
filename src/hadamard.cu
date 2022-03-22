#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "cuda_helpers.cuh"

namespace cg = cooperative_groups;

namespace cross
{

/**
 * Complex multiplication of "left" with complex conjugate of "right"
*/
template<typename T>
__device__ inline T multiply_conjugate(T left, T right) {
    return T{
        left.x * right.x + left.y * right.y,
        - left.x * right.y + left.y * right.x
    };
}

/**
 * Each thread is assigned a single entry from the ref matrix and computes
 * the hadamard product with the corresponding element in each deformed matrix
 *
 * So each thread does batch_size steps
 *
 */
template<typename T>
__global__ void hadamard_original(
    const T* __restrict__ ref,
    T* __restrict__ deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
) {
    cg::thread_block ctb = cg::this_thread_block();

    // Coordinates in the whole strip of deformed subregions
    unsigned int ref_idx = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;

    if (ref_idx >= subregions_per_pic * subregion_size.area()) {
        return;
    }

    for (dsize_t i = ref_idx; i < subregion_size.area() * subregions_per_pic * batch_size; i += subregion_size.area() * subregions_per_pic) {
        // Deformed complex conjugate
        deformed[i] = multiply_conjugate(ref[ref_idx], deformed[i]);
    }
}

/**
 * Expects all matrices to be the same size
 *
 * This implementation is optimized for m > n, i.e. right having more matrices than
 * left.
 * This leverages caching of the right matrix values, so that it can reuse the data
 * loaded into register to compute the output for all the corresponding data from left matrices.
 */
template<typename T>
__global__ void hadamard_n_to_m_over_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block ctb = cg::this_thread_block();

    unsigned int thread_start_offset = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;

    // Imagine the matrices just as one dimensional arrays one after another
    // and we just want to multiply the elements at the same offset in each array
    auto array_length = matrix_size.area();
    auto output_length_per_left_array = right_num * array_length;
    // TODO: Give each thread block continous part of the right data array to process instead of
    //  interleaving the data processed by different blocks
    for (dsize_t r_idx = thread_start_offset; r_idx < right_num * array_length; r_idx += ctb.group_dim().x * grid.group_dim().x) {
        auto r_val = right[r_idx];

        auto r_arr_idx = r_idx / array_length;
        // Output offset in the outputs for each left array
        // Outputs are ordered first all for the first left array agains all right arrays
        // then for the second left array agains all right arrays etc.
        // This offset is in the outputs for the given left array
        auto output_offset = r_arr_idx * array_length;
        auto input_offset = r_idx % array_length;
        for (dsize_t l_arr_idx = 0; l_arr_idx < left_num; ++l_arr_idx) {
            auto l_idx = l_arr_idx * array_length + input_offset;
            auto out_idx = l_arr_idx * output_length_per_left_array + output_offset;
            out[out_idx] =  multiply_conjugate(left[l_idx], r_val);
        }
    }
}

/**
 * Each output entry is computed by a single thread.
 * This leads to better data parallelism and simpler code,
 * but prevents any data reuse, as each thread has to fetch both values
 * for each output it computes.
 */
template<typename T>
__global__ void hadamard_n_to_m_over_output(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block ctb = cg::this_thread_block();

    unsigned int thread_start_offset = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;

    // Imagine the matrices just as one dimensional arrays one after another
    // and we just want to multiply the elements at the same offset in each array
    auto array_length = matrix_size.area();
    auto output_length_per_left_array = right_num * array_length;
    // TODO: Give each thread block continous part of the right data array to process instead of
    //  interleaving the data processed by different blocks
    for (dsize_t thread_offset = thread_start_offset; thread_offset < left_num * right_num * matrix_size.area(); thread_offset += ctb.group_dim().x * grid.group_dim().x) {
        auto l_array_idx = thread_offset / output_length_per_left_array;
        // Offset in the output for this left array
        // which corresponds to the offset in the right input data
        auto l_out_offset = thread_offset % output_length_per_left_array;
        // Offset in the left and right input arrays, which is the same for both
        auto l_offset = l_out_offset % array_length;
        out[thread_offset] = multiply_conjugate(left[l_array_idx * array_length + l_offset], right[l_out_offset]);
    }
}

template<typename T>
void run_hadamard_original(
    const T* ref,
    T* deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    dsize_t threads_per_block
) {
    dsize_t num_blocks = div_up(subregion_size.area() * subregions_per_pic, threads_per_block);
    hadamard_original<<<num_blocks, threads_per_block>>>(ref, deformed, subregion_size, subregions_per_pic, batch_size);
}

template<typename T>
void run_hadamard_n_to_m_over_right(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
) {
    // TODO: How many threads to run
    auto right_items = right_num * matrix_size.area();
    auto total_threads = right_items / min_items_per_thread;
    dsize_t num_blocks = div_up(total_threads, threads_per_block);
    hadamard_n_to_m_over_right<<<num_blocks, threads_per_block>>>(
        left,
        right,
        out,
        matrix_size,
        left_num,
        right_num
    );
}

/**
 *
 * If we hit the limit on number of blocks, we have to do more items per thread
 */
template<typename T>
void run_hadamard_n_to_m_over_output(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
) {
    auto output_items = left_num * right_num * matrix_size.area();
    auto total_threads = output_items / min_items_per_thread;
    // TODO: Clamp to the max number of blocks
    dsize_t num_blocks = div_up(total_threads, threads_per_block);
    hadamard_n_to_m_over_output<<<num_blocks, threads_per_block>>>(
        left,
        right,
        out,
        matrix_size,
        left_num,
        right_num
    );
}

template void run_hadamard_original<cufftComplex>(
    const cufftComplex* __restrict__ ref,
    cufftComplex* __restrict__ deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    // TODO: Change to something more general and less implementation dependant
    dsize_t threads_per_block
);

template void run_hadamard_original<cufftDoubleComplex>(
    const cufftDoubleComplex* __restrict__ ref,
    cufftDoubleComplex* __restrict__ deformed,
    dsize2_t subregion_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size,
    // TODO: Change to something more general and less implementation dependant
    dsize_t threads_per_block
);

template void run_hadamard_n_to_m_over_right<cufftComplex>(
    const cufftComplex* __restrict__ left,
    const cufftComplex* __restrict__ right,
    cufftComplex* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template void run_hadamard_n_to_m_over_right<cufftDoubleComplex>(
    const cufftDoubleComplex* __restrict__ left,
    const cufftDoubleComplex* __restrict__ right,
    cufftDoubleComplex* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template void run_hadamard_n_to_m_over_output<cufftComplex>(
    const cufftComplex* __restrict__ left,
    const cufftComplex* __restrict__ right,
    cufftComplex* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

template void run_hadamard_n_to_m_over_output<cufftDoubleComplex>(
    const cufftDoubleComplex* __restrict__ left,
    const cufftDoubleComplex* __restrict__ right,
    cufftDoubleComplex* __restrict__ out,
    dsize2_t matrix_size,
    dsize_t left_num,
    dsize_t right_num,
    // TODO: Benchmark these and set defaults
    dsize_t threads_per_block,
    dsize_t min_items_per_thread
);

}
