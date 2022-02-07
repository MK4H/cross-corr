#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <stdexcept>

#include "types.cuh"
#include "helpers.cuh"

namespace cg = cooperative_groups;

namespace cross {

constexpr unsigned int warp_size = 32;

__device__ bool bounds_check(int idx, dsize_t size) {
    return idx >= 0 && idx < size;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, int idx, dsize_t size) {
    return bounds_check(idx, size) ? source[idx] : 0;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, int x, int y, dsize2_t size) {
    return bounds_check(x, size.x) && bounds_check(y, size.y) ? source[y * size.x + x] : 0;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, dsize2_t idx, dsize2_t size) {
    return load_with_bounds_check(source, idx, size.x, size.y, size);
}




/**
 * This kernel first computes the range which should be
 * computed by the current warp in the left and right matrices
 * and then always loads 32 values
 */
template<typename T, typename RES>
__global__ void ccn_warp_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    // Initialize by loading a warp worth of data from left matrix
    // as we will be iterating over the left matrix

    // Then broadcast from the right data in sequence from all threads
    // With each broadcast, multiply and sum with the current value from
    // left matrix and then shuffle down the used values from left matrix.
    // Then shuffle the second warp worth of data from left matrix,
    // passing the last thread the value that is shuffled out of the thread 0
    // and would be forgotten
    // basically with warp size 4, it will go
    // 0 1 2 3 0 1 2 3, then 1 2 3 0 1 2 3 x, then 2 3 0 1 2 3 x x,
    // each time broadcasting first from thread 0, then 1, then 2
    // Once we get to 0 1 2 3 x x x x, we load one warp worth of values
    // from both left and right matrices

    // If the shift computed by the current thread does not overlap with the broadcast value
    // that means it tries to read from the left matrix out of bounds and thus will read 0
    // and ignore the broadcast value
    // By shifting the values down, when it reaches the part that overlaps it will receive
    // value shifted from the previous thread

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos = dsize2_t {
        ctb.group_index().x * ctb.group_dim().x,
        ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y
    };
    dsize2_t last_warp_thread_out_pos = thread0_out_pos +
            dsize2_t{warp.size() - 1, 0};

    // Position in the output matrix
    // This is unique for each thread, as each thread computes a single shift which
    // corresponds to a single output value
    dsize2_t output_pos = thread0_out_pos +
            dsize2_t{ctb.thread_index().x, 0};

    dsize2_t half_search_size = (search_size - 1) / 2;

    // Min of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 0
    vec2<int> warp_min_shift = {
            static_cast<int>(thread0_out_pos.x) - static_cast<int>(half_search_size.x),
            static_cast<int>(thread0_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Max of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 31
    // It is clamped into search size as matrix may not be of size divisible by warp_size
    vec2<int> warp_max_shift = {
            static_cast<int>(min(last_warp_thread_out_pos.x, search_size.x)) - static_cast<int>(half_search_size.x),
            static_cast<int>(min(last_warp_thread_out_pos.y, search_size.y)) - static_cast<int>(half_search_size.y)
    };


    // The start depends on the how far right the right matrix is shifted over the left matrix
    // if the right most shift, aka max shift is positive, that means that the left side of the right
    // matrix is inside the left matrix, so we need to start from the 0 element
    // if the max shift is negative, then absolute value tells us how many items of the right matrix are not needed
    // as they do not overlap in any shift computed by the matrix, as all smaller shifts have the right matrix more to the left
    // so they overlap less values
    dsize_t warp_x_right_start = warp_max_shift.x >= 0 ? 0 : -warp_max_shift.x;

    // The last value will be read by the min shift, so if it is larger than 0, the right side of the right matrix overhangs
    // the left matrix and so we don't need to reed the last abs(min_shift) values. Otherwise the right side of the right
    // matrix is inside the left matrix and we need to read it till the end.
    dsize_t warp_x_right_end = warp_min_shift.x >= 0 ? matrix_size.x - warp_min_shift.x : matrix_size.x;

    // All threads in a warp process the same range of rows, so warp_min_shift.y and warp_max_shift.y are the same
    dsize_t warp_y_right_start = max(-warp_min_shift.y, 0);
    dsize_t warp_y_right_end = min(matrix_size.y - warp_max_shift.y, matrix_size.y);

    RES sum = 0;
    for (dsize_t warp_y_right = warp_y_right_start; warp_y_right < warp_y_right_end; warp_y_right += 1) {
        // In y axis, both max and min shift are equal in the current implementation
        int warp_y_left = static_cast<int>(warp_y_right) + warp_min_shift.y;

        const T* right_row = right + warp_y_right * matrix_size.x;
        const T* left_row = left + warp_y_left * matrix_size.x;

        int warp_x_left = static_cast<int>(warp_x_right_start) + warp_min_shift.x;

        // Preload the first values from right matrix
        T thread_left_bottom = load_with_bounds_check(
                left_row,
                warp_x_left + warp.thread_rank(),
                matrix_size.x
        );

        for (
            dsize_t warp_x_right = warp_x_right_start;
            warp_x_right < warp_x_right_end;
            warp_x_right += warp.size(), warp_x_left += warp.size()
            ) {

            // Load next warp_size values
            // Load 0 if out of bounds

            // Right index will always be greater than 0 as we only
            // iterate over part of the matrix
            dsize_t right_idx = warp_x_right + warp.thread_rank();

            // Left index might be out of bounds even below 0, depending on the shift
            // It is also reading warp.size() next values, as we have warp.size() values already loaded
            // from the initialization before the for loop
            int left_idx = warp_x_left + warp.thread_rank() + warp.size();
            // TODO: Either do bounds check or limit the for loop below
            T thread_right = load_with_bounds_check(right_row, right_idx, matrix_size.x);
            T thread_left_top = load_with_bounds_check(left_row, left_idx, matrix_size.x);

            for (dsize_t i = 0; i < warp.size(); ++i) {
                // Broadcast
                auto right_val = warp.shfl(thread_right, i);

                // No need to mask, if either values is out of bounds the value will be 0
                sum += thread_left_bottom * right_val;

                // Shuffle does modulo srcLane automatically
                // Lane 0 pushes the bottom-most value of the top buffer to the top of the bottom buffer
                //  making it behave as one continuous buffer
                thread_left_bottom = warp.shfl(warp.thread_rank() != 0 ? thread_left_bottom : thread_left_top,
                                               warp.thread_rank() + 1);
                thread_left_top = warp.shfl_down(thread_left_top, 1);
            }
        }
    }

    if (output_pos.x < search_size.x && output_pos.y < search_size.y) {
        out[output_pos.linear_idx(search_size.x)] = sum;
    }
}

template<typename T, typename RES>
void run_ccn_warp_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_rows_per_block
) {
    if (cuda_rows_per_block > 32) {
        throw std::runtime_error("Too many rows per block: "s + std::to_string(cuda_rows_per_block) + " (max 32)");
    }

    dim3 num_threads(32, cuda_rows_per_block);
    dim3 num_blocks(
            div_up(search_size.x, num_threads.x),
            div_up(search_size.y, num_threads.y)
    );

    ccn_warp_shuffle<<<num_blocks, num_threads>>>(
            left,
            right,
            out,
            matrix_size,
            search_size
    );
}

template void run_ccn_warp_shuffle<int, int>(
        const int* __restrict__ left,
        const int* __restrict__ right,
        int* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t cuda_rows_per_block
);

template void run_ccn_warp_shuffle<float, float>(
        const float* __restrict__ left,
        const float* __restrict__ right,
        float* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t cuda_rows_per_block
);

template void run_ccn_warp_shuffle<double, double>(
        const double* __restrict__ left,
        const double* __restrict__ right,
        double* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t cuda_rows_per_block
);

}