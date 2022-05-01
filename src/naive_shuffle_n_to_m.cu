#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <stdexcept>
#include <cassert>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "bound_checked_loads.cuh"

#include "row_distribution.cuh"
#include "warp_size.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

constexpr dsize_t max_num_left_matrices = 4;
constexpr dsize_t max_num_right_matrices = 4;

/**
 * Arguments for the warp_shuffle_impl function.
 * As we need to write many calls for different constant values of NUM_RIGHTS which
 * all share the same argument values, we want to have each call as short as possible
 * This way, we can create the arguments with a single call and then use it in any of the calls in the switch statement
 *
 * @tparam T
 * @tparam RES
 */
template<typename T, typename RES>
struct warp_shuffle_impl_args {
    const T* __restrict__ left;
    const T* __restrict__ right;
    RES* __restrict__ out;
    dsize2_t warp_right_start;
    dsize2_t warp_right_end;
    vec2<int> warp_min_shift;
    dsize2_t output_pos;
    dsize2_t matrix_size;
    dsize2_t search_size;
    dsize_t num_right_matrices;

    __device__ warp_shuffle_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t warp_right_start,
        dsize2_t warp_right_end,
        vec2<int> warp_min_shift,
        dsize2_t output_pos,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t num_right_matrices
    ) : left(left), right(right), out(out), warp_right_start(warp_right_start),
        warp_right_end(warp_right_end), warp_min_shift(warp_min_shift), output_pos(output_pos),
        matrix_size(matrix_size), search_size(search_size), num_right_matrices(num_right_matrices) {

    }
};

template<typename T, typename RES>
__device__ warp_shuffle_impl_args<T, RES> create_warp_shuffle_impl_args(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t warp_right_start,
    dsize2_t warp_right_end,
    vec2<int> warp_min_shift,
    dsize2_t output_pos,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices
) {
    return warp_shuffle_impl_args<T, RES>(
        left,
        right,
        out,
        warp_right_start,
        warp_right_end,
        warp_min_shift,
        output_pos,
        matrix_size,
        search_size,
        num_right_matrices
    );
}

template<dsize_t NUM_LEFTS, dsize_t NUM_RIGHTS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void warp_shuffle_impl(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    const warp_shuffle_impl_args<T, RES>& args
) {
    // Compute the given shift for num_rights right matrices
    RES sum[NUM_LEFTS * NUM_RIGHTS];
    #pragma unroll
    for (dsize_t i = 0; i < NUM_LEFTS * NUM_RIGHTS; ++i) {
        sum[i] = 0;
    }

    for (dsize_t warp_y_right = args.warp_right_start.y; warp_y_right < args.warp_right_end.y; warp_y_right += 1) {
        // In y axis, both max and min shift are equal in the current implementation
        dsize_t warp_y_left = warp_y_right + args.warp_min_shift.y;

        const dsize_t right_row_offset = warp_y_right * args.matrix_size.x;
        const T* left_row = args.left + warp_y_left * args.matrix_size.x;
        const T* right_row = args.right + right_row_offset;

        int warp_x_left = static_cast<int>(args.warp_right_start.x) + args.warp_min_shift.x;

        // Preload the first values from left matrix
        T thread_left_bottom[NUM_LEFTS];
        #pragma unroll
        for (dsize_t l = 0; l < NUM_LEFTS; ++l) {
            thread_left_bottom[l] = load_with_bounds_check(
                left_row + l * args.matrix_size.area(),
                warp_x_left + warp.thread_rank(),
                args.matrix_size.x
            );
        }


        for (
            dsize_t warp_x_right = args.warp_right_start.x;
            warp_x_right < args.warp_right_end.x;
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

            // Load values from num_rights right matrices
            T thread_right[NUM_RIGHTS];
            #pragma unroll
            for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
                // TODO: Either do bounds check or limit the for loop below
                thread_right[r] = load_with_bounds_check(
                    right_row + r * args.matrix_size.area(), right_idx, args.matrix_size.x
                );
            }

            T thread_left_top[NUM_LEFTS];
            #pragma unroll
            for (dsize_t l = 0; l < NUM_LEFTS; ++l) {
                thread_left_top[l] = load_with_bounds_check(
                    left_row + l * args.matrix_size.area(),
                    left_idx,
                    args.matrix_size.x
                );
            }

            // TODO: Maybe pragma unroll?
            for (dsize_t i = 0; i < warp.size(); ++i) {
                #pragma unroll
                for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
                    // Broadcast
                    auto right_val = warp.shfl(thread_right[r], i);

                    #pragma unroll
                    for (dsize_t l = 0; l < NUM_LEFTS; ++l) {
                        // No need to mask, if either values is out of bounds the value will be 0
                        sum[l * NUM_RIGHTS + r] += thread_left_bottom[l] * right_val;
                    }
                }

                #pragma unroll
                for (dsize_t l = 0; l < NUM_LEFTS; ++l) {

                    // This if cannot be changed into ternary operator
                    // as nvcc fails to optimize the two arrays into registers
                    // and instead puts them into local memory when ternary operator
                    // is used
                    T bottom_shift_val;
                    if (warp.thread_rank() != 0) {
                        bottom_shift_val = thread_left_bottom[l];
                    } else {
                        // Lane 0 pushes the bottom-most value of the top buffer to the top of the bottom buffer
                        //  making it behave as one continuous buffer
                        bottom_shift_val = thread_left_top[l];
                    }
                    // Shuffle does modulo srcLane automatically
                    thread_left_bottom[l] = warp.shfl(bottom_shift_val, warp.thread_rank() + 1);

                    thread_left_top[l] = warp.shfl_down(thread_left_top[l], 1);
                }
            }
        }
    }

    if (args.output_pos.x < args.search_size.x && args.output_pos.y < args.search_size.y) {
        auto output_offset = args.output_pos.linear_idx(args.search_size.x);
        #pragma unroll
        for (dsize_t l = 0; l < NUM_LEFTS; ++l) {
            #pragma unroll
            for (dsize_t r = 0; r < NUM_RIGHTS; ++r) {
                T* matrix = args.out + (l * args.num_right_matrices + r) * args.search_size.area();
                if (ATOMIC) {
                    atomicAdd(matrix + output_offset, sum[l * NUM_RIGHTS + r]);
                } else {
                    matrix[output_offset] = sum[l * NUM_RIGHTS + r];
                }
            }
        }
    }
}

template<dsize_t NUM_LEFTS, dsize_t NUM_RIGHTS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void warp_shuffle_impl_dispatch_num_rights(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t thread_num_rights,
    const warp_shuffle_impl_args<T, RES>& args
) {
    if constexpr(NUM_RIGHTS == 0) {
        // Silence the unused parameter warning
        (void)warp;
        (void)thread_num_rights;
        (void)args;
        assert(false);
    } else {
        if (NUM_RIGHTS == thread_num_rights) {
            warp_shuffle_impl<NUM_LEFTS, NUM_RIGHTS, ATOMIC>(
                warp,
                args
            );
        } else {
            warp_shuffle_impl_dispatch_num_rights<NUM_LEFTS, NUM_RIGHTS - 1, ATOMIC>(
                warp,
                thread_num_rights,
                args
            );
        }
    }
}

template<dsize_t NUM_LEFTS, dsize_t NUM_RIGHTS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void warp_shuffle_impl_dispatch_num_lefts(
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t thread_num_lefts,
    dsize_t thread_num_rights,
    const warp_shuffle_impl_args<T, RES>& args
) {
    if constexpr(NUM_LEFTS == 0) {
        // Silence the unused parameter warning
        (void)warp;
        (void)thread_num_lefts;
        (void)thread_num_rights;
        (void)args;
        assert(false);
    } else {
        if (NUM_LEFTS == thread_num_lefts) {
            warp_shuffle_impl_dispatch_num_rights<NUM_LEFTS, NUM_RIGHTS, ATOMIC>(
                warp,
                thread_num_rights,
                args
            );
        } else {
            warp_shuffle_impl_dispatch_num_lefts<NUM_LEFTS - 1, NUM_RIGHTS, ATOMIC>(
                warp,
                thread_num_lefts,
                thread_num_rights,
                args
            );
        }
    }
}

/**
 * For description of the functionality implemented by this kernel, see ccn_warp_shuffle kernel.
 * This kernel adds distribution of rows of a single shift between multiple threads.
 *
 * @tparam T
 * @tparam RES
 * @param left
 * @param right
 * @param out
 * @param matrix_size
 * @param search_size
 */
template<typename DIST, typename T, typename RES>
__global__ void ccn_warp_shuffle_n_to_m_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
) {

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    // Matrix group is the group of right matrices (of at most right_matrices_per_thread matrices)
    // for which current thread computes the given shift
    // All warps in a block process the same 32 shifts in the x axis, but on different rows
    // so warps in the first block compute shifts 0-31, warps in the second block compute shifts 32-63 etc.
    // So each matrix_group needs to have as many threads as there are shifts in the x axis
    // so number of shifts in the x axis / warp_size
    // TODO: This is precomputed on CPU so we could pass it from there
    dsize_t blocks_per_matrix_group = div_up(search_size.x, warp_size);

    // Which matrix group this block and all its warps will compute
    // THe X axis of block index encodes the shift in x axis together with the left matrix group the thread belongs to
    dsize_t left_matrix_group_idx = ctb.group_index().x / blocks_per_matrix_group;
    dsize_t right_matrix_group_idx = ctb.group_index().y;

    // Offset of the current block and all of its warps in its matrix group
    // This corresponds to the position to write to in the output and the shift
    // to compute
    dsize_t matrix_group_block_offset = ctb.group_index().x % blocks_per_matrix_group;
    dsize_t warp_output_x_offset = matrix_group_block_offset * warp_size;

    // Index of the first matrix in the group processed by the current thread
    dsize_t left_matrix_group_start_idx = left_matrix_group_idx * left_matrices_per_thread;
    dsize_t right_matrix_group_start_idx = right_matrix_group_idx * right_matrices_per_thread;

    // Distribute rows of a single shift between multiple workers,
    // in this case threads
    // Return the assigned output row (which corresponds to a shift),
    // together with the number of workers computing this shift and
    // index of the current worker in range [0, number_of_workers_for_shift)
    assigned_work work = DIST::distribute_rows(
        ctb.group_index().z * ctb.group_dim().y + ctb.thread_index().y,
        max_rows_per_thread,
        matrix_size.y,
        search_size.y
    );

    // All threads of a warp should share the same worker_idx and workers_for_row
    // so either the whole warp continues or exists
    if (work.worker_idx >= work.workers_for_row) {
        return;
    }

    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos{
        warp_output_x_offset,
        work.output_row
    };
    dsize2_t last_warp_thread_out_pos = thread0_out_pos +
                                        dsize2_t{warp.size() - 1, 0};

    // Position in the output matrix
    // This is unique for each thread, as each thread computes a single shift which
    // corresponds to a single output value
    dsize2_t output_pos = thread0_out_pos +
                          dsize2_t{warp.thread_rank(), 0};

    dsize2_t half_search_size = (search_size - 1) / 2;

    // Min of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 0
    vec2<int> warp_min_shift{
        static_cast<int>(thread0_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(thread0_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Max of the shifts computed by the threads of the current warp
    // This will always be the shift computed by thread 31
    // It is clamped into search size as matrix may not be of size divisible by warp_size
    vec2<int> warp_max_shift{
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
    // Multiple threads from different warps may compute the same shift
    // These values are shared for all workers computing the same shift
    dsize_t shared_y_right_start = max(-warp_min_shift.y, 0);
    dsize_t shared_y_right_end = min(matrix_size.y - warp_max_shift.y, matrix_size.y);

    dsize_t shared_overlapping_rows = shared_y_right_end - shared_y_right_start;
    dsize_t rows_per_worker = div_up(shared_overlapping_rows, work.workers_for_row);


    // For the current worker
    dsize_t warp_y_right_start = shared_y_right_start + work.worker_idx * rows_per_worker;
    dsize_t warp_y_right_end = min(warp_y_right_start + rows_per_worker, shared_y_right_end);

    dsize_t thread_num_left_matrices = min(num_left_matrices - left_matrix_group_start_idx, left_matrices_per_thread);
    dsize_t thread_num_right_matrices = min(
        num_right_matrices - right_matrix_group_start_idx, right_matrices_per_thread
    );

    auto args = create_warp_shuffle_impl_args(
        left + left_matrix_group_start_idx * matrix_size.area(),
        right + right_matrix_group_start_idx * matrix_size.area(),
        out + (left_matrix_group_start_idx * num_right_matrices + right_matrix_group_start_idx) * search_size.area(),
        dsize2_t{warp_x_right_start, warp_y_right_start},
        dsize2_t{warp_x_right_end, warp_y_right_end},
        warp_min_shift,
        output_pos,
        matrix_size,
        search_size,
        num_right_matrices
    );

    warp_shuffle_impl_dispatch_num_lefts<max_num_left_matrices, max_num_right_matrices, true>(
        warp,
        thread_num_left_matrices,
        thread_num_right_matrices,
        args
    );
}

} // END anonymous namespace

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
) {
    if (cuda_rows_per_block > 32) {
        throw std::runtime_error("Too many rows per block: "s + std::to_string(cuda_rows_per_block) + " (max 32)");
    }

    if (left_matrices_per_thread > max_num_left_matrices) {
        throw std::runtime_error("Too many left matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " (max "s +
                                 std::to_string(max_num_left_matrices) +
                                 ")"s
        );
    }

    if (right_matrices_per_thread > max_num_right_matrices) {
        throw std::runtime_error("Too many right matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " (max "s +
                                 std::to_string(max_num_right_matrices) +
                                 ")"s
        );
    }

    dsize_t num_workers = DIST::num_workers(max_rows_per_thread, matrix_size.y, search_size.y);

    // Each row of cuda block corresponds to a single warp for simplified code
    constexpr dsize_t block_x_size = warp_size;

    // There will be total of num_left_matrix_groups * num_right_matrix_groups matrix_groups
    dsize_t num_left_matrix_groups = div_up(num_left_matrices, left_matrices_per_thread);
    dsize_t num_right_matrix_groups = div_up(num_right_matrices, right_matrices_per_thread);
    // Each shift is still computed by a single thread (in the x axis), so we need as many threads
    // as there are shifts, in each matrix group
    dsize_t blocks_per_matrix_group = div_up(search_size.x, block_x_size);

    dim3 num_threads(block_x_size, cuda_rows_per_block);
    dim3 num_blocks(
        // Encodes the shift in x direction and the left matrix group the thread belongs to
        blocks_per_matrix_group * num_left_matrix_groups,
        // Encodes the right matrix group the thread belongs to
        num_right_matrix_groups,
        // Encodes distribution of matrix rows between threads
        div_up(num_workers, num_threads.y)
    );

    ccn_warp_shuffle_n_to_m_work_distribution<DIST><<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        num_left_matrices,
        num_right_matrices,
        left_matrices_per_thread,
        right_matrices_per_thread,
        max_rows_per_thread
    );
}

template void run_ccn_warp_shuffle_n_to_m_work_distribution<triangle_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<triangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<triangle_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<rectangle_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<rectangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<rectangle_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<no_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<no_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

template void run_ccn_warp_shuffle_n_to_m_work_distribution<no_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_left_matrices,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t left_matrices_per_thread,
    dsize_t right_matrices_per_thread,
    dsize_t max_rows_per_thread
);

}
