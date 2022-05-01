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
#include "shared_mem.cuh"

#include "row_distribution.cuh"
#include "warp_size.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

constexpr dsize_t max_num_right_rows = 4;
constexpr dsize_t max_num_right_mats = 4;

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
    vec2<int> warp_max_shift;
    dsize2_t output_pos;
    dsize2_t matrix_size;
    dsize2_t search_size;

    __device__ warp_shuffle_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t warp_right_start,
        dsize2_t warp_right_end,
        vec2<int> warp_min_shift,
        vec2<int> warp_max_shift,
        dsize2_t output_pos,
        dsize2_t matrix_size,
        dsize2_t search_size
    ) : left(left), right(right), out(out), warp_right_start(warp_right_start),
        warp_right_end(warp_right_end), warp_min_shift(warp_min_shift), warp_max_shift(warp_max_shift),
        output_pos(output_pos), matrix_size(matrix_size), search_size(search_size) {

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
    vec2<int> warp_max_shift,
    dsize2_t output_pos,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    return warp_shuffle_impl_args<T, RES>(
        left,
        right,
        out,
        warp_right_start,
        warp_right_end,
        warp_min_shift,
        warp_max_shift,
        output_pos,
        matrix_size,
        search_size
    );
}

template<dsize_t NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, bool REVERSE_OUTPUT, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void compute_row_group(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    dsize_t warp_y_right_start,
    int y_shift,
    RES* __restrict__ res
) {
    dsize_t warp_y_left = warp_y_right_start + y_shift;
    const T* left_row = args.left + warp_y_left * args.matrix_size.x;

    const dsize_t first_right_row_offset = warp_y_right_start * args.matrix_size.x;
    const T* first_right_row = args.right + first_right_row_offset;

    int warp_x_left = static_cast<int>(args.warp_right_start.x) + args.warp_min_shift.x;

    // Preload the first values from left matrix
    T thread_left_bottom = load_with_bounds_check(
        left_row,
        warp_x_left + warp.thread_rank(),
        args.matrix_size.x
    );

    T sum[NUM_RIGHT_ROWS * NUM_RIGHT_MATS];
    for (dsize_t r = 0; r < NUM_RIGHT_ROWS * NUM_RIGHT_MATS; ++r) {
        sum[r] = 0;
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

        T thread_right[NUM_RIGHT_ROWS * NUM_RIGHT_MATS];
        for (dsize_t mat = 0; mat < NUM_RIGHT_MATS; ++mat) {
            for (dsize_t row = 0; row < NUM_RIGHT_ROWS; ++row) {
                thread_right[mat * NUM_RIGHT_ROWS + row] = load_with_bounds_check(
                    first_right_row + mat * args.matrix_size.area() + row * args.matrix_size.x,
                    right_idx,
                    args.matrix_size.x
                );
            }
        }

        T thread_left_top = load_with_bounds_check(left_row, left_idx, args.matrix_size.x);

        for (dsize_t i = 0; i < warp.size(); ++i) {
            for (dsize_t mat = 0; mat < NUM_RIGHT_MATS; ++mat) {
                for (dsize_t row = 0; row < NUM_RIGHT_ROWS; ++row) {
                    // Broadcast
                    auto right_val = warp.shfl(thread_right[mat * NUM_RIGHT_ROWS + row], i);

                    sum[mat * NUM_RIGHT_ROWS + row] += thread_left_bottom * right_val;
                }
            }
            // Shuffle does modulo srcLane automatically
            // Lane 0 pushes the bottom-most value of the top buffer to the top of the bottom buffer
            //  making it behave as one continuous buffer
            thread_left_bottom = warp.shfl(
                warp.thread_rank() != 0 ? thread_left_bottom : thread_left_top,
                warp.thread_rank() + 1
            );
            thread_left_top = warp.shfl_down(thread_left_top, 1);
        }
    }

    for (dsize_t mat = 0; mat < NUM_RIGHT_MATS; ++mat) {
        for (dsize_t row = 0; row < NUM_RIGHT_ROWS; ++row) {
            // Res contains first the results of min_shift from all matrices for all threads of the block,
            // then results of min_shift + 1 from all matrices for all threads of the block,
            // up to the results of min_shift + NUM_RIGHT_ROWS for all matrices in warp_shuffle_impl
            // This is done to allow access to a subset of shifts
            // TODO: Try substituting everywhere if compiler is not able
            //dsize_t result_idx = row * NUM_RIGHT_MATS + mat;
            // Whereas sum contains all rows for the first matrix, then all rows for the second matrix etc.
            // This is done as that minimizes the jumps around memory
            // TODO: Try substituting everywhere if compiler is not able
            dsize_t sum_idx = mat * NUM_RIGHT_ROWS + row;

            if constexpr(REVERSE_OUTPUT) {
                res[((NUM_RIGHT_ROWS - 1 - row) * NUM_RIGHT_MATS + mat) * ctb.size() +
                    ctb.thread_rank()] += sum[sum_idx];
            } else {
                res[(row * NUM_RIGHT_MATS + mat) * ctb.size() + ctb.thread_rank()] += sum[sum_idx];
            }
        }
    }
}

/*
 * First NUM_RIGHT_ROWS rows will only overlap in some of the shifts
 * If we start at the 0 row of the right matrix, then that means that the
 * top of the right matrix is inside the left matrix
 *
 * As we are computing NUM_RIGHT_ROWS shifts in consecutive rows with the same
 * x coordinate, the first shift will overlap given left row and no other shift
 * overlaps anything with the left row
 *
 * Next left row is overlapped with the args.warp_right_start.y by the following shift,
 * while the first shift overlaps the left row with args.warp_right_start.y + 1
 *
 * Then the third left row is overlapped with args.warp_right_start.y by the third shift,
 * with args.warp_right_start.y + 1 by second shift and with args.warp_right_start.y + 2 by
 * first shift etc.
 *
 * If the top of the right matrix starts outside the left matrix, which can only be above the
 * left matrix, some of the steps may be skipped, for example if it is one row above,
 * the first left row is overlapped by the first shift with row args.warp_right_start.y + 1
 * and by the second shift with row args.warp_right_start.y, which is exactly the second step described above
 *
 * Similar principle, but in reverse, applies when bottom of the right matrix is inside the left matrix.
 * There the left row stays the same, but we change the number of right rows it runs against,
 * getting progressively smaller.
 *
 * These ifs should cover all possibilities up to NUM_RIGHT_ROWS
 * Because max_shift.y - min_shift.y == NUM_RIGHT_ROWS, min_shift.y + NUM_RIGHT_ROWS == max_shift.y
 *
 */
template<int NUM_RIGHT_ROWS, dsize_t MAX_NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void startup(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    RES* __restrict__ res
) {
    if constexpr(NUM_RIGHT_ROWS < MAX_NUM_RIGHT_ROWS) {
        if (static_cast<int>(args.warp_right_start.y) + args.warp_min_shift.y + NUM_RIGHT_ROWS - 1 >= 0) {
            compute_row_group<NUM_RIGHT_ROWS, NUM_RIGHT_MATS, true>(
                ctb,
                warp,
                args,
                args.warp_right_start.y,
                args.warp_min_shift.y + NUM_RIGHT_ROWS - 1,
                res
            );
        }
        startup<NUM_RIGHT_ROWS + 1, MAX_NUM_RIGHT_ROWS, NUM_RIGHT_MATS>(ctb, warp, args, res);
    } else {
        // Silence the unused parameter warning
        (void)ctb;
        (void)warp;
        (void)args;
        (void)res;
    }
}

template<int NUM_RIGHT_ROWS, dsize_t MAX_NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void wind_down(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    RES* __restrict__ res
) {
    if constexpr(NUM_RIGHT_ROWS > 0) {
        if (args.warp_right_end.y - NUM_RIGHT_ROWS + args.warp_max_shift.y < args.matrix_size.y) {
            compute_row_group<NUM_RIGHT_ROWS, NUM_RIGHT_MATS, true>(
                ctb,
                warp,
                args,
                args.warp_right_end.y - NUM_RIGHT_ROWS,
                args.warp_max_shift.y,
                res + ((MAX_NUM_RIGHT_ROWS - NUM_RIGHT_ROWS) * NUM_RIGHT_MATS) * ctb.size()
            );
        }
        wind_down<NUM_RIGHT_ROWS - 1, MAX_NUM_RIGHT_ROWS, NUM_RIGHT_MATS>(ctb, warp, args, res);
    } else {
        // Silence the unused parameter warning
        (void)ctb;
        (void)warp;
        (void)args;
        (void)res;
    }
}

template<dsize_t NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void multirow_multiright_shuffle_impl(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    warp_shuffle_impl_args<T, RES> args,
    RES* __restrict__ res
) {
    startup<1, NUM_RIGHT_ROWS, NUM_RIGHT_MATS>(ctb, warp, args, res);

    /*
     * The startup gets us to the situation where we have the first
     * left row at max_shift (== min_shift + NUM_RIGHTS_ROW) which is
     * to be processed with all NUM_RIGHT_ROWS
     *
     * As we are always loading warp_y_right and the following (NUM_RIGHT_ROWS - 1) rows,
     * we need to stop NUM_RIGHT_ROWS before the end
     */
    int end = args.warp_right_end.y - (NUM_RIGHT_ROWS - 1);

    for (int warp_y_right = args.warp_right_start.y; warp_y_right < end; warp_y_right += 1) {
        compute_row_group<NUM_RIGHT_ROWS, NUM_RIGHT_MATS, true>(
            ctb,
            warp,
            args,
            warp_y_right,
            args.warp_max_shift.y,
            res
        );
    }

    wind_down<NUM_RIGHT_ROWS - 1, NUM_RIGHT_ROWS, NUM_RIGHT_MATS>(ctb, warp, args, res);

    auto first_output_offset = args.output_pos.linear_idx(args.search_size.x);
    RES* matrix = args.out;

    // TODO: Maybe just check the x axis, Y axis should be filtered out by 0 NUM_RIGHT_ROWS
    if (args.output_pos.x < args.search_size.x && args.output_pos.y < args.search_size.y) {
        for (dsize_t mat = 0; mat < NUM_RIGHT_MATS; ++mat) {
            for (dsize_t row = 0; row < NUM_RIGHT_ROWS; ++row) {
                auto output_offset = first_output_offset + mat * args.search_size.area() + row * args.search_size.x;
                auto val = res[(row * NUM_RIGHT_MATS + mat) * ctb.size() + ctb.thread_rank()];
                if constexpr(ATOMIC) {
                    atomicAdd(matrix + output_offset, val);
                } else {
                    matrix[output_offset] = val;
                }
            }
        }
    }
}

template<dsize_t NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void multirow_multiright_shuffle_impl_mats_dispatch(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_right_mats,
    const warp_shuffle_impl_args<T, RES>& args,
    RES* __restrict__ res
) {
    if constexpr(NUM_RIGHT_MATS == 0) {
        // Silence the unused parameter warning
        (void)ctb;
        (void)warp;
        (void)num_right_mats;
        (void)args;
        (void)res;
        assert(false);
    } else {
        if (NUM_RIGHT_MATS == num_right_mats) {
            multirow_multiright_shuffle_impl<NUM_RIGHT_ROWS, NUM_RIGHT_MATS, ATOMIC>(
                ctb,
                warp,
                args,
                res
            );
        } else {
            multirow_multiright_shuffle_impl_mats_dispatch<NUM_RIGHT_ROWS, NUM_RIGHT_MATS - 1, ATOMIC>(
                ctb,
                warp,
                num_right_mats,
                args,
                res
            );
        }
    }
}

template<dsize_t NUM_RIGHT_ROWS, dsize_t NUM_RIGHT_MATS, bool ATOMIC, dsize_t WARP_SIZE, typename T, typename RES>
__device__ void multirow_multiright_shuffle_impl_rows_dispatch(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<WARP_SIZE>& warp,
    dsize_t num_right_rows,
    dsize_t num_right_mats,
    const warp_shuffle_impl_args<T, RES>& args,
    RES* __restrict__ res
) {
    if constexpr(NUM_RIGHT_ROWS == 0) {
        // Zero is valid, if the warp is completely outside the result matrix

        // Silence the unused parameter warning
        (void)ctb;
        (void)warp;
        (void)num_right_rows;
        (void)num_right_mats;
        (void)args;
        (void)res;
    } else {
        if (NUM_RIGHT_ROWS == num_right_rows) {
            multirow_multiright_shuffle_impl_mats_dispatch<NUM_RIGHT_ROWS, NUM_RIGHT_MATS, ATOMIC>(
                ctb,
                warp,
                num_right_mats,
                args,
                res
            );
        } else {
            multirow_multiright_shuffle_impl_rows_dispatch<NUM_RIGHT_ROWS - 1, NUM_RIGHT_MATS, ATOMIC>(
                ctb,
                warp,
                num_right_rows,
                num_right_mats,
                args,
                res
            );
        }
    }
}


template<typename T, typename RES>
__global__ void ccn_multirow_multiright_shuffle(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t max_right_rows,
    dsize_t right_matrices_per_thread
) {

    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t blocks_per_matrix_group = div_up(search_size.x, warp_size);
    dsize_t matrix_group_idx = ctb.group_index().x / blocks_per_matrix_group;
    dsize_t matrix_group_block_offset = ctb.group_index().x % blocks_per_matrix_group;

    dsize_t output_x_offset = matrix_group_block_offset * warp_size;
    dsize_t matrix_group_start_idx = matrix_group_idx * right_matrices_per_thread;

    // All warps of given block start at the same x, but each work on different row of output
    dsize2_t thread0_out_pos{
        output_x_offset,
        (ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y) * max_right_rows
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
    // This will always be the shift computed by thread 31 for the x axis
    //
    // It is clamped into search size as matrix may not be of size divisible by warp_size
    vec2<int> warp_max_shift{
        static_cast<int>(min(last_warp_thread_out_pos.x, search_size.x - 1)) - static_cast<int>(half_search_size.x),
        // max_right_rows - 1 because + max_right_rows is the min_shift of next warp
        static_cast<int>(min(last_warp_thread_out_pos.y + max_right_rows - 1, search_size.y - 1)) -
        static_cast<int>(half_search_size.y)
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

    dsize_t warp_y_right_start = max(-warp_max_shift.y, 0);
    dsize_t warp_y_right_end = min(matrix_size.y - warp_min_shift.y, matrix_size.y);


    RES* res = shared_memory_proxy<RES>();
    for (dsize_t i = ctb.thread_rank(); i < max_right_rows * right_matrices_per_thread * ctb.size(); i += ctb.size()) {
        res[i] = 0;
    }
    ctb.sync();

    // Max shift might be smaller than min shift if warp is completely outside the out matrix
    // +1 because max_shift is inclusive, it is the last shift computed by this warp
    // so to get the number of shifts with both sides inclusive, we need to add 1
    auto warp_num_right_rows = static_cast<dsize_t>(max(warp_max_shift.y - warp_min_shift.y + 1, 0));

    dsize_t warp_num_right_matrices = min(num_right_matrices - matrix_group_start_idx, right_matrices_per_thread);


    auto args = create_warp_shuffle_impl_args(
        left,
        right + matrix_group_start_idx * matrix_size.area(),
        out + matrix_group_start_idx * search_size.area(),
        dsize2_t{warp_x_right_start, warp_y_right_start},
        dsize2_t{warp_x_right_end, warp_y_right_end},
        warp_min_shift,
        warp_max_shift,
        output_pos,
        matrix_size,
        search_size
    );

    multirow_multiright_shuffle_impl_rows_dispatch<max_num_right_rows, max_num_right_mats, false>(
        ctb,
        warp,
        warp_num_right_rows,
        warp_num_right_matrices,
        args,
        res
    );
}

} // END anonymous namespace

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
) {
    if (cuda_rows_per_block > 32) {
        throw std::runtime_error("Too many rows per block: "s + std::to_string(cuda_rows_per_block) + " (max 32)");
    }

    if (right_matrices_per_thread == 0 || right_matrices_per_thread > max_num_right_mats) {
        throw std::runtime_error("Invalid number of right matrices per thread: "s +
                                 std::to_string(right_matrices_per_thread) +
                                 " [1-"s +
                                 std::to_string(max_num_right_mats) +
                                 "]"s
        );
    }

    dim3 num_threads(warp_size, cuda_rows_per_block);

    dsize_t num_matrix_groups = div_up(num_right_matrices, right_matrices_per_thread);
    dsize_t blocks_per_matrix_group = div_up(search_size.x, num_threads.x);

    dim3 num_blocks(
        blocks_per_matrix_group * num_matrix_groups,
        div_up(search_size.y, num_threads.y * max_right_rows)
    );

    dsize_t block_size = num_threads.x * num_threads.y;

    dsize_t shared_mem_size = block_size * max_right_rows * right_matrices_per_thread * sizeof(RES);

    ccn_multirow_multiright_shuffle<<<num_blocks, num_threads, shared_mem_size>>>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        num_right_matrices,
        max_right_rows,
        right_matrices_per_thread
    );
}

template void run_ccn_multirow_multiright_shuffle<int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t max_right_rows,
    dsize_t right_matrices_per_thread
);

template void run_ccn_multirow_multiright_shuffle<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t max_right_rows,
    dsize_t right_matrices_per_thread
);

template void run_ccn_multirow_multiright_shuffle<double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t cuda_rows_per_block,
    dsize_t max_right_rows,
    dsize_t right_matrices_per_thread
);

}
