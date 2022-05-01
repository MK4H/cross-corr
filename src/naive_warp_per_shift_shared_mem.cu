#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <stdexcept>

#include "types.cuh"
#include "cuda_helpers.cuh"
#include "shared_mem.cuh"
#include "row_distribution.cuh"
#include "argument_error.hpp"
#include "warp_size.hpp"

namespace cg = cooperative_groups;

namespace cross {

namespace {

constexpr dsize_t max_num_right_matrices = 8;

/**
 *
 * @tparam T
 * @tparam RES
 * @param warp
 * @param left
 * @param right
 * @param left_buffer_start_row Might be negative if the min shift of the current block can only process some of the values from the left buffer with the current values of right buffer
 * @param right_buffer_start_row
 * @param row_size
 * @param warp_left_start_row
 * @param warp_left_end_row
 * @param num_left_loaded_rows
 * @param warp_y_shift
 * @param sum
 */
template<dsize_t NUM_RIGHT, typename T, typename RES>
__device__ void compute_from_shared_mem_buffers(
    const cg::thread_block_tile<warp_size>& warp,
    const shared_mem_buffer<T>& left_buffer,
    const shared_mem_buffer<T>(& right_buffers)[NUM_RIGHT],
    int left_buffer_start_row,
    dsize_t right_buffer_start_row,
    dsize_t row_size,
    dsize_t warp_right_buffer_start_row,
    dsize_t warp_right_buffer_end_row,
    dsize_t num_left_loaded_rows,
    int warp_y_shift,
    RES (& sum)[NUM_RIGHT]
) {
    // Limit the right buffer rows by the available corresponding rows in the left buffer

    // Right row which corresponds to the first row in the left buffer
    int right_row_for_first_left = left_buffer_start_row - warp_y_shift - static_cast<int>(right_buffer_start_row);
    // Even though this is int, the warp_right_buffer_start_row should always pull
    //  it above 0
    int warp_right_start_row = max(
        static_cast<int>(warp_right_buffer_start_row),
        right_row_for_first_left
    );
    // This might be negative
    int warp_right_end_row = min(
        static_cast<int>(warp_right_buffer_end_row),
        right_row_for_first_left + static_cast<int>(num_left_loaded_rows)
    );

    int buffer_row_shift = -right_row_for_first_left;

    int warp_right_start_offset = warp_right_start_row * static_cast<int>(row_size);
    int warp_right_end_offset = warp_right_end_row * static_cast<int>(row_size);
    int buffer_offset = buffer_row_shift * static_cast<int>(row_size);

    for (
        int right_idx = warp_right_start_offset + warp.thread_rank();
        right_idx < warp_right_end_offset;
        right_idx += warp.size()
    ) {
        auto l = left_buffer[right_idx + buffer_offset];
        for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
            const auto& right_buffer = right_buffers[i];
            sum[i] += l * right_buffer[right_idx];
        }
    }
}

template<typename T, typename RES>
struct shared_mem_rows_impl_args {
    const T* __restrict__ left;
    const T* __restrict__ right;
    RES* __restrict__ out;
    dsize2_t matrix_size;
    dsize2_t search_size;
    dsize_t shared_mem_row_size;
    dsize_t shared_mem_rows;
    dsize2_t block_right_start;
    dsize2_t block_right_end;
    int block_x_shift;
    int block_min_y_shift;
    dsize2_t warp_out_pos;

    __device__ shared_mem_rows_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t shared_mem_row_size,
        dsize_t shared_mem_rows,
        dsize2_t block_right_start,
        dsize2_t block_right_end,
        int block_x_shift,
        int block_min_y_shift,
        dsize2_t warp_out_pos
    ) : left(left), right(right), out(out), matrix_size(matrix_size),
        search_size(search_size), shared_mem_row_size(shared_mem_row_size),
        shared_mem_rows(shared_mem_rows), block_right_start(block_right_start),
        block_right_end(block_right_end), block_x_shift(block_x_shift), block_min_y_shift(block_min_y_shift),
        warp_out_pos(warp_out_pos) {

    }
};

template<typename T, typename RES>
__device__ shared_mem_rows_impl_args<T, RES> create_shared_mem_rows_impl_args(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shared_mem_row_size,
    dsize_t shared_mem_rows,
    dsize2_t block_right_start,
    dsize2_t block_right_end,
    int block_x_shift,
    int block_min_y_shift,
    dsize2_t warp_out_pos
) {
    return shared_mem_rows_impl_args<T, RES>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        shared_mem_row_size,
        shared_mem_rows,
        block_right_start,
        block_right_end,
        block_x_shift,
        block_min_y_shift,
        warp_out_pos
    );
}

template<dsize_t NUM_RIGHT, bool STRIDED_LOAD, bool ATOMIC, typename T, typename RES>
__device__ void shared_mem_rows_impl(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<warp_size>& warp,
    shared_mem_rows_impl_args<T, RES> args
) {
    int warp_y_shift = args.block_min_y_shift + static_cast<int>(warp.meta_group_rank());

    dsize_t warp_right_start_row = max(0, -warp_y_shift);
    dsize_t warp_right_end_row = min(args.matrix_size.y - warp_y_shift, args.matrix_size.y);

    // We need to limit the number of values preloaded into left buffer so that we don't load any values
    // which are to be used by the second right buffer
    // For example with min y shift -1, we cannot load the whole left buffer from 0 to shared_mem_rows,
    // as the warp with shift -1 will only use the first 7 values from the left buffer and the 8th value
    // should be used by the second load of right buffer, but it will be overwritten by the time we load the second
    // right buffer
    // So we need to read only the number of values used by the min shift and offset them in the left buffer
    // so that the top part continues seamlessly
    //
    // This might be negative, as we need the left buffer to start before the left matrix itself
    // We just need to handle this when preloading, after that everything should work as intended
    // as we only touch the left rows corresponding to the right rows, so nothing should touch then
    // negative rows
    // TODO: This needs to be fixed if we want this to work with fewer shared_mem_rows
    //  than there are warps in a block
    //  Currently left_buffer_preload_start_row is the row to be loaded corresponding to the
    //  the value in right buffer to be computed by warp 0 in the block
    int left_buffer_preload_start_row = static_cast<int>(args.block_right_start.y) + args.block_min_y_shift;
    dsize_t left_src_preload_start_row = left_buffer_preload_start_row >= 0 ? left_buffer_preload_start_row : 0;
    dsize_t preload_offset_rows = left_buffer_preload_start_row >= 0 ? 0 : -left_buffer_preload_start_row;

    T* shared = shared_memory_proxy<T>();
    shared_mem_buffer<T> left_bottom_s = shared_mem_buffer<T>::allocate(
        &shared, args.shared_mem_row_size * args.shared_mem_rows
    );
    shared_mem_buffer<T> left_top_s = shared_mem_buffer<T>::allocate(
        &shared, args.shared_mem_row_size * args.shared_mem_rows
    );

    // TODO: Remove size from the shared_mem_buffer struct as it is shared between all the buffers
    shared_mem_buffer<T> right_s[NUM_RIGHT];
    for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
        right_s[i] = shared_mem_buffer<T>::allocate(&shared, args.shared_mem_row_size * args.shared_mem_rows);
    }

    RES sum[NUM_RIGHT];
    for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
        sum[i] = 0;
    }

    for (
        dsize_t iter_block_right_start_x = args.block_right_start.x;
        iter_block_right_start_x < args.block_right_end.x;
        iter_block_right_start_x += args.shared_mem_row_size
    ) {

        dsize_t row_size = min(args.shared_mem_row_size, args.block_right_end.x - iter_block_right_start_x);
        dsize_t stride = args.matrix_size.x - row_size;
        // This should always be inside the left matrix due to bound checking when computing block_right_start
        dsize_t block_left_x_start = iter_block_right_start_x + args.block_x_shift;

        // This needs to be bound checked explicitly as block_right_start depends on the block_max_shift,
        // so block_min_shift might be outside the left matrix for first few rows of right matrix
        dsize2_t left_preload_start(
            block_left_x_start,
            left_src_preload_start_row
        );
        // Size of the last load, for preload includes the prefixed zeroes when preload offset is not 0
        dsize_t last_load_size = min(args.shared_mem_rows, args.matrix_size.y - left_buffer_preload_start_row);

        // Preload first values into the bottom buffer
        left_bottom_s.template load_strided_chunks<STRIDED_LOAD>(
            ctb,
            warp,
            args.left + left_preload_start.linear_idx(args.matrix_size.x),
            row_size,
            last_load_size - preload_offset_rows,
            stride,
            preload_offset_rows * row_size
        );

        int left_buffer_start_row = left_buffer_preload_start_row;
        // TODO: Unroll into three loops, start-up, core, finish
        //  where core can get rid of some of the range checks and run faster
        for (
            dsize_t right_buffer_start_row = args.block_right_start.y;
            right_buffer_start_row < args.block_right_end.y;
            right_buffer_start_row += args.shared_mem_rows, left_buffer_start_row += args.shared_mem_rows
        ) {
            dsize2_t left_load_start{
                block_left_x_start,
                // The first preload_size rows are already in the bottom buffer
                left_buffer_start_row + last_load_size
            };
            // TODO: This should be the same as right load size
            dsize_t left_load_size = min(args.shared_mem_rows, args.matrix_size.y - left_load_start.y);

            left_top_s.template load_strided_chunks<STRIDED_LOAD>(
                ctb,
                warp,
                args.left + left_load_start.linear_idx(args.matrix_size.x),
                row_size,
                left_load_size,
                stride
            );

            dsize2_t right_load_start{
                iter_block_right_start_x,
                right_buffer_start_row
            };
            dsize_t right_load_size = min(args.shared_mem_rows, args.block_right_end.y - right_buffer_start_row);
            // Load the rows from left and possibly multiple right matrices
            for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
                right_s[i].template load_strided_chunks<STRIDED_LOAD>(
                    ctb,
                    warp,
                    args.right + right_load_start.linear_idx(args.matrix_size.x) + i * args.matrix_size.area(),
                    row_size,
                    right_load_size,
                    stride
                );
            }

            ctb.sync();

            dsize_t warp_right_buffer_start_row =
                max(warp_right_start_row, right_buffer_start_row) - right_buffer_start_row;
            dsize_t warp_right_buffer_end_row =
                min(warp_right_end_row, right_buffer_start_row + right_load_size) - right_buffer_start_row;

            compute_from_shared_mem_buffers(
                warp,
                left_bottom_s,
                right_s,
                left_buffer_start_row,
                right_buffer_start_row,
                row_size,
                warp_right_buffer_start_row,
                warp_right_buffer_end_row,
                last_load_size,
                warp_y_shift,
                sum
            );

            compute_from_shared_mem_buffers(
                warp,
                left_top_s,
                right_s,
                left_load_start.y,
                right_buffer_start_row,
                row_size,
                warp_right_buffer_start_row,
                warp_right_buffer_end_row,
                left_load_size,
                warp_y_shift,
                sum
            );

            swap(left_bottom_s, left_top_s);
            last_load_size = left_load_size;

            ctb.sync();
        }
    }

    // On the X axis there should be only precisely the required number of warps
    // On Y axis, due to the set block size, some warps are out of bounds
    // We need those warps for loading shared memory, so we cannot just return at the start
    // so we need to check here
    if (args.warp_out_pos.y < args.search_size.y) {
        auto output_offset = args.warp_out_pos.linear_idx(args.search_size.x);
        for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
            auto result = cg::reduce(warp, sum[i], cg::plus<RES>());
            if (warp.thread_rank() == 0) {
                RES* out_matrix = args.out + i * args.search_size.area();
                if constexpr(ATOMIC) {
                    atomicAdd(out_matrix + output_offset, result);
                } else {
                    out_matrix[output_offset] = result;
                }
            }
        }
    }
}

template<dsize_t NUM_RIGHT, bool STRIDED_LOAD, bool ATOMIC, typename T, typename RES>
__device__ void shared_mem_rows_impl_dispatch(
    const cg::thread_block& ctb,
    const cg::thread_block_tile<warp_size>& warp,
    dsize_t right_matrices_per_block,
    shared_mem_rows_impl_args<T, RES> args
) {
    if constexpr(NUM_RIGHT == 0) {
        // Silence the unused parameter warning
        (void)ctb;
        (void)warp;
        (void)right_matrices_per_block;
        (void)args;
        assert(false);
    } else {
        if (NUM_RIGHT == right_matrices_per_block) {
            shared_mem_rows_impl<NUM_RIGHT, STRIDED_LOAD, ATOMIC>(
                ctb,
                warp,
                args
            );
        } else {
            shared_mem_rows_impl_dispatch<NUM_RIGHT - 1, STRIDED_LOAD, ATOMIC>(
                ctb,
                warp,
                right_matrices_per_block,
                args
            );
        }
    }
}

/**
 * In this implementation, the warps in the given block
 * compute the same shifts on the x axis, but on consecutive rows. This allows for
 * offset access to shared memory, which compared to strided access of implementations
 * with warps in the same block computed shifts with the same y axis does not lead
 * to bank conflicts.
 *
 * TODO: This does not work with shared_mem_rows < shift_per_block
 * @tparam T
 * @tparam RES
 * @param left
 * @param right
 * @param out
 * @param matrix_size
 * @param search_size
 * @param shared_mem_buffer_size
 */
template<bool STRIDED_LOAD, typename T, typename RES>
__global__ void ccn_shift_per_warp_shared_mem(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_right_matrices,
    dsize_t shared_mem_row_size,
    dsize_t shared_mem_rows,
    dsize_t right_matrices_per_block
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_block = warp.meta_group_size();
    dsize2_t half_search_size = (search_size - 1) / 2;

    // Each warp computes the same shift in right_matrices_per_warp consecutive matrices
    // This index tells us which group is to be computed by the current warp
    // For this work distribution, we imagine the result matrices laid left to right
    // There will be num_right_matrices / right_matrices_per_warp times more blocks than is
    // required, so if we divide the block x index by search_size.x, we get which group of right matrices
    // of size right_matrices_per_warp this warp is to compute
    dsize_t block_x_index = ctb.group_index().x;
    dsize_t matrix_group_idx = block_x_index / search_size.x;
    // Offset in the output matrix, which will tell us the shift to be computed
    dsize_t output_x_offset = block_x_index % search_size.x;

    // Index of the first right matrix to be processed by the current warp
    dsize_t matrix_group_start_idx = matrix_group_idx * right_matrices_per_block;

    dsize_t block_num_right_matrices = min(num_right_matrices - matrix_group_start_idx, right_matrices_per_block);

    // These shifts are from right matrix to left, i.e. if we have index into the right matrix
    //  we need to add this value to get corresponding index in left matrix.
    //  Inversely if we have index into the left matrix, we need to subtract this to get index
    //  into the left matrix
    int block_x_shift = static_cast<int>(output_x_offset) - half_search_size.x;
    // Shift of the first warp in the block
    int block_min_y_shift =
        static_cast<int>(ctb.group_index().y * shifts_per_block) - static_cast<int>(half_search_size.y);
    // Shift of the last warp in the block
    int block_max_y_shift =
        min(
            static_cast<int>(ctb.group_index().y * shifts_per_block) + shifts_per_block - 1,
            static_cast<int>(search_size.y)
        ) -
        static_cast<int>(half_search_size.y);
    dsize2_t block_right_start(
        max(0, -block_x_shift),
        max(0, -block_max_y_shift)
    );

    dsize2_t block_right_end(
        min(matrix_size.x - block_x_shift, matrix_size.x),
        min(matrix_size.y - block_min_y_shift, matrix_size.y)
    );

    if (grid.group_dim().z != 1) {
        dsize_t assigned_column_group = ctb.group_index().z;

        block_right_start.x += assigned_column_group * shared_mem_row_size;
        block_right_end.x = min(block_right_start.x + shared_mem_row_size, block_right_end.x);

        if (block_right_start.x >= block_right_end.x) {
            return;
        }
    }

    dsize2_t warp_out_pos{
        output_x_offset,
        ctb.group_index().y * shifts_per_block + warp.meta_group_rank()
    };

    auto args = create_shared_mem_rows_impl_args(
        left,
        right + matrix_group_start_idx * matrix_size.area(),
        out + matrix_group_start_idx * search_size.area(),
        matrix_size,
        search_size,
        shared_mem_row_size,
        shared_mem_rows,
        block_right_start,
        block_right_end,
        block_x_shift,
        block_min_y_shift,
        warp_out_pos
    );

    if (grid.group_dim().z != 1) {
        shared_mem_rows_impl_dispatch<max_num_right_matrices, STRIDED_LOAD, true>(
            ctb,
            warp,
            block_num_right_matrices,
            args
        );
    } else {
        shared_mem_rows_impl_dispatch<max_num_right_matrices, STRIDED_LOAD, false>(
            ctb,
            warp,
            block_num_right_matrices,
            args
        );
    }
}

} // END anonymous namespace


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
) {
    if (shifts_per_cuda_block > 32) {
        throw std::runtime_error("Too many shifts per block: "s + std::to_string(shifts_per_cuda_block) + " (max 32)");
    }

    if (right_matrices_per_block == 0 || right_matrices_per_block > max_num_right_matrices) {
        throw std::runtime_error("Invalid number of right matrices per block: "s +
                                 std::to_string(right_matrices_per_block) +
                                 " [1-"s +
                                 std::to_string(max_num_right_matrices) +
                                 "]"s
        );
    }

    dsize_t num_matrix_groups = div_up(num_right_matrices, right_matrices_per_block);

    dsize_t max_num_column_groups = div_up(matrix_size.x, shared_mem_row_size);

    dim3 num_threads(warp_size, shifts_per_cuda_block);
    dim3 num_blocks(
        search_size.x * num_matrix_groups,
        div_up(search_size.y, num_threads.y),
        column_group_per_block ? max_num_column_groups : 1
    );

    dsize_t shared_mem_buffer_size = shared_mem_row_size * shared_mem_rows * sizeof(T);
    // Two buffers for ring buffer of submatrices from left input
    dsize_t shared_mem_size = (2 + right_matrices_per_block) * shared_mem_buffer_size;

    if (strided_load) {
        ccn_shift_per_warp_shared_mem<true><<<num_blocks, num_threads, shared_mem_size>>>(
            left,
            right,
            out,
            matrix_size,
            search_size,
            num_right_matrices,
            shared_mem_row_size,
            shared_mem_rows,
            right_matrices_per_block
        );
    } else {
        ccn_shift_per_warp_shared_mem<false><<<num_blocks, num_threads, shared_mem_size>>>(
            left,
            right,
            out,
            matrix_size,
            search_size,
            num_right_matrices,
            shared_mem_row_size,
            shared_mem_rows,
            right_matrices_per_block
        );
    }
}

template void run_ccn_shift_per_warp_shared_mem<int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
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

template void run_ccn_shift_per_warp_shared_mem<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
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

template void run_ccn_shift_per_warp_shared_mem<double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
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

}
