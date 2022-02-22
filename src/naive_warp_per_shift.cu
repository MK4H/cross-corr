#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <stdexcept>

#include "types.cuh"
#include "helpers.cuh"
#include "shared_mem.cuh"
#include "row_distribution.cuh"

namespace cg = cooperative_groups;

namespace cross {

constexpr unsigned int warp_size = 32;

template<typename T, typename RES>
__global__ void ccn_shift_per_warp(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_block = ctb.group_dim().y;

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_block,
        ctb.group_index().y
    };

    if (warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    dsize2_t right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    dsize2_t overlap_size = right_end - right_start;
    dsize_t total_items = overlap_size.area();

    RES sum = 0;
    // Simpler internal loop, as is done in simple_indexing version,
    // leads to high thread divergence and much slower overall speed
    // so even though this is bottlenecked by the index computations,
    // it still runs much faster
    for (dsize_t i = warp.thread_rank(); i < total_items; i += warp.size()) {
        dsize_t overlap_row =  i / overlap_size.x;
        dsize_t overlap_row_offset = i % overlap_size.x;

        dsize2_t right_idx = right_start + dsize2_t{overlap_row_offset, overlap_row};
        dsize2_t left_idx = dsize2_t{
            right_idx.x + warp_shift.x,
            right_idx.y + warp_shift.y
        };


        sum += left[left_idx.linear_idx(matrix_size.x)] * right[right_idx.linear_idx(matrix_size.x)];
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        out[warp_out_pos.linear_idx(search_size.x)] = sum;
    }
}

/**
 *
 * @tparam T
 * @tparam RES
 * @param left
 * @param right
 * @param out Buffer for the output matrix, MUST BE ZEROED out
 * @param matrix_size
 * @param search_size
 * @param max_rows_per_warp
 */
template<typename DIST, typename T, typename RES>
__global__ void ccn_shift_per_warp_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t max_rows_per_warp
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_block = ctb.group_dim().y;

    // Distribute rows of a single shift between multiple workers,
    // in this case warps
    // Return the assigned output row (which corresponds to a shift),
    // together with the number of workers computing this shift and
    // index of the current worker in range [0, number_of_workers_for_shift)
    assigned_work work = DIST::distribute_rows(
        ctb.group_index().y,
        max_rows_per_warp,
        matrix_size.y,
        search_size.y
    );

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_block,
        work.output_row
    };

    // Either explicit check here for workers with idx above workers_for_row, or clamp right_start to be at most shared_right_end
    if (work.worker_idx >= work.workers_for_row || warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    // Properties shared by all workers computing this shift
    dsize2_t shared_right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t shared_right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    dsize_t shared_overlapping_rows = shared_right_end.y - shared_right_start.y;
    dsize_t rows_per_worker = div_up(shared_overlapping_rows, work.workers_for_row);

    // Properties specific for the current worker
    dsize2_t right_start(
        shared_right_start.x,
        shared_right_start.y + work.worker_idx * rows_per_worker
    );

    dsize2_t right_end(
        shared_right_end.x,
        min(right_start.y + rows_per_worker, shared_right_end.y)
    );

    dsize2_t overlap_size = right_end - right_start;
    dsize_t total_items = overlap_size.area();

    RES sum = 0;
    for (dsize_t i = warp.thread_rank(); i < total_items; i += warp.size()) {
        dsize_t overlap_row =  i / overlap_size.x;
        dsize_t overlap_row_offset = i % overlap_size.x;

        dsize2_t right_idx = right_start + dsize2_t{overlap_row_offset, overlap_row};
        dsize2_t left_idx = dsize2_t{
            right_idx.x + warp_shift.x,
            right_idx.y + warp_shift.y
        };


        sum += left[left_idx.linear_idx(matrix_size.x)] * right[right_idx.linear_idx(matrix_size.x)];
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        atomicAdd(out + warp_out_pos.linear_idx(search_size.x), sum);
    }
}

template<typename T, typename RES>
__global__ void ccn_shift_per_warp_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize_t shifts_per_block = ctb.group_dim().y;

    dsize2_t warp_out_pos{
        ctb.thread_index().y + ctb.group_index().x * shifts_per_block,
        ctb.group_index().y
    };

    if (warp_out_pos.x >= search_size.x || warp_out_pos.y >= search_size.y) {
        return;
    }

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> warp_shift = {
        static_cast<int>(warp_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(warp_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    dsize2_t right_start(
        max(0, -warp_shift.x),
        max(0, -warp_shift.y)
    );

    dsize2_t right_end(
        min(matrix_size.x - warp_shift.x, matrix_size.x),
        min(matrix_size.y - warp_shift.y, matrix_size.y)
    );

    RES sum = 0;
    for (dsize_t right_y = right_start.y; right_y < right_end.y; ++right_y) {
        for (dsize_t right_x = right_start.x + warp.thread_rank(); right_x < right_end.x; right_x += warp.size()) {
            auto left_x = right_x + warp_shift.x;
            auto left_y = right_y + warp_shift.y;

            sum += left[left_y * matrix_size.x + left_x] * right[right_y * matrix_size.x + right_x];
        }
    }

    sum = cg::reduce(warp, sum, cg::plus<RES>());
    if (warp.thread_rank() == 0) {
        out[warp_out_pos.linear_idx(search_size.x)] = sum;
    }
}

/**
 * Each shift is computed by a single warp
 * This solution increases parallelism and thus occupancy massively, but has to use
 * shared memory to reduce the redundancy in reading from global memory
 *
 * We use 2D thread blocks, where first dimension is 32 so each is
 * a warp, and second dimension determines the offset in x where the warps of the thread
 * block compute consecutive shifts in x
 *
 * Each block computes ctb.num_threads() / 32 shifts consecutive in the x dimension
 * The grid is 2D, where the y dimensions should span search_size.y and x dimension should span
 * search_size.x / (ctb.num_threads() / 32)
 *
 * TODO: Allow subwarp granularity for shift computation
 */
//template<typename T, typename RES>
//__global__ void ccn_shift_per_warp_shared_mem(
//    const T* __restrict__ left,
//    const T* __restrict__ right,
//    RES* __restrict__ out,
//    dsize2_t matrix_size,
//    dsize2_t search_size,
//    dsize_t shared_mem_buffer_size
//) {
//    cg::thread_block ctb = cg::this_thread_block();
//    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);
//
//    dsize_t shifts_per_block = ctb.group_dim().y;
//
//    // First we compute the parts of source matrices required by any warp in the current block
//    // so that we can read all the required data cooperatively into shared memory
//    // and reuse it by all warps of the current block
//
//    // Output position (corresponding to shift) of the first warp of this block
//    dsize2_t first_warp_out_pos{
//        ctb.group_index().x * shifts_per_block,
//        ctb.group_index().y
//    };
//
//    dsize2_t last_warp_out_pos{
//        min(first_warp_out_pos.x + shifts_per_block, search_size.x),
//        first_warp_out_pos.y
//    };
//
//    dsize2_t half_search_size = (search_size - 1) / 2;
//
//    vec2<int> block_min_shift{
//        static_cast<int>(first_warp_out_pos.x) - static_cast<int>(half_search_size.x),
//        static_cast<int>(first_warp_out_pos.y) - static_cast<int>(half_search_size.y)
//    };
//
//    vec2<int> block_max_shift{
//        static_cast<int>(min(last_warp_out_pos.x, search_size.x)) - static_cast<int>(half_search_size.x),
//        static_cast<int>(min(last_warp_out_pos.y, search_size.y)) - static_cast<int>(half_search_size.y)
//    };
//
//    T* shared = shared_memory_proxy<T>();
//
//    shared_mem_buffer<T> left_s = shared_mem_buffer<T>::allocate(&shared, shared_mem_buffer_size);
//    shared_mem_buffer<T> right_s = shared_mem_buffer<T>::allocate(&shared, shared_mem_buffer_size);
//
//    // If the right matrix is shifted to the left, i.e. the shift is negative,
//    // the block does not use the first few rows and/or columns, as they are shifted
//    // outside the left matrix and do not overlap it
//    // For min [-2,-3] and max [-1,-2],
//    // the max shift uses more items from the left and top of the right matrix,
//    // so we need to derive start from the max shift
//    // If max shift is positive, the right matrix top left corner is inside the
//    // left matrix, so we need to start from index 0 of the right matrix
//    dsize2_t block_right_start(
//        max(0, -block_max_shift.x),
//        max(0, -block_max_shift.y)
//    );
//
//    // The right matrix end, i.e. the bottom right corner, is derived from the min
//    // shift, as max shift will use less of the matrix as it will be shifted more outside
//    // the left matrix overlap than with the min shift
//    //
//    // If min shift is negative, then the corner is inside the left matrix, as the matrices
//    // are the same size and the top left corner is outside, as described above in the right_start
//    //
//    // For positive shifts, the corner is outside, so we don't need to process the last few
//    // rows and/or columns
//    dsize2_t block_right_end(
//        min(matrix_size.x - block_min_shift.x, matrix_size.x),
//        min(matrix_size.y - block_min_shift.y, matrix_size.y)
//    );
//
//    dsize_t right_row_size = block_right_end.x - block_right_start.x;
//    dsize_t right_stride = matrix_size.x - block_right_end.x + block_right_start.x;
//
//
//
//    // The shared_mem_buffer_size MUST be larger than matrix_size.x, so at least a single row will
//    // fit into shared memory
//    dsize_t rows_per_load = shared_mem_buffer_size / row_size;
//
//
//    // Block is 32 by y threads, so x coordinate is thread rank in a warp, y coordinate is
//    // shift along input x axis compared to the first warp
//    dsize2_t warp_out_pos{
//        first_warp_out_pos.x + ctb.thread_index().y,
//        ctb.group_index().y
//    };
//
//    vec2<int> warp_shift(
//        block_min_shift.x + ctb.thread_index().y,
//        block_min_shift.y
//    );
//
//    // TODO: Simplify these computations as the only thing needed is the x offset
//    dsize2_t warp_right_start(
//        max(0, -warp_shift.x),
//        block_right_start.y
//    );
//
//    dsize2_t warp_right_end(
//        // TODO: Either cut warps with shifts that do not overlap the two matrices here
//        //  or try something like subthread_block group and stop them at the start of the kernel
//        //warp_out_pos.x < search_size.x ? min(matrix_size.x - warp_shift.x, matrix_size.x) : 0,
//        warp_shift.x < static_cast<int>(matrix_size.x) ? min(matrix_size.x - warp_shift.x, matrix_size.x) : 0,
//        block_right_end.y
//    );
//
//    // Offsets in the shared memory submatrix
//    // As all warps in a block process the same rows,
//    // we need to only track the offset on x axis
//    dsize_t warp_right_start_offset = warp_right_start.x - block_right_start.x;
//    dsize_t warp_right_end_offset = warp_right_end.x - block_right_start.x;
//
//    dsize_t warp_row_size = warp_right_end_offset - warp_right_start_offset;
//
//
//    RES sum = 0;
//    for (dsize_t loaded_row = block_right_start.y; loaded_row < block_right_end.y; loaded_row += rows_per_load) {
//        dsize2_t right_load_start{
//            block_right_start.x,
//            loaded_row
//        };
//
//        dsize2_t left_load_start = right_load_start + block
//
//        dsize_t num_rows = min(rows_per_load, block_right_end.y - right_load_start.y);
//        right_s.load_strided_chunks(
//            ctb,
//            right + right_load_start.linear_idx(matrix_size.x),
//            row_size * num_rows,
//            row_size,
//            stride
//        );
//
//
//
//        left_s.load_strided_chunks(
//            ctb,
//            left,
//            row_size * num_rows,
//            row_size,
//            stride
//        );
//
//        ctb.sync();
//
//        continue;
//
//        dsize_t warp_items = warp_row_size * num_rows;
//        for (dsize_t i = warp.thread_rank(); i < warp_items; i += warp.size()) {
//            dsize_t thread_row =  i / warp_row_size;
//            dsize_t thread_row_offset = i % warp_row_size;
//
//            dsize2_t right_idx{
//                warp_right_start_offset + thread_row_offset,
//                thread_row
//            };
//            dsize2_t left_idx{
//                right_idx.x + warp_shift.x,
//                thread_row
//            };
//
//            sum += left_s[left_idx.linear_idx(matrix_size.x)] * right_s[right_idx.linear_idx(matrix_size.x)];
//        }
//
//        sum = cg::reduce(warp, sum, cg::plus<RES>());
//        if (warp.thread_rank() == 0) {
//            atomicAdd(out + warp_out_pos.linear_idx(search_size.x), sum);
//        }
//    }
//}

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
    cg::thread_block_tile<warp_size> warp,
    const shared_mem_buffer<T>& left,
    const shared_mem_buffer<T>(&right)[NUM_RIGHT],
    int left_buffer_start_row,
    dsize_t right_buffer_start_row,
    dsize_t row_size,
    dsize_t warp_right_buffer_start_row,
    dsize_t warp_right_buffer_end_row,
    dsize_t num_left_loaded_rows,
    int warp_y_shift,
    RES (&sum)[NUM_RIGHT]
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
        auto l = left[right_idx + buffer_offset];
        for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
            sum[i] += l * right[i][right_idx];
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
    dsize_t shifts_per_block;
    dsize_t shared_mem_row_size;
    dsize_t shared_mem_rows;
    dsize2_t block_right_start;
    dsize2_t block_right_end;
    int block_x_shift;
    int block_min_y_shift;

    __device__ shared_mem_rows_impl_args(
        const T* __restrict__ left,
        const T* __restrict__ right,
        RES* __restrict__ out,
        dsize2_t matrix_size,
        dsize2_t search_size,
        dsize_t shifts_per_block,
        dsize_t shared_mem_row_size,
        dsize_t shared_mem_rows,
        dsize2_t block_right_start,
        dsize2_t block_right_end,
        int block_x_shift,
        int block_min_y_shift
    ) : left(left), right(right), out(out), matrix_size(matrix_size),
        search_size(search_size), shifts_per_block(shifts_per_block),
        shared_mem_row_size(shared_mem_row_size), shared_mem_rows(shared_mem_rows),
        block_right_start(block_right_start), block_right_end(block_right_end),
        block_x_shift(block_x_shift), block_min_y_shift(block_min_y_shift)
    {

    }
};

template<typename T, typename RES>
__device__ shared_mem_rows_impl_args<T, RES> create_shared_mem_rows_impl_args(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t shifts_per_block,
    dsize_t shared_mem_row_size,
    dsize_t shared_mem_rows,
    dsize2_t block_right_start,
    dsize2_t block_right_end,
    int block_x_shift,
    int block_min_y_shift
) {
    return shared_mem_rows_impl_args<T, RES>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        shifts_per_block,
        shared_mem_row_size,
        shared_mem_rows,
        block_right_start,
        block_right_end,
        block_x_shift,
        block_min_y_shift
    );
}

template<dsize_t NUM_RIGHT, bool STRIDED_LOAD, typename T, typename RES>
__device__ void shared_mem_rows_impl(
    cg::thread_block ctb,
    cg::thread_block_tile<warp_size> warp,
    shared_mem_rows_impl_args<T, RES> args
) {
    int warp_y_shift = args.block_min_y_shift + static_cast<int>(warp.meta_group_rank());

    dsize_t warp_right_start_row = max(0, -warp_y_shift);
    dsize_t warp_right_end_row = min(args.matrix_size.y - warp_y_shift,args. matrix_size.y);

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
    int left_buffer_preload_start_row = static_cast<int>(args.block_right_start.y) + args.block_min_y_shift;
    dsize_t left_src_preload_start_row = left_buffer_preload_start_row >= 0 ? left_buffer_preload_start_row : 0;
    dsize_t preload_offset_rows = left_buffer_preload_start_row >= 0 ? 0 : -left_buffer_preload_start_row;

    T* shared = shared_memory_proxy<T>();
    shared_mem_buffer<T> left_bottom_s = shared_mem_buffer<T>::allocate(&shared, args.shared_mem_row_size * args.shared_mem_rows);
    shared_mem_buffer<T> left_top_s = shared_mem_buffer<T>::allocate(&shared, args.shared_mem_row_size * args.shared_mem_rows);

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
                    args.right + right_load_start.linear_idx(args.matrix_size.x),
                    row_size,
                    right_load_size,
                    stride
                );
            }

            ctb.sync();

            dsize_t warp_right_buffer_start_row = max(warp_right_start_row, right_buffer_start_row) - right_buffer_start_row;
            dsize_t warp_right_buffer_end_row = min(warp_right_end_row, right_buffer_start_row + right_load_size) - right_buffer_start_row;

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

    dsize2_t warp_out_pos{
        ctb.group_index().x,
        ctb.group_index().y * args.shifts_per_block + warp.meta_group_rank()
    };

    for (dsize_t i = 0; i < NUM_RIGHT; ++i) {
        sum[i] = cg::reduce(warp, sum[i], cg::plus<RES>());
        if (warp.thread_rank() == 0) {
            args.out[warp_out_pos.linear_idx(args.search_size.x)] = sum[i];
        }
    }
}

constexpr dsize_t max_num_right_matrices = 8;

/**
 * In this implementation, the warps in the given block
 * compute the same shifts on the x axis, but on consecutive rows. This allows for
 * offset access to shared memory, which compared to strided access of implementations
 * with warps in the same block computed shifts with the same y axis does not lead
 * to bank conflicts.
 *
 * TODO: This does not work with shared_mem_rows < shift_per_block
 * TODO: Add version which iterates through all right matrices with the single left matrix buffer
 *  which will need to store per thread sums in shared memory using atomic instructions
 *
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
__global__ void ccn_shift_per_warp_shared_mem_rows(
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
    int block_min_y_shift = static_cast<int>(ctb.group_index().y * shifts_per_block) - static_cast<int>(half_search_size.y);
    // Shift of the last warp in the block
    int block_max_y_shift =
        min(
            static_cast<int>(ctb.group_index().y * shifts_per_block) + shifts_per_block - 1,
            static_cast<int>(search_size.y)
        ) -
        static_cast<int>(half_search_size.y);

    // Block is reading submatrices from right sequentially
    dsize2_t block_right_start(
        max(0, -block_x_shift),
        max(0, -block_max_y_shift)
    );

    dsize2_t block_right_end(
        min(matrix_size.x - block_x_shift, matrix_size.x),
        min(matrix_size.y - block_min_y_shift, matrix_size.y)
    );

    auto args = create_shared_mem_rows_impl_args(
        left,
        right + matrix_group_start_idx * matrix_size.area(),
        out + matrix_group_start_idx * search_size.area(),
        matrix_size,
        search_size,
        shifts_per_block,
        shared_mem_row_size,
        shared_mem_rows,
        block_right_start,
        block_right_end,
        block_x_shift,
        block_min_y_shift
    );

    switch (block_num_right_matrices) {
        case 1: shared_mem_rows_impl<1, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 2: shared_mem_rows_impl<2, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 3: shared_mem_rows_impl<3, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 4: shared_mem_rows_impl<4, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 5: shared_mem_rows_impl<5, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 6: shared_mem_rows_impl<6, STRIDED_LOAD>(ctb, warp, args);
            break;
        case 7: shared_mem_rows_impl<7, STRIDED_LOAD>(ctb, warp, args);
            break;
        case max_num_right_matrices: shared_mem_rows_impl<max_num_right_matrices, STRIDED_LOAD>(ctb, warp, args);
            break;
        default:
            assert(false);
    }
}

template<typename T, typename RES>
void run_ccn_shift_per_warp(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
) {
    if (cuda_shifts_per_block > 32) {
        throw std::runtime_error("Too many shifts per block: "s + std::to_string(cuda_shifts_per_block) + " (max 32)");
    }

    dim3 num_threads(32, cuda_shifts_per_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        search_size.y
    );

    ccn_shift_per_warp<<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size
    );
}

template<typename DIST, typename T, typename RES>
void run_ccn_shift_per_warp_work_distribution(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
) {
    if (cuda_shifts_per_block > 32) {
        throw std::runtime_error("Too many shifts per block: "s + std::to_string(cuda_shifts_per_block) + " (max 32)");
    }

    dsize_t num_workers = DIST::num_workers(max_rows_per_warp, matrix_size.y, search_size.y);

    dim3 num_threads(32, cuda_shifts_per_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        num_workers
    );

    ccn_shift_per_warp_work_distribution<DIST><<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size,
        max_rows_per_warp
    );
}

template<typename T, typename RES>
void run_ccn_shift_per_warp_simple_indexing(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
) {
    if (cuda_shifts_per_block > 32) {
        throw std::runtime_error("Too many shifts per block: "s + std::to_string(cuda_shifts_per_block) + " (max 32)");
    }

    dim3 num_threads(32, cuda_shifts_per_block);
    dim3 num_blocks(
        div_up(search_size.x, num_threads.y),
        search_size.y
    );

    ccn_shift_per_warp_simple_indexing<<<num_blocks, num_threads>>>(
        left,
        right,
        out,
        matrix_size,
        search_size
    );
}

//template<typename T, typename RES>
//void run_ccn_shift_per_warp_shared_mem(
//    const T* __restrict__ left,
//    const T* __restrict__ right,
//    RES* __restrict__ out,
//    dsize2_t matrix_size,
//    dsize2_t search_size,
//    dsize_t cuda_shifts_per_block,
//    dsize_t shared_mem_buffer_size
//) {
//    if (cuda_shifts_per_block > 32) {
//        throw std::runtime_error("Too many shifts per block: "s + std::to_string(cuda_shifts_per_block) + " (max 32)");
//    }
//
//    if (shared_mem_buffer_size < matrix_size.x) {
//        throw std::runtime_error("Shared memory buffer must be large enough to contain at least a single row of input");
//    }
//
//    dim3 num_threads(32, cuda_shifts_per_block);
//    dim3 num_blocks(
//        div_up(search_size.x, num_threads.y),
//        search_size.y
//    );
//
//    // There are two shared memory buffers, one for left and one for right input matrix
//    ccn_shift_per_warp_shared_mem<<<num_blocks, num_threads, 2 * shared_mem_buffer_size * sizeof(T)>>>(
//        left,
//        right,
//        out,
//        matrix_size,
//        search_size,
//        shared_mem_buffer_size
//    );
//}

template<typename T, typename RES>
void run_ccn_shift_per_warp_shared_mem_rows(
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
    bool strided_load
    // dsize_t cols_per_warp,
    // dsize_t rows_per_warp
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

    dim3 num_threads(32, shifts_per_cuda_block);
    dim3 num_blocks(
        search_size.x * div_up(num_right_matrices, right_matrices_per_block),
        div_up(search_size.y, num_threads.y)
    );

    dsize_t shared_mem_buffer_size = shared_mem_row_size * shared_mem_rows * sizeof(T);
    // Two buffers for ring buffer of submatrices from left input
    dsize_t shared_mem_size = (2 + right_matrices_per_block) * shared_mem_buffer_size;
    if (strided_load) {
        ccn_shift_per_warp_shared_mem_rows<true><<<num_blocks, num_threads, shared_mem_size>>>(
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
        ccn_shift_per_warp_shared_mem_rows<false><<<num_blocks, num_threads, shared_mem_size>>>(
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

template void run_ccn_shift_per_warp<int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp<double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp_simple_indexing<int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp_simple_indexing<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp_simple_indexing<double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block
);

template void run_ccn_shift_per_warp_work_distribution<triangle_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<triangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<triangle_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<rectangle_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<rectangle_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<rectangle_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<no_distribution, int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<no_distribution, float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

template void run_ccn_shift_per_warp_work_distribution<no_distribution, double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_shifts_per_block,
    dsize_t max_rows_per_warp
);

//template void run_ccn_shift_per_warp_shared_mem<int, int>(
//    const int* __restrict__ left,
//    const int* __restrict__ right,
//    int* __restrict__ out,
//    dsize2_t matrix_size,
//    dsize2_t search_size,
//    dsize_t cuda_shifts_per_block,
//    dsize_t shared_mem_buffer_size
//);
//
//template void run_ccn_shift_per_warp_shared_mem<float, float>(
//    const float* __restrict__ left,
//    const float* __restrict__ right,
//    float* __restrict__ out,
//    dsize2_t matrix_size,
//    dsize2_t search_size,
//    dsize_t cuda_shifts_per_block,
//    dsize_t shared_mem_buffer_size
//);
//
//template void run_ccn_shift_per_warp_shared_mem<double, double>(
//    const double* __restrict__ left,
//    const double* __restrict__ right,
//    double* __restrict__ out,
//    dsize2_t matrix_size,
//    dsize2_t search_size,
//    dsize_t cuda_shifts_per_block,
//    dsize_t shared_mem_buffer_size
//);

template void run_ccn_shift_per_warp_shared_mem_rows<int, int>(
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
    bool strided_load
);

template void run_ccn_shift_per_warp_shared_mem_rows<float, float>(
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
    bool strided_load
);

template void run_ccn_shift_per_warp_shared_mem_rows<double, double>(
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
    bool strided_load
);

}