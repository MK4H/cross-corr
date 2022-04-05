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

namespace cg = cooperative_groups;

namespace cross {

constexpr unsigned int warp_size = 32;

template<typename T, typename RES>
__global__ void ccn_shift_per_block(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size
) {
    cg::thread_block ctb = cg::this_thread_block();
    cg::thread_block_tile<warp_size> warp = cg::tiled_partition<warp_size>(ctb);

    dsize2_t block_out_pos{
        ctb.group_index().x,
        ctb.group_index().y
    };

    dsize2_t half_search_size = (search_size - 1) / 2;

    vec2<int> block_shift = {
        static_cast<int>(block_out_pos.x) - static_cast<int>(half_search_size.x),
        static_cast<int>(block_out_pos.y) - static_cast<int>(half_search_size.y)
    };

    dsize2_t right_start(
        max(0, -block_shift.x),
        max(0, -block_shift.y)
    );

    dsize2_t right_end(
        min(matrix_size.x - block_shift.x, matrix_size.x),
        min(matrix_size.y - block_shift.y, matrix_size.y)
    );

    dsize2_t overlap_size = right_end - right_start;
    dsize_t total_items = overlap_size.area();

    RES sum = 0;
    // Simpler internal loop, as is done in simple_indexing version,
    // leads to high thread divergence and much slower overall speed
    // so even though this is bottlenecked by the index computations,
    // it still runs much faster
    for (dsize_t i = ctb.thread_rank(); i < total_items; i += ctb.size()) {
        dsize_t overlap_row =  i / overlap_size.x;
        dsize_t overlap_row_offset = i % overlap_size.x;

        dsize2_t right_idx = right_start + dsize2_t{overlap_row_offset, overlap_row};
        dsize2_t left_idx = dsize2_t{
            right_idx.x + block_shift.x,
            right_idx.y + block_shift.y
        };


        sum += left[left_idx.linear_idx(matrix_size.x)] * right[right_idx.linear_idx(matrix_size.x)];
    }

    // Reduce in each warp
    sum = cg::reduce(warp, sum, cg::plus<RES>());
    RES* shared = shared_memory_proxy<RES>();

    // First thread of each warp writes the warp sum into shared memory
    if (warp.thread_rank() == 0) {
        shared[warp.meta_group_rank()] = sum;
    }
    ctb.sync();

    // The first warp of the block reduces the values from shared memory into a single result
    if (warp.meta_group_rank() == 0) {
        // TODO: This expects max size of thread block to be 1024, which is true for all
        //  current compute capabilities
        if (warp.thread_rank() < warp.meta_group_size()) {
            sum = shared[warp.thread_rank()];
        } else {
            sum = 0;
        }
        sum = cg::reduce(warp, sum, cg::plus<RES>());

        if (warp.thread_rank() == 0) {
            out[block_out_pos.linear_idx(search_size.x)] = sum;
        }
    }
}

template<typename T, typename RES>
void run_ccn_shift_per_block(
    const T* __restrict__ left,
    const T* __restrict__ right,
    RES* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
) {
    if (cuda_block_size > 1024) {
        throw std::runtime_error("CUDA block too large: "s + std::to_string(cuda_block_size) + " (max 1024)");
    }

    dim3 num_threads(cuda_block_size);
    dim3 num_blocks(
        search_size.x,
        search_size.y
    );

    // One item for each wapr in a block
    dsize_t shared_mem_size = (cuda_block_size / warp_size) * sizeof(RES);
    ccn_shift_per_block<<<num_blocks, num_threads, shared_mem_size>>>(
        left,
        right,
        out,
        matrix_size,
        search_size
    );
}

template void run_ccn_shift_per_block<int, int>(
    const int* __restrict__ left,
    const int* __restrict__ right,
    int* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
);

template void run_ccn_shift_per_block<float, float>(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
);

template void run_ccn_shift_per_block<double, double>(
    const double* __restrict__ left,
    const double* __restrict__ right,
    double* __restrict__ out,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t cuda_block_size
);

}
