#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "shared_mem.cuh"
#include "types.cuh"
#include "matrix.cuh"
#include "clamps.cuh"
#include "helpers.cuh"

namespace cross {

template<typename T, typename RES>
__device__ void cross_corr_serial_shifts(
    cg::thread_block ctb,
    const matrix_slice<T>& m1,
    const matrix_slice<T>& m2,
    matrix_slice<RES>& out
) {
    auto warp = cg::tiled_partition<32>(ctb);
    auto half_size = vec2<int>((out.size() - 1) / 2);

    for (auto i = warp.meta_group_rank(); i < out.size().area(); i += warp.meta_group_size()) {
        vec2<int> out_pos{
            (int)(i % out.size().x),
            (int)(i / out.size().x)
        };
        vec2<int> shift = out_pos - half_size;

        // Part of m1 which overlaps m2 shifted by shift_x and shift_y
        auto m1_overlap = m1.submatrix_from_pos(
            clamp_to_nonnegative(shift.x),
            clamp_to_nonnegative(shift.y),
            clamp_down(m1.size().x + shift.x, m1.size().x),
            clamp_down(m2.size().y + shift.y, m2.size().y)
        );
        RES sum = 0;
        // TODO: This may lead to consistent divergence at the end of each row
        // maybe try to map linear index to position in the overlap matrix
        for (dsize_t y = m1_overlap.begin_y_src_idx(); y < m1_overlap.end_y_src_idx(); ++y) {
            for (dsize_t x = m1_overlap.begin_x_src_idx() + warp.thread_rank(); x < m1_overlap.end_x_src_idx(); x += warp.size()){
                // TODO: Check for overflow
                dsize2_t shifted{x - shift.x, y - shift.y};
                sum += m1[dsize2_t{x, y}] * m2[shifted];
            }
        }

        out[out_pos] = cg::reduce(warp, sum, cg::plus<RES>());
    }
}


template<typename T, typename RES>
__global__ void ccn_def_per_block(
    const T* __restrict__ ref_mat,
    const T* __restrict__ def_mats,
    RES* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats
) {
    cg::grid_group g = cg::this_grid();
    cg::thread_block ctb = cg::this_thread_block();
    T* shared = shared_memory_proxy<T>();

    shared_mem_buffer<T> ref_s = shared_mem_buffer<T>::allocate(&shared, matrix_size.area());
    shared_mem_buffer<T> def_s = shared_mem_buffer<T>::allocate(&shared, matrix_size.area());

    ref_s.load_continuous(ctb, ref_mat, matrix_size.area());

    for (auto def_mat_idx = ctb.group_index().x; def_mat_idx < num_def_mats; def_mat_idx += g.group_dim().x) {
        auto def_offset = def_mat_idx * matrix_size.area();
        def_s.load_continuous(ctb, def_mats + def_offset, matrix_size.area());

        auto out_offset = def_mat_idx * search_size.area();
        auto out_mat = matrix_slice<RES>::from_position_size(
            dsize2_t{0, 0},
            search_size,
            search_size.x,
            out_mats + out_offset
        );

        cross_corr_serial_shifts(
            ctb,
            matrix_slice<const T>::from_position_size(
                dsize2_t{0,0},
                matrix_size,
                matrix_size.x,
                ref_s.data()
            ),
            matrix_slice<const T>::from_position_size(
                dsize2_t{0,0},
                matrix_size,
                matrix_size.x,
                def_s.data()
            ),
            out_mat
        );
    }
}



template<typename T, typename RES>
void run_ccn_def_per_block(
    const T* __restrict__ ref_mat,
    const T* __restrict__ def_mats,
    RES* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t items_per_block,
    dsize_t threads_per_block
) {

    dsize_t shared_mem_size = 2 * matrix_size.area() * sizeof(T);

    dsize_t num_blocks = div_up(num_def_mats, items_per_block);

    ccn_def_per_block<<<num_blocks, threads_per_block, shared_mem_size>>>(
        ref_mat,
        def_mats,
        out_mats,
        matrix_size,
        search_size,
        num_def_mats
    );
}

template void run_ccn_def_per_block<int, int>(
    const int* __restrict__ ref_mat,
    const int* __restrict__ def_mats,
    int* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t total_num_blocks,
    dsize_t threads_per_block
);

template void run_ccn_def_per_block<float, float>(
    const float* __restrict__ ref_mat,
    const float* __restrict__ def_mats,
    float* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t total_num_blocks,
    dsize_t threads_per_block
);

template void run_ccn_def_per_block<double, double>(
    const double* __restrict__ ref_mat,
    const double* __restrict__ def_mats,
    double* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t total_num_blocks,
    dsize_t threads_per_block
);

}