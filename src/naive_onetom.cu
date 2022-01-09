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

/**
 * This kernel is a reimplementation of the original naive cross_corr kernel
 * The kernel receives reference subregions, each in row major order all stacked one after another
 * into a single array "ref". "deformed" contains corresponding subregions from "batch_size" of the deformed  pictures
 * which are to be cross-correlated with the reference subregions. All subregions are in row major order, first
 * all subregions of the first deformed image, then all subregions of the second deformed image up to the "batch_size"th
 * deformed image. Number of subregions from the reference and all the deformed images is the same.
 * The input arrays ref and deformed contain only the subregions themselfs, and we must
 * clamp the computation to use only the overlapping parts.
 *
 * For each subregion we search an area of the size "search_size" for cross-correlation maximum.
 * The whole strip of deformed subregions is partitioned into a 16x16 CUDA blocks,
 * where each thread computes one possible shift of the reference image.
 * Output contains an an array of "search_size" results in row major order
 * corresponding to the result of cross correlation for each position in the search area.
 *
 * The memory access patterns are not ideal. Due to the 16x16 size of each block,
 * each half of the warp accesses different row of the "picture", most likely leading to two 128 byte
 * global memory accesses. The implementation also does not use shared memory in any way.
 */
template<typename T, typename RES>
__global__ void cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size

) {
    cg::thread_block ctb = cg::this_thread_block();

    // Coordinates in the whole strip of deformed subregions
    unsigned int def_strip_x = ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x;
    unsigned int def_strip_y = ctb.group_index().y * ctb.group_dim().y + ctb.thread_index().y;

    unsigned int region_idx = def_strip_x / search_size.x;

    if (region_idx >= subregions_per_pic || def_strip_y >= search_size.y) {
        return;
    }

    // Position of the centre of the subregion
    dsize2_t in_region_pos = { def_strip_x % search_size.x, def_strip_y };
    dsize_t ref_idx = region_idx % subregions_per_pic;
    dsize2_t half_size = (search_size - 1) / 2;

    vec2<int> shift = {(int)in_region_pos.x - (int)half_size.x, (int)in_region_pos.y - (int)half_size.y};

    ref += ref_idx * subregion_size.area();
    deformed += region_idx * subregion_size.area();
    out += region_idx * search_size.area();

    for (dsize_t i = 0; i < batch_size; ++i) {
        // The code is different from the original as here we are sliding the
        // deformed region over the reference region, whereas the original
        // did it the other way, which is incorrect in my opinion
        // or at least inconsistent with the text of the thesis
        // where it is defined as reference * deformed
        // and the algorithm clearly states that this means sliding the deformed
        //
        // The results also now match the results of matlab xcorr2
        dsize_t x_ref_start = max(shift.x, 0);
        dsize_t x_ref_end = min(subregion_size.x + shift.x, subregion_size.x);
        dsize_t y_ref_start = max(shift.y, 0);
        dsize_t y_ref_end = min(subregion_size.y + shift.y, subregion_size.y);

        RES sum = 0;
        for (dsize_t y_ref = y_ref_start; y_ref < y_ref_end; ++y_ref) {
            for (dsize_t x_ref = x_ref_start; x_ref < x_ref_end; ++x_ref) {
                // If deformed is shifted by -10, the we are starting from [0,0] in ref
                // and need to start from [10,10] in deformed, as there are 10
                // values to the left and on top outside the reference matrix
                int x_shifted = x_ref - shift.x;
                int y_shifted = y_ref - shift.y;

                sum += deformed[y_shifted * subregion_size.x + x_shifted] * ref[y_ref * subregion_size.x + x_ref];
            }
        }

        out[in_region_pos.linear_idx(search_size.x)] = sum;

        deformed += subregions_per_pic * subregion_size.area();
        out += subregions_per_pic * search_size.area();
    }
}


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

    ref_s.load(ctb, ref_mat, matrix_size.area());

    for (auto def_mat_idx = ctb.group_index().x; def_mat_idx < num_def_mats; def_mat_idx += g.group_dim().x) {
        auto def_offset = def_mat_idx * matrix_size.area();
        def_s.load(ctb, def_mats + def_offset, matrix_size.area());

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
void run_cross_corr_naive_original(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
) {
    dim3 num_threads(16, 16);
    dim3 num_blocks(
        div_up(search_size.x * subregions_per_pic, num_threads.x),
        div_up(search_size.y, num_threads.y)
    );

    // TODO: DEBUG
    //std::cout << "[" << num_blocks.x << ", " << num_blocks.y << "]\n";

    cross_corr_naive_original<<<num_blocks, num_threads>>>(
        ref,
        deformed,
        out,
        subregion_size,
        search_size,
        subregions_per_pic,
        batch_size
    );
}

template<typename T, typename RES>
void run_ccn_def_per_block(
    const T* __restrict__ ref_mat,
    const T* __restrict__ def_mats,
    RES* __restrict__ out_mats,
    dsize2_t matrix_size,
    dsize2_t search_size,
    dsize_t num_def_mats,
    dsize_t total_num_blocks,
    dsize_t threads_per_block
) {

    dsize_t shared_mem_size = 2 * matrix_size.area() * sizeof(T);

    ccn_def_per_block<<<total_num_blocks, threads_per_block, shared_mem_size>>>(
        ref_mat,
        def_mats,
        out_mats,
        matrix_size,
        search_size,
        num_def_mats
    );
}

template void run_cross_corr_naive_original<int, int>(
    const int* __restrict__ ref,
    const int* __restrict__ deformed,
    int* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<float, float>(
    const float* __restrict__ ref,
    const float* __restrict__ deformed,
    float* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

template void run_cross_corr_naive_original<double, double>(
    const double* __restrict__ ref,
    const double* __restrict__ deformed,
    double* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t subregions_per_pic,
    dsize_t batch_size
);

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