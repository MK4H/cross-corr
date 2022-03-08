#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "helpers.cuh"
#include "shared_mem.cuh"
#include "matrix.cuh"
#include "clamps.cuh"

namespace cg = cooperative_groups;

namespace cross
{


/*
TODO: Try loading the row of reference subregion to shared memory
then do the multiplication of each ref row element with each deformed row element,
either materializing this matrix or computing it during a sum.

As each pair of ref and deformed are multiplied only in a signle shift, it may be better to just
do it during the sum computation.

So with single dimensional blocks, we just load the ref row (or part of it) and
each thread goes linearily through it's deformed row and computes the
multiplication of each element with the corresponding ref element.

Threads iterate this computation over the ref elemens to use shared memory broadcast.
So that each thread in a warp accesses the same ref element and multiplies it with the elements
in its deformed row.

As each thread in a warp will be processing a successive elements of the deformed row,
we could do thread shuffle to reduce the accesses to global memory.


TODO: Try using the whole warp for computation of a single shift
*/

template<typename T>
__device__ dsize_t load_row_chunk(const cg::thread_block& ctb, T* dst, const T* src, dsize_t row_start, dsize_t row_size, dsize_t chunk_start, dsize_t chunk_size) {
    dsize_t copy_size = min(row_size - chunk_start, chunk_size);
    for (; chunk_start + ctb.thread_index().x < copy_size; chunk_start += ctb.size()) {
        dst[ctb.thread_index().x] = src[row_start + chunk_start + ctb.thread_index().x];
    }
    return copy_size;
}

/** Limited to one reference subregion and one deformed subregion
 *
 * This is just a first simple implementation, future implementations will build on top
 * of this and optimize it
 * Each block processes one row of the ref matrix
 *
 * Each block loads a <block_size> number of elements from a row of the reference subregion and
 * then goes through the deformed subregion, loading it in <block_size> chunks into shared memory.
 *
 * Each
 */
// template<typename T, typename RES>
// __global__ void ccn_shared_mem_rows(
//     const T* __restrict__ ref,
//     const T* __restrict__ deformed,
//     RES* __restrict__ out,
//     dsize2_t subregion_size,
//     dsize2_t search_size
// ) {
//     cg::thread_block ctb = cg::this_thread_block();

//     const dsize_t ref_chunk_size = ctb.size();
//     const dsize_t def_chunk_size = ctb.size();

//     // Split shared memory
//     extern __shared__ T shared[];
//     T* ref_s = shared;
//     T* def_s = ref_s + ref_chunk_size;
//     RES* shift_sums = def_s + 2*def_chunk_size + ref_chunk_size;

//     dsize2_t half_search_size = (search_size - 1) / 2;

//     const dsize_t ref_row = ctb.group_index();
//     const dsize_t ref_row_start = ref_row * subregion_size.x;

//     for (dsize_t ref_chunk_start = 0; ref_chunk_start < subregion_size.x; ref_chunk_start += ref_chunk_size) {
//         // Load part of the ref row into shared memory
//         load_row_chunk(ctb, ref_s, ref, ref_row_start, subregion_size.x, ref_chunk_start, ref_chunk_size);
//         ctb.sync();

//         // Load ref chunk into thread registers
//         // T ref_val = ref_chunk_start + ctb.thread_index().x < subregion_size.x ?
//         //     ref[ref_row_start + ref_chunk_start + ctb.thread_index().x] :
//         //     0;

//         // First deformed row to compute cc with
//         const dsize_t def_begin_row = max((int)ref_row - (int)half_search_size.y, 0);
//         const dsize_t def_end_row = min(ref_row + half_search_size.y, subregion_size.y);
//         for (dsize_t def_row = def_begin_row; def_row < def_end_row; ++def_row) {

//             const dsize_t def_row_start = def_row * subregion_size.x;
//             // For given ref row chunk, we only need to load parts of the def row
//             // that are at most search_size.x before or after
//             const dsize_t def_row_part_start = max(def_row_start, (int)def_row_start + ref_chunk_start - half_search_size.x);
//             const dsize_t def_row_part_end = min(def_row_start + subregion_size.x, def_row_part_start + ref_chunk_size + search_size.x);
//             const dsize_t def_row_part_size = def_row_part_end - def_row_part_start;

//             for (dsize_t def_chunk_start = 0; def_chunk_start < def_row_part_size; def_chunk_start += def_chunk_size) {
//                 load_row_chunk(ctb, def_s, deformed, def_row_part_start, def_row_part_size, def_chunk_start, def_chunk_size);
//                 ctb.sync();



//                 auto ctw = cg::tiled_partition<32>(ctb);
//                 auto num_warps = ctw.meta_group_size();
//                 RES shift_sum = 0;
//                 for (dsize_t shift = 0; shift < 32; ++shift) {

//                 }

//                 // TODO: Add the number of threads as param


//             }
//         }
//     }
// }

/** Two part ring buffer
 *
 * This buffer holds two submatrices, which should be neighbours in the source matrix.
 * When loading, the older submatrix is overwritten
 *
 * matrices are stored in row major order and should be continuous in the x axis
 *
 * Currently this datastructure is designed to work with blocks with 32x32 threads.
 */
// template<typename T>
// class ring_buffer {
// public:
//     __host__ __device__ ring_buffer(T* data, dsize2_t size)
//         :data_(data), size_(size)
//     {

//     }

//     __device__ void load(const cg::thread_block& ctb, const T* src, dsize2_t pos) {

//     }

//     __host__ __device__ T operator[](dsize2_t pos) {

//     }
// private:
//     T* data_;
//     dsize2_t pos_;
//     dsize2_t size_;
// };

dsize_t __device__ get_x_def_block_start(dsize_t block_start_x, int max_block_shift_x) {
    return max((int)block_start_x - max_block_shift_x, 0);
}

/**
 * This kernel takes advantage of the access pattern in the original implementation
 * where threads in the same thread block read 15/16 of the same values as different
 * threads from the same thread block in previous step. In the original implementation
 * it is done in the form of 16x16 matrix, which shares 15 columns with the previous matrix.
 *
 * Here we use 1D thread block, so each step would share all but one value from ref and def
 * matrices from the previous step. We use this to preload the next thread block worth of
 * data in one step to shared memory and use this to prevent repeated accesses to global memory,
 * even if those might have been cached.
 *
 * Whereas all threads in a single block access the same <thread_block_size> values from ref,
 * they start at <thread_block_size> different values in the corresponding def row and read
 * another <thread_block_size> values from their initial value. This means that each loop,
 * the whole block accesses <thread_block_size> values from ref but 2x<thread_block_size> values
 * from def. Important note here is that half of the def values is accessed in the next loop too,
 * so we need something like a ring buffer where each loop we load just half of the 2x<thread_block_size>
 * values, allowing us to do this load in one step.
 *
 * TODO: The following is no longer true
 * All threads read from the same <thread_block_size> values of ref
 * mostly using broadcast, but multiplying them with different but continuous
 * values from def matrix
 * If perfectly aligned, they would all do broadcast from ref and then each read
 * different value from def, where consecutive threads would read consecutive values
 */
template<typename T, typename RES>
__global__ void ccn_ring_buffer_row(
    const T* __restrict__ ref,
    const T* __restrict__ def,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size
) {
    // TODO: Check block-wide shuffle down from cooperative groups


    auto ref_mat = matrix_slice<const T>::from_position_size(
        dsize2_t{0,0},
        subregion_size,
        subregion_size.x,
        ref
    );

    auto def_mat = matrix_slice<const T>::from_position_size(
        dsize2_t{0,0},
        subregion_size,
        subregion_size.x,
        def
    );

    cg::thread_block ctb = cg::this_thread_block();

    // Each offset between ref and def is computed by a single thread
    // so each thread corresponds to a single value in the resulting matrix
    // same as the original
    // As this kernel is simplified and only has 1D thread blocks, each thread block
    // works in a single row
    dsize2_t result_pos{
        ctb.group_index().x * ctb.group_dim().x + ctb.thread_index().x,
        ctb.group_index().y * ctb.group_dim().y
    };

    // Index of the leftmost thread in the block
    unsigned int block_res_x = ctb.group_index().x * ctb.group_dim().x;

    T* shared = shared_memory_proxy<T>();

    shared_mem_buffer<T> ref_s = shared_mem_buffer<T>::allocate(&shared, ctb.size() * 2);
    shared_mem_buffer<T> def_s = shared_mem_buffer<T>::allocate(&shared, ctb.size() * 2);

    dsize2_t half_size = (search_size - 1) / 2;
    // As thread indicies are only positive, and half of the resulting matrix
    // represents negative shifts, we need to shift by half the search size to left
    vec2<int> shift{(int)result_pos.x - (int)half_size.x, (int)result_pos.y - (int)half_size.y};

    // Minimal shift computed by any thread in the same block
    // As all threads share shift in the y axis, and value of shift along the x axis
    // is dependent on threadId in x axis, thread 0 will always have the lowest shift
    vec2<int> min_block_shift{(int)block_res_x - (int)half_size.x, (int)result_pos.y - (int)half_size.y};

    // Similarly with the min shift, max shift will always be computed by the
    // last thread in the current block
    // ctb.size() - 1 is index of the last thread in this block
    vec2<int> max_block_shift{min_block_shift.x + (int)ctb.size() - 1, min_block_shift.y};

    // Slice of the ref matrix which overlaps with the deformed matrix shifted by
    // <shift> and thus needs to be computed by the current thread
    // Specific for current thread
    auto ref_slice = matrix_slice<const T>::from_positions(
        dsize2_t{clamp_to_nonnegative(shift.x), clamp_to_nonnegative(shift.y)},
        dsize2_t{
            clamp_down(subregion_size.x + shift.x, subregion_size.x),
            clamp_down(subregion_size.y + shift.y, subregion_size.y)
        },
        subregion_size.x,
        ref
    );

    // Slice of ref matrix containing union of all ref_slices for all threads within current thread block
    // Basically part of the ref matrix accessed by any thread from the current thread block
    auto ref_slice_block = matrix_slice<const T>::from_positions(
        dsize2_t{
            clamp_to_nonnegative(min_block_shift.x),
            ref_slice.begin_y_src_idx()},
        dsize2_t{
            clamp_down(subregion_size.x + max_block_shift.x, subregion_size.x),
            ref_slice.end_y_src_idx()
        },
        subregion_size.x,
        ref
    );


    RES sum = 0;
    for (dsize_t row = 0; row < ref_slice_block.size().y; ++row) {
        // As we are only going through the parts of ref matrix that overlap def matrix
        // any shifted_. indicies should be valid
        int def_y = ref_slice_block.begin_y_src_idx() + row - shift.y;
        auto def_row = def_mat.row(def_y);
        // TODO: Most things are shared between rows, so no need to recompute them each time
        auto ref_buffer = make_row_ring_buffer<2>(
            ctb,
            ref_slice_block.row(row),
            std::move(ref_s)
        );

        // When def is shifted by -5 on x, and we access item 0 in ref, we want item 5 in def,
        // so ref.x - shift.x gives us the index in def matrix
        // Preloads the first 2 parts of the buffer
        dsize_t def_row_start_idx = max((int)ref_buffer.start_offset() - max_block_shift.x, 0);
        auto def_buffer = make_row_ring_buffer<2>(
            ctb,
            def_row.subslice(
                min(ref_slice_block.size().x, subregion_size.x - def_row_start_idx),
                def_row_start_idx
            ),
            std::move(def_s)
        );

        // Relative offsets of the two buffers stay the same during the whole row processing
        int def_buffer_thread_offset = (int)ref_buffer.start_offset() - (int)def_buffer.start_offset() - shift.x;
        do {
            // Sync after load
            ctb.sync();

            // Indicies in the ref_s buffer which should be processed by the current thread
            dsize_t ref_buffer_thread_start = max((int)ref_slice.begin_x_src_idx() - (int)ref_buffer.start_offset(), 0);
            dsize_t ref_buffer_thread_end = min(
                (int)ref_slice.end_x_src_idx() - (int)ref_buffer.start_offset(),
                min(
                    ref_buffer.num_loaded(),
                    (int)def_buffer.num_loaded() - ((int)ref_buffer_thread_start + (int)def_buffer_thread_offset)
                )
            );



            // if (ctb.group_index().y == 0 && ctb.thread_rank() == 0) {
            //     printf("Block: %u, Shift: %d, Start: %u, End: %u, Offset: %d, Ref start: %u, Def start: %u\n", ctb.group_index().x,  shift.x, ref_buffer_thread_start, ref_buffer_thread_end, def_buffer_thread_offset, ref_buffer.start_offset(), def_buffer.start_offset());
            // }

            for (dsize_t ref_buffer_index = ref_buffer_thread_start; ref_buffer_index < ref_buffer_thread_end; ++ref_buffer_index) {

                sum += ref_buffer[ref_buffer_index] * def_buffer[(int)ref_buffer_index + def_buffer_thread_offset];
            }



            // Sync after computation
            ctb.sync();
        } while (ref_buffer.load_next(ctb) && def_buffer.load_next(ctb));
    }

    if (result_pos.x < search_size.x) {
        out[result_pos.linear_idx(search_size.x)] = sum;
    }
}

/** Ring buffers
 *
 * The original implementation does pretty chaotic memory accesses and code divergence due to the x_ref and y_ref clamping
 *
 * We would like to manipulate the original implementation so that the whole 16x16 threadblock accesses two 16x16
 * submatrices each cycle and does elementwise multiplication, where each thread gets
 * one result element and adds it to its private sum
 *
 * The following cycle the block accesses two 16x16 matrices, which each share
 * 16x15 submatrix with the previous matrix. We could basically just do a ring buffer,
 * where the first warp adds to the ringbuffer each cycle, overwriting the oldest data
 *

 */
// template<typename T, typename RES>
// __global__ void ccn_ring_buffer_matrix(
//     const T* __restrict__ ref,
//     const T* __restrict__ deformed,
//     RES* __restrict__ out,
//     dsize2_t subregion_size,
//     dsize2_t search_size
// ) {
//     // TODO: Check block-wide shuffle down from cooperative groups


//     cg::thread_block ctb = cg::this_thread_block();

//     const dsize_t buffer_size = ctb.size() * 2;

//     extern __shared__ T shared[];

//     //
//     for (dsize_t )
// }




template<typename T, typename RES>
void run_ccn_ring_buffer_row(
    const T* __restrict__ ref,
    const T* __restrict__ deformed,
    RES* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
) {
    // TODO: Launch multiple kernels
    dim3 num_blocks(
        div_up(search_size.x, threads_per_block),
        search_size.y
    );

    std::cout << "[" << num_blocks.x << ", " << num_blocks.y << "]\n";

    dsize_t shared_mem_size = 4 * threads_per_block * sizeof(T);

    ccn_ring_buffer_row<<<num_blocks, threads_per_block, shared_mem_size>>>(
        ref,
        deformed,
        out,
        subregion_size,
        search_size
    );
}

template void run_ccn_ring_buffer_row<int, int>(
    const int* __restrict__ ref,
    const int* __restrict__ deformed,
    int* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

template void run_ccn_ring_buffer_row<float, float>(
    const float* __restrict__ ref,
    const float* __restrict__ deformed,
    float* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

template void run_ccn_ring_buffer_row<double, double>(
    const double* __restrict__ ref,
    const double* __restrict__ deformed,
    double* __restrict__ out,
    dsize2_t subregion_size,
    dsize2_t search_size,
    dsize_t threads_per_block
);

}
