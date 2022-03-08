#pragma once

#include "cuda.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "types.cuh"
#include "helpers.cuh"

namespace cg = cooperative_groups;

namespace cross {

template <typename T>
__device__ T* shared_memory_proxy()
{
    // double2 to ensure 16B alignment
	extern __shared__ double2 memory[];
	return reinterpret_cast<T*>(memory);
}

template<typename T>
class shared_mem_buffer{
public:

    using value_type = T;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    __device__ shared_mem_buffer()
        :data_(nullptr), size_(0)
    {

    }

    __device__ shared_mem_buffer(T* data, dsize_t size)
        :data_(data), size_(size)
    {

    }

    __device__ shared_mem_buffer(shared_mem_buffer<T>&& r) noexcept
        :data_(r.data_), size_(r.size_)
    {
        r.data_ = nullptr;
        r.size_ = 0;
    }

    __device__ shared_mem_buffer<T>& operator =(shared_mem_buffer<T>&& r) noexcept
    {
        this->data_ = r.data_;
        this->size_ = r.size_;
        r.data_ = nullptr;
        r.size_ = 0;
        return *this;
    }

    __device__ static shared_mem_buffer<T> allocate(T** shared, dsize_t size) {
        T* data = *shared;
        *shared += size;
        return shared_mem_buffer<T>(data, size);
    }

    __device__ reference operator [](dsize_t i) {
        return data_[i];
    }

    __device__ const_reference operator [](dsize_t i) const {
        return data_[i];
    }

    __device__ size_type size() const {
        return size_;
    }

    // __device__ size_type load(const cg::thread_block& ctb, row_slice<T>&& slice, dsize_t size, dsize_t offset = 0) {
    //     // TODO: Asserts
    //     size_type copy_size = min(size, slice.size());
    //     auto data = data_ + offset;
    //     for (size_type i = ctb.thread_index().x; i < copy_size; i += ctb.size()) {
    //         data[i] = slice[i];
    //     }
    //     return copy_size;
    // }

    template<bool STRIDED_LOAD>
    __device__ size_type load_continuous(const cg::thread_block& ctb, const cg::thread_block_tile<32>& warp, const T* src, dsize_t size, dsize_t offset = 0) {
        if (STRIDED_LOAD) {
            load_continuous_chunk_strided_warps(ctb, src, size, offset);
        } else {
            load_continuous_chunk_continuous_warps(warp, src, size, offset);
        }
    }

    /**
     * Expects all warps to call this function, as each warp loads different part of the buffer
     * Each warp can call it at different times, but all warps have to call it before the buffer is loaded
     *
     * @param warp
     * @param src
     * @param size
     * @param offset
     * @return
     */
    __device__ size_type load_continuous_chunk_continuous_warps(
        const cg::thread_block_tile<32>& warp,
        const T* src,
        dsize_t size,
        dsize_t offset = 0
    ) {
        size_type copy_size = min(size, size_ - offset);
        auto data = data_ + offset;

        // Amount of data loaded by each warp (except the last)
        dsize_t warp_load_size = div_up(copy_size, warp.meta_group_size());
        dsize_t warp_start = warp.meta_group_rank() * warp_load_size;
        dsize_t warp_end = warp_start + warp_load_size;
        for (size_type i = warp_start + warp.thread_rank(); i < warp_end; i += warp.size()) {
            data[i] = src[i];
        }
        return copy_size;
    }

    __device__ size_type load_continuous_chunk_strided_warps(const cg::thread_block& ctb, const T* src, dsize_t size, dsize_t offset = 0) {
        size_type copy_size = min(size, size_ - offset);
        auto data = data_ + offset;
        for (size_type i = ctb.thread_rank(); i < copy_size; i += ctb.size()) {
            data[i] = src[i];
        }
        return copy_size;
    }

    template<bool STRIDED_LOAD>
    __device__ size_type load_strided_chunks(
        const cg::thread_block& ctb,
        const cg::thread_block_tile<32>& warp,
        const T* src,
        dsize_t chunk_size,
        dsize_t num_chunks,
        dsize_t chunk_stride,
        dsize_t offset_items = 0
    ) {
        if (STRIDED_LOAD) {
            return load_strided_chunks_strided_warps(ctb, src, chunk_size, num_chunks, chunk_stride, offset_items);
        } else {
            return load_strided_chunks_continuous_warps(warp, src, chunk_size, num_chunks, chunk_stride, offset_items);
        }
    }


    /**
     * Expects all warps to call this function, as each warp loads different part of the buffer
     * Each warp can call it at different times, but all warps have to call it before the buffer is loaded
     * @param ctb
     * @param src
     * @param chunk_size
     * @param num_chunks
     * @param chunk_stride
     * @param offset_items
     * @return
     */
    __device__ size_type load_strided_chunks_continuous_warps(
        const cg::thread_block_tile<32>& warp,
        const T* src,
        dsize_t chunk_size,
        dsize_t num_chunks,
        dsize_t chunk_stride,
        dsize_t offset_items = 0
    ) {
        size_type copy_size = min(chunk_size * num_chunks, size_ - offset_items);
        auto data = data_ + offset_items;

        // Amount of data loaded by each warp (except the last)
        dsize_t warp_load_size = div_up(copy_size, warp.meta_group_size());
        dsize_t warp_start = warp.meta_group_rank() * warp_load_size;
        dsize_t warp_end = min(warp_start + warp_load_size, copy_size);

//        if (ctb.group_index().x == 1 && ctb.group_index().y == 2 && warp.thread_rank() == 0) {
//            printf(
//                "Block: [%u, %u], Warp: %u, Chunk size: %u, Num chunks: %u, Chunk stride: %u, Copy size: %u, Warp load size: %u, Warp start: %u, Warp end: %u\n",
//                ctb.group_index().x,
//                ctb.group_index().y,
//                warp.meta_group_rank(),
//                chunk_size,
//                num_chunks,
//                chunk_stride,
//                copy_size,
//                warp_load_size,
//                warp_start,
//                warp_end
//            );
//        }

        for (size_type i = warp_start + warp.thread_rank(); i < warp_end; i += warp.size()) {
            auto chunk_idx = i / chunk_size;
            auto chunk_offset = i % chunk_size;
            auto src_idx =  (chunk_size + chunk_stride) * chunk_idx + chunk_offset;

//            if (ctb.group_index().x == 1 && ctb.group_index().y == 2 && num_chunks == 8) {
//                printf(
//                    "Block: [%u, %u], Warp: %u, i: %u, Src idx: %u, Src value: %f\n",
//                    ctb.group_index().x,
//                    ctb.group_index().y,
//                    warp.meta_group_rank(),
//                    i,
//                    src_idx,
//                    src[src_idx]
//                );
//            }

            data[i] = src[src_idx];
        }
        return copy_size;
    }

    __device__ size_type load_strided_chunks_strided_warps(
        const cg::thread_block& ctb,
        const T* src,
        dsize_t chunk_size,
        dsize_t num_chunks,
        dsize_t chunk_stride,
        dsize_t offset_items = 0
    ) {
        size_type copy_size = min(chunk_size * num_chunks, size_ - offset_items);
        auto data = data_ + offset_items;

        for (size_type i = ctb.thread_rank(); i < copy_size; i += ctb.size()) {
            auto chunk_idx = i / chunk_size;
            auto chunk_offset = i % chunk_size;
            auto src_idx =  (chunk_size + chunk_stride) * chunk_idx + chunk_offset;
            data[i] = src[src_idx];
        }
        return copy_size;
    }

    __device__ value_type* data() const {
        return data_;
    }
private:
    value_type* data_;
    dsize_t size_;
};

template<typename T>
__device__ void swap(shared_mem_buffer<T>& a, shared_mem_buffer<T>& b) {
    shared_mem_buffer<T> c{std::move(a)};
    a = std::move(b);
    b = std::move(c);
}

}
