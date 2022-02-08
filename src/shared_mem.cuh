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

    __device__ shared_mem_buffer(T* data, dsize_t size)
        :data_(data), size_(size)
    {

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

    // __device__ size_type load(cg::thread_block ctb, row_slice<T>&& slice, dsize_t size, dsize_t offset = 0) {
    //     // TODO: Asserts
    //     size_type copy_size = min(size, slice.size());
    //     auto data = data_ + offset;
    //     for (size_type i = ctb.thread_index().x; i < copy_size; i += ctb.size()) {
    //         data[i] = slice[i];
    //     }
    //     return copy_size;
    // }

    __device__ size_type load_continuous(cg::thread_block ctb, const T* src, dsize_t size, dsize_t offset = 0) {
        return load_warp_continuous_impl(ctb, src, size, offset);
    }

    __device__ size_type load_strided_chunks(
        cg::thread_block ctb,
        const T* src,
        dsize_t total_chunk_size,
        dsize_t chunk_size,
        dsize_t chunk_stride,
        dsize_t offset = 0
    ) {
        constexpr dsize_t warp_size = 32;
        auto warp = cg::tiled_partition<warp_size>(ctb);

        size_type copy_size = min(total_chunk_size, size_ - offset);
        auto data = data_ + offset;

        // Number of times each warp will load values
        auto warp_loads = div_up(copy_size, warp.size());
        auto warp_start = warp.meta_group_rank() * warp_loads * warp.size();
        auto warp_end = warp_start + warp_loads * warp.size();

        auto warp_offset_in_start_chunk = warp_start % chunk_size;

        for (size_type i = warp_start + warp.thread_rank(); i < warp_end; i += warp.size()) {
            auto chunk_idx = (i + warp_offset_in_start_chunk) / chunk_size;
            auto chunk_offset = (i + warp_offset_in_start_chunk) % chunk_size;
            auto data_idx =  (chunk_size + chunk_stride) * chunk_idx + chunk_offset;
            data[data_idx] = src[i];
        }
        return copy_size;
    }

    __device__ value_type* data() const {
        return data_;
    }
private:
    value_type* data_;
    dsize_t size_;

    __device__ size_type load_warp_continuous_impl(cg::thread_block ctb, const T* src, dsize_t size, dsize_t offset = 0) {
        constexpr dsize_t warp_size = 32;
        auto warp = cg::tiled_partition<warp_size>(ctb);

        size_type copy_size = min(size, size_ - offset);
        auto data = data_ + offset;

        // Number of times each warp will load values
        auto warp_loads = div_up(copy_size, warp.size());
        auto warp_start = warp.meta_group_rank() * warp_loads * warp.size();
        auto warp_end = warp_start + warp_loads * warp.size();
        for (size_type i = warp_start + warp.thread_rank(); i < warp_end; i += warp.size()) {
            data[i] = src[i];
        }
        return copy_size;
    }

    __device__ size_type load_warp_strided_impl(cg::thread_block ctb, const T* src, dsize_t size, dsize_t offset = 0) {
        size_type copy_size = min(size, size_ - offset);
        auto data = data_ + offset;
        for (size_type i = ctb.thread_index().x; i < copy_size; i += ctb.size()) {
            data[i] = src[i];
        }
        return copy_size;
    }
};


}