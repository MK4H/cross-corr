#pragma once

#include <cuda_runtime.h>

#include "types.cuh"

namespace cross {
inline __device__ bool bounds_check(int idx, dsize_t size) {
    return idx >= 0 && idx < size;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, int idx, dsize_t size) {
    return bounds_check(idx, size) ? source[idx] : 0;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, int x, int y, dsize2_t size) {
    return bounds_check(x, size.x) && bounds_check(y, size.y) ? source[y * size.x + x] : 0;
}

/**
 * Loads data with bounds check. If out of bounds, returns 0
 *
 * @tparam T
 * @param source Source data array
 * @param idx Index to load from the source data array
 * @param size Size of the source array
 * @return source[idx] or 0 if idx is out of bounds
 */
template<typename T>
__device__ T load_with_bounds_check(const T* source, dsize2_t idx, dsize2_t size) {
    return load_with_bounds_check(source, idx, size.x, size.y, size);
}
}
