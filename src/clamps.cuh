#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "types.cuh"

namespace cross
{

template<typename T>
__device__ T clamp_up(const T& val, const T& low_bound) {
    return max(val, low_bound);
}

template<typename T>
__device__ T clamp_down(const T& val, const T&  high_bound) {
    return min(val, high_bound);
}

template<typename T>
__device__ T clamp(const T& val, const T& low_bound, const T& high_bound) {
    return clamp_up(clamp_down(val, high_bound), low_bound);
}

template<typename T, typename RES = dsize_t>
__device__ RES clamp_to_nonnegative(const T& val) {
    return (RES)clamp_up(val, 0);
}

}
