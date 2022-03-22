#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <exception>
#include <cuda_runtime.h>
#include <vector>

#include "types.cuh"

namespace cross {
#define CUCH(status) cross::cuda_check(status, __LINE__, __FILE__, #status)

inline void cuda_check(cudaError_t status, int line, const char* src_filename, const char* line_str = nullptr)
{
    if (status != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA Error " << status << ":" << cudaGetErrorString(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
        std::cerr << ss.str();
        throw std::runtime_error(ss.str());
    }
}

// Divides two integers and rounds the result up
template<typename T, typename U>
inline __host__ __device__ T div_up(T a, U b)
{
    return (a + b - 1) / b;
}

/** Allocates device buffer large enough to hold \p num instances of T
 *
 * This helper prevents common error of forgetting the sizeof(T)
 * when allocating buffers
 */
template<typename T>
void cuda_malloc(T** p, dsize_t num) {
    CUCH(cudaMalloc(p, num * sizeof(T)));
}

template<typename T>
void cuda_memcpy_to_device(T* dst, T* src, dsize_t num) {
    CUCH(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename DATA>
void cuda_memcpy_to_device(typename DATA::value_type* dst, DATA& src) {
    cuda_memcpy_to_device(dst, src.data(), src.size());
}

template<typename T>
void cuda_memset(T* p, int value, dsize_t num_elements) {
    cudaMemset(p, value, num_elements * sizeof(T));
}

template<typename T>
void cuda_memcpy2D_to_device(T* dst, dsize_t dst_width, T* src, dsize_t src_width, dsize_t columns, dsize_t rows) {
    CUCH(cudaMemcpy2D(
        dst,
        dst_width * sizeof(T),
        src,
        src_width * sizeof(T),
        columns * sizeof(T),
        rows,
        cudaMemcpyHostToDevice));
}

template<typename DATA>
void cuda_memcpy2D_to_device(typename DATA::value_type* dst, dsize_t dst_width, DATA& src) {
    cuda_memcpy2D_to_device(dst, dst_width, src, src.matrix_size().x, src.matrix_size().x, src.matrix_size().y);
}

template<typename T>
void cuda_memcpy_from_device(T* dst, T* src, dsize_t num) {
    CUCH(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename DATA>
void cuda_memcpy_from_device(DATA& dst, typename DATA::value_type* src) {
    cuda_memcpy_from_device(dst.data(), src, dst.size());
}

inline std::ostream& operator<<(std::ostream& out, const std::vector<float2>& vec) {
    std::string sep = "";
    for(auto&& val: vec) {
        out << sep << "[" << val.x << "," << val.y <<"]";
        sep = ", ";
    }
    return out;
}

}
