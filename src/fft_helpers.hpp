#pragma once

#include <iostream>
#include <sstream>

#include <cufft.h>

namespace cross {


// Taken from original thesis src/cufft_helpers.hpp

#define FFTCH(status) cross::fft_check(status, __LINE__, __FILE__, #status)

static const char* cufft_get_error_message(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}

inline void fft_check(cufftResult status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != CUFFT_SUCCESS)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cufft_get_error_message(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cerr << ss.str();
		throw std::runtime_error(ss.str());
	}
}

template<typename T>
inline cufftType fft_type_R2C();
template<>
inline cufftType fft_type_R2C<float>()
{
    return cufftType::CUFFT_R2C;
}
template<>
inline cufftType fft_type_R2C<double>()
{
    return cufftType::CUFFT_D2Z;
}


template<typename T>
inline cufftType fft_type_C2R();
template<>
inline cufftType fft_type_C2R<float>()
{
    return cufftType::CUFFT_C2R;
}
template<>
inline cufftType fft_type_C2R<double>()
{
    return cufftType::CUFFT_Z2D;
}

template<typename T>
struct real_trait
{
    using type = float;
};

template<>
struct real_trait<float>
{
    using type = cufftReal;
};
template<>
struct real_trait<double>
{
    using type = cufftDoubleReal;
};

template<typename T>
struct complex_trait
{
    using type = float;
};

template<>
struct complex_trait<cufftReal>
{
    using type = cufftComplex;
};
template<>
struct complex_trait<cufftDoubleReal>
{
    using type = cufftDoubleComplex;
};

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in, typename complex_trait<T>::type* out);

template<>
inline void fft_real_to_complex<cufftReal>(cufftHandle plan, cufftReal* in, complex_trait<cufftReal>::type* out)
{
    FFTCH(cufftExecR2C(plan, in, out));
}
template<>
inline void fft_real_to_complex<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in, complex_trait<cufftDoubleReal>::type* out)
{
    FFTCH(cufftExecD2Z(plan, in, out));
}

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in_out);
template<>
inline void fft_real_to_complex<cufftReal>(cufftHandle plan, cufftReal* in_out)
{
    FFTCH(cufftExecR2C(plan, in_out, (cufftComplex*)in_out));
}
template<>
inline void fft_real_to_complex<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in_out)
{
    FFTCH(cufftExecD2Z(plan, in_out, (cufftDoubleComplex*)in_out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, typename complex_trait<T>::type* in, T* out);

template<>
inline void fft_complex_to_real<cufftReal>(cufftHandle plan, typename complex_trait<cufftReal>::type* in, float* out)
{
    FFTCH(cufftExecC2R(plan, in, out));
}
template<>
inline void fft_complex_to_real<cufftDoubleReal>(cufftHandle plan, typename complex_trait<cufftDoubleReal>::type* in, double* out)
{
    FFTCH(cufftExecZ2D(plan, in, out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, T* in_out);
template<>
inline void fft_complex_to_real<cufftReal>(cufftHandle plan, cufftReal* in_out)
{
    FFTCH(cufftExecC2R(plan, (cufftComplex*)in_out, in_out));
}
template<>
inline void fft_complex_to_real<cufftDoubleReal>(cufftHandle plan, cufftDoubleReal* in_out)
{
    FFTCH(cufftExecZ2D(plan, (cufftDoubleComplex*)in_out, in_out));
}

template<typename MAT_IN, typename MAT_OUT>
void copy_quadrant(const MAT_IN& src, MAT_OUT& tgt, dsize2_t src_pos, dsize2_t size, dsize2_t tgt_pos) {
    for (dsize_t y = 0; y < size.y; ++y) {
        for (dsize_t x = 0; x < size.x; ++x) {
            dsize2_t offset{x, y};
            // Normalize the results from cuFFT by dividing by the number of elements
            tgt[tgt_pos + offset] = src[src_pos + offset] / src.area();
        }
    }
}

template<typename MAT_IN, typename MAT_OUT>
void normalize_fft_results(const MAT_IN& res, MAT_OUT &&norm) {
    // We need to swap top left with bottom right and top right with bottom left quadrants
    // We also need to remove "empty" row and column both at res.size() / 2,
    // which split the res matrix into the afforementioned quadrants
    // So for example with res matrix being 20x20, the top left is 10x10
    // top right is 9x10, bottom left is 10x9 and bottom right is 9x9
    // and we need to put them into a 19x19 matrix
    // It fits, as both 19*19 = 361 = 10*10 + 9*10 + 10*9 + 9*9

    // For odd sized matrices, the empty row and column should also be at floor(res.size() / 2)
    // so for 19x19, they should be at index 9 and all quadrants should be the same 9x9

    // We also need to normalize the results, as cuFFT computes unnormalized FFT, so every
    // element of the resulting matrix is multiplied by the number of elements of the matrix
    // so for 20x20 result, each element is multiplied by 400

    // Row and column in the res matrix containing empty values
    // Also tells us the size of the top left quadrant
    auto empty_x = res.size().x / 2;
    auto empty_y = res.size().y / 2;

    // res top left quadrant
    copy_quadrant(res, norm,
        dsize2_t{0, 0},
        dsize2_t{empty_x, empty_y},
        dsize2_t{norm.size().x / 2, norm.size().y / 2}
    );
    // res top right quadrant
    copy_quadrant(res, norm,
        dsize2_t{empty_x + 1, 0},
        dsize2_t{res.size().x - empty_x - 1, empty_y},
        dsize2_t{0, norm.size().y / 2}
    );
    // res bottom left quadrant
    copy_quadrant(res, norm,
        dsize2_t{0, empty_y + 1},
        dsize2_t{empty_x, res.size().y - empty_y - 1},
        dsize2_t{norm.size().x / 2, 0}
    );
    // res bottom right quadrant
    copy_quadrant(res, norm,
        dsize2_t{empty_x + 1, empty_y + 1},
        dsize2_t{res.size().x - empty_x - 1, res.size().y - empty_y - 1},
        dsize2_t{0, 0}
    );
}

template<typename T, typename ALLOC>
data_array<T> normalize_fft_results(const data_array<T, ALLOC>& res) {
    data_array<T> norm{res.matrix_size() - 1, res.num_matrices()};
    for (dsize_t i = 0; i < norm.num_matrices(); ++i) {
        // TOOD: In parallel
        normalize_fft_results(res.view(i), norm.view(i));
    }

    return norm;
}

}