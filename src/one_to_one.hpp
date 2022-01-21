#pragma once

#include <vector>
#include <chrono>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class one_to_one: public cross_corr_alg<T, ALLOC> {
public:
    one_to_one(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T, ALLOC>(is_fft, num_measurements)
    {}

    static bool validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
        return rows > 0 && cols > 0 && left_matrices == 1 && right_matrices == 1;
    }

protected:
    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_one_to_one(this->refs(), this->targets());
    }
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class cpu_one_to_one: public one_to_one<T, ALLOC> {
public:
    cpu_one_to_one()
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return result_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(def_path);

        auto result_matrix_size = ref_.matrix_size() + target_.matrix_size() - 1;
        result_ = data_array<T, ALLOC>{result_matrix_size};
    }

    void run_impl() override {
        cpu_cross_corr_one_to_one(ref_, target_, result_);
    }

    void finalize_impl() override {
    }


private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_one_to_one<T, DEBUG, ALLOC>::labels{
    "Total",
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_one: public one_to_one<T, ALLOC> {
public:
    naive_original_alg_one_to_one()
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return result_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(def_path);
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_ref_,
                d_target_,
                d_result_,
                target_.matrix_size(),
                result_.matrix_size(),
                1,
                1
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_one_to_one<T, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_ring_buffer_row_alg: public one_to_one<T, ALLOC> {
public:
    naive_ring_buffer_row_alg()
        :one_to_one<T, ALLOC> (false, labels.size()), ref_(), target_(), res_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return res_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);
        res_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_res_, res_.size());

        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_ccn_ring_buffer_row(
                d_ref_,
                d_target_,
                d_res_,
                target_.matrix_size(),
                res_.matrix_size(),
                THREADS_PER_BLOCK
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(res_, d_res_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> res_;

    T* d_ref_;
    T* d_target_;
    T* d_res_;
};

template<typename T, dsize_t THREADS_PER_BLOCK, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_ring_buffer_row_alg<T, THREADS_PER_BLOCK, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_original_alg_one_to_one: public one_to_one<T, ALLOC> {
public:
    fft_original_alg_one_to_one()
        :one_to_one<T, ALLOC>(true, labels.size()), ref_(), target_(), result_(), fft_buffer_size_(0)
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return target_;
    }

    const data_array<T, ALLOC>& results() const override {
        return result_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {

        ref_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        if (ref_.matrix_size() != target_.matrix_size()) {
            throw std::runtime_error(
                "Invalid input matrix sizes, expected ref and target to be the same size: ref = "s +
                to_string(ref_.matrix_size()) +
                " target = "s +
                to_string(target_.matrix_size())
            );
        }

        // if (DEBUG)
        // {
        //     std::ofstream out_file("../data/ref.csv");
        //     ref_.store_to_csv(out_file);

        //     out_file = std::ofstream("../data/target.csv");
        //     target_.store_to_csv(out_file);
        // }

        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        result_ = data_array<T, ALLOC>{ref_.matrix_size()};

        fft_buffer_size_ = ref_.matrix_size().y * (ref_.matrix_size().x / 2 + 1);

        cuda_malloc(&d_inputs_, ref_.size() + target_.size());
        cuda_malloc(&d_result_, result_.size());

        // 2 * as we have two input matrices we are doing FFT on
        cuda_malloc(&d_inputs_fft_, 2 * fft_buffer_size_);

        cuda_memcpy_to_device(d_inputs_, ref_);
        cuda_memcpy_to_device(d_inputs_ + ref_.size(), target_);

        int sizes[2] = {static_cast<int>(ref_.matrix_size().y), static_cast<int>(ref_.matrix_size().x)};
        // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
        FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), 2));
        FFTCH(cufftPlan2d(&fft_inv_plan_, result_.matrix_size().y, result_.matrix_size().x, fft_type_C2R<T>()));
    }

    void run_impl() override {
        CPU_MEASURE(1,
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        // if (DEBUG)
        // {
        //     std::vector<fft_complex_t> tmp(fft_buffer_size_);
        //     cuda_memcpy_from_device(tmp.data(), d_ref_fft_, tmp.size());

        //     std::ofstream out("../data/ref_fft.csv");
        //     out << tmp << std::endl;

        //     cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

        //     out = std::ofstream("../data/target_fft.csv");
        //     out << tmp << std::endl;
        // }


        CUDA_MEASURE(2,
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_,
                {ref_.matrix_size().y, (ref_.matrix_size().x / 2) + 1},
                1,
                1,
                256)
        );

        // if (DEBUG)
        // {
        //     CUCH(cudaDeviceSynchronize());
        //     std::vector<fft_complex_t> tmp(fft_buffer_size_);
        //     cuda_memcpy_from_device(tmp.data(), d_target_fft_, tmp.size());

        //     std::ofstream out("../data/hadamard.csv");
        //     out << tmp << std::endl;
        // }

        CPU_MEASURE(3,
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_, d_result_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;
    data_array<T, ALLOC> result_;

    T* d_inputs_;
    T* d_result_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_inputs_fft_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_original_alg_one_to_one<T, DEBUG, ALLOC>::labels{
    "Total",
    "Forward FFT",
    "Hadamard",
    "Inverse FFT"
};


}