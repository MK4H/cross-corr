#pragma once

#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"

#include "kernels.cuh"

using namespace std::string_literals;

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class n_to_mn: public cross_corr_alg<T, ALLOC> {
public:
    n_to_mn(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_n_to_mn(this->refs(), this->targets());
    }
protected:
    static void check_input_sizes_divisible(dsize_t ref_num_matrices, dsize_t target_num_matrices) {
        if (target_num_matrices % ref_num_matrices != 0) {
            throw std::runtime_error(
                "Invalid input data counts, "s +
                std::to_string(target_num_matrices) +
                " is not divisible by "s +
                std::to_string(ref_num_matrices)
            );
        }
    }
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class cpu_n_to_mn: public n_to_mn<T, ALLOC> {
public:
    cpu_n_to_mn()
        :n_to_mn<T, ALLOC>(false, labels.size()), refs_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_n_to_mn(refs_, targets_, results_);
    }

    void finalize_impl() override {
    }


private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_n_to_mn<T, DEBUG, ALLOC>::labels{
    "Total",
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_n_to_mn: public n_to_mn<T, ALLOC> {
public:
    naive_original_alg_n_to_mn()
        :n_to_mn<T, ALLOC>(false, labels.size()), refs_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());

        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(1,
            run_cross_corr_naive_original(
                d_refs_,
                d_targets_,
                d_results_,
                targets_.matrix_size(),
                results_.matrix_size(),
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices()
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_n_to_mn<T, DEBUG, ALLOC>::labels{
    "Total",
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_original_alg_n_to_mn: public n_to_mn<T, ALLOC> {
public:
    fft_original_alg_n_to_mn()
        :n_to_mn<T, ALLOC>(true, labels.size()), refs_(), targets_(), results_(), fft_buffer_size_(0)
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    const std::vector<std::string>& measurement_labels() const override {
        return labels;
    }

protected:
    void prepare_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {

        refs_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());

        if (refs_.matrix_size() != targets_.matrix_size()) {
            throw std::runtime_error(
                "Invalid input matrix sizes, expected ref and target to be the same size: ref = "s +
                to_string(refs_.matrix_size()) +
                " target = "s +
                to_string(targets_.matrix_size())
            );
        }

        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        results_ = data_array<T, ALLOC>{refs_.matrix_size(), targets_.num_matrices()};

        fft_buffer_size_ = refs_.matrix_size().y * (refs_.matrix_size().x / 2 + 1);

        cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
        cuda_malloc(&d_results_, results_.size());

        auto num_inputs = refs_.num_matrices() + targets_.num_matrices();

        cuda_malloc(&d_inputs_fft_, num_inputs * fft_buffer_size_);

        cuda_memcpy_to_device(d_inputs_, refs_);
        cuda_memcpy_to_device(d_inputs_ + refs_.size(), targets_);

        int input_sizes[2] = {static_cast<int>(refs_.matrix_size().y), static_cast<int>(refs_.matrix_size().x)};
        // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
        FFTCH(cufftPlanMany(&fft_plan_, 2, input_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs));

        int result_sizes[2] = {static_cast<int>(results_.matrix_size().y), static_cast<int>(results_.matrix_size().x)};
        FFTCH(cufftPlanMany(&fft_inv_plan_, 2, result_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
    }

    void run_impl() override {
        CPU_MEASURE(1,
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        CUDA_MEASURE(2,
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                {refs_.matrix_size().y, (refs_.matrix_size().x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices(),
                // TODO: Number of threads
                256)
        );

        CUDA_MEASURE(3,
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(), d_results_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_inputs_;
    T* d_results_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_inputs_fft_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_original_alg_n_to_mn<T, DEBUG, ALLOC>::labels{
    "Total",
    "Forward FFT",
    "Hadamard",
    "Inverse FFT"
};


}