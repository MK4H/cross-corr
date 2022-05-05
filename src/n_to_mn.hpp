#pragma once

#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "stopwatch.hpp"

#include "kernels.cuh"

using namespace std::string_literals;

namespace cross {

inline bool input_sizes_divisible(dsize_t ref_num_matrices, dsize_t target_num_matrices) {
    return target_num_matrices % ref_num_matrices == 0;
}

inline bool n_to_mn_validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
    return rows > 0 && cols > 0 && left_matrices > 0 && right_matrices > 0 && input_sizes_divisible(left_matrices, right_matrices);
}

/**
 * The results should be ordered so we first have the cross-correlation of
 * the n left/reference matrices with the corresponding matrix from the first
 * n right/deformed matrices, then results of the cross-correlation of the n
 * left/reference matrices with the corresponding matrices with index [n,2*n),
 * etc. ending with the cross-correlation of the n left/reference matrices with
 * right matrices [(m-1)*n,m*n).
 *
 * When indexing, this means multiply the index of the right matrix by "m" and add left matrix index,
 * i.e. right_idx * m + left_idx
 */
template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class n_to_mn: public cross_corr_alg<T, BENCH_TYPE, ALLOC> {
public:
    n_to_mn(bool is_fft, std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :cross_corr_alg<T, BENCH_TYPE, ALLOC>(is_fft, measure_alg() ? num_measurements : 0, min_measured_time)
    {}

    static bool validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
        return rows > 0 && cols > 0 && left_matrices > 0 && right_matrices > 0 && input_sizes_divisible(left_matrices, right_matrices);
    }

protected:
    static constexpr bool measure_alg() {
        return BENCH_TYPE == BenchmarkType::Algorithm;
    }

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_n_to_mn(this->refs(), this->targets());
    }
protected:


    static void check_input_sizes_divisible(dsize_t ref_num_matrices, dsize_t target_num_matrices) {
        if (!input_sizes_divisible(ref_num_matrices, target_num_matrices)) {
            throw std::runtime_error(
                "Invalid input data counts, "s +
                std::to_string(target_num_matrices) +
                " is not divisible by "s +
                std::to_string(ref_num_matrices)
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class cpu_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit cpu_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), refs_(), targets_(), results_()
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

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_n_to_mn(refs_, targets_, results_);
    }

private:
    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_gpu_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    naive_gpu_n_to_mn(std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(false, num_measurements, min_measured_time), refs_(), targets_(), results_()
    {}

    const data_array<T, ALLOC>& refs() const override {
        return refs_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }
protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_targets_));
        CUCH(cudaFree(d_refs_));
    }

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_original_alg_n_to_mn: public naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_original_alg_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :naive_gpu_n_to_mn<T, BENCH_TYPE, ALLOC>(std::size(labels), min_measured_time)
    {

    }
protected:
    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
            run_cross_corr_naive_original(
                this->d_refs_,
                this->d_targets_,
                this->d_results_,
                this->targets_.matrix_size(),
                this->results_.matrix_size(),
                this->refs_.num_matrices(),
                this->targets_.num_matrices() / this->refs_.num_matrices()
            )
        );
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    inline static const char* labels[] = {
        "Kernel"
    };
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_original_alg_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit fft_original_alg_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(true, std::size(labels), min_measured_time),
            refs_(), targets_(), results_(), fft_buffer_size_(0),
            fft_plan_(), fft_inv_plan_()

    {
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
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

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        results_ = data_array<T, ALLOC>{refs_.matrix_size(), targets_.num_matrices()};

        fft_buffer_size_ = refs_.matrix_size().y * (refs_.matrix_size().x / 2 + 1);

        auto num_inputs = refs_.num_matrices() + targets_.num_matrices();

        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_inputs_fft_, num_inputs * fft_buffer_size_);
        );

        int input_sizes[2] = {static_cast<int>(refs_.matrix_size().y), static_cast<int>(refs_.matrix_size().x)};
        int result_sizes[2] = {static_cast<int>(results_.matrix_size().y), static_cast<int>(results_.matrix_size().x)};
        CPU_MEASURE(4, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, input_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs));

            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, result_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_inputs_, refs_);
        cuda_memcpy_to_device(d_inputs_ + refs_.size(), targets_);
    }

    void run_impl() override {
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, false,
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        CUDA_ADAPTIVE_MEASURE(1, this->measure_alg(), this->sw_,
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                {refs_.matrix_size().y, (refs_.matrix_size().x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(), d_results_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_inputs_fft_));
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_inputs_));
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    inline static const char* labels[] = {
        "Forward FFT",
        "Hadamard",
        "Inverse FFT",
        "Allocation",
        "Plan"
    };

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_inputs_;
    T* d_results_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t fft_buffer_size_;

    fft_complex_t* d_inputs_fft_;

    dsize_t hadamard_threads_per_block_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_reduced_transfer_n_to_mn: public n_to_mn<T, BENCH_TYPE, ALLOC> {
public:
    explicit fft_reduced_transfer_n_to_mn([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_mn<T, BENCH_TYPE, ALLOC>(true, std::size(labels), min_measured_time),
            refs_(), targets_(), results_(), fft_buffer_size_(0), num_inputs_(0),
            fft_plan_(), fft_inv_plan_()
    {
        scatter_threads_per_block_  = args.value("scatter_threads_per_block", 256);
        scatter_items_per_thread_ = args.value("scatter_items_per_thread", 10);
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
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

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("scatter_threads_per_block", std::to_string(scatter_threads_per_block_)),
            std::make_pair("scatter_items_per_thread", std::to_string(scatter_items_per_thread_)),
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:

    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_input_sizes_divisible(refs_.num_matrices(), targets_.num_matrices());
        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        padded_matrix_size_ = 2 * refs_.matrix_size();
        fft_buffer_size_ = padded_matrix_size_.y * (padded_matrix_size_.x / 2 + 1);
        // 1 for the ref matrix
        num_inputs_ = refs_.num_matrices() + targets_.num_matrices();

        // Input matrices are NOT padded
        results_ = data_array<T, ALLOC>{padded_matrix_size_, targets_.num_matrices()};

        CPU_MEASURE(5, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
            cuda_malloc(&d_padded_inputs_, num_inputs_ * padded_matrix_size_.area());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_padded_inputs_fft_, num_inputs_ * fft_buffer_size_);
        );

        int sizes[2] = {static_cast<int>(padded_matrix_size_.y), static_cast<int>(padded_matrix_size_.x)};
        CPU_MEASURE(6, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs_));

            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_memcpy_to_device(d_inputs_, refs_);
            cuda_memcpy_to_device(d_inputs_ + refs_.size(), targets_);
        );

        cuda_memset(d_padded_inputs_, 0, num_inputs_ * padded_matrix_size_.area());

        CUDA_ADAPTIVE_MEASURE(4, this->measure_alg(), this->sw_,
            run_scatter(
                d_inputs_,
                d_padded_inputs_,
                refs_.matrix_size(),
                num_inputs_,
                padded_matrix_size_,
                dsize2_t{0,0},
                scatter_threads_per_block_,
                scatter_items_per_thread_
            );
        );
    }

    void run_impl() override {
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, false,
            fft_real_to_complex(fft_plan_, d_padded_inputs_, d_padded_inputs_fft_);
        );

        CUDA_ADAPTIVE_MEASURE(1, this->measure_alg(), this->sw_,
            run_hadamard_original(
                d_padded_inputs_fft_,
                d_padded_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                {padded_matrix_size_.y, (padded_matrix_size_.x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices() / refs_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_padded_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(), d_results_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_padded_inputs_fft_));
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_padded_inputs_));
        CUCH(cudaFree(d_inputs_));
    }

    [[nodiscard]] std::vector<const char*> measurement_labels_impl() const override {
        return this->measure_alg() ?
               std::vector<const char*>(std::begin(labels), std::end(labels)) :
               std::vector<const char*>{};
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    inline static const char* labels[] = {
        "Forward FFT",
        "Hadamard",
        "Inverse FFT",
        "ToDevice",
        "Scatter",
        "Allocation",
        "Plan"
    };

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;
    data_array<T, ALLOC> results_;

    T* d_inputs_;
    T* d_padded_inputs_;
    T* d_results_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize_t num_inputs_;
    dsize2_t padded_matrix_size_;
    dsize_t fft_buffer_size_;

    fft_complex_t* d_padded_inputs_fft_;

    dsize_t scatter_threads_per_block_;
    dsize_t scatter_items_per_thread_;
    dsize_t hadamard_threads_per_block_;

};

}
