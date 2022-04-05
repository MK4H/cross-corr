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

inline bool n_to_m_validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
    return rows > 0 && cols > 0 && left_matrices > 0 && right_matrices > 0;
}

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class n_to_m: public cross_corr_alg<T, BENCH_TYPE, ALLOC> {
public:
    n_to_m(bool is_fft, std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :cross_corr_alg<T, BENCH_TYPE, ALLOC>(is_fft, measure_alg() ? num_measurements : 0, min_measured_time)
    {}

protected:
    static constexpr bool measure_alg() {
        return BENCH_TYPE == BenchmarkType::Algorithm;
    }

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_n_to_m(this->refs(), this->targets());
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class cpu_n_to_m: public n_to_m<T, BENCH_TYPE, ALLOC> {
public:
    explicit cpu_n_to_m([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_m<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), refs_(), targets_(), results_()
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

        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, refs_.num_matrices() * targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_n_to_m(refs_, targets_, results_);
    }

private:

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_original_alg_n_to_m: public n_to_m<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_original_alg_n_to_m(const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_m<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), cudaStreams(), refs_(), targets_(), results_()
    {
        num_cuda_streams_ = args.value("num_cuda_streams", 8);
        cudaStreams.resize(num_cuda_streams_);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("num_cuda_streams", std::to_string(num_cuda_streams_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, refs_.num_matrices() * targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        for (auto & cudaStream : cudaStreams) {
            CUCH(cudaStreamCreate(&cudaStream));
        }
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
         start_kernels();
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        for (auto & cudaStream : cudaStreams) {
            CUCH(cudaStreamDestroy(cudaStream));
        }

        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_targets_));
        CUCH(cudaFree(d_refs_));
    }

private:

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    std::vector<cudaStream_t> cudaStreams;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;

    dsize_t num_cuda_streams_;

    void start_kernels() {
        for (dsize_t ref = 0; ref < refs_.num_matrices(); ++ref) {

            run_cross_corr_naive_original(
                d_refs_ + ref * refs_.matrix_size().area(),
                d_targets_,
                d_results_ + (ref * targets_.num_matrices()) * results_.matrix_size().area(),
                targets_.matrix_size(),
                results_.matrix_size(),
                1,
                targets_.num_matrices(),
                cudaStreams[ref % cudaStreams.size()]
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_work_distribution_n_to_m: public n_to_m<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_warp_shuffle_work_distribution_n_to_m([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_m<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), cudaStreams(), refs_(), targets_(), results_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 2);
        rows_per_thread_ = args.value("rows_per_thread", 10);
        distribution_type_ = from_string(args.value("distribution_type", "rectangle"));
        num_cuda_streams_ = args.value("num_cuda_streams", 8);

        cudaStreams.resize(num_cuda_streams_);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_)),
            std::make_pair("rows_per_thread", std::to_string(rows_per_thread_)),
            std::make_pair("distribution_type", to_string(distribution_type_)),
            std::make_pair("num_cuda_streams", std::to_string(num_cuda_streams_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, refs_.num_matrices() * targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());

        for (auto & cudaStream : cudaStreams) {
            CUCH(cudaStreamCreate(&cudaStream));
        }
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        switch (distribution_type_) {
            case distribution::none:
                start_kernels<no_distribution>();
                break;
            case distribution::rectangle:
                start_kernels<rectangle_distribution>();
                break;
            case distribution::triangle:
                start_kernels<triangle_distribution>();
                break;
        }
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        for (auto & cudaStream : cudaStreams) {
            CUCH(cudaStreamDestroy(cudaStream));
        }

        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_targets_));
        CUCH(cudaFree(d_refs_));
    }

private:
    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    std::vector<cudaStream_t> cudaStreams;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;

    dsize_t block_y_size_;
    dsize_t right_matrices_per_thread_;
    dsize_t rows_per_thread_;
    distribution distribution_type_;
    dsize_t num_cuda_streams_;

    template<typename DISTRIBUTION>
    void start_kernels() {
        for (dsize_t ref = 0; ref < refs_.num_matrices(); ++ref) {
            run_ccn_warp_shuffle_work_distribution<DISTRIBUTION>(
                d_refs_ + ref * refs_.matrix_size().area(),
                d_targets_,
                d_results_ + (ref * targets_.num_matrices()) * results_.matrix_size().area(),
                targets_.matrix_size(),
                results_.matrix_size(),
                targets_.num_matrices(),
                block_y_size_,
                right_matrices_per_thread_,
                rows_per_thread_,
                cudaStreams[ref % cudaStreams.size()]
            );
        }
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shuffle_specialized_n_to_m: public n_to_m<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shuffle_specialized_n_to_m(const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_m<T, BENCH_TYPE, ALLOC>(false, 0, min_measured_time), refs_(), targets_(), results_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        left_matrices_per_thread_ = args.value("left_matrices_per_thread", 2);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 2);
        rows_per_thread_ = args.value("rows_per_thread", 10);
        // TODO: Add default work distribution
        distribution_type_ = from_string(args.value("distribution_type", "none"));
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("left_matrices_per_thread", std::to_string(left_matrices_per_thread_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_)),
            std::make_pair("rows_per_thread", std::to_string(rows_per_thread_)),
            std::make_pair("distribution_type", to_string(distribution_type_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& def_path) override {
        refs_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(def_path);

        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = refs_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, refs_.num_matrices() * targets_.num_matrices()};

        cuda_malloc(&d_refs_, refs_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_refs_, refs_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        switch (distribution_type_) {
            case distribution::none:
                start_kernels<no_distribution>();
                break;
            case distribution::rectangle:
                start_kernels<rectangle_distribution>();
                break;
            case distribution::triangle:
                start_kernels<triangle_distribution>();
                break;
        }
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_results_));
        CUCH(cudaFree(d_targets_));
        CUCH(cudaFree(d_refs_));
    }

private:

    data_array<T, ALLOC> refs_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_refs_;
    T* d_targets_;
    T* d_results_;

    dsize_t block_y_size_;
    dsize_t left_matrices_per_thread_;
    dsize_t right_matrices_per_thread_;
    dsize_t rows_per_thread_;
    distribution distribution_type_;

    template<typename DISTRIBUTION>
    void start_kernels() {
        run_ccn_warp_shuffle_n_to_m_work_distribution<DISTRIBUTION>(
            d_refs_,
            d_targets_,
            d_results_,
            targets_.matrix_size(),
            results_.matrix_size(),
            refs_.num_matrices(),
            targets_.num_matrices(),
            block_y_size_,
            left_matrices_per_thread_,
            right_matrices_per_thread_,
            rows_per_thread_
        );
    }
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_better_hadamard_alg_n_to_m: public n_to_m<T, BENCH_TYPE, ALLOC> {
public:
    explicit fft_better_hadamard_alg_n_to_m([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :n_to_m<T, BENCH_TYPE, ALLOC>(true, labels.size(), min_measured_time), refs_(), targets_(), results_(), fft_buffer_size_(0)
    {
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
        hadamard_items_per_thread_ = args.value("hadamard_items_per_thread", 10);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_)),
            std::make_pair("hadamard_items_per_thread", std::to_string(hadamard_items_per_thread_))
        };
    }


protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        refs_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_matrices_same_size(refs_, targets_);
    }

    void prepare_impl() override {
        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        results_ = data_array<T, ALLOC>{refs_.matrix_size(), refs_.num_matrices() * targets_.num_matrices()};

        fft_buffer_size_ = refs_.matrix_size().y * (refs_.matrix_size().x / 2 + 1);

        auto num_inputs = refs_.num_matrices() + targets_.num_matrices();

        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, refs_.size() + targets_.size());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_inputs_fft_, num_inputs * fft_buffer_size_);
            cuda_malloc(&d_haddamard_results_, results_.num_matrices() * fft_buffer_size_);
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
            run_hadamard_n_to_m_over_output(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_ * refs_.num_matrices(),
                d_haddamard_results_,
                {refs_.matrix_size().y, (refs_.matrix_size().x / 2) + 1},
                refs_.num_matrices(),
                targets_.num_matrices(),
                hadamard_threads_per_block_,
                hadamard_items_per_thread_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_haddamard_results_, d_results_)
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

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
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
    fft_complex_t* d_haddamard_results_;

    dsize_t hadamard_threads_per_block_;
    dsize_t hadamard_items_per_thread_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> fft_better_hadamard_alg_n_to_m<T, BENCH_TYPE, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "Allocation",
    "Plan"
};



}
