#pragma once

#include <vector>
#include <chrono>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "stopwatch.hpp"
#include "row_distribution.cuh"
#include "argument_error.hpp"

#include "kernels.cuh"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace cross {

inline bool one_to_one_validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
    return rows > 0 && cols > 0 && left_matrices == 1 && right_matrices == 1;
}

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class one_to_one: public cross_corr_alg<T, BENCH_TYPE, ALLOC> {
public:
    one_to_one(bool is_fft, std::size_t num_measurements, std::chrono::nanoseconds min_measured_time)
        :cross_corr_alg<T, BENCH_TYPE, ALLOC>(is_fft, measure_alg() ? num_measurements : 0, min_measured_time)
    {}

protected:
    static constexpr bool measure_alg() {
        return BENCH_TYPE == BenchmarkType::Algorithm;
    }

    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_one_to_one(this->refs(), this->targets());
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class cpu_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit cpu_one_to_one([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
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

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        auto result_matrix_size = ref_.matrix_size() + target_.matrix_size() - 1;
        result_ = data_array<T, ALLOC>{result_matrix_size};
    }

    void run_impl() override {
        cpu_cross_corr_one_to_one(ref_, target_, result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> cpu_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_original_alg_one_to_one([[maybe_unused]] const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
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

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
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
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
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

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_original_alg_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_warp_shuffle_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_y_size_ = args.value("block_y_size", 8);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
                std::make_pair("block_y_size", std::to_string(block_y_size_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_warp_shuffle(
                             d_ref_,
                             d_target_,
                             d_result_,
                             target_.matrix_size(),
                             result_.matrix_size(),
                             1,
                             block_y_size_,
                             1
                     )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_y_size_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_warp_shuffle_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
        "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_work_distribution_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_warp_shuffle_work_distribution_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        rows_per_thread_ = args.value("rows_per_thread", 10);
        distribution_type_ = from_string(args.value("distribution_type", "rectangle"));
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("rows_per_thread", std::to_string(rows_per_thread_)),
            std::make_pair("distribution_type", to_string(distribution_type_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        switch (distribution_type_) {
            case distribution::none:
                run_kernel<no_distribution>();
                break;
            case distribution::rectangle:
                run_kernel<rectangle_distribution>();
                break;
            case distribution::triangle:
                run_kernel<triangle_distribution>();
                break;
        }
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_y_size_;
    dsize_t rows_per_thread_;
    distribution distribution_type_;

    template<typename DISTRIBUTION>
    void run_kernel() {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_warp_shuffle_work_distribution<DISTRIBUTION>(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         1,
                         block_y_size_,
                         1,
                         rows_per_thread_
                     )
        );
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_warp_shuffle_work_distribution_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shift_per_warp_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        shifts_per_block_ = args.value("shifts_per_block", 8);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("shifts_per_block", std::to_string(shifts_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_shift_per_warp(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         shifts_per_block_
                     )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t shifts_per_block_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_work_distribution_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shift_per_warp_work_distribution_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        shifts_per_block_ = args.value("shifts_per_block", 8);
        rows_per_warp_ = args.value("rows_per_warp", 3);
        distribution_type_ = from_string(args.value("distribution_type", "rectangle"));
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("shifts_per_block", std::to_string(shifts_per_block_)),
            std::make_pair("rows_per_warp", std::to_string(rows_per_warp_)),
            std::make_pair("distribution_type", to_string(distribution_type_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        switch (distribution_type_) {
            case distribution::none:
                run_kernel<no_distribution>();
                break;
            case distribution::rectangle:
                run_kernel<rectangle_distribution>();
                break;
            case distribution::triangle:
                run_kernel<triangle_distribution>();
                break;
        }
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t shifts_per_block_;
    dsize_t rows_per_warp_;
    distribution distribution_type_;

    template<typename DISTRIBUTION>
    void run_kernel() {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_shift_per_warp_work_distribution<DISTRIBUTION>(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         shifts_per_block_,
                         rows_per_warp_
                     )
        );
    }
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_work_distribution_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_simple_indexing_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shift_per_warp_simple_indexing_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        shifts_per_block_ = args.value("shifts_per_block", 8);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("shifts_per_block", std::to_string(shifts_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_shift_per_warp_simple_indexing(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         shifts_per_block_
                     )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t shifts_per_block_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_simple_indexing_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};


//template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
//class naive_shift_per_warp_shared_mem_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
//public:
//    explicit naive_shift_per_warp_shared_mem_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
//        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
//    {
//        shifts_per_block_ = args.value("shifts_per_block", 8);
//        shared_mem_buffer_rows_ = args.value("shared_mem_buffer_rows", 4);
//    }
//
//    const data_array<T, ALLOC>& refs() const override {
//        return ref_;
//    }
//
//    const data_array<T, ALLOC>& targets() const override {
//        return target_;
//    }
//
//    const data_array<T, ALLOC>& results() const override {
//        return result_;
//    }
//
//    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
//        return std::vector<std::pair<std::string, std::string>>{
//            std::make_pair("shifts_per_block", std::to_string(shifts_per_block_)),
//            std::make_pair("shared_mem_buffer_rows", std::to_string(shared_mem_buffer_rows_))
//        };
//    }
//
//protected:
//    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
//        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
//        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);
//
//        this->check_matrices_same_size(ref_, target_);
//    }
//
//    void prepare_impl() override {
//        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};
//
//        cuda_malloc(&d_ref_, ref_.size());
//        cuda_malloc(&d_target_, target_.size());
//        cuda_malloc(&d_result_, result_.size());
//    }
//
//    void transfer_impl() {
//        cuda_memcpy_to_device(d_ref_, ref_);
//        cuda_memcpy_to_device(d_target_, target_);
//    }
//
//    void run_impl() override {
//        CUDA_MEASURE(0, BENCH_TYPE, this->sw_,
//                     run_ccn_shift_per_warp_shared_mem(
//                         d_ref_,
//                         d_target_,
//                         d_result_,
//                         target_.matrix_size(),
//                         result_.matrix_size(),
//                         shifts_per_block_,
//                         shared_mem_buffer_rows_ * target_.matrix_size().x
//                     )
//        );
//    }
//
//    void finalize_impl() override {
//        cuda_memcpy_from_device(result_, d_result_);
//    }
//
//
//private:
//
//    static std::vector<std::string> labels;
//
//    data_array<T, ALLOC> ref_;
//    data_array<T, ALLOC> target_;
//
//    data_array<T, ALLOC> result_;
//
//    T* d_ref_;
//    T* d_target_;
//    T* d_result_;
//
//    dsize_t shifts_per_block_;
//    dsize_t shared_mem_buffer_rows_;
//};
//
//template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
//std::vector<std::string> naive_shift_per_warp_shared_mem_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
//    "Kernel"
//};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_shared_mem_rows_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shift_per_warp_shared_mem_rows_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        // These arguments DO NOT WORK with DEBUG build, as unoptimized kernel requires too many registers
        // TODO: Change back to this, as it is the most performant
        // shifts_per_block_ = args.value(SHIFTS_PER_BLOCK_ARG, 32);
        shifts_per_block_ = args.value(SHIFTS_PER_BLOCK_ARG, 16);
        shared_mem_row_size_ = args.value(SHARED_MEM_ROW_SIZE_ARG, 128);
        shared_mem_rows_ = args.value(SHARED_MEM_ROWS_ARG, shifts_per_block_);
        strided_load_ = args.value(STRIDED_LOAD_ARG, true);

        if (shared_mem_rows_ == 0) {
            shared_mem_rows_ = shifts_per_block_;
        }

        // TODO: Remove this if we change the implementation to work with fewer
        //  shared mem rows than shifts per block
        if (shared_mem_rows_ < shifts_per_block_) {
            throw argument_error("Invalid number of shared memory rows ["s +
                std::to_string(shared_mem_rows_) +
                "], must be greater than shifts per block [" +
                std::to_string(shifts_per_block_) +
                "]",
                 SHARED_MEM_ROWS_ARG);
        }
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair(SHIFTS_PER_BLOCK_ARG, std::to_string(shifts_per_block_)),
            std::make_pair(SHARED_MEM_ROW_SIZE_ARG, std::to_string(shared_mem_row_size_)),
            std::make_pair(SHARED_MEM_ROWS_ARG, std::to_string(shared_mem_rows_)),
            std::make_pair(STRIDED_LOAD_ARG, std::to_string(strided_load_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                     run_ccn_shift_per_warp_shared_mem_rows(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         1,
                         shifts_per_block_,
                         shared_mem_row_size_,
                         shared_mem_rows_,
                         1,
                         strided_load_
                     )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    inline static const std::string SHIFTS_PER_BLOCK_ARG = "shifts_per_block";
    inline static const std::string SHARED_MEM_ROW_SIZE_ARG = "shared_mem_row_size";
    inline static const std::string SHARED_MEM_ROWS_ARG = "shared_mem_rows";
    inline static const std::string STRIDED_LOAD_ARG = "strided_load";

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t shifts_per_block_;
    dsize_t shared_mem_row_size_;
    dsize_t shared_mem_rows_;
    bool strided_load_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_shared_mem_rows_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_shift_per_block_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_shift_per_block_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_size_ = args.value("block_size", 128);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_size", std::to_string(block_size_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                              run_ccn_shift_per_block(
                                  d_ref_,
                                  d_target_,
                                  d_result_,
                                  target_.matrix_size(),
                                  result_.matrix_size(),
                                  block_size_
                              )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_size_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_shift_per_block_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_multirow_shuffle_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_multirow_shuffle_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        right_rows_per_thread_ = args.value("right_rows_per_thread", 4);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("right_rows_per_thread", std::to_string(right_rows_per_thread_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                              run_ccn_multirow_shuffle(
                                  d_ref_,
                                  d_target_,
                                  d_result_,
                                  target_.matrix_size(),
                                  result_.matrix_size(),
                                  block_y_size_,
                                  right_rows_per_thread_
                              )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_y_size_;
    dsize_t right_rows_per_thread_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_multirow_shuffle_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_multileft_shuffle_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_multileft_shuffle_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        shifts_per_thread_ = args.value("shifts_per_thread", 4);
        left_rows_per_thread_ = args.value("left_rows_per_thread", 3);

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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("shifts_per_thread", std::to_string(shifts_per_thread_)),
            std::make_pair("left_rows_per_thread", std::to_string(left_rows_per_thread_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                              run_ccn_multileft_shuffle(
                                  d_ref_,
                                  d_target_,
                                  d_result_,
                                  target_.matrix_size(),
                                  result_.matrix_size(),
                                  block_y_size_,
                                  shifts_per_thread_,
                                  left_rows_per_thread_
                              )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_y_size_;
    dsize_t shifts_per_thread_;
    dsize_t left_rows_per_thread_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_multileft_shuffle_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class naive_multirow_multiright_shuffle_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit naive_multirow_multiright_shuffle_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(false, labels.size(), min_measured_time), ref_(), target_(), result_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        right_rows_per_thread_ = args.value("right_rows_per_thread", 4);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("block_y_size", std::to_string(block_y_size_)),
            std::make_pair("right_rows_per_thread", std::to_string(right_rows_per_thread_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        result_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1, 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_result_, result_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_,
                              run_ccn_multirow_multiright_shuffle(
                                  d_ref_,
                                  d_target_,
                                  d_result_,
                                  target_.matrix_size(),
                                  result_.matrix_size(),
                                  1,
                                  block_y_size_,
                                  right_rows_per_thread_,
                                  1
                              )
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_target_));
        CUCH(cudaFree(d_ref_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t block_y_size_;
    dsize_t right_rows_per_thread_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> naive_multirow_multiright_shuffle_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Kernel"
};



template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_original_alg_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit fft_original_alg_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(true, labels.size(), min_measured_time), ref_(), target_(), result_(), fft_buffer_size_(0)
    {
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:

    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        result_ = data_array<T, ALLOC>{ref_.matrix_size()};

        fft_buffer_size_ = ref_.matrix_size().y * (ref_.matrix_size().x / 2 + 1);
        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, ref_.size() + target_.size());
            cuda_malloc(&d_result_, result_.size());

            // 2 * as we have two input matrices we are doing FFT on
            cuda_malloc(&d_inputs_fft_, 2 * fft_buffer_size_);
        );


        int sizes[2] = {static_cast<int>(ref_.matrix_size().y), static_cast<int>(ref_.matrix_size().x)};
        CPU_MEASURE(4, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), 2));
            FFTCH(cufftPlan2d(&fft_inv_plan_, result_.matrix_size().y, result_.matrix_size().x, fft_type_C2R<T>()));
        );
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_inputs_, ref_);
        cuda_memcpy_to_device(d_inputs_ + ref_.size(), target_);
    }

    void run_impl() override {
        CPU_ADAPTIVE_MEASURE(0, this->measure_alg(), this->sw_, false,
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        CUDA_ADAPTIVE_MEASURE(1, this->measure_alg(), this->sw_,
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_,
                {ref_.matrix_size().y, (ref_.matrix_size().x / 2) + 1},
                1,
                1,
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_, d_result_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_inputs_fft_));
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_inputs_));
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
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

    dsize_t hadamard_threads_per_block_;
};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> fft_original_alg_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "Allocation",
    "Plan"
};


template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC = std::allocator<T>>
class fft_reduced_transfer_one_to_one: public one_to_one<T, BENCH_TYPE, ALLOC> {
public:
    explicit fft_reduced_transfer_one_to_one(const json& args, std::chrono::nanoseconds min_measured_time)
        :one_to_one<T, BENCH_TYPE, ALLOC>(true, labels.size(), min_measured_time), ref_(), target_(), result_(), fft_buffer_size_(0)
    {
        scatter_threads_per_block_  = args.value("scatter_threads_per_block", 256);
        scatter_items_per_thread_ = args.value("scatter_items_per_thread", 10);
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("scatter_threads_per_block", std::to_string(scatter_threads_per_block_)),
            std::make_pair("scatter_items_per_thread", std::to_string(scatter_items_per_thread_)),
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:

    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        padded_matrix_size_ = 2 * ref_.matrix_size();
        fft_buffer_size_ = padded_matrix_size_.y * (padded_matrix_size_.x / 2 + 1);

        // Input matrices are NOT padded
        result_ = data_array<T, ALLOC>{padded_matrix_size_};
        CPU_MEASURE(5, this->measure_alg(), this->sw_, false,
            cuda_malloc(&d_inputs_, ref_.size() + target_.size());
            cuda_malloc(&d_padded_inputs_, 2 * padded_matrix_size_.area());
            cuda_malloc(&d_result_, result_.size());

            // 2 * as we have two input matrices we are doing FFT on
            cuda_malloc(&d_padded_inputs_fft_, 2 * fft_buffer_size_);
        );

        int sizes[2] = {static_cast<int>(padded_matrix_size_.y), static_cast<int>(padded_matrix_size_.x)};
        CPU_MEASURE(6, this->measure_alg(), this->sw_, false,
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), 2));
            FFTCH(cufftPlan2d(&fft_inv_plan_, result_.matrix_size().y, result_.matrix_size().x, fft_type_C2R<T>()));
        );
    }

    void transfer_impl() override {
        CPU_MEASURE(3, this->measure_alg(), this->sw_, false,
            cuda_memcpy_to_device(d_inputs_, ref_);
            cuda_memcpy_to_device(d_inputs_ + ref_.size(), target_);
        );

        cuda_memset(d_padded_inputs_, 0, 2 * padded_matrix_size_.area());

        CUDA_ADAPTIVE_MEASURE(4, this->measure_alg(), this->sw_,
            run_scatter(
                d_inputs_,
                d_padded_inputs_,
                ref_.matrix_size(),
                ref_.num_matrices() + target_.num_matrices(),
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
                d_padded_inputs_fft_ + fft_buffer_size_,
                {padded_matrix_size_.y, (padded_matrix_size_.x / 2) + 1},
                1,
                1,
                hadamard_threads_per_block_)
        );

        CPU_ADAPTIVE_MEASURE(2, this->measure_alg(), this->sw_, false,
            fft_complex_to_real(fft_inv_plan_, d_padded_inputs_fft_ + fft_buffer_size_, d_result_)
        );
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    void free_impl() override {
        FFTCH(cufftDestroy(fft_inv_plan_));
        FFTCH(cufftDestroy(fft_plan_));
        CUCH(cudaFree(d_padded_inputs_fft_));
        CUCH(cudaFree(d_result_));
        CUCH(cudaFree(d_padded_inputs_));
        CUCH(cudaFree(d_inputs_));
    }

        std::vector<std::string> measurement_labels_impl() const override {
        return this->measure_alg() ? labels : std::vector<std::string>{};
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;
    data_array<T, ALLOC> result_;

    T* d_inputs_;
    T* d_padded_inputs_;
    T* d_result_;

    cufftHandle fft_plan_;
    cufftHandle fft_inv_plan_;

    dsize2_t padded_matrix_size_;
    dsize_t fft_buffer_size_;

    fft_complex_t* d_padded_inputs_fft_;

    dsize_t scatter_threads_per_block_;
    dsize_t scatter_items_per_thread_;
    dsize_t hadamard_threads_per_block_;

};

template<typename T, BenchmarkType BENCH_TYPE, typename ALLOC>
std::vector<std::string> fft_reduced_transfer_one_to_one<T, BENCH_TYPE, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "ToDevice",
    "Scatter",
    "Allocation",
    "Plan"
};

}
