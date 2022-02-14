#pragma once

#include <vector>
#include <chrono>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"
#include "row_distribution.cuh"

#include "kernels.cuh"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

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
    explicit cpu_one_to_one([[maybe_unused]] const json& args)
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
        return labels;
    }
private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_one_to_one<T, DEBUG, ALLOC>::labels{
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit naive_original_alg_one_to_one([[maybe_unused]] const json& args)
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
        CUDA_MEASURE(this->label_index(0),
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

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_ring_buffer_row_alg: public one_to_one<T, ALLOC> {
public:
    explicit naive_ring_buffer_row_alg(const json& args)
        :one_to_one<T, ALLOC> (false, labels.size()), ref_(), target_(), res_()
    {
        threads_per_block_ = args.value("threads_per_block", 256);
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

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("threads_per_block", std::to_string(threads_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        target_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, target_);
    }

    void prepare_impl() override {
        res_ = data_array<T, ALLOC>{ref_.matrix_size() + target_.matrix_size() - 1};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_target_, target_.size());
        cuda_malloc(&d_res_, res_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_target_, target_);
    }

    void run_impl() override {
        CUDA_MEASURE(this->label_index(0),
            run_ccn_ring_buffer_row(
                d_ref_,
                d_target_,
                d_res_,
                target_.matrix_size(),
                res_.matrix_size(),
                threads_per_block_
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(res_, d_res_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> res_;

    T* d_ref_;
    T* d_target_;
    T* d_res_;

    dsize_t threads_per_block_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_ring_buffer_row_alg<T, DEBUG, ALLOC>::labels{
    "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit naive_warp_shuffle_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
    {
        rows_per_block_ = args.value("rows_per_block", 8);
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
                std::make_pair("rows_per_block", std::to_string(rows_per_block_))
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
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_warp_shuffle(
                             d_ref_,
                             d_target_,
                             d_result_,
                             target_.matrix_size(),
                             result_.matrix_size(),
                             1,
                             rows_per_block_,
                             1
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> target_;

    data_array<T, ALLOC> result_;

    T* d_ref_;
    T* d_target_;
    T* d_result_;

    dsize_t rows_per_block_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_warp_shuffle_one_to_one<T, DEBUG, ALLOC>::labels{
        "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit naive_shift_per_warp_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
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
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_shift_per_warp(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         shifts_per_block_
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_one_to_one<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_work_distribution_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit naive_shift_per_warp_work_distribution_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
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

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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
        CUDA_MEASURE(this->label_index(0),
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

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_work_distribution_one_to_one<T, DEBUG, ALLOC>::labels{
    "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_simple_indexing_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit naive_shift_per_warp_simple_indexing_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(false, labels.size()), ref_(), target_(), result_()
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
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_shift_per_warp_simple_indexing(
                         d_ref_,
                         d_target_,
                         d_result_,
                         target_.matrix_size(),
                         result_.matrix_size(),
                         shifts_per_block_
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_simple_indexing_one_to_one<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_original_alg_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit fft_original_alg_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(true, labels.size()), ref_(), target_(), result_(), fft_buffer_size_(0)
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

        cuda_malloc(&d_inputs_, ref_.size() + target_.size());
        cuda_malloc(&d_result_, result_.size());

        // 2 * as we have two input matrices we are doing FFT on
        cuda_malloc(&d_inputs_fft_, 2 * fft_buffer_size_);

        int sizes[2] = {static_cast<int>(ref_.matrix_size().y), static_cast<int>(ref_.matrix_size().x)};
        // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
        FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), 2));
        FFTCH(cufftPlan2d(&fft_inv_plan_, result_.matrix_size().y, result_.matrix_size().x, fft_type_C2R<T>()));
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_inputs_, ref_);
        cuda_memcpy_to_device(d_inputs_ + ref_.size(), target_);
    }

    void run_impl() override {
        CPU_MEASURE(this->label_index(0),
            fft_real_to_complex(fft_plan_, d_inputs_, d_inputs_fft_);
        );

        CUDA_MEASURE(this->label_index(1),
            run_hadamard_original(
                d_inputs_fft_,
                d_inputs_fft_ + fft_buffer_size_,
                {ref_.matrix_size().y, (ref_.matrix_size().x / 2) + 1},
                1,
                1,
                hadamard_threads_per_block_)
        );

        CPU_MEASURE(this->label_index(2),
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_, d_result_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_original_alg_one_to_one<T, DEBUG, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_reduced_transfer_one_to_one: public one_to_one<T, ALLOC> {
public:
    explicit fft_reduced_transfer_one_to_one(const json& args)
        :one_to_one<T, ALLOC>(true, labels.size()), ref_(), target_(), result_(), fft_buffer_size_(0)
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

        cuda_malloc(&d_inputs_, ref_.size() + target_.size());
        cuda_malloc(&d_padded_inputs_, 2 * padded_matrix_size_.area());
        cuda_malloc(&d_result_, result_.size());

        // 2 * as we have two input matrices we are doing FFT on
        cuda_malloc(&d_padded_inputs_fft_, 2 * fft_buffer_size_);

        int sizes[2] = {static_cast<int>(padded_matrix_size_.y), static_cast<int>(padded_matrix_size_.x)};
        // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
        FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), 2));
        FFTCH(cufftPlan2d(&fft_inv_plan_, result_.matrix_size().y, result_.matrix_size().x, fft_type_C2R<T>()));
    }

    void transfer_impl() override {
        CPU_MEASURE(this->label_index(3),
            cuda_memcpy_to_device(d_inputs_, ref_);
            cuda_memcpy_to_device(d_inputs_ + ref_.size(), target_);
        );

        cuda_memset(d_padded_inputs_, 0, 2 * padded_matrix_size_.area());

        CUDA_MEASURE(this->label_index(4),
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

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());

        // DEBUG
        // data_array<T> tmp{padded_matrix_size_, ref_.num_matrices() + target_.num_matrices()};
        // cuda_memcpy_from_device(tmp.data(), d_padded_inputs_, tmp.size());

        // std::ofstream out("./scattered.csv");
        // tmp.store_to_csv(out);

        // END DEBUG
    }

    void run_impl() override {
        CPU_MEASURE(this->label_index(0),
            fft_real_to_complex(fft_plan_, d_padded_inputs_, d_padded_inputs_fft_);
        );

        CUDA_MEASURE(this->label_index(1),
            run_hadamard_original(
                d_padded_inputs_fft_,
                d_padded_inputs_fft_ + fft_buffer_size_,
                {padded_matrix_size_.y, (padded_matrix_size_.x / 2) + 1},
                1,
                1,
                hadamard_threads_per_block_)
        );

        CPU_MEASURE(this->label_index(2),
            fft_complex_to_real(fft_inv_plan_, d_padded_inputs_fft_ + fft_buffer_size_, d_result_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(result_, d_result_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_reduced_transfer_one_to_one<T, DEBUG, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "ToDevice",
    "Scatter"
};

}