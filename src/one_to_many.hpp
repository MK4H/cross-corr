#pragma once

#include <vector>
#include <chrono>

#include "cross_corr.hpp"
#include "matrix.hpp"
#include "helpers.cuh"
#include "stopwatch.hpp"
#include "argument_error.hpp"

#include "kernels.cuh"

namespace cross {

template<typename T, typename ALLOC = std::allocator<T>>
class one_to_many: public cross_corr_alg<T, ALLOC> {
public:
    one_to_many(bool is_fft, std::size_t num_measurements)
        :cross_corr_alg<T,ALLOC>(is_fft, num_measurements)
    {}

    static bool validate_input_size(dsize_t rows, dsize_t cols, dsize_t left_matrices, dsize_t right_matrices) {
        return rows > 0 && cols > 0 && left_matrices == 1 && right_matrices > 0;
    }

protected:
    data_array<T> get_valid_results() const override {
        return cpu_cross_corr_one_to_many(this->refs(), this->targets());
    }
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class cpu_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit cpu_one_to_many([[maybe_unused]] const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};
    }

    void run_impl() override {
        cpu_cross_corr_one_to_many(ref_, targets_, results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }
private:
    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> cpu_one_to_many<T, DEBUG, ALLOC>::labels{
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_original_alg_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit naive_original_alg_one_to_many([[maybe_unused]] const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {

    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(this->label_index(0),
            run_cross_corr_naive_original(
                d_ref_,
                d_targets_,
                d_results_,
                targets_.matrix_size(),
                results_.matrix_size(),
                // Subregions_per_pic tells us the number of reference subregions from the picture
                1,
                // Batch size is the number of deformed subregions for each reference subregion
                targets_.num_matrices()
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_original_alg_one_to_many<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_def_per_block: public one_to_many<T, ALLOC> {
public:
    explicit naive_def_per_block([[maybe_unused]] const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_(), items_per_block_(), threads_per_block_()
    {
        items_per_block_ = args.value("items_per_block", 10);
        threads_per_block_ = args.value("threads_per_block", 256);
    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }
    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("items_per_block", std::to_string(items_per_block_)),
            std::make_pair("threads_per_block", std::to_string(threads_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        auto result_matrix_size = ref_.matrix_size() + targets_.matrix_size() - 1;
        results_ = data_array<T, ALLOC>{result_matrix_size, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(this->label_index(0),
            run_ccn_def_per_block(
                d_ref_,
                d_targets_,
                d_results_,
                ref_.matrix_size(),
                results_.matrix_size(),
                targets_.num_matrices(),
                items_per_block_,
                threads_per_block_
            )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    dsize_t items_per_block_;
    dsize_t threads_per_block_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_def_per_block<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit naive_warp_shuffle_one_to_many(const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {
        rows_per_block_ = args.value("rows_per_block", 8);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 2);
    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("rows_per_block", std::to_string(rows_per_block_)),
            std::make_pair("right_matrices_per_thread", std::to_string(right_matrices_per_thread_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        results_ = data_array<T, ALLOC>{ref_.matrix_size() + targets_.matrix_size() - 1, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_warp_shuffle(
                         d_ref_,
                         d_targets_,
                         d_results_,
                         targets_.matrix_size(),
                         results_.matrix_size(),
                         targets_.num_matrices(),
                         rows_per_block_,
                         right_matrices_per_thread_
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    dsize_t rows_per_block_;
    dsize_t right_matrices_per_thread_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_warp_shuffle_one_to_many<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_warp_shuffle_work_distribution_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit naive_warp_shuffle_work_distribution_one_to_many(const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {
        block_y_size_ = args.value("block_y_size", 8);
        right_matrices_per_thread_ = args.value("right_matrices_per_thread", 2);
        rows_per_thread_ = args.value("rows_per_thread", 10);
        distribution_type_ = from_string(args.value("distribution_type", "rectangle"));
    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
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
            std::make_pair("distribution_type", to_string(distribution_type_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        results_ = data_array<T, ALLOC>{ref_.matrix_size() + targets_.matrix_size() - 1, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
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
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    dsize_t block_y_size_;
    dsize_t right_matrices_per_thread_;
    dsize_t rows_per_thread_;
    distribution distribution_type_;

    template<typename DISTRIBUTION>
    void run_kernel() {
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_warp_shuffle_work_distribution<DISTRIBUTION>(
                         d_ref_,
                         d_targets_,
                         d_results_,
                         targets_.matrix_size(),
                         results_.matrix_size(),
                         targets_.num_matrices(),
                         block_y_size_,
                         right_matrices_per_thread_,
                         rows_per_thread_
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_warp_shuffle_work_distribution_one_to_many<T, DEBUG, ALLOC>::labels{
    "Kernel"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class naive_shift_per_warp_shared_mem_rows_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit naive_shift_per_warp_shared_mem_rows_one_to_many(const json& args)
        :one_to_many<T, ALLOC>(false, labels.size()), ref_(), targets_(), results_()
    {
        shifts_per_block_ = args.value(SHIFTS_PER_BLOCK_ARG, 8);
        shared_mem_row_size_ = args.value(SHARED_MEM_ROW_SIZE_ARG, 32);
        shared_mem_rows_ = args.value(SHARED_MEM_ROWS_ARG, shifts_per_block_);
        strided_load_ = args.value(STRIDED_LOAD_ARG, true);
        right_matrices_per_block_ = args.value(RIGHT_MATRICES_PER_BLOCK_ARG, 8);

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
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair(SHIFTS_PER_BLOCK_ARG, std::to_string(shifts_per_block_)),
            std::make_pair(SHARED_MEM_ROW_SIZE_ARG, std::to_string(shared_mem_row_size_)),
            std::make_pair(SHARED_MEM_ROWS_ARG, std::to_string(shared_mem_rows_)),
            std::make_pair(STRIDED_LOAD_ARG, std::to_string(strided_load_)),
            std::make_pair(RIGHT_MATRICES_PER_BLOCK_ARG, std::to_string(right_matrices_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, no_padding, ALLOC>(ref_path);
        targets_ = load_matrix_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        results_ = data_array<T, ALLOC>{ref_.matrix_size() + targets_.matrix_size() - 1, targets_.num_matrices()};

        cuda_malloc(&d_ref_, ref_.size());
        cuda_malloc(&d_targets_, targets_.size());
        cuda_malloc(&d_results_, results_.size());
    }

    void transfer_impl() {
        cuda_memcpy_to_device(d_ref_, ref_);
        cuda_memcpy_to_device(d_targets_, targets_);
    }

    void run_impl() override {
        CUDA_MEASURE(this->label_index(0),
                     run_ccn_shift_per_warp_shared_mem_rows(
                         d_ref_,
                         d_targets_,
                         d_results_,
                         targets_.matrix_size(),
                         results_.matrix_size(),
                         targets_.num_matrices(),
                         shifts_per_block_,
                         shared_mem_row_size_,
                         shared_mem_rows_,
                         right_matrices_per_block_,
                         strided_load_
                     )
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:

    static std::vector<std::string> labels;

    inline static const std::string SHIFTS_PER_BLOCK_ARG = "shifts_per_block";
    inline static const std::string SHARED_MEM_ROW_SIZE_ARG = "shared_mem_row_size";
    inline static const std::string SHARED_MEM_ROWS_ARG = "shared_mem_rows";
    inline static const std::string STRIDED_LOAD_ARG = "strided_load";
    inline static const std::string RIGHT_MATRICES_PER_BLOCK_ARG = "right_matrices_per_block";

    data_array<T, ALLOC> ref_;
    data_array<T, ALLOC> targets_;

    data_array<T, ALLOC> results_;

    T* d_ref_;
    T* d_targets_;
    T* d_results_;

    dsize_t shifts_per_block_;
    dsize_t shared_mem_row_size_;
    dsize_t shared_mem_rows_;
    dsize_t right_matrices_per_block_;
    bool strided_load_;
};

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> naive_shift_per_warp_shared_mem_rows_one_to_many<T, DEBUG, ALLOC>::labels{
    "Kernel"
};


template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_original_alg_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit fft_original_alg_one_to_many([[maybe_unused]] const json& args)
        :one_to_many<T, ALLOC>(true, labels.size()), ref_(), targets_(), results_(), fft_buffer_size_(0)
    {
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
    }

    std::vector<std::pair<std::string, std::string>> additional_properties() const override {
        return std::vector<std::pair<std::string, std::string>>{
            std::make_pair("hadamard_threads_per_block", std::to_string(hadamard_threads_per_block_))
        };
    }

protected:
    void load_impl(const std::filesystem::path& ref_path, const std::filesystem::path& target_path) override {
        ref_ = load_matrix_from_csv<T, relative_zero_padding<2>, ALLOC>(ref_path);
        targets_ = load_matrix_array_from_csv<T, relative_zero_padding<2>, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        // Input matrices are padded with zeroes to twice their size
        // so that we can just do FFT, hadamard and inverse and have the resutls
        results_ = data_array<T, ALLOC>{ref_.matrix_size(), targets_.num_matrices()};

        fft_buffer_size_ = ref_.matrix_size().y * (ref_.matrix_size().x / 2 + 1);

        // 1 for the ref matrix
        auto num_inputs = 1 + targets_.num_matrices();
        CPU_MEASURE(this->label_index(3),
            cuda_malloc(&d_inputs_, ref_.size() + targets_.size());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_inputs_fft_, num_inputs * fft_buffer_size_);
        );

        int input_sizes[2] = {static_cast<int>(ref_.matrix_size().y), static_cast<int>(ref_.matrix_size().x)};
        int result_sizes[2] = {static_cast<int>(results_.matrix_size().y), static_cast<int>(results_.matrix_size().x)};
        CPU_MEASURE(this->label_index(4),
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, input_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs));


            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, result_sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        cuda_memcpy_to_device(d_inputs_, ref_);
        cuda_memcpy_to_device(d_inputs_ + ref_.size(), targets_);
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
                targets_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_MEASURE(this->label_index(2),
            fft_complex_to_real(fft_inv_plan_, d_inputs_fft_ + fft_buffer_size_, d_results_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_original_alg_one_to_many<T, DEBUG, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "Allocation",
    "Plan"
};

template<typename T, bool DEBUG = false, typename ALLOC = std::allocator<T>>
class fft_reduced_transfer_one_to_many: public one_to_many<T, ALLOC> {
public:
    explicit fft_reduced_transfer_one_to_many([[maybe_unused]] const json& args)
        :one_to_many<T, ALLOC>(true, labels.size()), ref_(), targets_(), results_(), fft_buffer_size_(0)
    {
        scatter_threads_per_block_  = args.value("scatter_threads_per_block", 256);
        scatter_items_per_thread_ = args.value("scatter_items_per_thread", 10);
        hadamard_threads_per_block_ = args.value("hadamard_threads_per_block", 256);
    }

    const data_array<T, ALLOC>& refs() const override {
        return ref_;
    }

    const data_array<T, ALLOC>& targets() const override {
        return targets_;
    }

    const data_array<T, ALLOC>& results() const override {
        return results_;
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
        targets_ = load_matrix_array_from_csv<T, no_padding, ALLOC>(target_path);

        this->check_matrices_same_size(ref_, targets_);
    }

    void prepare_impl() override {
        padded_matrix_size_ = 2 * ref_.matrix_size();
        fft_buffer_size_ = padded_matrix_size_.y * (padded_matrix_size_.x / 2 + 1);
        // 1 for the ref matrix
        num_inputs_ = 1 + targets_.num_matrices();

        // Input matrices are NOT padded
        results_ = data_array<T, ALLOC>{padded_matrix_size_, targets_.num_matrices()};

        CPU_MEASURE(this->label_index(5),
            cuda_malloc(&d_inputs_, ref_.size() + targets_.size());
            cuda_malloc(&d_padded_inputs_, num_inputs_ * padded_matrix_size_.area());
            cuda_malloc(&d_results_, results_.size());

            cuda_malloc(&d_padded_inputs_fft_, num_inputs_ * fft_buffer_size_);
        );

        int sizes[2] = {static_cast<int>(padded_matrix_size_.y), static_cast<int>(padded_matrix_size_.x)};
        CPU_MEASURE(this->label_index(6),
            // With nullptr inembed and onembed, the values for istride, idist, ostride and odist are ignored
            FFTCH(cufftPlanMany(&fft_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_R2C<T>(), num_inputs_));

            FFTCH(cufftPlanMany(&fft_inv_plan_, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, fft_type_C2R<T>(), results_.num_matrices()));
        );
    }

    void transfer_impl() override {
        CPU_MEASURE(this->label_index(3),
            cuda_memcpy_to_device(d_inputs_, ref_);
            cuda_memcpy_to_device(d_inputs_ + ref_.size(), targets_);
        );

        cuda_memset(d_padded_inputs_, 0, num_inputs_ * padded_matrix_size_.area());

        CUDA_MEASURE(this->label_index(4),
            run_scatter(
                d_inputs_,
                d_padded_inputs_,
                ref_.matrix_size(),
                num_inputs_,
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
                targets_.num_matrices(),
                hadamard_threads_per_block_)
        );

        CPU_MEASURE(this->label_index(2),
            fft_complex_to_real(fft_inv_plan_, d_padded_inputs_fft_ + fft_buffer_size_, d_results_)
        );

        CUCH(cudaDeviceSynchronize());
        CUCH(cudaGetLastError());
    }

    void finalize_impl() override {
        cuda_memcpy_from_device(results_, d_results_);
    }

    std::vector<std::string> measurement_labels_impl() const override {
        return labels;
    }

private:
    using fft_real_t = typename real_trait<T>::type;
    using fft_complex_t = typename complex_trait<T>::type;

    static std::vector<std::string> labels;

    data_array<T, ALLOC> ref_;
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

template<typename T, bool DEBUG, typename ALLOC>
std::vector<std::string> fft_reduced_transfer_one_to_many<T, DEBUG, ALLOC>::labels{
    "Forward FFT",
    "Hadamard",
    "Inverse FFT",
    "ToDevice",
    "Scatter",
    "Allocation",
    "Plan"
};


}
