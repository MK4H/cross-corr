#pragma once

#include <vector>
#include <chrono>
#include <string>

#include <cuda_runtime.h>

#include "cuda_helpers.cuh"

namespace cross {

template<typename DURATION>
class measurement_result {
public:
    measurement_result()
        :measurement_result<DURATION>(DURATION::zero(), 0)
    {
    }

    measurement_result(DURATION total_time, std::size_t iterations)
        :total_time_(total_time), iterations_(iterations)
    {
    }

    DURATION get_iteration_time() const {
        if (total_time_ == DURATION::zero()) {
            return DURATION::zero();
        }
        if (iterations_ == 0) {
            throw std::runtime_error("Zero iterations with non-zero total time");
        }

        return total_time_ / iterations_;
    }

    DURATION get_total_time() const {
        return total_time_;
    }

    std::size_t get_iterations() const {
        return iterations_;
    }
private:
    DURATION total_time_;
    std::size_t iterations_;
};

template<typename DURATION>
measurement_result<DURATION> create_result(DURATION total_time, std::size_t iterations) {
    return measurement_result<DURATION>{
        total_time,
        iterations
    };
}

template<typename MEASURED_DURATION>
std::size_t get_next_iteration_count(std::chrono::nanoseconds min_time, const measurement_result<MEASURED_DURATION>& res) {
    auto total_time = res.get_total_time();
    if (total_time >= min_time) {
        return 0;
    }

    auto total_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_time);

    auto ratio = static_cast<double>(min_time.count()) / static_cast<double>(total_nanoseconds.count());
    auto next_iterations = std::ceil(res.get_iterations() * std::max(std::min(ratio * 1.5, 100.0), 1.5));

    return static_cast<std::size_t>(next_iterations);

}

#define CUDA_SYNC_MEASURE(label, enable, stopwatch, block)  \
    do {                                                    \
        if (enable) {                                       \
            auto m__ = (stopwatch).cuda_start(label);       \
            block;                                          \
            m__.insert_stop();                              \
            (stopwatch).cuda_sync_measure(m__);             \
        } else {                                            \
            block;                                          \
        }                                                   \
    } while(0)

#define CUDA_SYNC_REPEATED_MEASURE(label, enable, num_repeats, stopwatch, block)\
    do {                                                                        \
        if (enable) {                                                           \
            auto m__ = (stopwatch).cuda_start(label);                           \
            for (auto i__ = 0; i__ < (num_repeats); ++i__ ) {                   \
                block;                                                          \
            }                                                                   \
            m__.insert_stop();                                                  \
            (stopwatch).cuda_sync_measure(m__, num_repeats);                    \
        } else {                                                                \
            block;                                                              \
        }                                                                       \
    } while(0)

#define CUDA_ASYNC_MEASURE(label, enable, stopwatch, block) \
    do {                                                    \
        if (enable) {                                       \
            auto m__ = (stopwatch).cuda_start(label);       \
            block;                                          \
            m__.insert_stop();                              \
            (stopwatch).cuda_async_measure(m__);            \
        } else {                                            \
            block;                                          \
        }                                                   \
    } while(0)

#define CUDA_ASYNC_REPEATED_MEASURE(label, enable, num_repeats, stopwatch, block)   \
    do {                                                                            \
        if (enable) {                                                               \
            auto m__ = (stopwatch).cuda_start(label);                               \
            for (auto i__ = 0; i__ < (num_repeats); ++i__ ) {                       \
                block;                                                              \
            }                                                                       \
            m__.insert_stop();                                                      \
            (stopwatch).cuda_async_measure(m__, num_repeats);                       \
        } else {                                                                    \
            block;                                                                  \
        }                                                                           \
    } while(0)

#define CUDA_ADAPTIVE_MEASURE(label, enable, stopwatch, block)                              \
    do {                                                                                    \
        if (enable) {                                                                       \
            std::size_t iterations__ = 1;                                                   \
            do {                                                                            \
                auto m__ = (stopwatch).cuda_start(label);                                   \
                for (std::size_t i__ = 0; i__ < iterations__; ++i__ ) {                     \
                    block;                                                                  \
                }                                                                           \
                m__.insert_stop();                                                          \
                auto t__ = (stopwatch).cuda_sync_measure(m__, iterations__);                \
                iterations__ = get_next_iteration_count((stopwatch).get_min_time(), t__);   \
            } while(iterations__ != 0);                                                     \
        } else {                                                                            \
            block;                                                                          \
        }                                                                                   \
    } while(0)

#define CPU_MEASURE(label, enable, stopwatch, gpu_sync, block)  \
    do {                                                        \
        if (enable) {                                           \
            auto m__ = (stopwatch).cpu_start(label);            \
            block;                                              \
            if ((gpu_sync)) {                                   \
                CUCH(cudaDeviceSynchronize());                  \
            }                                                   \
            (stopwatch).cpu_measure(m__);                       \
            if ((gpu_sync)) {                                   \
                CUCH(cudaGetLastError());                       \
            }                                                   \
        } else {                                                \
            block;                                              \
        }                                                       \
    } while(0)

#define CPU_REPEATED_MEASURE(label, enable, stopwatch, gpu_sync, num_repeats, block)  \
    do {                                                                    \
        if (enable) {                                                       \
            auto m__ = (stopwatch).cpu_start(label);                        \
            for (auto i__ = 0; i__ < (num_repeats); ++i__ ) {               \
                block;                                                      \
                if ((gpu_sync)) {                                           \
                    CUCH(cudaDeviceSynchronize());                          \
                }                                                           \
            }                                                               \
            (stopwatch).cpu_measure(m__, (num_repeats));                    \
            if ((gpu_sync)) {                                               \
                CUCH(cudaGetLastError());                                   \
            }                                                               \
        } else {                                                            \
            block;                                                          \
        }                                                                   \
    } while(0)

#define CPU_ADAPTIVE_MEASURE(label, enable, stopwatch, gpu_sync, block)                     \
    do {                                                                                    \
        if (enable) {                                                                       \
            std::size_t iterations__ = 1;                                                   \
            do {                                                                            \
                auto m__ = (stopwatch).cpu_start(label);                                    \
                for (std::size_t i__ = 0; i__ < (iterations__); ++i__ ) {                  \
                    block;                                                                  \
                    if ((gpu_sync)) {                                                       \
                        CUCH(cudaDeviceSynchronize());                                      \
                    }                                                                       \
                }                                                                           \
                auto t__ = (stopwatch).cpu_measure(m__, iterations__);                      \
                if ((gpu_sync)) {                                                           \
                    CUCH(cudaGetLastError());                                               \
                }                                                                           \
                iterations__ = get_next_iteration_count((stopwatch).get_min_time(), t__);   \
            } while(iterations__ != 0);                                                     \
        } else {                                                                            \
            block;                                                                          \
        }                                                                                   \
    } while(0)



class measurement {
public:
    explicit measurement(std::size_t label)
        :label_(label)
    {

    }

    std::size_t get_label() const {
        return label_;
    }
private:
    std::size_t label_;
};

template<typename CLOCK>
class cpu_measurement: public measurement {
public:
    cpu_measurement(std::size_t label, typename CLOCK::time_point start)
        :measurement(label), start_(start)
    {

    }

    typename CLOCK::time_point get_start() const {
        return start_;
    }

private:
    typename CLOCK::time_point start_;
};

class cuda_measurement: public measurement {
public:
    cuda_measurement(std::size_t label, cudaEvent_t start, cudaEvent_t stop)
        :measurement(label), start_(start), stop_(stop)
    {
        CUCH(cudaEventRecord(start_));
    }

    void insert_stop() {
        CUCH(cudaEventRecord(stop_));
    }

    cudaEvent_t get_start() const {
        return start_;
    }

    cudaEvent_t get_stop() const {
        return stop_;
    }
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

class cuda_event_pair {
public:
    cuda_event_pair()
        :start_(), stop_(), iterations_(0)
    {
        CUCH(cudaEventCreateWithFlags(&start_, cudaEventBlockingSync));
        CUCH(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
    }

    ~cuda_event_pair() {
        CUCH(cudaEventDestroy(stop_));
        CUCH(cudaEventDestroy(start_));
    }

    cudaEvent_t get_start() const {
        return start_;
    }

    cudaEvent_t get_stop() const {
        return stop_;
    }

    void set_iterations(std::size_t iterations) {
        iterations_ = iterations;
    }

    bool is_used() const {
        return iterations_ != 0;
    }

    std::size_t get_iterations() const {
        return iterations_;
    }

    void reset() {
        iterations_ = 0;
    }
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    std::size_t iterations_;
};

template<typename CLOCK>
class stopwatch {
public:
    using time_point = typename CLOCK::time_point;
    using duration = typename CLOCK::duration;
    using result = measurement_result<duration>;

    stopwatch(std::size_t num_measurements, std::chrono::nanoseconds min_time)
        :measurements_(num_measurements), min_time_(min_time), events_(num_measurements)
    {

    }

    std::chrono::nanoseconds get_min_time() const {
        return min_time_;
    }

    time_point now() {
        return CLOCK::now();
    }

    cpu_measurement<CLOCK> cpu_start(std::size_t label) {
        return cpu_measurement<CLOCK>{label, now()};
    }

    result cpu_measure(const cpu_measurement<CLOCK>& measurement, std::size_t iterations = 1) {
        return cpu_measure(measurement.get_label(), measurement.get_start(), iterations);
    }

    result cpu_measure(std::size_t label, time_point start, std::size_t iterations = 1) {
        auto t = CLOCK::now() - start;
        measurements_[label] = create_result(t, iterations);
        return measurements_[label];
    }

    cuda_measurement cuda_start(std::size_t label) {
        return cuda_measurement{label, events_[label].get_start(), events_[label].get_stop()};
    }

    result cuda_sync_measure(const cuda_measurement& measurement, std::size_t iterations = 1) {
        return cuda_sync_measure(measurement.get_label(), iterations);
    }

    result cuda_sync_measure(std::size_t label, std::size_t iterations = 1) {
        auto& events = events_[label];
        events.set_iterations(iterations);

        auto t = cuda_measure_events(events);
        measurements_[label] = create_result(t, events.get_iterations());

        return measurements_[label];
    }

    void cuda_async_measure(const cuda_measurement& measurement, std::size_t iterations = 1) {
        cuda_async_measure(measurement.get_label(), iterations);
    }

    void cuda_async_measure(std::size_t label, std::size_t iterations = 1) {
        events_[label].set_iterations(iterations);
    }

    void cuda_collect() {
        for (dsize_t i = 0; i < events_.size(); ++i) {
            const cuda_event_pair& events = events_[i];
            if (events.is_used()) {
                auto t = cuda_measure_events(events);
                measurements_[i] = create_result(t, events.get_iterations());
            }
        }
    }

    void store_measurement(std::size_t label, duration total_time, std::size_t iterations = 1) {
        measurements_[label] = create_result(total_time, iterations);
    }

    void reset() {
        std::fill(measurements_.begin(), measurements_.end(), measurement_result<typename CLOCK::duration>());
        for (auto&& event: events_) {
            event.reset();
        }
    }

    /**
     * cuda_collect MUST be called before retrieving results
     * otherwise the cuda measurements will not be present
     */
    const std::vector<result>& results() const {
        return measurements_;
    }
private:
    std::vector<result> measurements_;
    std::chrono::nanoseconds min_time_;
    std::vector<cuda_event_pair> events_;

    duration cuda_measure_events(const cuda_event_pair& events) {
        CUCH(cudaEventSynchronize(events.get_stop()));
        float milliseconds = 0;
        CUCH(cudaEventElapsedTime(&milliseconds, events.get_start(), events.get_stop()));

        return std::chrono::duration_cast<duration>(std::chrono::duration<double, std::milli>(milliseconds));
    }
};

}
