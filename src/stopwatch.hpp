#pragma once

#include <vector>
#include <chrono>
#include <string>

#include <cuda_runtime.h>

namespace cross {

#define CUDA_MEASURE(label, block)              \
    do {                                        \
        auto m__ = this->sw_.cuda_start(label); \
        block;                                  \
        m__.insert_stop();                      \
        this->sw_.cuda_measure(m__);            \
    } while(0)

#define CPU_MEASURE(label, block)               \
    do {                                        \
        auto m__ = this->sw_.cpu_start(label);  \
        block;                                  \
        this->sw_.cpu_measure(m__);             \
    } while(0)

template<typename CLOCK>
class cpu_measurement {
public:
    cpu_measurement(std::size_t label, typename CLOCK::time_point start)
        :label_(label), start_(start)
    {

    }

    std::size_t get_label() const {
        return label_;
    }

    typename CLOCK::time_point get_start() const {
        return start_;
    }

private:
    std::size_t label_;
    typename CLOCK::time_point start_;
};

class cuda_measurement {
public:
    cuda_measurement(std::size_t label, cudaEvent_t start, cudaEvent_t stop)
        :label_(label), start_(start), stop_(stop)
    {
        CUCH(cudaEventRecord(start_));
    }

    void insert_stop() {
        CUCH(cudaEventRecord(stop_));
    }

    std::size_t get_label() const {
        return label_;
    }

    cudaEvent_t get_start() const {
        return start_;
    }

    cudaEvent_t get_stop() const {
        return stop_;
    }
private:
    std::size_t label_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

template<typename CLOCK>
class stopwatch {
public:
    stopwatch(std::size_t num_measurements)
        :measurements_(num_measurements), events_(2*num_measurements)
    {
        for (auto&& event: events_) {
            CUCH(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
        }
    }

    ~stopwatch() {
        for (auto&& event: events_) {
            CUCH(cudaEventDestroy(event));
        }
    }

    typename CLOCK::time_point now() {
        return CLOCK::now();
    }

    cpu_measurement<CLOCK> cpu_start(std::size_t label) {
        return cpu_measurement<CLOCK>{label, now()};
    }

    void cpu_measure(const cpu_measurement<CLOCK>& measurement) {
        cpu_measure(measurement.get_label(), measurement.get_start());
    }

    void cpu_measure(std::size_t label, typename CLOCK::time_point start) {
        measurements_[label] = CLOCK::now() - start;
    }

    cuda_measurement cuda_start(std::size_t label) {
        return cuda_measurement{label, events_[2*label], events_[2*label + 1]};
    }

    void cuda_measure(const cuda_measurement& measurement) {
        cuda_measure(measurement.get_label(), measurement.get_start(), measurement.get_stop());
    }

    void cuda_measure(std::size_t label, cudaEvent_t start, cudaEvent_t stop) {
        CUCH(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUCH(cudaEventElapsedTime(&milliseconds, start, stop));
        measurements_[label] = std::chrono::duration_cast<typename CLOCK::duration>(std::chrono::duration<double, std::milli>(milliseconds));
    }

    void store_measurement(std::size_t label, typename CLOCK::duration measurement) {
        measurements_[label] = measurement;
    }

    const std::vector<typename CLOCK::duration> results() const {
        return measurements_;
    }
private:
    std::vector<typename CLOCK::duration> measurements_;
    std::vector<cudaEvent_t> events_;

};

}