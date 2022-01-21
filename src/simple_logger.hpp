#pragma once

#include <iostream>
#include <string>

#include "validate.hpp"

/**
 * Toggleable ostream which serves as a decorator for another ostream.
 * Can be toggled on or off, ignoring any message print requests if toggled off
 */
namespace cross {

class simple_logger {
public:
    simple_logger(bool print_progress, bool append_result_stats)
        :print_progress_(print_progress), append_result_stats_(append_result_stats)
    {

    }

    void log(const std::string& message) {
        if (print_progress_) {
            std::cout << message << std::endl;
        }
    }

    void result_stats(const validation_results& results) {
        if (print_progress_) {
            std::cout << results;
        } else {
            to_csv(std::cout, results, !append_result_stats_);
        }
    }
private:
    bool print_progress_;
    bool append_result_stats_;
};

}