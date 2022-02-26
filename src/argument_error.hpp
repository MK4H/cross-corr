#pragma once

#include <string>
#include <stdexcept>

class argument_error: public std::runtime_error {
public:
    argument_error(const std::string& msg, std::string arg_name);

    [[nodiscard]] const std::string& arg_name() const {
        return arg_name_;
    }
private:
    std::string arg_name_;
};
