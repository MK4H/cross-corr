#include "argument_error.hpp"

argument_error::argument_error(const std::string &msg, std::string arg_name)
    :std::runtime_error(msg), arg_name_(std::move(arg_name))
{

}
