#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "types.hpp"

#ifndef UTILS_H
#define UTILS_H

namespace py = pybind11;

Matrix read_input_data(const std::vector<std::vector<int> > list);
Vector read_input_label(const std::vector<int> list);

#endif // UTILS_H