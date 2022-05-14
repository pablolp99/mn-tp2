#include <pybind11/pybind11.h>
#include "types.hpp"

#ifndef UTILS_H
#define UTILS_H

namespace py = pybind11;

Matrix read_input_data(const std::vector<std::vector<int> > list, uint imgs, uint img_size);

#endif // UTILS_H