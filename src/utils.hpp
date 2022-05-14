#include <pybind11/pybind11.h>
#include "types.hpp"

#ifndef UTILS_H
#define UTILS_H

namespace py = pybind11;

// Parser de imagenes. Que reciba el vector con todos los pixeles
// y convierta eso a una Matriz de Eigen
// Ver: https://stackoverflow.com/questions/50883703/pybind-how-can-i-operate-over-a-pylist-object
// TL;DR: usar el tipo `py::list` para comprender una lista. Iterador: `py::handle obj : l`

Matrix read_input_data(const std::vector<std::vector<float> > list, uint imgs, uint img_size);

#endif // UTILS_H