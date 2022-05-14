#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "utils.hpp"

namespace py = pybind11;

Matrix read_input_data(const std::vector<std::vector<int> > list, uint imgs, uint img_size) {
    Matrix result = Matrix(imgs, img_size);

    uint i = 0;
    for (auto img : list) {
        uint j = 0;
        for (auto pxl : img) {
            result(i, j) = double(pxl);
            ++j;
        }
        ++i;
    }

    return result;
}