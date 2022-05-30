#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "utils.hpp"

namespace py = pybind11;

Matrix read_input_data(const std::vector<std::vector<double> > list) {
    uint imgs = list.size();
    uint img_size = list.at(0).size();

    Matrix result(imgs, img_size);

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

Vector read_input_label(const std::vector<int> list) {
    uint imgs = list.size();
    Vector result(imgs);

    uint i = 0;
    for (auto label : list) {
        result(i) = double(label);
        ++i;
    }
    return result;
}

Vector init_random_vector(uint size){
    return Eigen::VectorXd::Random(size);
}