#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "knn.hpp"
#include "pca.hpp"
#include "utils.hpp"

namespace py = pybind11;

// Documentation: https://pybind11.readthedocs.io/en/stable/classes.html

PYBIND11_MODULE(mnpkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    py::class_<KNNClassifier> knn(m, "KNNClassifierCpp");
    knn.def(py::init<uint &>(), py::arg("k_neighbors"))
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def_readwrite("k_", &KNNClassifier::k)
        .def_readwrite("train_", &KNNClassifier::train)
        .def_readwrite("target_", &KNNClassifier::target);

    py::class_<PCA> pca(m, "PCACpp");
    pca.def(py::init<uint &>(), py::arg("k_neighbors"))
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def_readwrite("alpha_", &PCA::alpha)
        .def_readwrite("eigenvectors_", &PCA::eigenvectors);


// This function should not be exported to python
    // m.def("read_img", &read_input_data, "Convert Python list to Eigen representation");
}
