#include <pybind11/pybind11.h>
#include "knn.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mnpkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<uint &>(), py::arg("k_neighbors"))
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def("set_params", &KNNClassifier::set_params)
        .def("get_params", &KNNClassifier::get_params);
}
