#include <pybind11/pybind11.h>
#include "knn.hpp"

namespace py = pybind11;

// Documentation: https://pybind11.readthedocs.io/en/stable/classes.html

PYBIND11_MODULE(mnpkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    py::class_<KNNClassifier> knn(m, "KNNClassifier");
    knn.def(py::init<uint &>(), py::arg("k_neighbors"))
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def_readwrite("k_neighbors", &KNNClassifier::k);
}
