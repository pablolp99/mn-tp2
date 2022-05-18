#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "knn.hpp"
#include "pca.hpp"
#include "utils.hpp"

namespace py = pybind11;

// Documentation: https://pybind11.readthedocs.io/en/stable/classes.html

PYBIND11_MODULE(metnum_pkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    py::class_<KNNClassifier> knn(m, "KNNClassifierCpp");
    knn.def(py::init<uint &>(), py::arg("k_neighbors"))
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def_readwrite("k_", &KNNClassifier::k)
        .def_readwrite("train_", &KNNClassifier::train)
        .def_readwrite("target_", &KNNClassifier::target)
        // Pickle support
        .def(py::pickle(
            [](const KNNClassifier &knn_classifier) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(knn_classifier.k, knn_classifier.train, knn_classifier.target, knn_classifier.train_size);
            },
            [](py::tuple knn_tuple_representation) { // __setstate__
                if (knn_tuple_representation.size() != 4)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                KNNClassifier knn_classifier(knn_tuple_representation[0].cast<uint>());

                /* Assign any additional state */
                knn_classifier.set_train(knn_tuple_representation[1].cast<Matrix>());
                knn_classifier.set_target(knn_tuple_representation[2].cast<Vector>());
                knn_classifier.set_train_size(knn_tuple_representation[3].cast<uint>());

                return knn_classifier;
            }
        ));

    py::class_<PCA> pca(m, "PCACpp");
    pca.def(py::init<uint &, double &>(), py::arg("alpha"), py::arg("epsilon"))
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def_readwrite("alpha_", &PCA::alpha)
        .def_readwrite("eigenvectors_", &PCA::eigenvectors)
        .def(py::pickle(
            [](const PCA &pca_classifier) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(pca_classifier.alpha, pca_classifier.epsilon, pca_classifier.eigenvectors);
            },
            [](py::tuple pca_tuple_representation) { // __setstate__
                if (pca_tuple_representation.size() != 3)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                PCA pca_classifier(pca_tuple_representation[0].cast<uint>(), pca_tuple_representation[1].cast<double>());

                /* Assign any additional state */
                pca_classifier.set_eigenvectors(pca_tuple_representation[2].cast<Matrix>());

                return pca_classifier;
            }
        ));
}