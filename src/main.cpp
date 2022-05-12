#include <pybind11/pybind11.h>

int add_example(int a, int b){
    return a + b;
}

PYBIND11_MODULE(mnpkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    m.def("add_example", &add_example);
}
