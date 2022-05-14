#include <iostream>
#include "pca.h"
#include "utils.hpp"

using namespace std

PCA::PCA(uint n_components) {
    this->alpha = n_components;
}

void PCA::fit(const std::vector<std::vector<int> > list) {
    Matrix X = read_input_data(list, 5, 5);

    // n = filas de X

    // Generar vector \mu = (x_1 + ... + x_n) / n (promedio de las imagenes)

    // X_i = (x_i - \mu) / sqrt(n-1)

    // M = X^t * X

    // Conseguir los eigenvalues con el metodo de la potencia para todas las columnas de M (eigenvectors)
}

pair<double, Vector> PCA::_power_method(Matrix A, Vector x0, uint iter) {
    Vector v(x0);
    uint i = 0;

    while(i < iter) {
        v = (A * v) / (A * v).norm();

        // TODO: Agregar otro parametro de parada (no hay cambio hace tantas iteraciones por ejemplo)

        i++;
    }

    int lambda = (v.transpose() * A * v) / ((A * v).norm()) ^ 2;

    return make_pair(lambda, v)
}

Matrix PCA::transform(Matrix X) {
    return X * this->eigenvectors;
}