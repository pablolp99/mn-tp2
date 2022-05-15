#include <iostream>
#include "pca.hpp"
#include "utils.hpp"

using namespace std

#define SEEN_VECTORS_AMOUNT 5

PCA::PCA(uint n_components) {
    this->alpha = n_components;
}

void PCA::fit(const std::vector<std::vector<int> > list) {
    Matrix X = read_input_data(list, 5, 5);

    // Promedio de las imagenes
    Vector u = X.colwise().mean();

    X.
    // X_i = (x_i - \mu) / sqrt(n-1)

    Matrix M = X^t * X

    // Conseguir los eigenvalues con el metodo de la potencia para todas las columnas de M (eigenvectors)
}

pair<Vector, Matrix> calculate_eigenvalues(const Matrix X, uint num, uint num_iter, double epsilon) {
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for(uint i = 0; i < num; i++){

        pair<double, Vector> eigen = _power_method(A, num_iter);
        eigvalues(i) = get<0>(eigen);
        eigvectors.col(i) = get<1>(eigen);
        A = _deflate(A, eigen);
    }

    return make_pair(eigvalues, eigvectors);
}

pair<double, Vector> PCA::_power_method(Matrix A, uint iter) {
    Vector v = Vector::Random(A.cols());;
    uint i = 0;

    queue<Vector> last_vectors;
    queue.push(v);

    while(i < iter) {
        v = (A * v) / (A * v).norm();

        // Miro los ultimos n autovectores encontrados y veo si no cambio en mucho tiempo
        for (Vector x : last_vectors) {
            if(x != v) {
                break;
            }
            if(x == last_vectors.last()) {
                int lambda = (v.transpose() * A * v) / ((A * v).norm()) ^ 2;

                return make_pair(lambda, v)
            }
        }

        // Agrego el vector encontrado a la lista de los ultimos n autovectores encontrados
        queue.push(v);
        if (last_vectors.size() > SEEN_VECTORS_AMOUNT) {
            queue.pop();
        }

        i++;
    }

    throw invalid_argument( "No se pudo encontrar el eigenvalue" );
}

Matrix _deflate(const Matrix& A, pair<double, Vector> eigen) {
    double eigenval = get<0>(eigen);
    Vector eigenvec = get<1>(eigen);
    return  A - (eigenval * eigenvec * eigenvec.transpose());
}

Matrix PCA::transform(Matrix X) {
    return X * this->eigenvectors;
}