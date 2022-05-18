#include <iostream>
#include "pca.hpp"
#include "utils.hpp"

#include <deque>

using namespace std;

#define SEEN_VECTORS_AMOUNT 5
#define ITERACIONES 5000

PCA::PCA(uint n_components) {
    this->alpha = n_components;
}

void PCA::fit(const std::vector<std::vector<int>> list) {
    Matrix X = read_input_data(list);

    // Promedio de las imagenes
    Vector u = X.colwise().mean();

    // X_i = (x_i - \mu)^t / sqrt(n-1)
    X.rowwise() -= u.transpose(); // TODO: Revisar si esto funciona bien. Deberia tomar cada fila y restarle u
    X /= sqrt(X.rows()-1);

    // M = X^t*X
    Matrix M = X.transpose() * X;

    // Conseguir los eigenvalues con el metodo de la potencia para todas las columnas de M (eigenvectors)
    eigenvectors = get<1>(_calculate_eigenvalues(M, alpha, ITERACIONES));
}

pair<Vector, Matrix> PCA::_calculate_eigenvalues(const Matrix &X, uint num, uint num_iter) {
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

    deque<Vector> last_vectors;
    last_vectors.push_back(v);

    while(i < iter) {
        v = (A * v) / (A * v).norm();

        // Miro los ultimos n autovectores encontrados y veo si no cambio en mucho tiempo
        for (Vector x : last_vectors) {
            if(x != v) {
                break;
            }
            if(x == last_vectors.back()) {
                double lambda = v.transpose() * A * v;

                lambda /= pow((A * v).squaredNorm(), 2);

                return make_pair(lambda, v);
            }
        }

        // Agrego el vector encontrado a la lista de los ultimos n autovectores encontrados
        last_vectors.push_back(v);
        if (last_vectors.size() > SEEN_VECTORS_AMOUNT) {
            last_vectors.pop_front();
        }

        i++;
    }

    throw invalid_argument( "No se pudo encontrar el eigenvalue" );
}

Matrix PCA::_deflate(const Matrix& A, pair<double, Vector> eigen) {
    double eigenval = get<0>(eigen);
    Vector eigenvec = get<1>(eigen);
    return  A - (eigenval * eigenvec * eigenvec.transpose());
}

Matrix PCA::transform(Matrix X) {
    return X * eigenvectors;
}