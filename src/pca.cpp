#include <iostream>
#include <deque>

#include "progressbar.hpp"
#include "pca.hpp"
#include "utils.hpp"

using namespace std;

#define SEEN_VECTORS_AMOUNT 5
#define ITERACIONES 5000

PCA::PCA(uint alpha, double epsilon) {
    this->alpha = alpha;
    this->epsilon = epsilon;
}

void PCA::set_eigenvectors(Matrix eigenvectors) {
    this->eigenvectors = eigenvectors;
}

void PCA::set_covariance_by_component(Vector covariance_by_component) {
    this->covariance_by_component = covariance_by_component;
}

void PCA::fit(const std::vector<std::vector<int>> list) {
    Matrix X = read_input_data(list);

    // Promedio de las imagenes
    Vector u = X.colwise().mean();

    // X_i = (x_i - \mu)^t / sqrt(n-1)
    X.rowwise() -= u.transpose();
    X /= sqrt(X.rows()-1);

    // M = X^t*X
    Matrix M = X.transpose() * X;
    cout << M.diagonal() << endl;

    // Calcular los autovectores mediante el metodo de la potencia
    eigenvectors = get<1>(_calculate_eigenvalues(M));

    Matrix M_x = eigenvectors.transpose() * M * eigenvectors;
    this->covariance_by_component = M_x.diagonal();
}

pair<Vector, Matrix> PCA::_calculate_eigenvalues(const Matrix &X) {
    Matrix A(X);
    Vector eigvalues(alpha);
    Matrix eigvectors(A.rows(), alpha);

    // progressbar bar(alpha);

    for(uint i = 0; i < alpha; i++){
        pair<double, Vector> eigen_val_and_vec = _power_method(A);
        eigvalues(i) = get<0>(eigen_val_and_vec);
        eigvectors.col(i) = get<1>(eigen_val_and_vec);
        A = _deflate(A, eigen_val_and_vec);
        // bar.update();
    }
    std::cout << std::endl;

    return make_pair(eigvalues, eigvectors);
}

pair<double, Vector> PCA::_power_method(Matrix A) {
    Vector v = init_random_vector(A.rows());
    double lambda = 0;
    uint i = 0;

    Vector last_vector;

    while(i < ITERACIONES) {
        v = (A * v) / (A * v).norm();
        double _tmp_num = (v.transpose() * A * v);
        double _tmp_den = (v.transpose() * v);
        lambda = _tmp_num / _tmp_den ;
        
        if((i > 0) && (v - last_vector).isZero(epsilon)) {
            return make_pair(lambda, v);
        }

        last_vector = v;

        i++;
    }

    return make_pair(lambda, v);
}

Matrix PCA::_deflate(const Matrix& A, pair<double, Vector> eigen_val_and_vec) {
    double eigenval = get<0>(eigen_val_and_vec);
    Vector eigenvec = get<1>(eigen_val_and_vec);
    return  A - (eigenval * eigenvec * eigenvec.transpose());
}

Matrix PCA::transform(const std::vector<std::vector<int>> list) {
    Matrix X = read_input_data(list);
    return X * eigenvectors;
}