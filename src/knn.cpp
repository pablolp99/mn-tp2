#include <algorithm>
#include <iostream>
#include <map>
#include "knn.hpp"
#include "utils.hpp"

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

void KNNClassifier::fit(const std::vector<std::vector<int> > list, const std::vector<int> label, uint imgs, uint img_size){
    Matrix X = read_input_data(list, imgs, img_size);
    Vector y = read_input_label(label, imgs);
    this->_fit(X, y);
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    this->train_size = uint (X.rows());
}

Vector KNNClassifier::predict(const std::vector<std::vector<int> > list, uint imgs, uint img_size){
    Matrix X = read_input_data(list, imgs, img_size);
    return this->_predict(X);
}

Vector KNNClassifier::_predict(Matrix X) {
    Vector res(this->train_size);
    
    for (uint i = 0; i < this->train_size; ++i){
        uint pred = _predict_vector(X.row(i));
        res(i) = pred;
    }

    return res;
}

uint KNNClassifier::_predict_vector(Vector x) {
    // KNN Algorithm
    uint res;

    // Calculate distances to all training cases

    // Sort (smallest to largest)

    // Keep the k closest vectors

    // Vote for the most suitable result

    return res;
}