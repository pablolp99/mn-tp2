#include <algorithm>
#include <iostream>
#include <map>
#include "knn.hpp"
#include "utils.hpp"

using namespace std;

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
}

Vector KNNClassifier::predict(const std::vector<std::vector<int> > list, uint imgs, uint img_size){
    Matrix X = read_input_data(list, imgs, img_size);
    return this->_predict(X);
}

Vector KNNClassifier::_predict(Matrix X) {
    Vector res = Vector(1);
    // KNN Algorithm
    return res;
}