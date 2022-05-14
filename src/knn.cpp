#include <algorithm>
#include <iostream>
#include <map>
#include "knn.hpp"
#include "utils.hpp"

using namespace std;

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

KNNClassifier& KNNClassifier::fit(const std::vector<std::vector<int> > list, const std::vector<int> y, uint imgs, uint img_size){
    Matrix X = read_input_data(list, imgs, img_size);
    Vector y = read_input_label(y, imgs);
    return this->_fit(X, y);
}

KNNClassifier& KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    return *this;
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