#include <algorithm>
#include <iostream>
#include <map>
#include "knn.hpp"

using namespace std;

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
    map<string, uint> params = {{'k_neighbors', this->k}};
    this->params = params;
}

KNNClassifier& KNNClassifier::fit(Matrix X, Vector y) {
    this->_fit(X, y);
    return *this;
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
}

Vector KNNClassifier::predict(Matrix X) {
    Vector res = Vector(1);
    return res;
}

map<string, uint> get_params(bool deep) {
    return this->params;
}

KNNClassifier& KNNClassifier::set_params(map<string, uint> params) {
    try {
        this->params = params;
        this->k = params.at("k_neighbors");
    }
    catch (int e) {} ;
    return *this;
}