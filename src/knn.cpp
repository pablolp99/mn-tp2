#include <algorithm>
#include <iostream>
#include <map>
#include "knn.h"

using namespace std;

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

KNNClassifier KNNClassifier::fit(Matrix X, Vector y) {
    this->_fit(X, y);
    return this;
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
}

// Vector KNNClassifier::predict(Matrix X) {
//     return this->_predict(X);
// }

// Vector KNNClassifier::_predict(Matrix X) {
//     // KNN Algorithm implementation
//     // Return a Vector
//     Vector res = {};
//     return res;
// }

// map<string, string> get_params(bool deep) {
//     map<string, string> params = {{'k_neighbors', this->k.to_string()}}
//     return params;
// }

// KNNClassifier KNNClassifier::set_params(map<string, string> params) {
//     try {
//         this->k = params.at("k_neighbors")
//     }
//     catch () ;

//     return this;
// }