#include <algorithm>
#include <iostream>
#include <map>
#include "knn.hpp"

using namespace std;

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

KNNClassifier& KNNClassifier::fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    return *this;
}

Vector KNNClassifier::predict(Matrix X) {
    Vector res = Vector(1);

    // KNN Algorithm

    return res;
}