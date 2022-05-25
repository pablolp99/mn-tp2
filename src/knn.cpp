#include <algorithm>
#include <iostream>
#include <map>
#include "omp.h"

#include "knn.hpp"
#include "utils.hpp"

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

void KNNClassifier::set_train(const Matrix& train) {
    this->train = train;
}

void KNNClassifier::set_target(const Vector& target) {
    this->target = target;
}

void KNNClassifier::set_train_size(const uint& train_size) {
    this->train_size = train_size;
}

void KNNClassifier::fit(const std::vector<std::vector<double> > list, const std::vector<int> label){
    Matrix X = read_input_data(list);
    Vector y = read_input_label(label);
    cout << "--KNN-- fitting with k: " << this->k << " and alpha: " << X.cols() << endl;
    _fit(X, y);
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    this->train_size = uint(X.rows());
}

Vector KNNClassifier::predict(const std::vector<std::vector<double> > list){
    Matrix X = read_input_data(list);

    Vector res(X.rows());

    #pragma omp parallel for
    for (uint i = 0; i < X.rows(); ++i){
        Vector x = X.row(i);
        res(i) = _predict(x);
    }

    return res;
}

int KNNClassifier::_predict(Vector x) {
    // KNN Algorithm
    Vector distances(train_size);
    Vector indexes(train_size);

    // Calculate distances to all training cases
    #pragma omp parallel for
    for (int j = 0; j < train_size; ++j){
        Vector v = train.row(j);
        distances(j) = ((x - v).norm());
        indexes(j) = target(j);
    }

    std::vector<pair<float, int> > dist_idx;
    for (int j = 0; j < train_size; ++j){
        dist_idx.push_back(make_pair(distances(j), int(indexes(j))));
    }

    // Sort (smallest to largest)
    sort(dist_idx.begin(), dist_idx.end());

    // Vote for the most suitable result
    std::map<uint, uint> count_map;

    for (uint j = 0; j < k; j++) {
        // Key exists in map
        if ( count_map.find(dist_idx[j].second) == count_map.end() ) {
            count_map[dist_idx[j].second] = 1;
        } else {
            count_map[dist_idx[j].second] += 1;
        }
    }

    // Find the class with the most frequency.
    int current_max_freq = count_map[dist_idx[0].second];
    int current_class = dist_idx[0].second;

    for(auto it = count_map.begin(); it != count_map.end(); ++it ) {
        // If there is a tie, we choose the first one
        // to appear in the search
        if (it->second > current_max_freq){
            current_max_freq = it->second;
            current_class = it->first;
        }
    }
    
    return current_class;
}