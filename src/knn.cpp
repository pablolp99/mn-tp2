#include <algorithm>
#include <iostream>
#include <map>
#include "omp.h"

#include "knn.hpp"
#include "progressbar.hpp"
#include "utils.hpp"

KNNClassifier::KNNClassifier(uint k_neighbors) {
    this->k = k_neighbors;
}

void KNNClassifier::fit(const std::vector<std::vector<int> > list, const std::vector<int> label){
    Matrix X = read_input_data(list);
    Vector y = read_input_label(label);
    _fit(X, y);
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    this->train_size = uint(X.rows());
}

Vector KNNClassifier::predict(const std::vector<std::vector<int> > list){
    Matrix X = read_input_data(list);

    Vector res(X.rows());
    progressbar bar(X.rows());

    #pragma omp parallel for
    for (uint i = 0; i < X.rows(); ++i){
        Vector x = X.row(i);
        res(i) = _predict(x);
        bar.update();
    }

    return res;
}

int KNNClassifier::_predict(Vector x) {
    // KNN Algorithm
    std::vector<pair<float, int> > dist;

    // Calculate distances to all training cases
    for (int j = 0; j < train_size; ++j){
        Vector v = train.row(j);
        dist.push_back(make_pair( ((x - v).norm()) , target(j)));
    }

    // Sort (smallest to largest)
    sort(dist.begin(), dist.end());

    // Vote for the most suitable result
    std::map<uint, uint> count_map;

    for (uint j = 0; j < k; j++) {
        // Key exists in map
        if ( count_map.find(dist[j].second) == count_map.end() ) {
            count_map[dist[j].second] = 1;
        } else {
            count_map[dist[j].second] += 1;
        }
    }

    // Find the class with the most frequency.
    int current_max_freq = count_map[dist[0].second];
    int current_class = dist[0].second;

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