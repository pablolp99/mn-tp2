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
    _fit(X, y);
}

void KNNClassifier::_fit(Matrix X, Vector y) {
    this->train = X;
    this->target = y;
    this->train_size = uint(X.rows());
}

Vector KNNClassifier::predict(const std::vector<std::vector<int> > list, uint imgs, uint img_size){
    Matrix X = read_input_data(list, imgs, img_size);

    Vector res(X.rows());
    
    for (uint i = 0; i < X.rows(); ++i){
    //     // KNN Algorithm
    //     std::vector<pair<float, int> > dist;
        
    //     // Calculate distances to all training cases
    //     for (int i = 0; i < train_size; ++i){
    //         dist.push_back(make_pair(( train.row(i) - X.row(i) ).norm(), target(i)));
    //     };

    //     // Sort (smallest to largest)
    //     sort(dist.begin(), dist.end());

    //     // Vote for the most suitable result
    //     std::map<uint, uint> count_map;

    //     for (uint i = 0; i < k; i++) {
    //         uint cls = dist[i].second;
    //         auto it = count_map.find(cls);
            
    //         if (it != count_map.end()) {
    //             count_map[cls]++;
    //         } else {
    //             count_map[cls] = 1;
    //         }
    //     } 

    //     int current_max_freq = -1;
    //     int current_class = 0;
    //     for(auto it = count_map.begin(); it != count_map.end(); ++it ) {
    //         if ((int)it->second > current_max_freq) {
    //             current_max_freq = it->first;
    //             current_class = it->second;
    //         }
    //     }

        res(i) = -1;
    }

    return res;
}