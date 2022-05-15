#include <vector>
#include "types.hpp"

using namespace std;


class KNNClassifier {
    public:
        // Constructor
        KNNClassifier(uint k_neighbors);

        // Methods
        void fit(const std::vector<std::vector<int> > list, const std::vector<int> y);
        Vector predict(const std::vector<std::vector<int> > list);
        
        // Attributes
        uint k;
        Matrix train;
        Vector target;
        uint train_size;

    private:
        Vector pred_vector_;

        // Private methods
        void _fit(Matrix X, Vector y);
        int _predict(Vector x);
        std::vector<pair<float, int> > calculate_distances(Vector x);
};
