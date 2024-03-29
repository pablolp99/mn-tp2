#include <vector>
#include "types.hpp"

using namespace std;


class KNNClassifier {
    public:
        // Constructor
        KNNClassifier(uint k_neighbors);

        // Methods
        void fit(const std::vector<std::vector<double> > list, const std::vector<int> y);
        Vector predict(const std::vector<std::vector<double> > list);
        
        // Attributes
        uint k;
        Matrix train;
        Vector target;
        uint train_size;

        // Setters for pybind11 integration
        void set_train(const Matrix& train);
        void set_target(const Vector& target);
        void set_train_size(const uint& train_size);

    private:
        // Private methods
        void _fit(Matrix X, Vector y);
        int _predict(Vector x);
        std::vector<pair<float, int> > calculate_distances(Vector x);
};
