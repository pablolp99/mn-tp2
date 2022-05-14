#include <vector>
#include "types.hpp"

using namespace std;

class KNNClassifier {
    public:
        // Constructor
        KNNClassifier(uint k_neighbors);

        // Methods
        void fit(const std::vector<std::vector<int> > list, const std::vector<int> y, uint imgs, uint img_size);
        Vector predict(const std::vector<std::vector<int> > list, uint imgs, uint img_size);
        
        // Attributes
        uint k;
        Matrix train;
        Vector target;
        uint train_size;

    private:
        Vector pred_vector_;

        // Private methods
        void _fit(Matrix X, Vector y);
        // void _predict(Matrix X);
        // int _predict_vector(Vector x);
};
