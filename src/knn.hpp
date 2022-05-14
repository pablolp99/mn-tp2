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
        // Private methods
        void _fit(Matrix X, Vector y);
        Vector _predict(Matrix X);
        uint _predict_vector(Vector x);
};
