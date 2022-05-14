#include <vector>
#include "types.hpp"

using namespace std;

class KNNClassifier {
    public:
        KNNClassifier(uint k_neighbors);
        void fit(const std::vector<std::vector<int> > list, const std::vector<int> y, uint imgs, uint img_size);
        Vector predict(const std::vector<std::vector<int> > list, uint imgs, uint img_size);
        uint k;

    private:
        Matrix train;
        Vector target;
        uint train_size;
        void _fit(Matrix X, Vector y);
        Vector _predict(Matrix X);
        uint _predict_vector(Vector x);
};
