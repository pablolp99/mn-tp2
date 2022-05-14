#include "types.hpp"
#include <map>

using namespace std;

class KNNClassifier {
    public:
        KNNClassifier(uint k_neighbors);
        KNNClassifier& fit(Matrix X, Vector y);
        Vector predict(Matrix X);
        uint k;

    private:
        Matrix train;
        Vector target;
};
