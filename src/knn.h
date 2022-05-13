#include "types.h"
#include <map>

using namespace std;

class KNNClassifier {
    public:
        KNNClassifier(uint k_neighbors);
        KNNClassifier fit(Matrix X, Vector y);
        // Vector predict(Matrix X);
        // map<string, string> get_params(bool deep);
        // KNNClassifier set_params(map<string, string> params);

    private:
        uint k;
        Matrix train;
        Vector target;
        void _fit(Matrix X, Vector y);
        // Vector _predict(Matrix X);
};
