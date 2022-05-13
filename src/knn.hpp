#include "types.hpp"
#include <map>

using namespace std;

class KNNClassifier {
    public:
        KNNClassifier(uint k_neighbors);
        KNNClassifier& fit(Matrix X, Vector y);
        Vector predict(Matrix X);
        map<string, uint>& get_params(bool deep);
        KNNClassifier& set_params(map<string, uint> params);

    private:
        uint k;
        Matrix train;
        Vector target;
        map<string, uint> params;
        void _fit(Matrix X, Vector y);
};
