#include "types.h"

using namespace std;

class PCA {
public:
    PCA(uint n_components);

    void fit(Matrix X);

    Matrix transform(Matrix X);
private:
    uint alpha;
    Matrix eigenvectors;
    void power_method();
};
