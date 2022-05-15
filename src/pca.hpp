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

    pair<double, Vector> PCA::_power_method(Matrix A, Vector x0, uint iter)
    Matrix _deflate(const Matrix& A, pair<double, Vector> eigen);
};
