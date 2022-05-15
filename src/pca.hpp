#include "types.hpp"

using namespace std;

class PCA {
public:
    // Constructor
    PCA(uint n_components);

    // Methods
    void fit(Matrix X);
    Matrix transform(Matrix X);

    //Attributes
    uint alpha;
    Matrix eigenvectors;
private:
    pair<double, Vector> _power_method(Matrix A, Vector x0, uint iter);
    Matrix _deflate(const Matrix& A, pair<double, Vector> eigen);
};
