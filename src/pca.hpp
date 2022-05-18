#include "types.hpp"

using namespace std;

class PCA {
public:
    // Constructor
    PCA(uint n_components);

    // Methods
    void fit(const std::vector<std::vector<int>> list);
    Matrix transform(Matrix X);

    //Attributes
    uint alpha;
    Matrix eigenvectors;
private:
    pair<Vector, Matrix> _calculate_eigenvalues(const Matrix& X, uint num, uint num_iter);
    pair<double, Vector> _power_method(Matrix A, uint iter);
    Matrix _deflate(const Matrix& A, pair<double, Vector> eigen);
};
