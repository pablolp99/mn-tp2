#include "types.hpp"

using namespace std;

class PCA {
public:
    // Constructor
    PCA(uint alpha, double epsilon);

    // Methods
    void fit(const std::vector<std::vector<int>> list);
    Matrix transform(const std::vector<std::vector<int>> list);

    void set_alpha(uint alpha);
    void set_eigenvectors(Matrix eigenvectors);

    //Attributes
    uint alpha;
    Matrix eigenvectors;
private:
    double eps;
    pair<Vector, Matrix> _calculate_eigenvalues(const Matrix& X);
    pair<double, Vector> _power_method(Matrix A);
    Matrix _deflate(const Matrix& A, pair<double, Vector> eigen);
};
