#include "types.hpp"

using namespace std;

class PCA {
public:
    // Constructor
    PCA(uint alpha, double epsilon);

    // Methods
    void fit(const std::vector<std::vector<double> > list);
    Matrix transform(const std::vector<std::vector<double> > list);

    //Attributes
    uint alpha;
    double epsilon;
    Matrix eigenvectors;

    // Setters for pybind11 integration
    void set_eigenvectors(Matrix eigenvectors);

private:
    pair<Vector, Matrix> _calculate_eigenvalues(const Matrix& X);
    pair<double, Vector> _power_method(Matrix A);
    Matrix _deflate(const Matrix& A, pair<double, Vector> eigen);
};
