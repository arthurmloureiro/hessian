#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <numeric>
#include <cmath>
#include <random>
#include <ctime> //for random seeds
#include <functional> // to use std::function
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <boost/multi_array.hpp>
//#include <eigen3/Eigen/Eigenvalues>
//#include "eigenmvn.h" // for some reason a dependency on this random file was created

using namespace Eigen;
using namespace std;

// typedef std::vector<double> Vec;
typedef Matrix<double, Dynamic, Dynamic> matrixType;
typedef Matrix<double, Dynamic, 1> vectorType;
typedef boost::multi_array<double, 3> NArrayType;

// functions declared
int intPow(int x, int p);
double neglnpost(matrixType a, matrixType G, matrixType x, matrixType sigma,matrixType invN, matrixType Z, int L, int q);
double neglnlike(matrixType a, matrixType x, matrixType invN, matrixType Z);
double neglnprior(matrixType G, double q);
void GtoC(matrixType G, vectorType& W, matrixType& U,matrixType& C,matrixType& invC);
matrixType sigma_from_a(matrixType a, int L, int nbins);
matrixType tilde(matrixType sigma, matrixType U);
double target(matrixType sigma, matrixType G);
matrixType perturb(matrixType G, int ii, int jj, double DeltaG);
double OrtizFirstDerivatives(matrixType sigmaTilde, matrixType U, matrixType phi, int ii, int jj, int nbins);
void Ortiz_matrices(matrixType& phi, NArrayType &psi, matrixType W, int m);
double grad_G(matrixType sigmaTilde, matrixType U, matrixType phi, int ii, int jj, int q, int L, int nbins);
matrixType grad_a(matrixType a, matrixType x, matrixType invC,  matrixType invN, matrixType Z, int L, int q);
MatrixXd blkdiag(const MatrixXd& a, int count);

// for the multivariate Gaussian
// taken from: https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c,
struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

