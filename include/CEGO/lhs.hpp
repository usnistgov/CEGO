#ifndef CEGO_LHS_HPP
#define CEGO_LHS_HPP

#include "Eigen/Dense"
#include <algorithm>

/**
  @brief Generate samples for each parameter in [0,1] with the use of Latin-Hypercube sampling
  @param Npop The number of members in the population
  @param Nparam The number of independent variables
  @retuns population The matrix of the population
  @note This code is based upon the Python implementation of LHS sampling in scipy.optimize, available with a PSF license
 */
inline Eigen::ArrayXXd LHS_samples(Eigen::Index Npop, Eigen::Index Nparam) {
    double segsize = 1.0 / Npop;
    
    // Each entry in a1 is a random number between zero and 1 / Npop;
    // this is the random location of the point within the grid cell
    Eigen::ArrayXXd a1 = segsize*(0.5*(Eigen::MatrixXd::Random(Npop, Nparam).array() +1.0));
    // Each entry in a2 is a shift amount, the same amount in each parameter
    Eigen::ArrayXd a2 = Eigen::ArrayXd::LinSpaced(Npop+1,0,1.0).head(Npop);
    // samples puts the data points along the main - diagonal of the parameter space
    // see the figure in the notebook for samples
    Eigen::ArrayXXd samples = a1.colwise() + a2;

    // A function to generate shuffled (permuted) indices
    auto random_indices = [](Eigen::Index Npop) {
        std::vector<Eigen::Index> indices;
        for (auto j = 0; j < Npop; ++j){ indices.emplace_back(j); }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        return indices;
    };

    // Initialize population of candidate solutions by permutation of the
    // random samples in each variable, one at a time
    Eigen::ArrayXXd population(Npop, Nparam);
    for (auto i = 0; i < Nparam; ++i) {
        Eigen::Index j = 0;
        for (auto index : random_indices(Npop)) {
            population(j,i) = samples(index,i);
            j++;
        }
    }
    return population;
}

#endif