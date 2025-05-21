//
// Created by jjgomez on 30/01/23.
//

#ifndef LAB0_REGRESSION_H
#define LAB0_REGRESSION_H

#include <string>

#include <eigen3/Eigen/Core>

// This class implements a simple solver for a multivariate regression. It offers
// methods to load data from a file, to compute the optimal weights, to compute the
// RMSE of the estimated solution and accessors for the solution itself.

class Regression {
public:
    // Constructs a regression object. It loads data from the file located at <data_file>.
    Regression(const std::string& data_file);

    // Resets to 0 the current solution of the regression.
    void ResetSolution();

    // Computes the optimal regression weights using non-linear optimization.
    // DISCLAIMER: this is an overkill solution. Just for teaching purposes.
    void ComputeRegressionWeights();

    // Returns the regression weights computed.
    Eigen::Vector3f GetComputedWeights();

    // Computes the Root-Mean-Squared-Error of the current computed weights.
    float ComputeRMSE();

    // Makes an estimation using the data provided.
    float Estimate(const Eigen::Vector3f x) const;

private:
    // Loads data into the data_ field of the class.
    void LoadDataFromFile(const std::string& data_file);

    // Matrix with the data to solve the regression.
    Eigen::MatrixXf x_;

    // Vector with the labels of the data.
    Eigen::VectorXf y_;

    // Estimated weights of the regression.
    Eigen::Vector3f theta_;
};


#endif //LAB0_REGRESSION_H
