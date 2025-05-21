//
// Created by jjgomez on 30/01/23.
//

#include "Regression.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Optimization/RegressionEdge.h"
#include "Optimization/WeightVertex.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/core/robust_kernel_impl.h"

using namespace std;

Regression::Regression(const std::string &data_file) {
    LoadDataFromFile(data_file);

    ResetSolution();
}

void Regression::ResetSolution() {
    theta_ = Eigen::Vector3f::Zero();
}

void Regression::ComputeRegressionWeights() {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);
    // optimizer.setVerbose(true);

    // Set weights vertex
    WeightVertex* weight_vertex = new WeightVertex();
    Eigen::Vector3d seed(-1000.39, 200.21, 200);
    weight_vertex->setEstimate(seed);
    weight_vertex->setId(0);

    optimizer.addVertex(weight_vertex);

    // Set error terms
    for (int idx = 0; idx < x_.rows(); idx++) {
        RegresionEdge* error_term = new RegresionEdge();

        error_term->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        error_term->setMeasurement(y_(idx));
        error_term->x_ = Eigen::Vector3d(1.0, x_(idx, 0), x_(idx, 1));

        Eigen::Matrix<double, 1, 1> information_matrix =
                Eigen::Matrix<double, 1, 1>::Identity();
        error_term->setInformation(information_matrix);

        optimizer.addEdge(error_term);
    }


    optimizer.initializeOptimization(0);
    optimizer.optimize(100);

    // Recover the optimized weights.
    theta_ = weight_vertex->estimate().cast<float>();

    // cerr << "Computed weights: " << solution_.transpose() << endl;
}

Eigen::Vector3f Regression::GetComputedWeights() {
    return theta_;
}

float Regression::ComputeRMSE() {
    return 0;
}

float Regression::Estimate(const Eigen::Vector3f x) const {
    return 0;
}

void Regression::LoadDataFromFile(const std::string &data_file) {
    ifstream data_file_reader(data_file);

    if (!data_file_reader.is_open()) {
        throw std::runtime_error("File not found");
    }

    // File format is as follows:
    // <m²> #rooms price
    vector<Eigen::Vector3f> data;

    while (!data_file_reader.eof()){
        string row_as_string;
        getline(data_file_reader, row_as_string);

        stringstream ss;
        ss << row_as_string;

        Eigen::Vector3f row;
        ss >> row(0) >> row(1) >> row(2);
        data.push_back(row);
    }

    const int n_entries = data.size();
    x_ = Eigen::MatrixXf(n_entries, 2);
    y_ = Eigen::VectorXf(n_entries);

    for (int idx = 0; idx < n_entries; idx++) {
        x_(idx, 0) = data[idx](0);
        x_(idx, 1) = data[idx](1);

        y_(idx) = data[idx](2);

    }

    data_file_reader.close();
}
