//
// Created by jjgomez on 30/01/23.
//

#ifndef LAB0_REGRESSION_EDGE_H
#define LAB0_REGRESSION_EDGE_H

#include "Optimization/WeightVertex.h"

#include <eigen3/Eigen/Core>

#include "g2o/core/base_unary_edge.h"

class RegresionEdge : public g2o::BaseUnaryEdge <1, double, WeightVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RegresionEdge();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError();

    Eigen::Vector3d x_;
};


#endif //LAB0_REGRESSION_EDGE_H
