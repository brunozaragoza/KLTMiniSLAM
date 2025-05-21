//
// Created by jjgomez on 30/01/23.
//

#include "WeightVertex.h"

WeightVertex::WeightVertex() : g2o::BaseVertex<3, Eigen::Vector3d>() {}

bool WeightVertex::read(std::istream& is) {
    is >> _estimate(0 ), _estimate(1 ), _estimate(2 );
    return true;
}

bool WeightVertex::write(std::ostream& os) const {
    os << _estimate;
    return true;
}

void WeightVertex::setToOriginImpl() {
    _estimate.fill(0);
}

void WeightVertex::oplusImpl(const double *update) {
    Eigen::Map<const Eigen::Vector3d> v(update);
    _estimate += v;
}
