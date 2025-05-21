//
// Created by jjgomez on 30/01/23.
//

#include "RegressionEdge.h"

RegresionEdge::RegresionEdge() {}

bool RegresionEdge::read(std::istream& is) {
    return true;
}

bool RegresionEdge::write(std::ostream& os) const {
    return true;
}

void RegresionEdge::computeError() {
    const WeightVertex* weight_vertex = static_cast<const WeightVertex*>(_vertices[0]);

    Eigen::Vector3d weights = weight_vertex->estimate();

    _error(0) = _measurement - x_.dot(weights);
}
