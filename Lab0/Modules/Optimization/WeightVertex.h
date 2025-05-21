//
// Created by jjgomez on 30/01/23.
//

#ifndef LAB0_WEIGHT_VERTEX_H
#define LAB0_WEIGHT_VERTEX_H

#include <eigen3/Eigen/Core>

#include "g2o/core/base_vertex.h"

class WeightVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    WeightVertex();

    virtual bool read(std::istream& is);

    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl();

    virtual void oplusImpl(const double * update);
};


#endif //LAB0_WEIGHT_VERTEX_H
