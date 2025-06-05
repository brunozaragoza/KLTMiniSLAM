/**
* This file is part of Mini-SLAM
*
* Copyright (C) 2021 Juan J. Gómez Rodríguez and Juan D. Tardós, University of Zaragoza.
*
* Mini-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Mini-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with Mini-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "KannalaBrandt8.h"

#define fx vParameters_[0]
#define fy vParameters_[1]
#define cx vParameters_[2]
#define cy vParameters_[3]
#define k0 vParameters_[4]
#define k1 vParameters_[5]
#define k2 vParameters_[6]
#define k3 vParameters_[7]

void KannalaBrandt8::project(const Eigen::Vector3f& p3D, Eigen::Vector2f& p2D){
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    const double x2_plus_y2 = p3D[0] * p3D[0] + p3D[1] * p3D[1];
    const double theta = atan2f(sqrtf(x2_plus_y2), p3D[2]);
    const double psi = atan2f(p3D[1], p3D[0]);

    const double theta2 = theta * theta;
    const double theta3 = theta * theta2;
    const double theta5 = theta3 * theta2;
    const double theta7 = theta5 * theta2;
    const double theta9 = theta7 * theta2;
    const double r = theta + k0 * theta3 + k1 * theta5
                    + k2 * theta7 + k3 * theta9;

    p2D[0] = fx * r * cos(psi) + cx;
    p2D[1] = fy * r * sin(psi) + cy;
}

void KannalaBrandt8::unproject(const Eigen::Vector2f& p2D, Eigen::Vector3f& p3D) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    
    //Use Newthon method to solve for theta with good precision (err ~ e-6)
    const double precision = 1e-6;
    const double x = (p2D[0] - cx) / fx;
    const double y = (p2D[1] - cy) / fy;
    float scale = 1.f;
    float theta_d = sqrtf(x * x + y * y);
    theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

    if (theta_d > 1e-8) {
        //Compensate distortion iteratively
        float theta = theta_d;

        for (int j = 0; j < 10; j++) {
            float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                    theta4 * theta4;
            float k0_theta2 = k0 * theta2, k1_theta4 = k1 * theta4;
            float k2_theta6 = k2 * theta6, k3_theta8 = k3 * theta8;
            float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                              (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < precision)
                break;
        }
        scale = std::tan(theta) / theta_d;
    }

    p3D[0] = x * scale;
    p3D[1] = y * scale;
    p3D[2] = 1.f;
}

void KannalaBrandt8::projectJac(const Eigen::Vector3f& p3D, Eigen::Matrix<float,2,3>& Jac) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */

    double x2 = p3D[0] * p3D[0], y2 = p3D[1] * p3D[1], z2 = p3D[2] * p3D[2];
    double r2 = x2 + y2;
    double r = sqrt(r2);
    double r3 = r2 * r;
    double theta = atan2(r, p3D[2]);

    double theta2 = theta * theta, theta3 = theta2 * theta;
    double theta4 = theta2 * theta2, theta5 = theta4 * theta;
    double theta6 = theta2 * theta4, theta7 = theta6 * theta;
    double theta8 = theta4 * theta4, theta9 = theta8 * theta;

    double f = theta + theta3 * k0 + theta5 * k1 + theta7 * k2 + theta9 * k3;
    double fd = 1 + 3 * k0 * theta2 + 5 * k1 * theta4 + 7 * k2 * theta6 + 9 * k3 * theta8;

    Jac(0, 0) = fx * (fd * p3D[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    Jac(1, 0) = fy * (fd * p3D[2] * p3D[1] * p3D[0] / (r2 * (r2 + z2)) - f * p3D[1] * p3D[0] / r3);

    Jac(0, 1) = fx * (fd * p3D[2] * p3D[1] * p3D[0] / (r2 * (r2 + z2)) - f * p3D[1] * p3D[0] / r3);
    Jac(1, 1) = fy * (fd * p3D[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

    Jac(0, 2) = -fx * fd * p3D[0] / (r2 + z2);
    Jac(1, 2) = -fy * fd * p3D[1] / (r2 + z2);

}

void KannalaBrandt8::unprojectJac(const Eigen::Vector2f& p2D, Eigen::Matrix<float,3,2>& Jac) {
    throw std::runtime_error("KannalaBrandt8::unprojectJac not implemented yet");
}