#include "KannalaBrandt.h"

#define fx vParameters_[0]
#define fy vParameters_[1]
#define cx vParameters_[2]
#define cy vParameters_[3]

//Extra parameters defined in the Kannala-Brandt model
#define k1 vParameters_[4]
#define k2 vParameters_[5]
#define k3 vParameters_[6]
#define k4 vParameters_[7]

void KannalaBrandt::project(const Eigen::Vector3f& p3D, Eigen::Vector2f& p2D)
{
    //Kannala-Brandt projection model
    float x = p3D(0);
    float y = p3D(1);
    float z = p3D(2);

    //Calculate incident angle theta
    float r = sqrt(x*x + y*y);
    float theta = atan(r/z);

    //Distorted radius
    float dr = theta + k1*pow(theta,3) + k2*pow(theta,5) + k3*pow(theta,7) + k4*pow(theta,9);

    //Pixel coordinates of the distorted point
    p2D(0) = fx * dr * (x / r) + cx;
    p2D(1) = fy * dr * (y / r) + cy;

}

void KannalaBrandt::unproject(const Eigen::Vector2f& p2D, Eigen::Vector3f& p3D)
{
    float mx = (p2D(0) - cx) / fx;
    float my = (p2D(1) - cy) / fy;

    float r_prime = sqrt(mx*mx + my*my);

    float theta = inverseDistortion(r_prime);

    //Unprojected point to obtain the direction vector
    p3D(0) = sin(theta) * mx / r_prime;
    p3D(1) = sin(theta) * my / r_prime;
    p3D(2) = cos(theta);

}


float KannalaBrandt::inverseDistortion(float r_prime){

    //https://personal.math.ubc.ca/~anstee/math104/newtonmethod.pdf
    //Initialize
    float theta = r_prime;
    //Max iters
    int max_iters = 10;

    const float eps = 1e-6; //Tolerance for convergence, just to break the loop.

    for (int i = 0; i < max_iters; i++){
        //function -> is there any way to obtain this instead of re-writing it?
        float f = theta - r_prime - k1*pow(theta,3) - k2*pow(theta,5) - k3*pow(theta,7) - k4*pow(theta,9);
        //Derivative
        float f_prime = 1 - 3*k1*pow(theta,2) - 5*k2*pow(theta,4) - 7*k3*pow(theta,6) - 9*k4*pow(theta,8);

        //Update theta
        theta = theta - f/f_prime;

        //Check convergence -> if the function is close to zero, we can stop
        if (fabs(f) < eps){
            break;
        }
    }
    return theta;
}



void KannalaBrandt::projectJac(const Eigen::Vector3f& p3D, Eigen::Matrix<float,2,3>& Jac) {


    //-Variables
    float x = p3D(0);
    float y = p3D(1);
    float z = p3D(2);

    //Calculate incident angle theta
    float r = sqrt(x*x + y*y);
    float theta = atan2(r,z);

    //Distorted radius
    float dr = theta + k1*pow(theta,3) + k2*pow(theta,5) + k3*pow(theta,7) + k4*pow(theta,9);
    float dr_prime = 1 + 3*k1*pow(theta,2) + 5*k2*pow(theta,4) + 7*k3*pow(theta,6) + 9*k4*pow(theta,8);

    //Partial derivatives
    float dtheta_dx = (z*x) / ((r*r + z*z) * r);
    float dtheta_dy = (z*y) / ((r*r + z*z) * r);
    float dtheta_dz = -r / (r*r + z*z);

    //More partial derivatives
    float ddr_dx = dr_prime * dtheta_dx;
    float ddr_dy = dr_prime * dtheta_dy;
    float ddr_dz = dr_prime * dtheta_dz;

    //Jacobian matrix
    Jac(0,0) = fx * (ddr_dx * (x / r) + dr * (y*y) / (r*r*r));
    Jac(0,1) = fx * (ddr_dy * (x / r) + dr * (-x*y) / (r*r*r));
    Jac(0,2) = fx * ddr_dz * (x / r);

    Jac(1,0) = fy * (ddr_dx * (y / r) + dr * (-x*y) / (r*r*r));
    Jac(1,1) = fy * (ddr_dy * (y / r) + dr * (x*x) / (r*r*r));
    Jac(1,2) = fy * ddr_dz * (y / r);

    
    

}


void KannalaBrandt::unprojectJac(const Eigen::Vector2f& p2D, Eigen::Matrix<float,3,2>& Jac) {
}