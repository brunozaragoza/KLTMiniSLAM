//
// Created by jjgomez on 1/02/23.
//

#include <sophus/se3.hpp>

using namespace std;

// Applies one motion to the robot position.
void MoveRobot(Sophus::SE3f& robot_transform_world,
               Sophus::SE3f& motion) {
}

// Applies all motions to the robot position stored in motions.
void MoveRobotSeveralTimes(Sophus::SE3f& robot_transform_world,
                           std::vector<Sophus::SE3f>& motions) {

}

int main() {
    // Robot position is initialized to the origin.
    Sophus::SE3f robot_transform_world;

    // Define next robot motion (3 meters in the x axis).
    Sophus::SE3f motion;
    motion.translation() = Eigen::Vector3f(3, 0, 0);
    motion.setRotationMatrix(Eigen::Matrix3f::Identity());

    // Apply motion.
    MoveRobot(robot_transform_world, motion);

    // Now define several motions and apply all of them with just one function call.
    std::vector<Sophus::SE3f> several_motions(10, motion);

    MoveRobotSeveralTimes(robot_transform_world, several_motions);

    return 0;
}