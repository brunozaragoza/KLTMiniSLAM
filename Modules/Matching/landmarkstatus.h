#ifndef JJSLAM_LANDMARK_H
#define JJSLAM_LANDMARK_H


enum LandmarkStatus {
    TRACKED_WITH_3D,
    TRACKED,
    JUST_TRIANGULATED,
    BAD,
    OUT_IMAGE_BOUNDARIES,
    BAD_FEATURE,
};

bool IsUsable(LandmarkStatus status);

#endif //NRSLAM_LANDMARK_STATUS_H
