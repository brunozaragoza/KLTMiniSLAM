#include "landmarkstatus.h"

bool IsUsable(LandmarkStatus status) {
    return status == TRACKED_WITH_3D || status == TRACKED || status == JUST_TRIANGULATED;
}
