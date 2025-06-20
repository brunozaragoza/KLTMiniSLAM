cmake_minimum_required(VERSION 3.5)
project(MiniSLAM)

set(CMAKE_CXX_STANDARD 17)
# Find dependencies
find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_library(MLPACK_LIBRARY
    NAMES mlpack
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu
)

find_path(MLPACK_INCLUDE_DIR
    NAMES mlpack/core.hpp
    PATHS /usr/include
)

find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu
)

find_path(ARMADILLO_INCLUDE_DIR
    NAMES armadillo
    PATHS /usr/include
)

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)




# Find libmlpack.so (or equivalent) and store it in MLPACK_LIBRARY.
# If this example script were smarter, it would also find other dependencies of mlpack and store them in
# an MLPACK_LIBRARIES variable instead.

#Set the name of the Library
set(LIBRARY_NAME mini_slam)

#Current library
set(root ${CMAKE_CURRENT_SOURCE_DIR})

#Define header and source files of the library
set(HEADER_FILES
        ${root}/Calibration/CameraModel.h
        ${root}/Calibration/PinHole.h
        ${root}/Calibration/KannalaBrandt.h
        ${root}/DatasetLoader/EurocVisualLoader.h
        ${root}/DatasetLoader/TUMRGBDLoader.h
        ${root}/DatasetLoader/TUMVILoader.h
        ${root}/DatasetLoader/VideoLoader.h
        ${root}/Features/Descriptor.h
        ${root}/Features/Feature.h
        ${root}/Features/FAST.h
        ${root}/Features/ORB.h
        ${root}/Features/ShiTomasi.h
        ${root}/Map/KeyFrame.h
        ${root}/Map/Map.h
        ${root}/Map/MapPoint.h
        ${root}/Mapping/LocalMapping.h
        ${root}/Matching/DescriptorMatching.h
        ${root}/Matching/landmarkstatus.h
        ${root}/Matching/lucas_kanade_tracker.h
        ${root}/Optimization/g2oBundleAdjustment.h
        ${root}/Optimization/g2oTypes.h
        ${root}/System/Settings.h
        ${root}/System/MiniSLAM.h
        ${root}/Tracking/Frame.h
        ${root}/Tracking/MonocularMapInitializer.h
        ${root}/Tracking/monocularmapinit.h
        ${root}/Tracking/Tracking.h
        ${root}/Tracking/TrackingKLT.h
        ${root}/Utils/Geometry.h
        ${root}/Utils/dbscan.h
        ${root}/Visualization/FrameVisualizer.h
        ${root}/Visualization/image_viz.h
        ${root}/Visualization/color_factory.h
        ${root}/Visualization/MapVisualizer.h)

set(SOURCE_FILES
        ${root}/Calibration/PinHole.cc
        ${root}/Calibration/KannalaBrandt.cc
        ${root}/DatasetLoader/EurocVisualLoader.cc
        ${root}/DatasetLoader/TUMRGBDLoader.cc
        ${root}/DatasetLoader/TUMVILoader.cc
        ${root}/DatasetLoader/VideoLoader.cc
        ${root}/Features/FAST.cc
        ${root}/Features/ShiTomasi.cc
        ${root}/Features/ORB.cc
        ${root}/Map/KeyFrame.cc
        ${root}/Map/Map.cc
        ${root}/Map/MapPoint.cc
        ${root}/Mapping/LocalMapping.cc
        ${root}/Matching/landmarkstatus.cc
        ${root}/Matching/lucas_kanade_tracker.cc
        ${root}/Matching/DescriptorMatching.cc
        ${root}/Optimization/g2oBundleAdjustment.cc
        ${root}/Optimization/g2oTypes.cc
        ${root}/System/Settings.cc
        ${root}/System/MiniSLAM.cc
        ${root}/Tracking/Frame.cc
        ${root}/Tracking/TrackingKLT.cc
        ${root}/Tracking/MonocularMapInitializer.cc
        ${root}/Tracking/monocularmapinit.cc
        ${root}/Tracking/Tracking.cc
        ${root}/Utils/Geometry.cc
        ${root}/Utils/dbscan.cc
        ${root}/Visualization/FrameVisualizer.cc
        ${root}/Visualization/image_viz.cc
        ${root}/Visualization/color_factory.cc
        ${root}/Visualization/MapVisualizer.cc)

set(MINI_SLAM_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/Thirdparty/Sophus
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o
        ${ARMADILLO_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${ARMADILLO_INCLUDE_DIRS})

#Add the library
add_library(${LIBRARY_NAME} SHARED
        ${SOURCE_FILES}
        ${HEADER_FILES})

#Define the link libraries used in the project
target_link_libraries(${LIBRARY_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${MLPACK_LIBRARY}
        ${ARMADILLO_LIBRARY}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_core.so
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_stuff.so
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_types_sba.so)
