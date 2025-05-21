cmake_minimum_required(VERSION 3.5)

set(LIBRARY_NAME Lab0)

# Current library
set(root ${CMAKE_CURRENT_SOURCE_DIR})

set(HEADER_FILES
        ${root}/Optimization/RegressionEdge.h
        ${root}/Optimization/WeightVertex.h
        ${root}/Regresion/Regression.h
        )

set(SOURCE_FILES
        ${root}/Optimization/RegressionEdge.cc
        ${root}/Optimization/WeightVertex.cc
        ${root}/Regresion/Regression.cc
        )

set(Lab0_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Define the include directory search file for this project
include_directories(${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o)

#Add the library
add_library(${LIBRARY_NAME} SHARED
        ${SOURCE_FILES}
        ${HEADER_FILES})

# Define the link libraries used in the project
target_link_libraries(${LIBRARY_NAME}
        ${EIGEN3_LIBS}
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_core.so
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_stuff.so
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_types_sba.so
        )

target_include_directories(Lab0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
