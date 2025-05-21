cmake_minimum_required(VERSION 3.5)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCV 4 REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_BINARY_DIR}
        ${Lab0_LIB_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/Thirdparty/Sophus)

#Set executalble directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Apps)

set(PROGRAM_NAME task_1)
add_executable(${PROGRAM_NAME} task_1.cc)
target_link_libraries(${PROGRAM_NAME} Lab0)

set(PROGRAM_NAME task_2)
add_executable(${PROGRAM_NAME} task_2.cc)
target_link_libraries(${PROGRAM_NAME} Lab0)

set(PROGRAM_NAME task_3)
add_executable(${PROGRAM_NAME} task_3.cc)
target_link_libraries(${PROGRAM_NAME} ${OpenCV_LIBS} Lab0)
