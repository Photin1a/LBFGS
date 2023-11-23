cmake_minimum_required(VERSION 3.0)
project(deepest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

message(STATUS "Start to deal with the project: ${PROJECT_NAME}")

include_directories(${PROJECT_SOURCE_DIR}/include)

#Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(rosenbrock main.cpp)
