cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 11)

project(mnpkg)

include_directories(eigen)
add_subdirectory(pybind11)

pybind11_add_module(mnpkg src/main.cpp)
#target_link_libraries(mnpkg Eigen3::Eigen)

