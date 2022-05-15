cmake_minimum_required(VERSION 3.23)
project(mnpkg)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find_package(PkgConfig)
# pkg_check_modules(EIGEN3 REQUIRED eigen3)

find_package (Eigen3 3.3 REQUIRED)
include_directories(eigen)

add_subdirectory(pybind11)

pybind11_add_module(mnpkg src/main.cpp src/knn.cpp src/utils.cpp)
# pybind11_add_module(mnpkg src/main.cpp src/knn.cpp src/pca.cpp src/utils.cpp)
target_link_libraries(mnpkg PRIVATE Eigen3::Eigen)