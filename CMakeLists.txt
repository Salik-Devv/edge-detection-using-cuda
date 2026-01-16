cmake_minimum_required(VERSION 3.18)

project(edge_detection_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE CPP_SOURCES src/host/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/kernels/*.cu)

add_executable(edge_detect
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(edge_detect ${OpenCV_LIBS})

set_target_properties(edge_detect PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)