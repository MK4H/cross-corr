cmake_minimum_required(VERSION 3.10)
project(CROSS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Requires cmake 3.18
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# TODO: Change to FindCUDAToolkit if we switch to cmake 3.17+
# See https://stackoverflow.com/questions/27018340/cmake-does-not-properly-find-cuda-library
FIND_PACKAGE(CUDA 11.4 REQUIRED)
FIND_PACKAGE(Boost 1.71 COMPONENTS program_options REQUIRED)
# apt install nlohmann-json3-dev
FIND_PACKAGE(nlohmann_json 3.7.3 REQUIRED)

file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cu")

add_executable(cross ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(cross PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cross PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(cross PUBLIC ${Boost_LIBRARIES})
target_link_libraries(cross PUBLIC ${CUDA_CUFFT_LIBRARIES})
target_link_libraries(cross PRIVATE nlohmann_json::nlohmann_json)

#target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=[compute_70,sm_70,sm_75,sm_80]>)
target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=[compute_70,sm_70,sm_75,sm_80] -lineinfo>)
#target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=[compute_70,sm_70,sm_75,sm_80] -G -Xcompiler -rdynamic>)
#target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g -rdynamic>)