cmake_minimum_required(VERSION 3.18)

# Handle superbuild first
option (USE_SUPERBUILD "Whether or not a superbuild should be invoked" OFF)
if (USE_SUPERBUILD)
    project (SUPERBUILD NONE)
    # execute the superbuild (this script will be invoked again without the
    # USE_SUPERBUILD option this time)
    include (cmake/superbuild.cmake)
    return()                      # stop processing this file further
else()
    project(CROSS LANGUAGES C CXX CUDA)
endif()



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Requires cmake 3.18
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# See https://stackoverflow.com/questions/27018340/cmake-does-not-properly-find-cuda-library
FIND_PACKAGE(CUDAToolkit 11.4 REQUIRED)
FIND_PACKAGE(Boost 1.71 COMPONENTS program_options REQUIRED)
# apt install nlohmann-json3-dev
FIND_PACKAGE(nlohmann_json 3.7.3 REQUIRED)


set(CPP_SOURCES
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/row_distribution.cpp
        ${CMAKE_SOURCE_DIR}/src/argument_error.cpp
        ${CMAKE_SOURCE_DIR}/src/run_args.cpp
)
# Route everything through nvcc
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CUDA)

set(CUDA_SOURCES
        ${CMAKE_SOURCE_DIR}/src/hadamard.cu
        ${CMAKE_SOURCE_DIR}/src/naive_original.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_n_to_m.cu
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_multirow.cu
        ${CMAKE_SOURCE_DIR}/src/naive_warp_per_shift.cu
        ${CMAKE_SOURCE_DIR}/src/scatter.cu
)

add_executable(cross ${CPP_SOURCES} ${CUDA_SOURCES})


target_include_directories(cross PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cross PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(cross PUBLIC ${Boost_LIBRARIES})
target_link_libraries(cross PUBLIC CUDA::cudart)
target_link_libraries(cross PUBLIC CUDA::cufft)
target_link_libraries(cross PRIVATE nlohmann_json::nlohmann_json)

target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
target_compile_options(cross PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>)

target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>)

target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -rdynamic>)
target_compile_options(cross PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -Xcompiler -rdynamic>)
