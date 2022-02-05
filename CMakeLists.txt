cmake_minimum_required(VERSION 3.20)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda main.cu src/stride_csr/brandes.cu src/stride_csr/graph.cu src/utils.cu)

set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)