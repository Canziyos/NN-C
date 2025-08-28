cmake_minimum_required(VERSION 3.10)

# Project name and C standard
project(NeuralNetworkProject C)
set(CMAKE_C_STANDARD 99)

# Add the source files
add_executable(nn_executable main.c Model.c init.c math_utils.c debug_utils.c)
