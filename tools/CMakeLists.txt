cmake_minimum_required(VERSION 3.10)
project(tools_project CXX)

set(CMAKE_CXX_STANDARD 17)

# Unified optimization flags (optional, for consistency)
set(COMMON_COMPILE_FLAGS -Ofast -march=native -w)

set(SOURCE_EXE xbin2xvecs.cpp)

add_executable(bin2vec ${SOURCE_EXE})

target_compile_options(bin2vec PRIVATE ${COMMON_COMPILE_FLAGS})