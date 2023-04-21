cmake_minimum_required(VERSION 3.0)
project(DGEMM_TILED Fortran)

# Set the compiler options
set(CMAKE_Fortran_COMPILER gfortran)
set(CMAKE_Fortran_FLAGS "-O3")

# Set the source files
set(SOURCES dgemm_tiled.f90)

# Set the name of the executable
set(EXECUTABLE dgemm_tiled)

# Create the executable
add_executable(${EXECUTABLE} ${SOURCES})
