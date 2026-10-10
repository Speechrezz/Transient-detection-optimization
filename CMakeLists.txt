cmake_minimum_required(VERSION 3.15)
project(detector_cpp LANGUAGES CXX)

# Find Pybind11 library
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE MY_SOURCES detector/*)
pybind11_add_module(detector_cpp ${MY_SOURCES})

target_compile_features(detector_cpp PRIVATE cxx_std_17)

install(TARGETS detector_cpp DESTINATION .)