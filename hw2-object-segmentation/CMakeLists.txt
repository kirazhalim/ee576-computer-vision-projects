# Author: Abdulhalim Kiraz
# Course: EE576
# Project No: 2
# File: CMakeLists.txt
# Description: CMake configuration file for Project 2.

cmake_minimum_required(VERSION 3.10)
project(object_segmentation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(.)

set(SOURCE_FILES
    main.cpp
    Functions.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
