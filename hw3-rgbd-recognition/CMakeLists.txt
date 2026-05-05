# Author: Abdulhalim Kiraz
# course: ee576
# project no: 3
# file: cmakelists.txt
# description: cmake configuration file for project 3.

cmake_minimum_required(VERSION 3.10)
project(rgbd_recognition)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(.)

set(SOURCE_FILES
    main.cpp
    Functions.cpp
    MatParser.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ZLIB::ZLIB)
