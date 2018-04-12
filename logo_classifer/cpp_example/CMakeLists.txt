cmake_minimum_required(VERSION 2.8.3)
project(logo_classifer)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(demo_logo_classifier src/demo_logo_classifier.cpp src/keras_model.cc)
target_link_libraries(demo_logo_classifier ${OpenCV_LIBS})


