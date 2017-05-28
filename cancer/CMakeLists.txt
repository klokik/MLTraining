cmake_minimum_required(VERSION 2.8)

project(CancerClassification)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "--std=c++14 -g -pedantic -Wall -Wuninitialized -Werror ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(start framework.cc)
target_link_libraries(start ${OpenCV_LIBS})
