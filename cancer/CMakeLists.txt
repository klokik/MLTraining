cmake_minimum_required(VERSION 2.8)

project(CancerClassification)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wuninitialized -Werror ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

include_directories("inc")

set(CLASSIFIERS_SRC
	src/DummyClassifier.cc
	src/NearestNeighborClassifiers.cc
	src/RosenblatPerceptronClassifier.cc
	src/LinearFisherDiscriminantClassifier.cc
	src/SVMClassifiers.cc
	src/EllipseClassifiers.cc)

add_executable(start
	start.cc
	src/framework.cc
	inc/framework.hh
	inc/cv_common.hh
	${CLASSIFIERS_SRC})

target_link_libraries(start ${OpenCV_LIBS})
