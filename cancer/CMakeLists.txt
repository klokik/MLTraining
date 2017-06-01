cmake_minimum_required(VERSION 2.8)

project(CancerClassification)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wuninitialized -Werror ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

set(CLASSIFIERS_SRC
	DummyClassifier.cc
	NearestNeighborClassifiers.cc
	RosenblatPerceptronClassifier.cc
	LinearFisherDiscriminantClassifier.cc
	SVMClassifiers.cc
	EllipseClassifiers.cc)

add_executable(start start.cc framework.cc framework.hh cv_common.hh ${CLASSIFIERS_SRC})
target_link_libraries(start ${OpenCV_LIBS})
