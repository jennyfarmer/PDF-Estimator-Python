cmake_minimum_required(VERSION 3.15)
project(pdf_estimator CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_host_system_information(RESULT N_CORES QUERY NUMBER_OF_LOGICAL_CORES)
set(CMAKE_BUILD_PARALLEL_LEVEL ${N_CORES})

find_package(pybind11 REQUIRED)

set(CPP_SOURCES
    src/callPDF.cpp
    extern/ChebyShev.cpp
    extern/InputData.cpp
    extern/InputParameters.cpp
    extern/MinimizeScore.cpp
    extern/OutputControl.cpp
    extern/Score.cpp
    extern/ScoreQZ.cpp
    extern/WriteResults.cpp
    extern/Variable.cpp
    extern/JointProbability.cpp
)

pybind11_add_module(_pdf_estimator src/bindings.cpp ${CPP_SOURCES})

target_include_directories(_pdf_estimator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern
)

if (MSVC)
    target_compile_options(_pdf_estimator PRIVATE /O2 /wd4715)
else()
    target_compile_options(_pdf_estimator PRIVATE -O2 -Wno-return-type)
endif()

install(TARGETS _pdf_estimator DESTINATION pdf_estimator)
