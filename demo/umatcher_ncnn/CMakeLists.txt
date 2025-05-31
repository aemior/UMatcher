cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)

project(umatcher_demo)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

set(ncnn_DIR /home/ming/workspace/ncnn-20250503/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND ")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

add_executable(detector_demo
    src/detector_demo.cpp
    src/center_crop.cpp
    src/umatcher.cpp
    src/udetector.cpp
)

target_link_libraries(detector_demo
    ncnn
    ${OpenCV_LIBS}
)

add_executable(tracker_demo
    src/tracker_demo.cpp
    src/center_crop.cpp
    src/umatcher.cpp
    src/utracker.cpp
)

target_link_libraries(tracker_demo
    ncnn
    ${OpenCV_LIBS}
)