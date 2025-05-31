cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)

project(umatcher_demo)

set(OpenCV_DIR /home/kylin/workspace/opencv420_install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)

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
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
)

add_executable(tracker_demo
    src/tracker_demo.cpp
    src/center_crop.cpp
    src/umatcher.cpp
    src/utracker.cpp
)

target_link_libraries(tracker_demo
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
)

# add_executable(debug_matcher
#     src/debug_matcher.cpp
#     src/center_crop.cpp
#     src/umatcher.cpp
# )

# target_link_libraries(debug_matcher
#     ${OpenCV_LIBS}
#     ${RKNN_RT_LIB}
# )