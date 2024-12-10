cmake_minimum_required(VERSION 3.0)
project(RealSenseExample)

# 设置编译器为C++11
set(CMAKE_CXX_STANDARD 14)

# 查找OpenCV库
find_package(OpenCV REQUIRED)
find_package(PCL 1.12 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# 添加包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

add_library(yolo SHARED src/yolo.cpp)
add_library(pcl SHARED  src/sgbm.cpp)
target_link_libraries(yolo ${OpenCV_LIBS} )
target_link_libraries(pcl ${PCL_LIBRARIES}  ${realsense2_LIBRARY} )

# 添加可执行文件
add_executable(main src/main.cpp)

target_link_libraries(main
    ${PCL_LIBRARIES} 
    pcl)
target_link_libraries(main
    ${OpenCV_LIBS}
    yolo )

