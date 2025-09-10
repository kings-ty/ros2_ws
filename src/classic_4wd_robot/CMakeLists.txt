cmake_minimum_required(VERSION 3.8)
project(classic_4wd_robot)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME}/)

ament_package()