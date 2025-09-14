cmake_minimum_required(VERSION 3.16)
project(ros2_ws_cpp_projects)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Automatically find all .cpp files in simple-slam-test directory only
file(GLOB CPP_SOURCES "src/simple-slam-test/*.cpp")

# Create executable for each .cpp file
foreach(cpp_file ${CPP_SOURCES})
    # Get filename without extension
    get_filename_component(exe_name ${cpp_file} NAME_WE)
    
    # Create executable
    add_executable(${exe_name} ${cpp_file})
    
    # Link OpenCV libraries to all executables
    target_link_libraries(${exe_name} ${OpenCV_LIBS})
    
    message(STATUS "Added executable: ${exe_name} from ${cpp_file}")
endforeach()

# Also include .cpp files in root directory
file(GLOB ROOT_CPP_SOURCES "*.cpp")
foreach(cpp_file ${ROOT_CPP_SOURCES})
    get_filename_component(exe_name ${cpp_file} NAME_WE)
    add_executable(${exe_name}_root ${cpp_file})
    target_link_libraries(${exe_name}_root ${OpenCV_LIBS})
    message(STATUS "Added root executable: ${exe_name}_root from ${cpp_file}")
endforeach()