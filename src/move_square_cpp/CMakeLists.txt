cmake_minimum_required(VERSION 3.8)
project(move_square_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(turtlesim REQUIRED)

# Add your C++ executable
add_executable(line_follower_node src/move_square_node.cpp)

# Define dependencies for the executable
ament_target_dependencies(line_follower_node rclcpp geometry_msgs turtlesim)

# Install the executable so it can be used with `ros2 run`
install(TARGETS
  line_follower_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
