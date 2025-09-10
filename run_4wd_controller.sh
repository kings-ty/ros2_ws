#!/bin/bash
# Run autonomous controller with 4WD rover topic remapping

echo "Starting autonomous controller for 4WD rover..."
echo "Make sure the 4WD rover simulation is running first!"

# Source workspace
source install/setup.bash

# Run controller with topic remapping for 4WD rover
ros2 run turtlebot_autonomous main_controller \
  --ros-args \
  --remap /scan:=/rover/scan \
  --remap /camera/image_raw:=/rover/camera/image_raw \
  --remap /cmd_vel:=/rover/cmd_vel

echo "Controller started with 4WD rover topic mapping"