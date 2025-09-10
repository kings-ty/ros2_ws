#!/bin/bash
# Start laptop camera for ROS2

echo "Starting laptop camera for 4WD rover..."

# Source ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

# Start USB camera node
ros2 run usb_cam usb_cam_node_exe \
  --ros-args \
  --param video_device:=/dev/video0 \
  --param image_width:=640 \
  --param image_height:=480 \
  --param pixel_format:=yuyv \
  --param camera_frame_id:=camera_link \
  --param framerate:=30.0 \
  --remap /image_raw:=/camera/image_raw

echo "Camera started on /camera/image_raw topic"