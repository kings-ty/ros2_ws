#!/bin/bash
# Complete setup for 4WD rover with camera

echo "Starting 4WD Rover + Camera System..."

# Function to start camera
start_camera() {
    echo "Starting camera in background..."
    
    # Method 1: Try USB camera first
    ros2 run usb_cam usb_cam_node_exe \
      --ros-args \
      --param video_device:=/dev/video0 \
      --param image_width:=640 \
      --param image_height:=480 \
      --param pixel_format:=yuyv \
      --param camera_frame_id:=camera_link \
      --remap /image_raw:=/camera/image_raw &
    
    CAMERA_PID=$!
    echo "Camera started with PID: $CAMERA_PID"
    return $CAMERA_PID
}

# Source workspace
source install/setup.bash

echo "Available video devices:"
ls /dev/video* 2>/dev/null || echo "No video devices found"

echo ""
echo "=== Starting Components ==="
echo "1. Starting 4WD rover simulation..."
echo "2. Starting camera..."
echo "3. You can then run your autonomous controller"

echo ""
echo "Commands to run in separate terminals:"
echo ""
echo "Terminal 1 - 4WD Rover:"
echo "  source install/setup.bash"
echo "  ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py"
echo ""
echo "Terminal 2 - Camera:"
echo "  source install/setup.bash" 
echo "  ros2 run usb_cam usb_cam_node_exe --ros-args --param video_device:=/dev/video0 --remap /image_raw:=/camera/image_raw"
echo ""
echo "Terminal 3 - Your Controller:"
echo "  source install/setup.bash"
echo "  ros2 run turtlebot_autonomous main_controller"
echo ""
echo "Alternative: Install raspicam if you have Raspberry Pi camera:"
echo "  sudo apt install ros-humble-raspicam2-node"