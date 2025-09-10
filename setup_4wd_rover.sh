#!/bin/bash
# Setup script for RoverRobotics 4WD skid steer robot

echo "Setting up RoverRobotics 4WD skid steer robot..."

# Clear TurtleBot environment variables
unset TURTLEBOT3_MODEL
unset TURTLEBOT4_MODEL

# Set Gazebo model path
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/gazebo_plugins/worlds

# Source the workspace
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    echo "✓ Workspace sourced"
else
    echo "✗ Warning: install/setup.bash not found"
fi

echo ""
echo "🚀 RoverRobotics 4WD Setup Complete!"
echo ""
echo "Available 4WD Robot Models:"
echo "  - 4WD Rover (recommended for your use case)"
echo "  - Flipper (4WD with flip capabilities)"
echo "  - Max series (various sizes)"
echo ""
echo "To run 4WD simulation + your autonomous controller:"
echo ""
echo "  # Terminal 1 - Start 4WD Rover simulation:"
echo "  ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py"
echo ""
echo "  # Terminal 2 - Run your enhanced autonomous controller:"
echo "  ros2 run turtlebot_autonomous main_controller"
echo ""
echo "Your controller will now work with TRUE 4WD skid steer!"
echo "Perfect for expanded maps and better stability 🎯"