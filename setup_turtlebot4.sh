#!/bin/bash
# Setup script for TurtleBot4 4WD skid steer configuration

echo "Setting up TurtleBot4 for 4WD skid steer operation..."

# Clear TurtleBot3 model (not needed for TurtleBot4)
unset TURTLEBOT3_MODEL

# Set TurtleBot4 specific environment variables
export TURTLEBOT4_MODEL=standard  # or 'lite' for lighter version
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

echo "Environment configured:"
echo "  TURTLEBOT3_MODEL: ${TURTLEBOT3_MODEL:-'unset'}"
echo "  TURTLEBOT4_MODEL: ${TURTLEBOT4_MODEL}"
echo "  RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION}"

# Source the workspace
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    echo "  Workspace sourced: ✓"
else
    echo "  Warning: install/setup.bash not found"
fi

echo ""
echo "TurtleBot4 setup complete!"
echo "Your autonomous controller is now configured for 4WD skid steer operation."
echo ""
echo "To run your autonomous controller:"
echo "  ros2 run turtlebot_autonomous main_controller"
echo ""
echo "To run in simulation (if available):"
echo "  ros2 launch turtlebot4_gazebo turtlebot4_world.launch.py"