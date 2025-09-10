#!/bin/bash
# Debug script to check 4WD rover topics

echo "=== Checking 4WD Rover Topics ==="
source install/setup.bash

echo "All topics:"
ros2 topic list

echo ""
echo "Scan/Laser topics:"
ros2 topic list | grep -E "(scan|laser|lidar)"

echo ""
echo "Camera/Image topics:"
ros2 topic list | grep -E "(image|camera)"

echo ""
echo "Command velocity topics:"
ros2 topic list | grep -E "(cmd_vel|twist)"

echo ""
echo "=== Topic Details ==="
echo "Let's check what specific topics exist..."

if ros2 topic list | grep -q "/scan"; then
    echo "✓ Found /scan topic"
    ros2 topic info /scan
elif ros2 topic list | grep -q "scan"; then
    echo "Found scan topic with different name:"
    ros2 topic list | grep scan
else
    echo "❌ No scan topic found"
fi

echo ""
echo "Run this script to see what topics your 4WD rover provides!"