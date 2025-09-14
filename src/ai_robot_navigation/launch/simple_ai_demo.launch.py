#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Robot simulation launch
    robot_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('classic_4wd_robot'), 'launch', 'robot_gazebo.launch.py')
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # YOLO8 Object Detector (delayed start)
    yolo_detector_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ai_robot_navigation',
                executable='yolo_detector',
                name='yolo_detector',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    # Simple test controller (instead of full DQN for now)
    test_controller = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='ai_robot_navigation',
                executable='navigation_controller',
                name='test_controller',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        robot_sim_launch,
        yolo_detector_node,
        test_controller,
    ])