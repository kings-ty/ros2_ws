#!/usr/bin/env python3
"""
Launch file for TurtleBot4 autonomous navigation with 4WD skid steer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    return LaunchDescription([
        # Set TurtleBot4 environment
        SetEnvironmentVariable(
            name='TURTLEBOT4_MODEL',
            value='standard'
        ),
        
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        
        DeclareLaunchArgument(
            'training_mode',
            default_value='false',
            description='Enable DQN training mode'
        ),
        
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),
        
        # TurtleBot4 Autonomous Controller Node
        Node(
            package='turtlebot_autonomous',
            executable='main_controller',
            name='turtlebot4_autonomous_controller',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'training_mode': LaunchConfiguration('training_mode'),
            }],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            remappings=[
                # TurtleBot4 uses standard topic names
                ('/cmd_vel', '/cmd_vel'),
                ('/scan', '/scan'),
                ('/camera/image_raw', '/camera/image_raw'),
            ]
        ),
        
        # Optional: Robot State Publisher (if needed)
        # Node(
        #     package='robot_state_publisher',
        #     executable='robot_state_publisher',
        #     name='robot_state_publisher',
        #     output='screen',
        #     parameters=[{
        #         'use_sim_time': LaunchConfiguration('use_sim_time'),
        #         'robot_description': ''  # Will be loaded from TurtleBot4 description
        #     }]
        # ),
    ])


if __name__ == '__main__':
    generate_launch_description()