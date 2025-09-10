#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directories
    turtlebot4_simulator_dir = get_package_share_directory('turtlebot4_gazebo')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='maze')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='maze',
        description='World to use for training')
    
    # TurtleBot4 simulation launch
    turtlebot4_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(turtlebot4_simulator_dir, 'launch', 'turtlebot4_ignition.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'world': world
        }.items()
    )
    
    # DQN Environment Node
    dqn_env_node = Node(
        package='turtlebot4_dqn',
        executable='dqn_environment',
        name='dqn_environment',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # DQN Trainer Node  
    dqn_trainer_node = Node(
        package='turtlebot4_dqn',
        executable='dqn_trainer',
        name='dqn_trainer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_world_cmd,
        turtlebot4_sim,
        dqn_env_node,
        dqn_trainer_node,
    ])