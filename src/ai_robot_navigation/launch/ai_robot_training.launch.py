#!/usr/bin/env python3

"""
Professional AI Robot Training Launch System
- Multi-node coordination
- Configurable training parameters
- Real-time monitoring
- Automatic model saving
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    robot_model = LaunchConfiguration('robot_model', default='classic_4wd')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    training_mode = LaunchConfiguration('training_mode', default='true')
    log_level = LaunchConfiguration('log_level', default='info')
    
    # Declare launch arguments
    declare_robot_model = DeclareLaunchArgument(
        'robot_model',
        default_value='classic_4wd',
        description='Robot model to use (classic_4wd, turtlebot3, rover)')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time')
    
    declare_training_mode = DeclareLaunchArgument(
        'training_mode',
        default_value='true',
        description='Enable training mode')
    
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level (debug, info, warn, error)')
    
    # Robot simulation launch
    robot_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('classic_4wd_robot'), 'launch', 'robot_gazebo.launch.py')
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # YOLO8 Object Detector
    yolo_detector_node = Node(
        package='ai_robot_navigation',
        executable='yolo_detector',
        name='yolo_detector',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'confidence_threshold': 0.5},
            {'model_path': 'yolov8n.pt'}
        ],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # Professional DQN Agent
    dqn_agent_node = Node(
        package='ai_robot_navigation',
        executable='dqn_agent',
        name='dqn_agent',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'training_mode': training_mode},
            {'learning_rate': 0.0001},
            {'batch_size': 64},
            {'gamma': 0.99}
        ],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # Sensor Fusion Node (optional)
    sensor_fusion_node = Node(
        package='ai_robot_navigation',
        executable='sensor_fusion',
        name='sensor_fusion',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # RViz for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('ai_robot_navigation'), 'config', 'ai_robot.rviz')]
    )
    
    # RQT Image View for YOLO visualization
    rqt_image_view = ExecuteProcess(
        cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/yolo/annotated_image'],
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        declare_robot_model,
        declare_use_sim_time,
        declare_training_mode,
        declare_log_level,
        
        # Robot simulation
        robot_sim_launch,
        
        # AI components (delayed start to ensure robot is ready)
        TimerAction(
            period=3.0,
            actions=[yolo_detector_node]
        ),
        
        TimerAction(
            period=5.0,
            actions=[dqn_agent_node]
        ),
        
        # Visualization (optional)
        TimerAction(
            period=7.0,
            actions=[rviz_node, rqt_image_view]
        ),
    ])