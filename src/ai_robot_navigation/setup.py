from setuptools import setup
import os
from glob import glob

package_name = 'ai_robot_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AI Robotics Developer',
    maintainer_email='dev@robotics.com',
    description='Professional AI Robot Navigation with YOLO8 + DQN',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = ai_robot_navigation.yolo_detector:main',
            'dqn_agent = ai_robot_navigation.dqn_agent:main',
            'sensor_fusion = ai_robot_navigation.sensor_fusion:main',
            'navigation_controller = ai_robot_navigation.navigation_controller:main',
            'training_manager = ai_robot_navigation.training_manager:main',
        ],
    },
)