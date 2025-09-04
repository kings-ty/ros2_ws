from setuptools import setup

package_name = 'turtlebot_autonomous'

setup(
    name=package_name,
    version='1.0.0',
    packages=[
        package_name,
        f'{package_name}.nodes',
        f'{package_name}.models', 
        f'{package_name}.core',
        f'{package_name}.utils',
        f'{package_name}.config'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/autonomous_navigation.launch.py']),
        ('share/' + package_name + '/config', ['config/robot_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='TurtleBot3 Autonomous Navigation System',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'autonomous_navigation = turtlebot_autonomous.nodes.autonomous_navigation:main',
            'camera_test = turtlebot_autonomous.nodes.camera_test:main',
            'laser_test = turtlebot_autonomous.nodes.laser_test:main',
        ],
    },
)