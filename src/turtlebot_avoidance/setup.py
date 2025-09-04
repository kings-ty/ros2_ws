from setuptools import setup

package_name = 'turtlebot_avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='TurtleBot obstacle avoidance',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_avoidance = turtlebot_avoidance.obstacle_avoidance:main',
	    'simple_camera_test = turtlebot_avoidance.simple_camera_test:main',
	    'yolo_navigation = turtlebot_avoidance.yolo_navigation:main',
        ],
    },
)
