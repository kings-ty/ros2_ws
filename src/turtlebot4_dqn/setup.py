from setuptools import setup
import os
from glob import glob

package_name = 'turtlebot4_dqn'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='TurtleBot4 DQN navigation training',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dqn_trainer = turtlebot4_dqn.dqn_trainer:main',
            'dqn_environment = turtlebot4_dqn.dqn_environment:main',
        ],
    },
)