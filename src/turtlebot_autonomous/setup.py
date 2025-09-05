from setuptools import setup

package_name = 'turtlebot_autonomous'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.nodes', f'{package_name}.config', 
          f'{package_name}.core', f'{package_name}.utils', f'{package_name}.models'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ty',
    maintainer_email='ty@example.com',
    description='TurtleBot autonomous navigation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_controller = turtlebot_autonomous.nodes.main_controller:main',
        ],
    },
)
