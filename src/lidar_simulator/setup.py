from setuptools import find_packages, setup

package_name = 'lidar_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tonto',
    maintainer_email='tnt.kosen2423@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_sim_node = lidar_simulator.lidar_sim_node:main',
            'joy_integrate_pose_node = lidar_simulator.joy_integrate_pose_node:main',
        ],
    },
)
