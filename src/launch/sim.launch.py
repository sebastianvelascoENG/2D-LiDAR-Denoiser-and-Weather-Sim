#!/usr/bin/env python3
"""Bring up Gazebo Sim (Harmonic) + tb3_waffle_c1 + ros_gz bridge.

Publishes the CLEAN scan on /scan_clean. Run weather_model_node (see
experiment.launch.py) to get the weather-noised /scan that SLAM/Nav2 consume.

Args:
  world         rain_room | rain_yard   (default rain_room)
  x_pose y_pose yaw                      spawn pose (default 0 0 0)
  gui           true | false             start the Gazebo GUI client
  use_sim_time  true | false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('thesis_rain_sim')
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw = LaunchConfiguration('yaw')
    gui = LaunchConfiguration('gui')
    use_sim_time = LaunchConfiguration('use_sim_time')

    world_path = PathJoinSubstitution([pkg, 'worlds', [world, '.sdf']])
    model_path = os.path.join(pkg, 'models', 'tb3_waffle_c1', 'model.sdf')
    urdf_path = os.path.join(tb3_gazebo, 'urdf', 'turtlebot3_waffle.urdf')
    bridge_config = os.path.join(pkg, 'config', 'bridge.yaml')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.pathsep.join([
            os.path.join(pkg, 'models'),
            os.path.join(tb3_gazebo, 'models'),
        ]),
    )

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': ['-r -s -v2 ', world_path],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
        condition=IfCondition(gui),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc,
        }],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'tb3_waffle_c1',
            '-file', model_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
            '-Y', yaw,
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='rain_room'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        set_resource_path,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn,
        bridge,
    ])
