#!/usr/bin/env python3
"""Nav2 bringup for tb3_waffle_c1, with the LiDAR topic chosen at launch.

  ros2 launch thesis_rain_sim nav2.launch.py scan_topic:=/scan_denoised

Uses the complete stock nav2_bringup params (so every lifecycle node - incl.
route_server / collision_monitor / docking_server - has a valid section) and
rewrites just what this study needs: the scan topic, sensor ranges for the
RPLiDAR C1, and use_sim_time. robot_radius (0.27) comes from the YAML as-is -
see the note at the top of nav2_waffle.yaml before overriding it here. SLAM supplies map->odom, so AMCL
and map_server are not started here (navigation_launch.py only).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('thesis_rain_sim')
    nav2_bringup = get_package_share_directory('nav2_bringup')

    scan_topic = LaunchConfiguration('scan_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')

    # DWB-based params tuned for the TB3 waffle (config/nav2_waffle.yaml),
    # completed with the route_server / smoother / collision_monitor / docking
    # sections navigation_launch.py requires.
    base_params = os.path.join(pkg, 'config', 'nav2_waffle.yaml')
    rewritten = RewrittenYaml(
        source_file=base_params,
        root_key='',
        param_rewrites={
            'topic': scan_topic,          # costmap + collision_monitor scan sources
            'scan_topic': scan_topic,     # amcl (unused with SLAM, kept consistent)
            'use_sim_time': use_sim_time,
            'raytrace_max_range': '10.0',
            'obstacle_max_range': '8.0',
            'max_laser_range': '12.0',
            'bond_timeout': '30.0',       # loaded machine: slow lifecycle bringup
        },
        convert_types=True,
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': rewritten,
            'autostart': autostart,
            'use_composition': 'False',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        navigation,
    ])
