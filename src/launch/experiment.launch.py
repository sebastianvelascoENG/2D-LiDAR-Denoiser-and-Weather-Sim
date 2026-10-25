#!/usr/bin/env python3
"""One weather x one pipeline run of the weather / denoising experiment.

  ros2 launch thesis_rain_sim experiment.launch.py \
       world:=rain_yard rain:=heavy fog:=light dust:=off pipeline:=denoised \
       nav2:=true rain_model:=hybrid seed:=3

Weather (each effect on or off independently):
  rain:=calm|light|heavy    (condition:= is a deprecated alias for rain:=)
  fog:=off|light|heavy      fog_visibility_m:=<m> replaces the level's visibility
  dust:=off|light|heavy     dust_visibility_m:=<m>; clouds from config/dust_clouds.yaml,
                            set dust_clouds:= (default: the world name)
Fog and dust need rain_model:=goodin2019|alavi2025|hybrid.

Brings up: sim (sim.launch.py) + weather_model_node + clustering_denoiser_node
+ slam_toolbox (consuming /scan or /scan_denoised) + ground_truth_logger
(+ optional Nav2 + RViz + rosbag record).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from nav2_common.launch import RewrittenYaml

from thesis_rain_sim import weather


def _weather_nodes(context, pkg):
    """weather_model_node + ground_truth_logger, with the weather resolved in Python
    (rain / deprecated condition alias, run label) so a bad value fails the launch."""
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    try:
        rain = weather.resolve_rain(arg('rain'), arg('condition'))
        label = weather.weather_label(rain, arg('fog'), arg('dust'))
    except ValueError as e:
        raise RuntimeError(f'experiment.launch.py: {e}') from None
    fog, dust = arg('fog').lower(), arg('dust').lower()
    actions = [LogInfo(msg=f"weather: {label}  (rain={rain} fog={fog} dust={dust}, "
                           f"rain_model={arg('rain_model')}, seed={arg('seed')})")]
    if arg('condition') and not arg('rain'):
        actions.append(LogInfo(msg="condition:= is a deprecated alias for rain:="))

    actions.append(Node(
        package='thesis_rain_sim', executable='weather_model_node.py',
        name='weather_model_node', output='screen',
        parameters=[os.path.join(pkg, 'config', 'weather_params.yaml'), {
            'use_sim_time': True,
            'rain': rain, 'condition': '',
            'fog': fog, 'fog_visibility_m': float(arg('fog_visibility_m')),
            'dust': dust, 'dust_visibility_m': float(arg('dust_visibility_m')),
            'dust_cloud_set': arg('dust_clouds') or arg('world'),
            'rain_model': arg('rain_model'), 'seed': int(arg('seed')),
        }],
    ))
    actions.append(Node(
        package='thesis_rain_sim', executable='ground_truth_logger.py',
        name='ground_truth_logger', output='screen',
        parameters=[{
            'use_sim_time': True,
            'output_csv': arg('log_csv'),
            'condition': label,
            'rain': rain, 'fog': fog, 'dust': dust,
            'pipeline': arg('pipeline'),
            'rate_hz': 2.0,
        }],
    ))
    return actions


def generate_launch_description():
    pkg = get_package_share_directory('thesis_rain_sim')

    world = LaunchConfiguration('world')
    pipeline = LaunchConfiguration('pipeline')
    use_slam = LaunchConfiguration('slam')
    use_nav2 = LaunchConfiguration('nav2')
    use_rviz = LaunchConfiguration('rviz')
    gui = LaunchConfiguration('gui')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw = LaunchConfiguration('yaw')
    record_bag = LaunchConfiguration('record_bag')
    bag_path = LaunchConfiguration('bag_path')

    scan_for_nav = PythonExpression(
        ["'/scan_denoised' if '", pipeline, "' == 'denoised' else '/scan'"])

    slam_params = RewrittenYaml(
        source_file=os.path.join(pkg, 'config', 'slam.yaml'),
        root_key='', convert_types=True,
        param_rewrites={'scan_topic': scan_for_nav, 'use_sim_time': 'true'})

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={
            'world': world, 'x_pose': x_pose, 'y_pose': y_pose, 'yaw': yaw,
            'gui': gui, 'use_sim_time': 'true',
        }.items(),
    )

    weather_nodes = OpaqueFunction(function=_weather_nodes, kwargs={'pkg': pkg})

    denoiser = Node(
        package='thesis_rain_sim', executable='clustering_denoiser_node.py',
        name='clustering_denoiser_node', output='screen',
        parameters=[os.path.join(pkg, 'config', 'denoiser_params.yaml')],
    )

    # slam_toolbox is a lifecycle node in Jazzy - start it, then drive
    # configure -> activate ourselves (no external lifecycle manager).
    #
    # async_slam_toolbox_node (Karto) can abort with "Mapper FATAL ERROR -
    # unable to get pointer in probability search!" on sparse/noisy scans
    # (observed under condition:=heavy) - it's an uncaught exception inside
    # slam_toolbox itself, not something this package's nodes feed it
    # (weather_model_node/clustering_denoiser_node always emit finite, clamped
    # ranges). respawn=True keeps localization (and therefore Nav2, which
    # otherwise aborts every goal once the map->odom transform goes stale)
    # recovering instead of the whole run dying on one bad scan.
    #
    # The configure/activate bring-up is driven by a plain `ros2 lifecycle
    # set` CLI call rather than launch's EmitEvent(ChangeState(...)) events:
    # the latter only reaches slam_toolbox's lifecycle service once per
    # launch - re-emitting the same event objects (or fresh ones built by a
    # callable on_start handler) on a respawn silently does nothing, leaving
    # the respawned node stuck 'unconfigured' forever. Spawning a short CLI
    # process each time via a callable on_start handler works reliably.
    slam = LifecycleNode(
        package='slam_toolbox', executable='async_slam_toolbox_node',
        name='slam_toolbox', namespace='', output='screen',
        condition=IfCondition(use_slam),
        parameters=[slam_params, {'use_lifecycle_manager': False, 'use_sim_time': True}],
        respawn=True, respawn_delay=2.0,
    )

    def _bring_up_slam(event, context):
        # OnProcessStart fires as soon as the OS process launches, well before
        # async_slam_toolbox_node has finished initializing and advertised its
        # lifecycle services - a single `ros2 lifecycle set` call here can lose
        # that race (service not found yet -> immediate non-zero exit), which
        # leaves the node stuck 'unconfigured' (no map frame, ever) for the
        # rest of the run since nothing retries it. Retry each call until it
        # succeeds instead of firing once.
        return [
            LogInfo(msg='slam_toolbox (re)started - configuring and activating'),
            ExecuteProcess(
                cmd=['bash', '-c',
                     'until ros2 lifecycle set /slam_toolbox configure; do sleep 0.5; done && '
                     'until ros2 lifecycle set /slam_toolbox activate; do sleep 0.5; done'],
                output='screen',
            ),
        ]

    slam_bring_up = RegisterEventHandler(
        OnProcessStart(target_action=slam, on_start=_bring_up_slam),
        condition=IfCondition(use_slam),
    )

    # Nav2 is 10 lifecycle nodes; on a loaded machine (sim < 1x real-time) they
    # lose the bringup race if started with everything else. Delay until the sim
    # + SLAM have settled.
    nav2 = TimerAction(
        period=LaunchConfiguration('nav2_delay'),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nav2.launch.py')),
            launch_arguments={'scan_topic': scan_for_nav, 'use_sim_time': 'true'}.items(),
        )],
        condition=IfCondition(use_nav2),
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', os.path.join(pkg, 'rviz', 'experiment.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    bag = ExecuteProcess(
        condition=IfCondition(record_bag),
        cmd=['ros2', 'bag', 'record', '-o', bag_path,
             '/scan_clean', '/scan', '/scan_denoised', '/ground_truth',
             '/odom', '/tf', '/tf_static', '/map', '/rain/rate', '/weather/state',
             '/scan_rain_labels', '/denoiser/keyframe_ratio'],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='rain_room'),
        DeclareLaunchArgument('rain', default_value='',
                              description="calm | light | heavy ('' = calm, or condition:=)"),
        DeclareLaunchArgument('condition', default_value='',
                              description='DEPRECATED alias for rain:='),
        DeclareLaunchArgument('fog', default_value='off', description='off | light | heavy'),
        DeclareLaunchArgument('dust', default_value='off', description='off | light | heavy'),
        DeclareLaunchArgument('fog_visibility_m', default_value='-1.0',
                              description='>0 replaces the fog level visibility [m]'),
        DeclareLaunchArgument('dust_visibility_m', default_value='-1.0',
                              description='>0 replaces the dust level visibility [m]'),
        DeclareLaunchArgument('dust_clouds', default_value='',
                              description="cloud set in config/dust_clouds.yaml ('' = world name)"),
        DeclareLaunchArgument('seed', default_value='1',
                              description='weather_model_node RNG seed'),
        DeclareLaunchArgument('rain_model', default_value='hybrid',
                              description='legacy | goodin2019 | alavi2025 | hybrid '
                                          '(fog / dust need one of the last three)'),
        DeclareLaunchArgument('pipeline', default_value='raw'),
        DeclareLaunchArgument('slam', default_value='true'),
        DeclareLaunchArgument('nav2', default_value='false'),
        DeclareLaunchArgument('nav2_delay', default_value='15.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('log_csv', default_value='rain_sim_log.csv'),
        DeclareLaunchArgument('record_bag', default_value='false'),
        DeclareLaunchArgument('bag_path', default_value='rain_sim_bag'),
        sim,
        weather_nodes,
        denoiser,
        slam,
        slam_bring_up,
        nav2,
        rviz,
        bag,
    ])
