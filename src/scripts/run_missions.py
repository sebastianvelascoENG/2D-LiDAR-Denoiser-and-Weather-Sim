#!/usr/bin/env python3
"""run_missions - drive the robot through a fixed set of Nav2 goals and log KPIs.

Uses nav2_simple_commander. Sets the initial pose from /ground_truth, then for
each goal (optionally looped) records: result, duration, straight-line distance,
distance travelled (from /odom), and number of recoveries.

Params:
  goals        (double array, flattened x,y,yaw triples)  default: a 4-corner loop
  n_loops      (int)   default 1
  output_csv   (str)   default 'rain_sim_missions.csv'
  condition    (str)   'calm'
  pipeline     (str)   'raw'
  settle_sec   (float) wait after setting initial pose (default 5.0)
  max_retries  (int)   retries per goal after a FAILED result, waiting for
                       slam_toolbox to come back active first (default 2).
                       Needed because async_slam_toolbox_node (Karto) can
                       abort mid-mission on noisy rain scans; the launch file
                       respawns it, but that takes a couple of seconds, so a
                       failed goal is retried instead of immediately trying
                       the next one against a still-recovering/dead map frame.
  goal_timeout (float) seconds before a goal attempt is cancelled and logged as
                       TIMEOUT (default 240.0)
  progress_file (str)  if set, a one-line 'phase|done|total|detail' status is
                       rewritten here as the mission advances (read by
                       run_sweep.sh to draw its progress bar)

Every ROS call here is bounded by a timeout. BasicNavigator's own
waitUntilNav2Active()/goToPose()/cancelTask()/lifecycleShutdown() wait on
service/action responses with no timeout, and a single dropped response
(seen as "failed to send response to /slam_toolbox/get_state (timeout)" in
the launch log) left the whole sweep hanging until Ctrl-C.
"""
import csv
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# rain_room loop; (2.5, 2.5) would sit on obstacle_box_1's corner, hence (3.5, 3.5).
DEFAULT_GOALS = [3.5, 3.5, 0.0, -2.5, 2.5, 0.0, -2.5, -2.5, 0.0, 2.5, -2.5, 0.0]


def quat_z(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MissionRunner(Node):
    def __init__(self):
        super().__init__('run_missions')
        self.declare_parameter('goals', DEFAULT_GOALS)
        self.declare_parameter('n_loops', 1)
        self.declare_parameter('output_csv', 'rain_sim_missions.csv')
        self.declare_parameter('condition', 'calm')
        self.declare_parameter('pipeline', 'raw')
        self.declare_parameter('settle_sec', 5.0)
        self.declare_parameter('localizer', 'slam_toolbox')  # 'amcl' or 'slam_toolbox'
        self.declare_parameter('max_retries', 2)
        self.declare_parameter('goal_timeout', 240.0)
        self.declare_parameter('progress_file', '')

        flat = list(self.get_parameter('goals').value)
        self.goals = [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
        self.n_loops = int(self.get_parameter('n_loops').value)
        self.path = os.path.expanduser(self.get_parameter('output_csv').value)
        self.condition = self.get_parameter('condition').value
        self.pipeline = self.get_parameter('pipeline').value
        self.settle = float(self.get_parameter('settle_sec').value)
        self.localizer = self.get_parameter('localizer').value
        self.max_retries = int(self.get_parameter('max_retries').value)
        self.goal_timeout = float(self.get_parameter('goal_timeout').value)
        prog = self.get_parameter('progress_file').value
        self.progress_path = os.path.expanduser(prog) if prog else ''
        self.total = self.n_loops * len(self.goals)

        self.gt = None
        self.odom = None
        self.travelled = 0.0
        self._last_odom_xy = None
        self.create_subscription(Odometry, '/ground_truth', self._gt, 10)
        self.create_subscription(Odometry, '/odom', self._odom, 20)
        self._state_clients = {}

    def _gt(self, m):
        self.gt = m

    def _odom(self, m):
        self.odom = m
        p = m.pose.pose.position
        if self._last_odom_xy is not None:
            self.travelled += math.hypot(p.x - self._last_odom_xy[0], p.y - self._last_odom_xy[1])
        self._last_odom_xy = (p.x, p.y)

    def _spin_until(self, pred, timeout=20.0):
        t0 = time.time()
        while rclpy.ok() and not pred() and time.time() - t0 < timeout:
            time.sleep(0.05)

    def _progress(self, phase, done, detail=''):
        if not self.progress_path:
            return
        try:
            tmp = self.progress_path + '.tmp'
            with open(tmp, 'w') as f:
                f.write(f'{phase}|{done}|{self.total}|{detail}\n')
            os.replace(tmp, self.progress_path)
        except OSError:
            pass

    def _wait_active(self, node_name, timeout):
        """Block until lifecycle node `node_name` reports 'active', or timeout
        elapses. Each get_state call is bounded, so a dropped response is
        retried instead of blocking forever."""
        client = self._state_clients.get(node_name)
        if client is None:
            client = self.create_client(GetState, f'/{node_name}/get_state')
            self._state_clients[node_name] = client
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            if client.wait_for_service(timeout_sec=1.0):
                future = client.call_async(GetState.Request())
                self._spin_until(future.done, timeout=3.0)
                if not future.done():
                    client.remove_pending_request(future)
                elif future.result() is not None and \
                        future.result().current_state.label == 'active':
                    return True
            time.sleep(0.5)
        return False

    def _wait_for_slam_recovery(self, timeout=20.0):
        """Block until slam_toolbox reports 'active' again (it self-respawns
        after a Karto crash per the launch file, but needs a couple of
        seconds), or timeout elapses. No-op for localizer != slam_toolbox."""
        if self.localizer != 'slam_toolbox':
            return True
        return self._wait_active('slam_toolbox', timeout)

    def _send_goal(self, nav, goal):
        """Bounded replacement for nav.goToPose(); returns False if the goal
        could not be sent or was rejected. Leaves nav's goal_handle /
        result_future set so nav.isTaskComplete()/getResult() work as usual."""
        nav.feedback = None
        nav.status = None
        nav.result_future = None
        if not nav.nav_to_pose_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("'NavigateToPose' action server not available")
            return False
        msg = NavigateToPose.Goal()
        msg.pose = goal
        future = nav.nav_to_pose_client.send_goal_async(msg, nav._feedbackCallback)
        rclpy.spin_until_future_complete(nav, future, timeout_sec=10.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            self.get_logger().error('goal was not accepted (or no response)')
            return False
        nav.goal_handle = handle
        nav.result_future = handle.get_result_async()
        return True

    def _cancel_goal(self, nav):
        """Bounded replacement for nav.cancelTask()."""
        if nav.goal_handle is None:
            return
        future = nav.goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(nav, future, timeout_sec=5.0)

    def run(self):
        # BasicNavigator's isTaskComplete()/waitUntilNav2Active() spin *their own*
        # node via rclpy's implicit process-global executor. If we spun `self`
        # (this node) through that same global executor too - e.g. via bare
        # rclpy.spin_once(self, ...) interleaved with nav's calls - the two nodes
        # fight over one shared executor and our /odom, /ground_truth callbacks
        # starve (travelled_m silently reads back as 0.0 even though the robot
        # moved). Give `self` its own dedicated executor spinning in the
        # background so its subscriptions are serviced independently of nav.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._spin_until(lambda: self.gt is not None, timeout=30.0)
        nav = BasicNavigator()

        if self.gt is not None and self.localizer == 'amcl':
            init = PoseWithCovarianceStamped()
            init.header.frame_id = 'map'
            init.header.stamp = self.get_clock().now().to_msg()
            init.pose.pose = self.gt.pose.pose
            nav.setInitialPose(init)

        self._progress('waiting', 0, f'waiting for {self.localizer} + bt_navigator')
        for node_name in (self.localizer, 'bt_navigator'):
            if not self._wait_active(node_name, timeout=120.0):
                self.get_logger().error(f'{node_name} did not become active within 120s')
                self._progress('failed', 0, f'{node_name} never became active')
                return False
        self.get_logger().info('Nav2 is ready for use!')
        time.sleep(self.settle)

        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        fh = open(self.path, 'a', newline='')
        w = csv.writer(fh)
        if new:
            w.writerow(['loop', 'goal_idx', 'condition', 'pipeline', 'goal_x', 'goal_y',
                        'result', 'duration_s', 'straight_m', 'travelled_m', 'recoveries'])

        done = 0
        for loop in range(self.n_loops):
            for gi, (gx, gy, gyaw) in enumerate(self.goals):
                where = f'loop {loop + 1}/{self.n_loops} goal {gi + 1}/{len(self.goals)}'
                start = self.gt.pose.pose.position if self.gt else None
                straight = math.hypot(gx - start.x, gy - start.y) if start else float('nan')

                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
                qz, qw = quat_z(float(gyaw))
                goal.pose.orientation.z, goal.pose.orientation.w = qz, qw

                for attempt in range(self.max_retries + 1):
                    goal.header.stamp = nav.get_clock().now().to_msg()
                    self.travelled = 0.0
                    self._last_odom_xy = None
                    t0 = time.time()
                    try_tag = f' (try {attempt + 1})' if attempt else ''
                    res = None
                    if not self._send_goal(nav, goal):
                        res = 'NOT_SENT'
                    else:
                        while not nav.isTaskComplete():
                            elapsed = time.time() - t0
                            fb = nav.getFeedback()
                            left = f'{fb.distance_remaining:.1f} m left' if fb else '-- m left'
                            self._progress('navigating', done,
                                           f'{where}{try_tag} | {left} | {elapsed:.0f}s')
                            if elapsed > self.goal_timeout:
                                self._cancel_goal(nav)
                                res = 'TIMEOUT'
                                break
                            time.sleep(0.1)
                    dur = time.time() - t0
                    if res is None:
                        res = {TaskResult.SUCCEEDED: 'SUCCEEDED',
                               TaskResult.CANCELED: 'CANCELED',
                               TaskResult.FAILED: 'FAILED'}.get(nav.getResult(), 'UNKNOWN')
                    if res == 'SUCCEEDED' or attempt == self.max_retries:
                        break
                    self.get_logger().warn(
                        f"loop {loop} goal {gi}: attempt {attempt + 1} {res} "
                        f"in {dur:.1f}s - waiting for slam_toolbox to recover before retry")
                    self._progress('retrying', done, f'{where} | {res}, waiting for SLAM')
                    self._wait_for_slam_recovery(timeout=20.0)
                    if dur < 2.0:
                        # Nav2's BT gives up instantly (no spin/wait/backup recovery
                        # attempted) when the goal is currently outside the still-
                        # growing SLAM global costmap's bounds ("Goal Coordinates
                        # of(...) was outside bounds") - that class of error is
                        # marked unrecoverable by nav2's own BT nodes, so retrying
                        # immediately just repeats the same instant failure. Back
                        # off longer here to give slam_toolbox real time to publish
                        # a bigger map before resending the same goal.
                        self.get_logger().warn(
                            f"loop {loop} goal {gi}: attempt failed in {dur:.1f}s (instant "
                            f"failure, likely costmap not yet covering the goal) - "
                            f"backing off 12s instead of the usual settle time")
                        time.sleep(12.0)
                    else:
                        time.sleep(self.settle)

                fb = nav.getFeedback()
                recov = getattr(fb, 'number_of_recoveries', -1) if fb else -1
                w.writerow([loop, gi, self.condition, self.pipeline, f'{gx:.2f}', f'{gy:.2f}',
                            res, f'{dur:.2f}', f'{straight:.2f}', f'{self.travelled:.2f}', recov])
                fh.flush()
                done += 1
                self._progress('navigating', done, f'{where} | {res} in {dur:.0f}s')
                self.get_logger().info(f"loop {loop} goal {gi}: {res} in {dur:.1f}s "
                                       f"({self.travelled:.1f} m travelled)")
        fh.close()
        self._progress('done', done, 'missions complete')
        # No nav.lifecycleShutdown(): it waits on lifecycle_manager with no
        # timeout, and run_sweep.sh tears the whole stack down right after.
        self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)
        nav.destroy_node()
        return True


def main(args=None):
    rclpy.init(args=args)
    node = MissionRunner()
    ok = False
    try:
        ok = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # Non-zero on failure/interrupt so run_sweep.sh doesn't mistake it for a completed run.
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
