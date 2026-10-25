#!/usr/bin/env python3
"""ground_truth_logger - record navigation error and scan quality to CSV.

Logs at a fixed rate:
  t_sec, condition, pipeline, rain_rate,
  gt_x, gt_y, gt_yaw,                     ground truth (from /ground_truth)
  est_x, est_y, est_yaw,                  SLAM estimate (tf map->base_footprint)
  err_pos_m, err_yaw_rad,
  n_clean, n_scan, n_denoised             valid-beam counts (scan quality)
  rain, fog, dust                         weather levels of the run

Params:
  output_csv   (str)  path to write (append if exists)
  condition    (str)  weather label - free text; experiment.launch.py passes
                      thesis_rain_sim.weather.weather_label (calm, heavy, fog_light,
                      rain_heavy+dust_light, ...)
  rain / fog / dust (str) the individual levels (calm|light|heavy, off|light|heavy)
  pipeline     (str)  raw|denoised       - free-text label
  rate_hz      (float) log rate (default 2.0)
  map_frame / base_frame
"""
import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import tf2_ros


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GroundTruthLogger(Node):

    def __init__(self):
        super().__init__('ground_truth_logger')
        self.declare_parameter('output_csv', 'rain_sim_log.csv')
        self.declare_parameter('condition', 'calm')
        self.declare_parameter('rain', '')
        self.declare_parameter('fog', 'off')
        self.declare_parameter('dust', 'off')
        self.declare_parameter('pipeline', 'raw')
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        self.path = os.path.expanduser(self.get_parameter('output_csv').value)
        self.condition = self.get_parameter('condition').value
        self.weather = [self.get_parameter(k).value for k in ('rain', 'fog', 'dust')]
        self.pipeline = self.get_parameter('pipeline').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate = float(self.get_parameter('rate_hz').value)

        self.gt = None
        self.rain_rate = float('nan')
        self.n_clean = self.n_scan = self.n_denoised = -1

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Odometry, '/ground_truth', self._gt, 20)
        self.create_subscription(Float64, '/rain/rate', self._rate, latched)
        self.create_subscription(LaserScan, '/scan_clean', lambda m: self._count(m, 'clean'), 10)
        self.create_subscription(LaserScan, '/scan', lambda m: self._count(m, 'scan'), 10)
        self.create_subscription(LaserScan, '/scan_denoised', lambda m: self._count(m, 'denoised'), 10)

        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.fh = open(self.path, 'a', newline='')
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(['t_sec', 'condition', 'pipeline', 'rain_rate',
                             'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y', 'est_yaw',
                             'err_pos_m', 'err_yaw_rad', 'n_clean', 'n_scan', 'n_denoised',
                             'rain', 'fog', 'dust'])
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"logging -> {self.path}  ({self.condition}/{self.pipeline})")

    def _gt(self, m):
        self.gt = m

    def _rate(self, m):
        self.rain_rate = m.data

    def _count(self, m, which):
        import numpy as np
        r = np.asarray(m.ranges)
        lo = m.range_min if m.range_min > 0 else 0.1
        hi = m.range_max if m.range_max > 0 else 12.0
        n = int(np.count_nonzero(np.isfinite(r) & (r >= lo) & (r < hi - 1e-3)))
        setattr(self, f'n_{which}', n)

    def _tick(self):
        if self.gt is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        gp = self.gt.pose.pose
        gx, gy, gyaw = gp.position.x, gp.position.y, yaw_from_quat(gp.orientation)

        ex = ey = eyaw = float('nan')
        err_pos = err_yaw = float('nan')
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame,
                                                 rclpy.time.Time())
            ex = tf.transform.translation.x
            ey = tf.transform.translation.y
            eyaw = yaw_from_quat(tf.transform.rotation)
            err_pos = math.hypot(gx - ex, gy - ey)
            d = (gyaw - eyaw + math.pi) % (2 * math.pi) - math.pi
            err_yaw = abs(d)
        except Exception:
            pass

        self.w.writerow([f'{t:.3f}', self.condition, self.pipeline, f'{self.rain_rate:.2f}',
                         f'{gx:.4f}', f'{gy:.4f}', f'{gyaw:.4f}',
                         f'{ex:.4f}', f'{ey:.4f}', f'{eyaw:.4f}',
                         f'{err_pos:.4f}', f'{err_yaw:.4f}',
                         self.n_clean, self.n_scan, self.n_denoised, *self.weather])
        self.fh.flush()

    def destroy_node(self):
        try:
            self.fh.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
