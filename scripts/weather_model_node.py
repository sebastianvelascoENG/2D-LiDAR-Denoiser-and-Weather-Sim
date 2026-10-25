#!/usr/bin/env python3
"""weather_model_node - inject rain, fog and dust into a clean 2D LiDAR scan.

(Formerly rain_model_node.)

Subscribes : /scan_clean        (sensor_msgs/LaserScan) - clean sim ranges
             /ground_truth      (nav_msgs/Odometry) - sensor pose for world-frame dust clouds
Publishes  : /scan              (sensor_msgs/LaserScan) - weather-corrupted ranges + modelled intensities
             /scan_rain_labels  (sensor_msgs/LaserScan) - per-beam ground truth, same header:
                                  ranges      = the CLEAN input ranges
                                  intensities = label code per beam (float):
                                    0 real return, no weather range error (sensor noise only)
                                    1 real return, weather range noise applied
                                    2 real surface lost to the weather (attenuation / dropout /
                                      replaced by a stronger weather echo)
                                    3 published return is a rain droplet (false return)
                                    4 no surface in the clean scan, no weather echo
                                    5 published return is fog backscatter (false return)
                                    6 published return is dust backscatter (false return)
             /rain/rate         (std_msgs/Float64, transient_local) - active rain rate [mm/h]
             /weather/state     (std_msgs/String, transient_local) - JSON: active rain / fog /
                                  dust levels, visibilities, cloud set, model, run label

The scan that SLAM / Nav2 / the denoiser consume is /scan.

Each weather effect is switched on and off on its own:

  rain  calm | light | heavy   (`condition` is a deprecated alias for `rain`)
  fog   off  | light | heavy
  dust  off  | light | heavy   (inside the clouds of `dust_cloud_set`)

calm + off + off is the clear-weather baseline. `rain_model` picks the rain
maths (thesis_rain_sim.rain_models, pure numpy, unit tested):

  legacy      original ad-hoc model, kept bit for bit as a baseline; rain only
  goodin2019  Goodin et al. 2019, Electronics 8(1):89
  alavi2025   Alavi et al. 2025, Appl. Sci. 15(15):8191
  hybrid      LISA drops (Kilic et al. 2021, arXiv:2107.07004) on the alavi2025 sensor

Fog and dust (thesis_rain_sim.fog_dust) run
on the same sensor and the same single-strongest-echo decision as the
physical rain models, so they need goodin2019, alavi2025 or hybrid.

All parameters can be changed at run time; invalid values are rejected. The
new configuration takes effect on the next scan and the RNG streams continue
(they are seeded once, from `seed`, at start-up: rain from `seed`, fog / dust
from [seed, 1], so switching fog or dust on does not change the rain draws).
"""
import json
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, String
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import tf2_ros

from thesis_rain_sim import weather
from thesis_rain_sim.dust_geometry import compose_pose, yaw_from_quaternion
from thesis_rain_sim.fog_dust import FogDustParams, WeatherMedia, load_cloud_sets
from thesis_rain_sim.rain_models import LEGACY_NO_MEDIA, clean_hits, make_model, rain_rate_for

# Node-level parameters (not passed to the models).
NODE_DEFAULTS = {
    'scan_in': '/scan_clean',
    'scan_out': '/scan',
    'labels_out': '/scan_rain_labels',
    'rain': '',                             # calm | light | heavy ('' = calm or `condition`)
    'condition': '',                        # DEPRECATED alias for `rain`
    'seed': 1,
    'fog': 'off',                           # off | light | heavy
    'fog_visibility_m': -1.0,               # >0 replaces the level's visibility
    'dust': 'off',                          # off | light | heavy
    'dust_visibility_m': -1.0,
    'dust_clouds_file': '',                 # '' -> share/thesis_rain_sim/config/dust_clouds.yaml
    'dust_cloud_set': 'sensor_ahead',
    'world_pose_source': 'ground_truth',    # ground_truth | tf
    'ground_truth_topic': '/ground_truth',
    'world_frame': 'map',                   # world_pose_source: tf
    'base_frame': 'base_footprint',         # world_pose_source: ground_truth
}

# name -> default. Everything here is passed to the model factories, which
# take the fields they know (rain_models.make_model, fog_dust.FogDustParams).
PARAM_DEFAULTS = {
    'rain_model': 'hybrid',                 # legacy | goodin2019 | alavi2025 | hybrid
    'rain_rate_mm_h': -1.0,                 # <0 -> derive from `rain`

    # --- legacy -------------------------------------------------------------
    'i0': 10000.0,                          # return intensity at 1 m, unit reflectivity
    'intensity_noise_frac': 0.05,
    'min_detectable_intensity': 8.0,
    'alpha_k': 0.01,                        # extinction: alpha = alpha_k * R^alpha_p
    'alpha_p': 0.6,
    'q_scatter': 0.6,                       # max extra dropout prob at long range
    'range_sigma_base_m': 0.012,
    'k_rain_range': 0.02,                   # range-noise growth per (mm/h) at r_max
    'p_false_k': 0.004,                     # droplet false-return prob per (mm/h)
    'false_range_min_m': 0.30,
    'false_range_max_m': 4.0,
    'false_intensity_mean': 6.0,
    'false_intensity_frac': 0.5,

    # --- goodin2019 + alavi2025 (shared) --------------------------------------
    'default_reflectivity': 0.7,            # gz gives no per-surface reflectivity
    'z_max': 12.0,                          # clear-weather max range [m] ...
    'z_max_reflectivity': 0.7,              # ... for a target of this reflectivity (C1: 12 m @ 70 %)
    'sensor_sigma_total_m': 0.012,          # target TOTAL sensor range noise
    'simulator_sigma_m': 0.010,             # already applied by gz (model.sdf <noise>)
    'intensity_output_scale': 10000.0 * math.pi,  # P_n -> published intensity

    # --- goodin2019 -----------------------------------------------------------
    'a': 0.01,                              # Eq. 5
    'b': 0.6,
    'range_noise_frac': 0.02,               # Eq. 9

    # --- alavi2025 (pulse_width_s and the overlap ramp are also fog / dust's) --
    'rain_type': 'continental',             # continental | thunderstorm | tropical
    'pulse_width_s': 5.0e-9,                # FWHM; bin width c*tau/2
    'overlap_r1_m': 0.05,                   # bistatic overlap: 0 below R1 ...
    'overlap_r2_m': 1.0,                    # ... 1 above R2
    'echo_mode': 'strongest',               # strongest | first
    'backscatter_mode': 'stochastic',       # stochastic | mean
    'use_incidence_angle': False,
    'incidence_window': 2,

    # --- hybrid (LISA, Kilic et al. 2021; also uses the shared block, rain_type
    #     and overlap_r1_m / overlap_r2_m) ---------------------------------------
    'extinction_source': 'dsd',             # dsd (Eq. 3, Q_ext = 2) | alavi (Table 2)
    'mp_n0': 8000.0,                        # Marshall & Palmer 1948 [m^-3 mm^-1]
    'mp_lambda_k': 4.1,                     # Lambda = k R^p [1/mm], Marshall & Palmer 1948
    'mp_lambda_p': -0.21,
    'min_drop_diameter_mm': 0.05,           # LISA Sec. 5.1
    'max_drop_diameter_mm': 6.0,            # (assumption) drop break-up
    'beam_divergence_rad': 3.0e-3,          # (assumption) LISA's Velodyne value
    'beam_exit_diameter_m': 0.0,            # (assumption)
    'drop_reflectivity': 0.0201,            # LISA Eq. 11, water n = 1.33
    'range_accuracy_m': 0.06,               # (assumption) C1 +/-30 mm as a full span
    'use_overlap': True,                    # Alavi Sec. 2.1 ramp on drop echoes

    # --- fog + dust (fog_dust.py) ----------------------------------------------
    'visibility_convention': 'kim',         # kim 3.912/V (Kim 2001) | mor 3.0/V (Hahner 2021)
    'fog_beta_model': 'rasshofer',          # rasshofer | alavi (per sr) | alavi_raw (as printed)
    'dust_beta_model': 'alavi',             # alavi (per sr) | alavi_raw (as printed)
    'soft_range_step_m': 0.02,
    'soft_integration_points': 101,         # odd (composite Simpson)
    'soft_range_convention': 'rising_edge',  # rising_edge | peak
    'soft_gain': 1.0,                       # sensitivity studies only
    'jitter_exponent_max': 1.0,             # Hahner: range * 2^U(-1, 1); 0 disables
}


# YAML reads a bare `off` as false, so `ros2 param set /weather_model_node fog off`
# arrives as a bool. fog / dust are declared with dynamic typing and false means off.
BOOL_AS_OFF = ('fog', 'dust')


def _level(value):
    if isinstance(value, bool):
        if value:
            raise ValueError("fog / dust must be off | light | heavy, got 'true'")
        return 'off'
    return str(value).strip().lower()


class Config:
    """Everything derived from one set of parameter values (no side effects)."""

    def __init__(self, values):
        self.values = values
        self.rain = weather.resolve_rain(values['rain'], values['condition'])
        self.used_alias = bool(values['condition']) and not values['rain']
        self.fog = _level(values['fog'])
        self.dust = _level(values['dust'])
        self.label = weather.weather_label(self.rain, self.fog, self.dust)
        params = {name: values[name] for name in PARAM_DEFAULTS}
        self.rain_rate = rain_rate_for(self.rain, float(params['rain_rate_mm_h']))
        self.model = make_model(params['rain_model'], params)
        fog_v = weather.visibility_for('fog', self.fog, values['fog_visibility_m'])
        dust_v = weather.visibility_for('dust', self.dust, values['dust_visibility_m'])
        self.cloud_set = str(values['dust_cloud_set'])
        clouds = []
        if dust_v is not None:
            path = values['dust_clouds_file'] or os.path.join(
                get_package_share_directory('thesis_rain_sim'), 'config', 'dust_clouds.yaml')
            sets = load_cloud_sets(os.path.expanduser(path))
            if self.cloud_set not in sets:
                raise ValueError(f"dust_cloud_set '{self.cloud_set}' is not in {path}; "
                                 f"available: {', '.join(sorted(sets))}")
            clouds = sets[self.cloud_set]
        self.media = WeatherMedia(FogDustParams.from_dict(params), fog_v, dust_v, clouds)
        if self.media.active and self.model.name == 'legacy':
            raise ValueError(f"fog:={self.fog} dust:={self.dust} with rain_model:=legacy: "
                             + LEGACY_NO_MEDIA)
        if values['world_pose_source'] not in ('ground_truth', 'tf'):
            raise ValueError("world_pose_source must be 'ground_truth' or 'tf', got "
                             f"'{values['world_pose_source']}'")

    def state(self):
        m = self.media
        return {
            'label': self.label, 'rain': self.rain, 'rain_rate_mm_h': self.rain_rate,
            'fog': self.fog, 'fog_visibility_m': m.fog_visibility_m,
            'dust': self.dust, 'dust_visibility_m': m.dust_visibility_m,
            'dust_cloud_set': self.cloud_set if m.dust_on else None,
            'rain_model': self.model.name, 'seed': int(self.values['seed']),
            'media': m.describe(),
        }


class WeatherModelNode(Node):

    def __init__(self):
        super().__init__('weather_model_node')

        for name, default in {**NODE_DEFAULTS, **PARAM_DEFAULTS}.items():
            if name in BOOL_AS_OFF:
                self.declare_parameter(name, default,
                                       ParameterDescriptor(dynamic_typing=True))
            else:
                self.declare_parameter(name, default)

        self._pending = None
        self._warned_intensity = False
        self._gt_pose = None
        self._sensor_offsets = {}
        self.cfg = Config(self._values())         # raises on bad input: fail loudly at startup
        self.seed = int(self.cfg.values['seed'])
        self.rng = np.random.default_rng(self.seed)
        self.weather_rng = np.random.default_rng([self.seed, 1])
        self.add_on_set_parameters_callback(self._on_params)

        g = self.get_parameter
        self.scan_in = g('scan_in').value
        self.scan_out = g('scan_out').value
        self.labels_out = g('labels_out').value
        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(LaserScan, self.scan_out, 10)
        self.label_pub = self.create_publisher(LaserScan, self.labels_out, 10)
        self.rate_pub = self.create_publisher(Float64, '/rain/rate', latched)
        self.state_pub = self.create_publisher(String, '/weather/state', latched)
        self.sub = self.create_subscription(LaserScan, self.scan_in, self.on_scan, 10)
        self.create_subscription(Odometry, g('ground_truth_topic').value, self._on_gt, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._announce()
        self.get_logger().info(f'weather: {self._describe()}  {self.scan_in} -> {self.scan_out}')

    # ---------------------------------------------------------------------
    def _values(self):
        return {n: self.get_parameter(n).value for n in {**NODE_DEFAULTS, **PARAM_DEFAULTS}}

    def _describe(self):
        c = self.cfg
        m = c.model
        s = (f"[{c.label}] rain='{c.rain}' fog='{c.fog}' dust='{c.dust}' model={m.name} "
             f"R={c.rain_rate:.1f} mm/h alpha_rain={m.alpha(c.rain_rate):.3e}/m seed={self.seed}")
        if m.name != 'legacy':
            s += f" P_min={m.p_min():.3e} sensor_sigma_extra={m.sensor_sigma_extra * 1000:.2f}mm"
        if m.name == 'alavi2025':
            s += (f" rain_type={m.rain_type} beta={m.beta(c.rain_rate):.3e}/(m sr)"
                  f" dr={m.bin_width:.3f}m echo={m.echo_mode} backscatter={m.backscatter_mode}")
        if m.name == 'hybrid':
            s += (f" extinction={m.extinction_source} N_tot={m.drop_density(c.rain_rate):.0f}/m^3"
                  f" divergence={m.beam_divergence_rad:.1e}rad dR={m.range_accuracy_m:.3f}m"
                  f" overlap={m.use_overlap}")
        w = c.media
        if w.fog_on:
            s += (f" | fog V={w.fog_visibility_m:g}m alpha={w.fog_alpha:.3e}/m"
                  f" beta={w.fog_beta:.3e}/(m sr) ({w.p.fog_beta_model})")
        if w.dust_on:
            s += (f" | dust V={w.dust_visibility_m:g}m alpha={w.dust_alpha:.3e}/m"
                  f" beta={w.dust_beta:.3e}/(m sr) set='{c.cloud_set}' ({len(w.clouds)} clouds,"
                  f" {len(w.world_clouds)} world-frame)")
        return s

    def _on_params(self, params):
        # The parameter store is updated only after this returns successfully,
        # so validate the merged values now and apply them on the next scan.
        values = self._values()
        values.update({p.name: p.value for p in params})
        try:
            self._pending = Config(values)
        except (ValueError, TypeError, OSError) as e:
            return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    def _announce(self):
        if self.cfg.used_alias:
            self.get_logger().warn("'condition' is a deprecated alias for 'rain'")
        self.rate_pub.publish(Float64(data=float(self.cfg.rain_rate)))
        self.state_pub.publish(String(data=json.dumps(self.cfg.state())))

    # --- sensor pose for world-frame dust clouds ------------------------------
    def _on_gt(self, msg: Odometry):
        p = msg.pose.pose
        q = p.orientation
        self._gt_pose = (p.position.x, p.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    @staticmethod
    def _xy_yaw(tf):
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _world_pose(self, scan):
        """(x, y, yaw) of the scan frame in the world, or None (world clouds skipped)."""
        frame = scan.header.frame_id
        v = self.cfg.values
        try:
            if v['world_pose_source'] == 'tf':
                last = None
                for stamp in (Time.from_msg(scan.header.stamp), Time()):  # scan stamp, then latest
                    try:
                        return self._xy_yaw(self.tf_buffer.lookup_transform(
                            v['world_frame'], frame, stamp))
                    except tf2_ros.TransformException as e:
                        last = e
                raise last
            if self._gt_pose is None:
                raise LookupError(f"no {v['ground_truth_topic']} message yet")
            base = v['base_frame']
            if frame not in self._sensor_offsets:               # static: look it up once
                self._sensor_offsets[frame] = (0.0, 0.0, 0.0) if frame == base else \
                    self._xy_yaw(self.tf_buffer.lookup_transform(base, frame, Time()))
            return compose_pose(self._gt_pose, self._sensor_offsets[frame])
        except (tf2_ros.TransformException, LookupError) as e:
            self.get_logger().warn(f'world-frame dust clouds skipped for this scan: {e}',
                                   throttle_duration_sec=2.0)
            return None

    # ---------------------------------------------------------------------
    def on_scan(self, msg: LaserScan):
        if self._pending is not None:
            self.cfg, self._pending = self._pending, None
            self.seed = int(self.cfg.values['seed'])
            self._announce()
            self.get_logger().info(f'weather reconfigured: {self._describe()}')

        model = self.cfg.model
        # gz gpu_lidar publishes all-zero intensities (no laser_retro in the
        # worlds), so the physical models use default_reflectivity for every
        # surface. Flag it if that ever changes.
        if not self._warned_intensity and model.name != 'legacy' and \
                len(msg.intensities) and max(msg.intensities) > 0.0:
            self._warned_intensity = True
            self.get_logger().warn(
                'scan_in carries non-zero intensities; they are ignored - every surface '
                'uses default_reflectivity')

        r_min = msg.range_min if msg.range_min > 0 else 0.1
        r_max = msg.range_max if msg.range_max > 0 else 12.0
        clean = np.asarray(msg.ranges, dtype=np.float64)
        media = None
        if self.cfg.media.active:
            pose = self._world_pose(msg) if self.cfg.media.has_world_clouds else None
            media = self.cfg.media.scan(
                clean, clean_hits(clean, r_min, r_max), r_min, r_max, msg.angle_min,
                msg.angle_increment, model.alpha(self.cfg.rain_rate), self.weather_rng, pose)
        out_r, out_i, labels = model.apply(
            clean, r_min, r_max, msg.angle_min, msg.angle_increment, self.cfg.rain_rate,
            self.rng, media=media)

        self.pub.publish(self._scan_like(msg, r_min, r_max, out_r, out_i))
        self.label_pub.publish(self._scan_like(msg, r_min, r_max, clean, labels))

    @staticmethod
    def _scan_like(msg, r_min, r_max, ranges, intensities):
        o = LaserScan()
        o.header = msg.header
        o.angle_min = msg.angle_min
        o.angle_max = msg.angle_max
        o.angle_increment = msg.angle_increment
        o.time_increment = msg.time_increment
        o.scan_time = msg.scan_time
        o.range_min = float(r_min)
        o.range_max = float(r_max)
        o.ranges = np.asarray(ranges).astype(np.float32).tolist()
        o.intensities = np.asarray(intensities).astype(np.float32).tolist()
        return o


def main(args=None):
    rclpy.init(args=args)
    try:
        node = WeatherModelNode()
    except (ValueError, TypeError, OSError) as e:
        rclpy.logging.get_logger('weather_model_node').fatal(f'bad parameters: {e}')
        rclpy.try_shutdown()
        raise SystemExit(1)
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
