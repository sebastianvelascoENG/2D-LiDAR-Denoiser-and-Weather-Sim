#!/usr/bin/env python3
"""clustering_denoiser_node - 2D LiDAR denoising by keyframe extraction +
angular clustering with low-intensity cluster removal.

Method after Hu, Zhang, Shao, Lin, Hua & Qin (2022), "Clustering denoising of
2D LiDAR scanning in indoor environment based on keyframe extraction", Sensors
23(1):18.

Subscribes : /scan           (sensor_msgs/LaserScan) - weather-noised scan
Publishes  : /scan_denoised   (sensor_msgs/LaserScan) - clusters below the
             intensity / size thresholds removed (set to +inf)
             /denoiser/keyframe_ratio (std_msgs/Float32) - running keyframe fraction

Pipeline per scan:
  1. distance filter                 keep r in [min_range, max_range]
  2. keyframe test (sliding window)   angular range-histogram, Pearson vs recent
                                      keyframes; drives an efficiency metric only
                                      (every frame is still denoised + published)
  3. region segmentation             t annular regions, r_i = sqrt(i) * r_max/sqrt(t)
  4. angular clustering per region    split where angular gap > k * angle_increment
  5. merge clusters across regions    adaptive Euclidean endpoint threshold
  6. removal                          drop a cluster if mean intensity < I_thr
                                      (primary, per thesis) or point count < m_thr
"""
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class ClusteringDenoiser(Node):

    def __init__(self):
        super().__init__('clustering_denoiser_node')

        self.declare_parameter('scan_in', '/scan')
        self.declare_parameter('scan_out', '/scan_denoised')
        self.declare_parameter('min_range', 0.10)
        self.declare_parameter('max_range', 12.0)

        # keyframe extraction
        self.declare_parameter('use_keyframe', True)
        self.declare_parameter('hist_bins', 36)
        self.declare_parameter('keyframe_window', 6)
        self.declare_parameter('keyframe_corr_threshold', 0.95)

        # clustering
        self.declare_parameter('num_regions', 4)
        self.declare_parameter('angle_threshold_mult', 3.0)   # theta_thr = k * angle_increment
        self.declare_parameter('merge_regions', True)

        # removal criteria
        self.declare_parameter('intensity_threshold', 15.0)   # primary (thesis)
        self.declare_parameter('min_cluster_points', 3)       # secondary guard
        self.declare_parameter('require_both', False)         # False: drop on either; True: need both

        g = self.get_parameter
        self.scan_in = g('scan_in').value
        self.scan_out = g('scan_out').value
        self.min_range = float(g('min_range').value)
        self.max_range = float(g('max_range').value)
        self.use_keyframe = bool(g('use_keyframe').value)
        self.hist_bins = int(g('hist_bins').value)
        self.kf_window = int(g('keyframe_window').value)
        self.kf_corr_thr = float(g('keyframe_corr_threshold').value)
        self.num_regions = max(1, int(g('num_regions').value))
        self.k_ang = float(g('angle_threshold_mult').value)
        self.merge_regions = bool(g('merge_regions').value)
        self.i_thr = float(g('intensity_threshold').value)
        self.m_thr = int(g('min_cluster_points').value)
        self.require_both = bool(g('require_both').value)

        self.kf_refs = deque(maxlen=self.kf_window)
        self.n_scans = 0
        self.n_keyframes = 0

        self.pub = self.create_publisher(LaserScan, self.scan_out, 10)
        self.kf_pub = self.create_publisher(Float32, '/denoiser/keyframe_ratio', 10)
        self.sub = self.create_subscription(LaserScan, self.scan_in, self.on_scan, 10)
        self.get_logger().info(
            f"clustering_denoiser: {self.scan_in} -> {self.scan_out}  "
            f"I_thr={self.i_thr} m_thr={self.m_thr} regions={self.num_regions}")

    # ------------------------------------------------------------------
    def _histogram(self, ranges, valid):
        """Angular mean-range descriptor over hist_bins, normalised."""
        n = len(ranges)
        h = np.zeros(self.hist_bins)
        if n == 0:
            return h
        bin_of = (np.arange(n) * self.hist_bins // n)
        vbins = bin_of[valid]
        h = np.bincount(vbins, weights=np.asarray(ranges)[valid],
                         minlength=self.hist_bins).astype(np.float64)[:self.hist_bins]
        c = np.bincount(vbins, minlength=self.hist_bins)[:self.hist_bins]
        nz = c > 0
        h[nz] /= c[nz]
        return h

    def _is_keyframe(self, hist):
        if not self.kf_refs:
            return True
        best = -1.0
        for ref in self.kf_refs:
            if np.std(hist) < 1e-9 or np.std(ref) < 1e-9:
                continue
            cc = float(np.corrcoef(hist, ref)[0, 1])
            best = max(best, cc)
        return best < self.kf_corr_thr

    # ------------------------------------------------------------------
    def _cluster_region(self, order_idx, angle_inc):
        """order_idx: beam indices in this region, sorted by angle. Returns list of lists."""
        if len(order_idx) == 0:
            return []
        thr = self.k_ang * angle_inc
        clusters = [[order_idx[0]]]
        for prev, cur in zip(order_idx[:-1], order_idx[1:]):
            if (cur - prev) * angle_inc > thr:
                clusters.append([cur])
            else:
                clusters[-1].append(cur)
        return clusters

    def _merge(self, clusters, r, angle_inc):
        """Merge consecutive clusters whose facing endpoints are Euclidean-close."""
        if not self.merge_regions or len(clusters) < 2:
            return clusters
        thr_ang = self.k_ang * angle_inc
        merged = [clusters[0]]
        for cl in clusters[1:]:
            a, b = merged[-1][-1], cl[0]
            dang = (b - a) * angle_inc
            ra, rb = r[a], r[b]
            d = math.sqrt(max(ra * ra + rb * rb - 2 * ra * rb * math.cos(dang), 0.0))
            dthr = math.sqrt(ra * ra + rb * rb - 2 * ra * rb * math.cos(thr_ang)) if ra > 0 and rb > 0 else thr_ang
            if dang <= thr_ang and d <= max(dthr, 0.15):
                merged[-1] = merged[-1] + cl
            else:
                merged.append(cl)
        return merged

    # ------------------------------------------------------------------
    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        r = np.asarray(msg.ranges, dtype=np.float64)
        inten = np.asarray(msg.intensities, dtype=np.float64) if len(msg.intensities) == n \
            else np.zeros(n)
        angle_inc = msg.angle_increment or (2 * math.pi / n)

        # 1. distance filter
        valid = np.isfinite(r) & (r >= self.min_range) & (r <= self.max_range)

        # 2. keyframe bookkeeping (efficiency metric; does not gate output)
        self.n_scans += 1
        if self.use_keyframe:
            hist = self._histogram(r, valid)
            if self._is_keyframe(hist):
                self.n_keyframes += 1
                self.kf_refs.append(hist)
            if self.n_scans % 20 == 0:
                self.kf_pub.publish(Float32(data=float(self.n_keyframes / self.n_scans)))

        keep = np.zeros(n, dtype=bool)
        vidx = np.flatnonzero(valid)
        if vidx.size:
            rv = r[vidx]
            rmax = float(rv.max())
            r1 = rmax / math.sqrt(self.num_regions) if rmax > 0 else 1.0
            # region id by range: largest i with r > sqrt(i)*r1
            reg = np.zeros(vidx.size, dtype=int)
            for i in range(1, self.num_regions):
                reg[rv > math.sqrt(i) * r1] = i

            all_clusters = []
            for region_id in range(self.num_regions):
                sel = vidx[reg == region_id]
                sel = sel[np.argsort(sel)]  # index order == angle order
                clusters = self._cluster_region(list(sel), angle_inc)
                all_clusters.extend(clusters)
            all_clusters = self._merge(all_clusters, r, angle_inc)

            # 6. removal
            for cl in all_clusters:
                cl_arr = np.asarray(cl, dtype=int)
                mean_i = float(inten[cl_arr].mean()) if cl_arr.size else 0.0
                low_intensity = mean_i < self.i_thr
                too_small = cl_arr.size < self.m_thr
                drop = (low_intensity and too_small) if self.require_both else (low_intensity or too_small)
                if not drop:
                    keep[cl_arr] = True

        out_r = np.where(keep, r, math.inf)
        out_i = np.where(keep, inten, 0.0)

        o = LaserScan()
        o.header = msg.header
        o.angle_min = msg.angle_min
        o.angle_max = msg.angle_max
        o.angle_increment = msg.angle_increment
        o.time_increment = msg.time_increment
        o.scan_time = msg.scan_time
        o.range_min = msg.range_min
        o.range_max = msg.range_max
        o.ranges = out_r.astype(np.float32).tolist()
        o.intensities = out_i.astype(np.float32).tolist()
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = ClusteringDenoiser()
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
