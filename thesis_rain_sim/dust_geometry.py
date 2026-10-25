"""dust_geometry - 2D ray / dust-cloud intersections, pure numpy (no ROS).

Rays start at origin (ox, oy) with unit direction (dx, dy). Every function
returns the entry and exit distances [r_in, r_out] along each ray, with
r_in clipped to 0 when the sensor is inside the cloud, and NaN for misses
(including grazing rays with a zero-length chord and clouds entirely behind).
"""
import math

import numpy as np


def beam_angles(angle_min, angle_increment, n):
    return angle_min + angle_increment * np.arange(n)


def ray_circle_chords(ox, oy, dx, dy, cx, cy, radius):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0, t1 = -b - sq, -b + sq
    hit = (disc > 0.0) & (t1 > 0.0)
    r_in = np.where(hit, np.maximum(t0, 0.0), np.nan)
    r_out = np.where(hit, t1, np.nan)
    return r_in, r_out


def ray_box_chords(ox, oy, dx, dy, cx, cy, size_x, size_y):
    """Axis-aligned box centred on (cx, cy) with full side lengths size_x, size_y."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    t_lo = np.full(dx.shape, -np.inf)
    t_hi = np.full(dx.shape, np.inf)
    for o, d, c, half in ((ox, dx, cx, size_x / 2.0), (oy, dy, cy, size_y / 2.0)):
        lo, hi = c - half, c + half
        parallel = np.abs(d) < 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (lo - o) / d
            tb = (hi - o) / d
        near = np.where(parallel, -np.inf, np.minimum(ta, tb))
        far = np.where(parallel, np.inf, np.maximum(ta, tb))
        if not lo < o < hi:  # parallel rays outside this slab never enter the box
            far = np.where(parallel, -np.inf, far)
        t_lo = np.maximum(t_lo, near)
        t_hi = np.minimum(t_hi, far)
    hit = (t_hi > t_lo) & (t_hi > 0.0)
    r_in = np.where(hit, np.maximum(t_lo, 0.0), np.nan)
    r_out = np.where(hit, t_hi, np.nan)
    return r_in, r_out


def cloud_chords(clouds, angles, origin=(0.0, 0.0), yaw=0.0):
    """Chords for a list of cloud dicts, as an (n_beams, n_clouds, 2) array.

    angles are beam angles in the sensor frame. origin/yaw give the sensor pose
    in the frame the clouds are written in (identity for frame: sensor clouds).
    Box clouds stay axis-aligned in their own frame.
    """
    angles = np.asarray(angles, dtype=float)
    out = np.full((angles.size, len(clouds), 2), np.nan)
    dx, dy = np.cos(angles + yaw), np.sin(angles + yaw)
    ox, oy = origin
    for j, cl in enumerate(clouds):
        shape = cl['shape']
        if shape == 'circle':
            r_in, r_out = ray_circle_chords(ox, oy, dx, dy, cl['x'], cl['y'], cl['radius'])
        elif shape == 'box':
            r_in, r_out = ray_box_chords(ox, oy, dx, dy, cl['x'], cl['y'],
                                         cl['size_x'], cl['size_y'])
        else:
            raise ValueError(f'unknown cloud shape {shape!r}')
        out[:, j, 0] = r_in
        out[:, j, 1] = r_out
    return out


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compose_pose(base, offset):
    """(x, y, yaw) of `offset` (given in the `base` frame) in base's parent frame."""
    bx, by, byaw = base
    ox, oy, oyaw = offset
    c, s = math.cos(byaw), math.sin(byaw)
    return bx + c * ox - s * oy, by + s * ox + c * oy, byaw + oyaw
