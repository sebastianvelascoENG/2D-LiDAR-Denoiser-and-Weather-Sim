"""fog_dust - fog and dust media for the weather model, pure numpy (no ROS).

Runs inside the rain models' sensor:

* physics: the coefficients below, Hahner et al. (2021) Eq. 18 soft
  return in the range domain, dust clouds as 2D circles / boxes, Hahner's
  range jitter;
* the SENSOR is the one the physical rain models use (goodin2019 / alavi2025 /
  hybrid): P_min = z_max_reflectivity / (pi z_max^2) (C1: 12 m at 70 %),
  default_reflectivity, pulse_width_s and the overlap ramp overlap_r1_m /
  overlap_r2_m. Every weather effect is seen by ONE sensor, so rain, fog and
  dust stay directly comparable. test_fog_dust.py also checks reference
  numbers for a second sensor (6 m at 10 %, 20 ns, R2 = 0.5 m) by passing
  those constants explicitly;
* one decision per beam: the rain model owns the hard target and the rain
  echoes and asks this module for (a) the extra optical depth of fog and dust
  along the beam and (b) the fog / dust soft echo. The strongest echo above
  P_min is published (ScanMedia.merge). Rain extinction also attenuates the
  fog / dust soft echo.

Coefficients (alpha extinction [1/m], beta backscatter [1/(m sr)], V visibility [m])
    fog   alpha = 3.912 / V   (Kim, McArthur & Korevaar 2001; 'mor': 3.0 / V, Hahner 2021)
          beta  = 0.046 / V   (Rasshofer et al. 2011; default)
                  alpha / (1.44 * 4 pi)    ('alavi', Alavi et al. 2025 Eq. 10 per sr)
                  alpha / 1.44             ('alavi_raw', as printed)
    dust  alpha = 5.26 V^-1.016           (Alavi et al. 2025, 905 nm, ISO 12103-1 A4)
          beta  = 5.38 V^-1.016 / (4 pi)  ('alavi'; 'alavi_raw' without the 4 pi)
Alavi's beta is built from Q_back (4 pi convention), so it is divided by 4 pi
to be per steradian, the same correction rain_models.beta_rain_per_sr applies
to Alavi's rain beta.

Soft return (Hahner et al. 2021, Eq. 18, range domain, s = c t / 2):
    P_soft(R) = int_0^{c tau_H} sin^2(pi s / (c tau_H)) beta(R-s) exp(-2 tau(R-s))
                xi(R-s) / (R-s)^2 [0 < R-s < R0] ds
with tau_H = pulse_width_s (the pulse FWHM, Hahner's half-power width) and xi
the overlap ramp. Peak over R on a grid range_min .. min(R0, range_max);
reported at the rising edge R_peak - c tau_H / 2, then jittered by 2^U(-1, 1)
(Hahner). Fog-only beams read an exact lookup table; beams through dust are
evaluated all at once by SoftReturnModel.direct_batch from running sums over
s (same values as the per-beam SoftReturnModel.direct, which tests keep as the
reference).
"""
import math
from dataclasses import dataclass, fields

import numpy as np
import yaml

from .dust_geometry import beam_angles, cloud_chords
from .rain_models import LABEL_DUST_FALSE, LABEL_FOG_FALSE, overlap

C_LIGHT = 299_792_458.0  # m/s
FOUR_PI = 4.0 * math.pi
ALAVI_BETA_MODELS = ('alavi', 'alavi_raw')
CLOUD_FIELDS = {'circle': ('x', 'y', 'radius'), 'box': ('x', 'y', 'size_x', 'size_y')}

# Stand-in for "the rest of the beam" when a dust chord is unbounded.
_FAR = 1.0e6


# =============================================================================
# Coefficients
# =============================================================================
def _check_visibility(visibility_m):
    if not visibility_m > 0.0:
        raise ValueError(f'visibility must be > 0 m, got {visibility_m}')


def alpha_fog(visibility_m, convention='kim'):
    """Fog extinction [1/m]: kim 3.912/V (Kim et al. 2001), mor 3.0/V (Hahner et al. 2021)."""
    _check_visibility(visibility_m)
    if convention == 'kim':
        return 3.912 / visibility_m
    if convention == 'mor':
        return 3.0 / visibility_m
    raise ValueError(f"visibility_convention must be 'kim' or 'mor', got {convention!r}")


def beta_fog(visibility_m, model='rasshofer', convention='kim'):
    """Fog backscatter [1/(m sr)]: rasshofer 0.046/V | alavi alpha/(1.44*4pi) | alavi_raw alpha/1.44."""
    _check_visibility(visibility_m)
    if model == 'rasshofer':
        return 0.046 / visibility_m
    if model == 'alavi':
        return alpha_fog(visibility_m, convention) / (1.44 * FOUR_PI)
    if model == 'alavi_raw':
        return alpha_fog(visibility_m, convention) / 1.44
    raise ValueError(
        f"fog_beta_model must be 'rasshofer', 'alavi' or 'alavi_raw', got {model!r}")


def alpha_dust(visibility_m):
    """Dust extinction [1/m], Alavi et al. (2025), 905 nm, ISO 12103-1 A4 dust."""
    _check_visibility(visibility_m)
    return 5.26 * visibility_m ** -1.016


def beta_dust(visibility_m, model='alavi'):
    """Dust backscatter [1/(m sr)], Alavi et al. (2025) Eq. 18: alavi /(4 pi) | alavi_raw as printed."""
    _check_visibility(visibility_m)
    raw = 5.38 * visibility_m ** -1.016
    if model == 'alavi':
        return raw / FOUR_PI
    if model == 'alavi_raw':
        return raw
    raise ValueError(f"dust_beta_model must be 'alavi' or 'alavi_raw', got {model!r}")


def simpson_weights(n):
    """Composite Simpson weights (without the h factor) for n odd points."""
    if n < 3 or n % 2 == 0:
        raise ValueError('Simpson needs an odd number of points >= 3')
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / 3.0


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class FogDustParams:
    """Fog / dust model parameters. The last block is the shared sensor (same
    names and defaults as the rain models' alavi2025 / hybrid fields)."""
    visibility_convention: str = 'kim'          # kim | mor
    fog_beta_model: str = 'rasshofer'           # rasshofer | alavi | alavi_raw
    dust_beta_model: str = 'alavi'              # alavi | alavi_raw
    soft_range_step_m: float = 0.02
    soft_integration_points: int = 101          # odd (composite Simpson)
    soft_range_convention: str = 'rising_edge'  # rising_edge | peak
    soft_gain: float = 1.0                      # sensitivity studies only
    jitter_exponent_max: float = 1.0            # Hahner: range * 2^U(-1, 1); 0 disables
    # shared sensor
    pulse_width_s: float = 5.0e-9               # FWHM = Hahner's tau_H
    overlap_r1_m: float = 0.05
    overlap_r2_m: float = 1.0

    @classmethod
    def from_dict(cls, d):
        """Fields this class knows from a flat params dict, cast to the field types."""
        return cls(**{f.name: type(f.default)(d[f.name]) for f in fields(cls) if f.name in d})

    def validate(self):
        if self.visibility_convention not in ('kim', 'mor'):
            raise ValueError(f'bad visibility_convention {self.visibility_convention!r}')
        if self.fog_beta_model not in ('rasshofer',) + ALAVI_BETA_MODELS:
            raise ValueError(f'bad fog_beta_model {self.fog_beta_model!r}')
        if self.dust_beta_model not in ALAVI_BETA_MODELS:
            raise ValueError(f'bad dust_beta_model {self.dust_beta_model!r}')
        if self.soft_range_convention not in ('rising_edge', 'peak'):
            raise ValueError(f'bad soft_range_convention {self.soft_range_convention!r}')
        n = self.soft_integration_points
        if n < 3 or n % 2 == 0:
            raise ValueError(f'soft_integration_points must be odd and >= 3, got {n}')
        if self.pulse_width_s <= 0.0:
            raise ValueError('pulse_width_s must be > 0')
        if self.soft_range_step_m <= 0.0:
            raise ValueError('soft_range_step_m must be > 0')
        if self.overlap_r2_m < self.overlap_r1_m:
            raise ValueError('overlap_r2_m must be >= overlap_r1_m')
        if self.jitter_exponent_max < 0.0:
            raise ValueError('jitter_exponent_max must be >= 0')
        return self


# =============================================================================
# Dust clouds
# =============================================================================
def check_cloud(cloud, where):
    if not isinstance(cloud, dict):
        raise ValueError(f'{where}: each cloud must be a mapping, got {cloud!r}')
    shape = cloud.get('shape')
    if shape not in CLOUD_FIELDS:
        raise ValueError(f"{where}: cloud shape must be 'circle' or 'box', got {shape!r}")
    frame = cloud.get('frame', 'sensor')
    if frame not in ('sensor', 'world'):
        raise ValueError(f"{where}: cloud frame must be 'sensor' or 'world', got {frame!r}")
    out = {'shape': shape, 'frame': frame}
    for key in CLOUD_FIELDS[shape]:
        if key not in cloud:
            raise ValueError(f'{where}: {shape} cloud needs {key!r}')
        out[key] = float(cloud[key])
    for key in ('radius', 'size_x', 'size_y'):
        if key in out and out[key] <= 0.0:
            raise ValueError(f'{where}: {key} must be > 0')
    return out


def load_cloud_sets(path):
    """{set name: [validated cloud, ...]} from a dust_clouds.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    sets = data.get('cloud_sets')
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path}: expected a top-level 'cloud_sets:' mapping")
    out = {}
    for name, clouds in sets.items():
        if not clouds:
            raise ValueError(f'{path}: cloud set {name!r} has no clouds')
        out[str(name)] = [check_cloud(c, f'{path}: {name}[{i}]') for i, c in enumerate(clouds)]
    return out


def _merge_intervals(iv):
    iv = sorted((float(a), float(b)) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class DustProfile:
    """Piecewise-constant dust alpha / beta along one beam.

    chords: (n_clouds, 2) array of [r_in, r_out], NaN for misses. Overlapping
    clouds are one medium (merged, no double counting).
    """

    def __init__(self, chords, alpha, beta):
        iv = chords[~np.isnan(chords[:, 0])]
        segments = _merge_intervals(np.clip(iv, 0.0, _FAR))
        self.alpha, self.beta, self.segments = alpha, beta, segments
        self.empty = not segments
        bps = sorted({0.0, *[s[0] for s in segments], *[s[1] for s in segments]})
        self.bps = np.asarray(bps)
        mids = 0.5 * (self.bps[:-1] + self.bps[1:])
        self.seg_alpha = np.zeros(mids.size)
        self.seg_beta = np.zeros(mids.size)
        for a, b in segments:
            inside = (mids > a) & (mids < b)
            self.seg_alpha[inside] += alpha
            self.seg_beta[inside] += beta
        self.cum_tau = np.concatenate([[0.0], np.cumsum(self.seg_alpha * np.diff(self.bps))])
        self.first_entry = min((s[0] for s in segments), default=math.inf)

    def tau(self, x):
        """Dust optical depth from 0 to x (piecewise linear, constant past the last cloud)."""
        return np.interp(x, self.bps, self.cum_tau)

    def tau_beta(self, x):
        """tau(x) and beta(x) together for x >= 0, with a single segment search."""
        idx = np.searchsorted(self.bps, x, side='right') - 1
        seg_a = np.append(self.seg_alpha, 0.0)  # trailing segment: past the last cloud
        seg_b = np.append(self.seg_beta, 0.0)
        tau = self.cum_tau[idx] + seg_a[idx] * (x - self.bps[idx])
        return tau, seg_b[idx]

    def transmission_beta(self, x, exp_x):
        """exp(-2 tau(x)) and beta(x), given exp_x = exp(-2 alpha x) on the same points.

        Same values as tau_beta, without an exp or a search per point: inside
        segment k, tau = alpha (x - a_k + cum_k), so exp(-2 tau) is exp_x times
        a constant; between and after segments it is a constant.
        """
        a2 = 2.0 * self.alpha
        trans = np.ones_like(x)
        inside = np.zeros(x.shape, bool)
        cum = 0.0
        for a, b in self.segments:
            seg = (x > a) & (x < b)
            trans = np.where(x >= b, math.exp(-a2 * (cum + b - a)),
                             np.where(seg, exp_x * math.exp(a2 * (a - cum)), trans))
            inside |= seg
            cum += b - a
        return trans, inside * self.beta


class DustBeams:
    """Dust along many beams at once: the vectorised form of one DustProfile per beam.

    beams: (B,) beam indices; chords: (B, n_clouds, 2) [r_in, r_out], NaN for
    misses. Each row keeps its merged segments (same merge as DustProfile)
    sorted in a, b: (B, M), padded with zero-length segments at _FAR.
    """

    def __init__(self, beams, chords, alpha, beta):
        a = np.clip(chords[:, :, 0], 0.0, _FAR)
        b = np.clip(chords[:, :, 1], 0.0, _FAR)
        pad = np.isnan(a) | ~(b > a)
        a[pad] = _FAR
        b[pad] = _FAR
        order = np.argsort(a, axis=1, kind='stable')
        a = np.take_along_axis(a, order, axis=1)
        b = np.take_along_axis(b, order, axis=1)
        rows = np.arange(a.shape[0])
        last = np.zeros(a.shape[0], int)  # last kept segment per row
        for k in range(1, a.shape[1]):
            ov = a[:, k] <= b[rows, last]
            b[rows[ov], last[ov]] = np.maximum(b[rows[ov], last[ov]], b[ov, k])
            a[ov, k] = _FAR
            b[ov, k] = _FAR
            last = np.where(ov, last, k)
        order = np.argsort(a, axis=1, kind='stable')
        self.a = np.take_along_axis(a, order, axis=1)
        self.b = np.take_along_axis(b, order, axis=1)
        self.beams = np.asarray(beams)
        self.alpha, self.beta = alpha, beta
        self.lengths = self.b - self.a
        self.first_entry = np.where(self.lengths[:, 0] > 0.0, self.a[:, 0], math.inf)

    def take(self, mask):
        out = object.__new__(DustBeams)
        out.a, out.b, out.beams = self.a[mask], self.b[mask], self.beams[mask]
        out.alpha, out.beta = self.alpha, self.beta
        out.lengths, out.first_entry = self.lengths[mask], self.first_entry[mask]
        return out

    def tau(self, rows, x):
        """Dust optical depth from 0 to x along the beams in `rows` (indices into this set)."""
        x = np.asarray(x, dtype=float)[..., None]
        return self.alpha * np.clip(x - self.a[rows], 0.0, self.lengths[rows]).sum(axis=-1)

    def profile(self, i):
        """Row i as a DustProfile (pads are zero-length, which DustProfile drops)."""
        return DustProfile(np.stack([self.a[i], self.b[i]], axis=1), self.alpha, self.beta)


# =============================================================================
# Soft return (Hahner et al. 2021, Eq. 18)
# =============================================================================
def _range_argmax_table(P):
    """Sparse table: level k holds the first argmax of P over [i, i + 2^k)."""
    levels = [np.arange(P.size)]
    span = 1
    while 2 * span <= P.size:
        prev = levels[-1]
        left, right = prev[:P.size - 2 * span + 1], prev[span:span + P.size - 2 * span + 1]
        levels.append(np.where(P[left] >= P[right], left, right))
        span *= 2
    return levels


def _range_argmax(P, levels, lo, hi):
    """First argmax of P over [lo, hi] (inclusive, lo <= hi), vectorised over lo / hi."""
    k = np.floor(np.log2(hi - lo + 1)).astype(int)
    i1 = np.empty_like(lo)
    i2 = np.empty_like(lo)
    for lvl in np.unique(k):
        m = k == lvl
        i1[m] = levels[lvl][lo[m]]
        i2[m] = levels[lvl][hi[m] - (1 << lvl) + 1]
    return np.where((P[i1] > P[i2]) | ((P[i1] == P[i2]) & (i1 <= i2)), i1, i2)


@dataclass(frozen=True)
class UniformMedium:
    """Extinction / backscatter that fill all space (fog, plus rain extinction)."""
    alpha: float
    beta: float


class SoftReturnModel:
    """Evaluates P_soft(R) on the R grid and finds its peak.

    The grid is R_k = range_min + k * step. Because s >= 0 and R <= R0 on the
    grid, R - s <= R0 always, so the [R - s < R0] factor never removes a point
    with non-zero weight (the s = 0 end has sin^2 = 0). P_soft(R) for a uniform
    medium alone is therefore independent of R0, and its lookup table over R0
    is exactly the running maximum of a single P_soft(R) curve.
    """

    def __init__(self, params: FogDustParams):
        self.p = params
        self.L = C_LIGHT * params.pulse_width_s  # c * tau_H [m]
        n = params.soft_integration_points
        self.s = np.linspace(0.0, self.L, n)
        h = self.L / (n - 1)
        self.w = np.sin(np.pi * self.s / self.L) ** 2 * simpson_weights(n) * h
        self._lut_key = None
        self._prefix_key = None

    def grid(self, range_min, r_end):
        k = self._last_index(range_min, r_end)
        if k < 0:
            return np.empty(0)
        return range_min + self.p.soft_range_step_m * np.arange(k + 1)

    def _last_index(self, range_min, r_end):
        return int(np.floor((r_end - range_min) / self.p.soft_range_step_m + 1e-9))

    def _mesh(self, R):
        X = R[:, None] - self.s[None, :]
        valid = X > 0.0
        return np.where(valid, X, 1.0), valid

    def _uniform_cache(self, uni, range_min, range_max):
        """Beam-independent pieces, rebuilt only when the medium or scan limits change.

        K0 = w(s) exp(-2 alpha x) xi(x) / x^2 on the (R, s) mesh, Pf = beta * sum_s K0,
        plus the running max / argmax of soft_gain * Pf (the lookup table).
        """
        key = (uni, range_min, range_max)
        if key != self._lut_key:
            R = self.grid(range_min, range_max)
            xs, valid = self._mesh(R)
            xi = overlap(xs, self.p.overlap_r1_m, self.p.overlap_r2_m)
            K0 = np.where(valid, self.w[None, :] * np.exp(-2.0 * uni.alpha * xs) * xi / xs ** 2,
                          0.0)
            Pf = uni.beta * K0.sum(axis=1)
            P = self.p.soft_gain * Pf
            if P.size:
                run_max = np.maximum.accumulate(P)
                prev = np.concatenate([[-np.inf], run_max[:-1]])
                run_arg = np.maximum.accumulate(np.where(P > prev, np.arange(P.size), 0))
            else:
                run_max, run_arg = P, P.astype(int)
            self._lut = (R, K0, Pf, run_max, run_arg)
            self._lut_key = key
            self._xs = xs
            self._exp_key = None
            self._P = P
            self._sparse = _range_argmax_table(P)
        return self._lut

    def _exp_mesh(self, alpha):
        """exp(-2 alpha x) on the (R, s) mesh, for the dust alpha (cached)."""
        if self._exp_key != alpha:
            self._exp = np.exp(-2.0 * alpha * self._xs)
            self._exp_key = alpha
        return self._exp

    def _prefix_sums(self, uni, range_min, range_max, alpha):
        """Running sums over s of K0 and K0 exp(-2 alpha x) on every R row (cached).

        Column 0 is zero, so the sum over s indices [p, q) is S[:, q] - S[:, p].
        """
        _, K0, _, _, _ = self._uniform_cache(uni, range_min, range_max)
        key = (self._lut_key, alpha)
        if key != self._prefix_key:
            zero = np.zeros((K0.shape[0], 1))
            self._S0 = np.concatenate([zero, np.cumsum(K0, axis=1)], axis=1)
            self._SE = np.concatenate(
                [zero, np.cumsum(K0 * self._exp_mesh(alpha), axis=1)], axis=1)
            self._prefix_key = key
        return self._S0, self._SE

    def uniform_lookup(self, uni, range_min, range_max, r_end):
        """(P_soft_max, R_soft) for beams with no dust whose grid ends at r_end (array).

        Exact, not interpolated: a beam's grid is a prefix of the full grid, so
        its peak is the running maximum at its last grid index.
        """
        R, _, _, run_max, run_arg = self._uniform_cache(uni, range_min, range_max)
        r_end = np.asarray(r_end, dtype=float)
        if R.size == 0:
            return np.zeros(r_end.shape), np.full(r_end.shape, np.nan)
        k = np.floor((r_end - range_min) / self.p.soft_range_step_m + 1e-9).astype(int)
        ok = k >= 0
        k = np.clip(k, 0, R.size - 1)
        return np.where(ok, run_max[k], 0.0), np.where(ok, R[run_arg[k]], np.nan)

    def direct(self, uni, dust, range_min, range_max, r_end, dust_beta_scale=1.0):
        """(P_soft_max, R_soft, fog_dominates) for one beam through dust.

        Rows whose window lies wholly before the first dust entry equal the
        uniform curve; rows wholly past the last exit equal it times
        exp(-2 tau_dust_total). Only the rows in between are integrated.
        """
        R_all, K0, Pf_all, _, _ = self._uniform_cache(uni, range_min, range_max)
        k_end = min(self._last_index(range_min, r_end), R_all.size - 1)
        if k_end < 0:
            return 0.0, math.nan, True
        R = R_all[:k_end + 1]
        Pf = Pf_all[:k_end + 1].copy()
        Pd = np.zeros_like(Pf)
        if not dust.empty:
            last_exit = dust.bps[-1]
            # rows with R <= first entry are untouched; rows with R - L >= last exit
            # see the whole cloud; the rows in between (a contiguous run) are integrated.
            k0 = int(np.searchsorted(R, dust.first_entry, side='right'))
            k1 = int(np.searchsorted(R - self.L, last_exit, side='left'))
            Pf[k1:] *= np.exp(-2.0 * dust.cum_tau[-1])
            if k1 > k0:
                xs = self._xs[k0:k1]
                if 2.0 * dust.alpha * max(R[-1], dust.segments[-1][0]) < 600.0:  # no over/underflow
                    trans, beta_d = dust.transmission_beta(xs, self._exp_mesh(dust.alpha)[k0:k1])
                else:
                    tau_d, beta_d = dust.tau_beta(xs)
                    trans = np.exp(-2.0 * tau_d)
                Kd = K0[k0:k1] * trans
                Pf[k0:k1] = uni.beta * Kd.sum(axis=1)
                if dust_beta_scale > 0.0:
                    Pd[k0:k1] = dust_beta_scale * (Kd * beta_d).sum(axis=1)
        P = self.p.soft_gain * (Pf + Pd)
        i = int(np.argmax(P))
        return float(P[i]), float(R[i]), bool(Pf[i] >= Pd[i])

    def direct_batch(self, uni, dust, range_min, range_max, r_end, dust_beta_scale):
        """direct() for every beam of a DustBeams at once; same values up to rounding.

        Along a beam everything depends on x = R - s alone, so on each R row the
        s-sum splits into contiguous s ranges: x <= a_1 (no dust yet, factor 1),
        a_k < x < b_k (inside segment k: exp(-2 alpha x) times a constant) and
        b_k <= x <= a_{k+1} (a constant). Each range is a difference of the cached
        running sums over s, so a beam costs O(rows x segments), not
        O(rows x integration points). Falls back to direct() per beam where
        exp(+-2 alpha x) could over/underflow (dust visibility below ~0.2 m).
        """
        r_end = np.asarray(r_end, dtype=float)
        scale = np.broadcast_to(np.asarray(dust_beta_scale, dtype=float), r_end.shape)
        nb = r_end.size
        p_out, r_out, fog_out = np.zeros(nb), np.full(nb, np.nan), np.ones(nb, bool)
        R_all = self._uniform_cache(uni, range_min, range_max)[0]
        if nb == 0 or R_all.size == 0:
            return p_out, r_out, fog_out
        k_end = np.minimum(np.floor((r_end - range_min) / self.p.soft_range_step_m + 1e-9),
                           R_all.size - 1).astype(int)
        ok = k_end >= 0
        if not ok.any():
            return p_out, r_out, fog_out
        real = dust.lengths > 0.0
        M = max(1, int(real.sum(axis=1).max()))            # pads are sorted to the end
        a, b, lengths = dust.a[:, :M], dust.b[:, :M], dust.lengths[:, :M]
        a_real = np.where(lengths > 0.0, a, 0.0)
        if 2.0 * dust.alpha * max(R_all[k_end.max()], float(a_real.max(initial=0.0))) >= 600.0:
            for i in np.flatnonzero(ok):
                p_out[i], r_out[i], fog_out[i] = self.direct(
                    uni, dust.profile(i), range_min, range_max, r_end[i], scale[i])
            return p_out, r_out, fog_out

        # Rows split as in direct(): [0, k0) R <= first entry, the plain fog curve;
        # [k0, k1) the pulse window overlaps dust; [k1, k_end] past the last exit,
        # the fog curve times the two-way dust transmission.
        P_fog = self._P
        last_exit = np.where(real, dust.b, -np.inf).max(axis=1)
        k_stop = k_end + 1
        k0 = np.minimum(np.searchsorted(R_all, dust.first_entry, side='right'), k_stop)
        k1 = np.minimum(np.maximum(np.searchsorted(R_all - self.L, last_exit, side='left'), k0),
                        k_stop)
        best_p = np.full(nb, -np.inf)
        best_k = np.zeros(nb, int)
        best_fog = np.ones(nb, bool)

        has = k0 > 0                                       # [0, k0): running max of the table
        best_p[has] = self._lut[3][k0[has] - 1]
        best_k[has] = self._lut[4][k0[has] - 1]

        width = k1 - k0
        busy = np.flatnonzero(width > 0)
        if busy.size:                                      # [k0, k1): running sums over s
            S0, SE = self._prefix_sums(uni, range_min, range_max, dust.alpha)
            n = self.s.size
            cum = np.cumsum(lengths, axis=1)               # dust path length up to b_k
            inside_f = np.where(lengths > 0.0,
                                np.exp(2.0 * dust.alpha * (a_real - (cum - lengths))), 0.0)
            after_c = np.exp(-2.0 * dust.alpha * cum)      # for b_k <= x <= a_{k+1}
            # chunks of similar window width, so few rows are padding
            busy = busy[np.argsort(width[busy], kind='stable')]
            for g in np.array_split(busy, -(-busy.size // 64)):
                W = int(width[g].max())
                rel = np.arange(W)[None, :]
                rows = np.minimum(k0[g, None] + rel, R_all.size - 1)   # (len(g), W)
                R = R_all[rows]
                jb = np.searchsorted(self.s, R[:, None, :] - b[g, :, None], side='right')
                ja = np.searchsorted(self.s, R[:, None, :] - a[g, :, None], side='left')
                # s indices [0, jb_k): x >= b_k; [jb_k, ja_k): inside k; [ja_k, n): x <= a_k
                ja_next = np.concatenate([ja[:, 1:], np.zeros_like(ja[:, :1])], axis=1)
                r3 = rows[:, None, :]
                inside = (inside_f[g, :, None] * (SE[r3, ja] - SE[r3, jb])).sum(axis=1)
                between = (after_c[g, :, None] * (S0[r3, jb] - S0[r3, ja_next])).sum(axis=1)
                before = S0[rows, n] - S0[rows, ja[:, 0]]
                Pf = uni.beta * (before + inside + between)
                Pd = (scale[g] * dust.beta)[:, None] * inside
                P = np.where(rel < width[g, None], self.p.soft_gain * (Pf + Pd), -np.inf)
                j = np.argmax(P, axis=1)
                bi = np.arange(g.size)
                pj = P[bi, j]
                win = pj > best_p[g]
                best_p[g[win]] = pj[win]
                best_k[g[win]] = rows[bi, j][win]
                best_fog[g[win]] = (Pf[bi, j] >= Pd[bi, j])[win]

        tail = k1 <= k_end                                 # [k1, k_end]: range max of the table
        if tail.any():
            kt = _range_argmax(P_fog, self._sparse, k1[tail], k_end[tail])
            trans = np.exp(-2.0 * dust.alpha * lengths[tail].sum(axis=1))
            pt = self.p.soft_gain * (self._lut[2][kt] * trans)
            idx = np.flatnonzero(tail)
            win = pt > best_p[idx]
            best_p[idx[win]] = pt[win]
            best_k[idx[win]] = kt[win]
            best_fog[idx[win]] = True

        p_out[ok], r_out[ok], fog_out[ok] = best_p[ok], R_all[best_k[ok]], best_fog[ok]
        return p_out, r_out, fog_out

    def reported_range(self, r_soft, range_min):
        """Rising edge (default) matches how hard returns are timed; or the peak."""
        if self.p.soft_range_convention == 'rising_edge':
            return np.maximum(range_min, r_soft - self.L / 2.0)
        return r_soft


# =============================================================================
# Fog + dust for one configuration, and per scan
# =============================================================================
class WeatherMedia:
    """Fog and dust for one configuration. `scan()` gives the per-scan ScanMedia.

    fog_visibility_m / dust_visibility_m: None switches that medium off.
    dust_clouds: validated clouds (load_cloud_sets); frame 'sensor' clouds move
    with the LiDAR, frame 'world' clouds need the sensor pose in the world.
    """

    def __init__(self, params: FogDustParams, fog_visibility_m=None, dust_visibility_m=None,
                 dust_clouds=()):
        self.p = params.validate()
        self.soft = SoftReturnModel(params)
        self.fog_visibility_m = fog_visibility_m
        self.dust_visibility_m = dust_visibility_m
        if fog_visibility_m is not None:
            self.fog_alpha = alpha_fog(fog_visibility_m, params.visibility_convention)
            self.fog_beta = beta_fog(fog_visibility_m, params.fog_beta_model,
                                     params.visibility_convention)
        else:
            self.fog_alpha = self.fog_beta = 0.0
        self.clouds = list(dust_clouds) if dust_visibility_m is not None else []
        if dust_visibility_m is not None:
            if not self.clouds:
                raise ValueError('dust is on but the dust cloud set has no clouds')
            self.dust_alpha = alpha_dust(dust_visibility_m)
            self.dust_beta = beta_dust(dust_visibility_m, params.dust_beta_model)
        else:
            self.dust_alpha = self.dust_beta = 0.0
        self.sensor_clouds = [c for c in self.clouds if c['frame'] == 'sensor']
        self.world_clouds = [c for c in self.clouds if c['frame'] == 'world']
        self._chord_key = None
        self._sensor_chords = None

    @property
    def fog_on(self):
        return self.fog_visibility_m is not None

    @property
    def dust_on(self):
        return self.dust_visibility_m is not None

    @property
    def active(self):
        return self.fog_on or self.dust_on

    @property
    def has_world_clouds(self):
        return bool(self.world_clouds)

    def describe(self):
        d = {'fog': None, 'dust': None}
        if self.fog_on:
            d['fog'] = {'visibility_m': self.fog_visibility_m, 'alpha_per_m': self.fog_alpha,
                        'beta_per_m_sr': self.fog_beta}
        if self.dust_on:
            d['dust'] = {'visibility_m': self.dust_visibility_m, 'alpha_per_m': self.dust_alpha,
                         'beta_per_m_sr': self.dust_beta, 'clouds': self.clouds}
        return d

    def chords(self, angle_min, angle_increment, n, world_pose=None):
        """(n, n_clouds, 2) dust chords; world clouds are skipped when world_pose is None."""
        key = (angle_min, angle_increment, n)
        angles = beam_angles(angle_min, angle_increment, n)
        if key != self._chord_key:
            self._sensor_chords = cloud_chords(self.sensor_clouds, angles)
            self._chord_key = key
        if not self.world_clouds or world_pose is None:
            return self._sensor_chords
        x, y, yaw = world_pose
        return np.concatenate(
            [self._sensor_chords, cloud_chords(self.world_clouds, angles, (x, y), yaw)], axis=1)

    def scan(self, ranges, hit, r_min, r_max, angle_min, angle_increment, rain_alpha, rng,
             world_pose=None):
        """Fog / dust along every beam of one scan, or None when both are off.

        ranges: clean ranges; hit: beams with a real surface (rain_models.clean_hits);
        rain_alpha: the rain model's extinction, which also attenuates the
        fog / dust soft echo; rng: draws exactly 3 uniforms per beam, so the
        stream depends only on the seed and beam count.
        """
        if not self.active:
            return None
        p = self.p
        r = np.asarray(ranges, dtype=float)
        n = r.size
        u_jitter = rng.uniform(-1.0, 1.0, n)
        # Two spare draws per beam, so the stream (and every seed's output) is
        # the same as in the sweeps already run.
        rng.random(n)
        rng.uniform(-1.0, 1.0, n)

        r_end = np.where(hit, r, r_max)  # medium integrated to the target, or to r_max
        dust = None
        if self.dust_on:
            ch = self.chords(angle_min, angle_increment, n, world_pose)
            if ch.shape[1]:
                r_in = np.nanmin(np.where(np.isnan(ch[:, :, 0]), np.inf, ch[:, :, 0]), axis=1)
                sel = np.flatnonzero(r_in < r_end)
                if sel.size:
                    dust = DustBeams(sel, ch[sel], self.dust_alpha, self.dust_beta)

        uni = UniformMedium(self.fog_alpha + float(rain_alpha), self.fog_beta)
        p_soft = np.zeros(n)
        r_soft = np.full(n, np.nan)
        fog_dom = np.ones(n, bool)
        if self.fog_on:
            no_dust = np.ones(n, bool)
            if dust is not None:
                no_dust[dust.beams] = False
            p_soft[no_dust], r_soft[no_dust] = self.soft.uniform_lookup(
                uni, r_min, r_max, r_end[no_dust])

        if dust is not None:
            sel = dust.beams
            p_soft[sel], r_soft[sel], fog_dom[sel] = self.soft.direct_batch(
                uni, dust, r_min, r_max, r_end[sel], 1.0)

        rep = np.full(n, np.nan)
        ok = np.isfinite(r_soft)
        rep[ok] = np.clip(self.soft.reported_range(r_soft[ok], r_min)
                          * 2.0 ** (p.jitter_exponent_max * u_jitter[ok]), r_min, r_max)
        label = np.where(fog_dom, LABEL_FOG_FALSE, LABEL_DUST_FALSE).astype(np.uint8)
        return ScanMedia(self.fog_alpha, dust, n, p_soft, rep, label)


class ScanMedia:
    """Fog / dust on the beams of one scan, as the rain models consume it.

    tau(beams, r)  extra optical depth from 0 to r along each beam (fog + dust)
    p_soft, r_soft the fog / dust soft echo per beam (power, reported range)
    label          LABEL_FOG_FALSE / LABEL_DUST_FALSE: which medium made the echo
    """

    def __init__(self, fog_alpha, dust, n, p_soft, r_soft, label):
        self.fog_alpha = fog_alpha
        self.dust = dust                       # DustBeams, or None when no beam meets dust
        self.dust_row = np.full(n, -1)         # beam -> row of `dust`, -1 for no dust
        if dust is not None:
            self.dust_row[dust.beams] = np.arange(dust.beams.size)
        self.p_soft = p_soft
        self.r_soft = r_soft
        self.label = label

    def tau(self, beams, r):
        beams, r = np.broadcast_arrays(np.asarray(beams), np.asarray(r, dtype=float))
        t = np.array(self.fog_alpha * r, dtype=float)       # writable, also for 0-d input
        if self.dust is not None:
            row = self.dust_row[beams]
            m = row >= 0
            if m.any():
                t[m] += self.dust.tau(row[m], r[m])
        return t

    def mean_tau(self, r_end):
        """Mean one-way fog + dust optical depth from 0 to r_end over the beams (for logging)."""
        return float(np.mean(self.tau(np.arange(len(r_end)), r_end))) if len(r_end) else 0.0

    def merge(self, out_r, out_i, labels, p_pub, p_min, intensity_scale, echo_mode='strongest'):
        """Let the fog / dust echo compete with what the rain model published (in place).

        p_pub: power of each published echo (-1 where the beam is +inf). With
        'strongest' the larger power wins (single strongest return, Alavi
        Sec. 5.1); with 'first' the nearer valid echo wins.
        """
        valid = self.p_soft >= p_min
        if echo_mode == 'first':
            wins = valid & ((p_pub < 0.0) | (self.r_soft < out_r))
        else:
            wins = valid & (self.p_soft > p_pub)
        out_r[wins] = self.r_soft[wins]
        out_i[wins] = self.p_soft[wins] * intensity_scale
        labels[wins] = self.label[wins]
        return wins
