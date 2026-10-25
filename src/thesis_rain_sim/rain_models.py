"""rain_models - 2D LiDAR rain models, pure numpy (no ROS), selectable by name.

Every model maps one clean scan to a rain-corrupted scan:

    model.apply(ranges, r_min, r_max, angle_min, angle_increment, rain_rate, rng)
        -> (out_ranges, out_intensities, labels)

`ranges` follows REP 117: a beam with no return is +inf. `rng` is a
numpy Generator owned by the caller, so a fixed seed gives a fixed output.
`labels` is a per-beam ground-truth code (LABEL_* below) for scoring a
denoiser's precision / recall.

Fog and dust (fog_dust.py) enter through the optional `media` argument of the
physical models (goodin2019 / alavi2025 / hybrid), a fog_dust.ScanMedia: its
optical depth attenuates the hard target and the rain echoes, and its soft
echo competes for the single published return. With media=None every model is
unchanged, bit for bit. `legacy` has no power-based sensor and rejects media.

Models
------
legacy      the original rain_model_node maths, kept bit for bit so sweeps
            recorded before this module existed stay comparable.
goodin2019  Goodin, Carruth, Doude, Hudson, "Predicting the Influence of Rain
            on LIDAR in ADAS", Electronics 8(1):89, 2019.
alavi2025   Alavi, Illing, Campean, Spencer, Abdullatif, "Quantifying
            Automotive Lidar System Uncertainty in Adverse Weather:
            Mathematical Models and Validation", Appl. Sci. 15(15):8191, 2025.
hybrid      drop physics from Kilic, Hegde, Sindagi, Cooper, Foster, Patel,
            "Lidar Light Scattering Augmentation (LISA): Physics-based
            Simulation of Adverse Weather Conditions for 3D Object Detection",
            arXiv:2107.07004, 2021 (Marshall-Palmer drops placed at random in
            the diverging beam, drop-vs-target power rule, SNR range noise),
            with the detection threshold, overlap ramp and single strongest
            echo of alavi2025. Each effect has exactly one source.

Relative power and detection threshold (shared by goodin2019 / alavi2025 / hybrid)
------------------------------------------------------------------------
Received power is normalised by the sensor constant (Goodin Eq. 2-3), so a
Lambertian hard target of reflectivity rho at range z returns

    P_n(z) = (rho / pi) / z^2 * exp(-2 alpha z)                 Goodin Eq. 3

and the sensor detects a return when P_n >= P_min, with

    P_min = rho_ref / (pi * z_max^2)                             Goodin Eq. 4

Goodin writes Eq. 4 for the usual data-sheet reference rho_ref = 0.9. The
RPLiDAR C1 data sheet instead rates 12 m at 70 % reflectivity (6 m at 10 %),
so rho_ref is a parameter (`z_max_reflectivity`, default 0.7).
"""
import math
from dataclasses import dataclass, fields

import numpy as np

C_LIGHT = 299_792_458.0                         # [m/s]
DB_PER_KM_TO_PER_M = math.log(10.0) / 10.0 / 1000.0

CONDITION_RATES = {'calm': 0.0, 'light': 6.0, 'heavy': 25.0}

# per-beam ground-truth labels (published on /scan_rain_labels). 0-4 are the
# original rain codes; with fog / dust on, "rain" in 0-4 reads as "weather".
LABEL_CLEAN = 0       # real return, no weather-induced range error (sensor noise only)
LABEL_RAIN_NOISE = 1  # real return, range perturbed by the weather range-noise term
LABEL_DROPPED = 2     # real surface in the clean scan, lost to the weather (attenuation /
#                       scatter dropout; out-competed by a weather echo is label 3, 5 or 6)
LABEL_RAIN_FALSE = 3  # published return is a rain droplet, not a surface
LABEL_NO_RETURN = 4   # no surface in the clean scan and no weather echo either
LABEL_FOG_FALSE = 5   # published return is fog backscatter (fog_dust.py)
LABEL_DUST_FALSE = 6  # published return is dust backscatter (fog_dust.py)

MODEL_NAMES = ('legacy', 'goodin2019', 'alavi2025', 'hybrid')
LEGACY_NO_MEDIA = ("fog / dust need a power-based sensor: use rain_model goodin2019, alavi2025 "
                   "or hybrid (legacy is the frozen ad-hoc rain model and has no detection "
                   "threshold in power units)")


def clean_hits(ranges, r_min, r_max):
    """Beams that hit a real surface in the clean scan (same rule as the legacy node)."""
    return np.isfinite(ranges) & (ranges >= r_min) & (ranges < r_max - 1e-3)


def rain_rate_for(condition, explicit_rate=-1.0):
    """Explicit rate if >= 0, otherwise the preset for `condition` (calm/light/heavy)."""
    if explicit_rate >= 0.0:
        return float(explicit_rate)
    return CONDITION_RATES.get(str(condition).lower(), 0.0)


def extra_sensor_sigma(total_sigma, simulator_sigma):
    """Std-dev to ADD so the total sensor noise equals `total_sigma`.

    Independent Gaussian errors add in variance, not in std-dev:
    sigma_total^2 = sigma_sim^2 + sigma_extra^2. Gazebo's gpu_lidar already
    applies `simulator_sigma` (model.sdf <noise>), so stacking the full target
    on top would over-shoot (sqrt(0.010^2 + 0.012^2) = 15.6 mm, not 12 mm).
    """
    return math.sqrt(max(0.0, total_sigma ** 2 - simulator_sigma ** 2))


def _pick(cls, params):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (params or {}).items() if k in names})


# =============================================================================
# legacy
# =============================================================================
@dataclass
class LegacyModel:
    """Original rain_model_node maths. DO NOT change the arithmetic or the order
    of RNG draws: `test_legacy_bit_for_bit` compares against a frozen copy.

        alpha(R)  = alpha_k * R**alpha_p
        I         = i0 / r^2 * exp(-2 alpha r) * (1 + N(0, i_noise))
        lost if I < min_detectable_intensity, or with prob q_scatter*(1-exp(-alpha r))
        sigma_r   = sqrt(sigma0^2 + (k_rain * R * r / r_max)^2)
        false return with prob min(p_false_k * R, 0.15), placed 0.3-4 m
    """
    i0: float = 10000.0
    intensity_noise_frac: float = 0.05
    min_detectable_intensity: float = 8.0
    alpha_k: float = 0.01
    alpha_p: float = 0.6
    q_scatter: float = 0.6
    range_sigma_base_m: float = 0.012
    k_rain_range: float = 0.02
    p_false_k: float = 0.004
    false_range_min_m: float = 0.30
    false_range_max_m: float = 4.0
    false_intensity_mean: float = 6.0
    false_intensity_frac: float = 0.5

    name = 'legacy'

    def alpha(self, rain_rate):
        return self.alpha_k * (rain_rate ** self.alpha_p) if rain_rate > 0 else 0.0

    def apply(self, ranges, r_min, r_max, angle_min, angle_increment, rain_rate, rng,
              media=None):
        if media is not None:
            raise ValueError(LEGACY_NO_MEDIA)
        n = len(ranges)
        r = np.asarray(ranges, dtype=np.float64)
        alpha = self.alpha(rain_rate)

        hit = clean_hits(r, r_min, r_max)
        out_r = np.full(n, math.inf, dtype=np.float64)
        out_i = np.zeros(n, dtype=np.float64)
        labels = np.full(n, LABEL_NO_RETURN, dtype=np.uint8)

        idx = np.flatnonzero(hit)
        if idx.size:
            rr = r[idx]
            inten = self.i0 / np.square(rr) * np.exp(-2.0 * alpha * rr)
            inten *= (1.0 + rng.normal(0.0, self.intensity_noise_frac, idx.size))
            inten = np.clip(inten, 0.0, None)

            p_drop = self.q_scatter * (1.0 - np.exp(-alpha * rr)) if alpha > 0 else 0.0
            detected = (inten >= self.min_detectable_intensity) & (rng.random(idx.size) > p_drop)

            sigma_r = np.sqrt(self.range_sigma_base_m ** 2
                              + (self.k_rain_range * rain_rate * rr / r_max) ** 2)
            meas = rr + rng.normal(0.0, sigma_r)
            meas = np.clip(meas, r_min, r_max)

            det = idx[detected]
            out_r[det] = meas[detected]
            out_i[det] = inten[detected]
            labels[idx] = LABEL_DROPPED
            labels[det] = LABEL_RAIN_NOISE if rain_rate > 0 else LABEL_CLEAN

        p_false = min(self.p_false_k * rain_rate, 0.15)
        if p_false > 0.0:
            draw = rng.random(n) < p_false
            fi = np.flatnonzero(draw)
            if fi.size:
                hi = self.false_range_max_m * np.ones(fi.size)
                real = np.isfinite(out_r[fi])
                hi[real] = np.minimum(hi[real], out_r[fi][real])
                lo = np.minimum(self.false_range_min_m, hi - 1e-2)
                frange = lo + (hi - lo) * rng.random(fi.size)
                finten = np.clip(
                    self.false_intensity_mean
                    * (1.0 + rng.normal(0.0, self.false_intensity_frac, fi.size)),
                    0.0, None)
                out_r[fi] = frange
                out_i[fi] = finten
                labels[fi] = LABEL_RAIN_FALSE
        return out_r, out_i, labels


# =============================================================================
# goodin2019
# =============================================================================
@dataclass
class Goodin2019:
    """Goodin et al. 2019, Sections 2.1-2.2.

    alpha(R) = a * R^b, a = 0.01, b = 0.6          Eq. 5 (fit to Filgueira et al.)
    P_n(z)   = rho/z^2 * exp(-2 alpha z)            Eq. 3 / Eq. 8, rho = reflectivity/pi
    P_min    = rho_ref / (pi z_max^2)               Eq. 4 (paper: rho_ref = 0.9)
    z'       = z + N(0, 0.02 z (1 - e^-R)^2)        Eq. 9

    Per-beam procedure, in the paper's order (Sec. 2.2, Fig. 1):
      1. clean range from the simulator,
      2. Eq. 9 range noise,
      3. attenuated intensity (Eq. 7/8), evaluated at the TRUE range z: Eq. 7
         is defined as the fractional reduction for "a surface at a distance z",
         and the clean intensity it scales is the simulator's, at z,
      4. drop the point if P_n < P_min (published as +inf, REP 117).

    Goodin models NO droplet false returns (Sec. 2: "false positives are very
    unlikely", citing Fersch et al. 2016), so this model adds none.

    Sensor noise is a separate, independent, non-rain term (see
    extra_sensor_sigma); Eq. 9 is zero at R = 0, so calm = sensor noise only.
    """
    a: float = 0.01
    b: float = 0.6
    range_noise_frac: float = 0.02
    default_reflectivity: float = 0.7
    z_max: float = 12.0
    z_max_reflectivity: float = 0.7
    sensor_sigma_total_m: float = 0.012
    simulator_sigma_m: float = 0.010
    intensity_output_scale: float = 10000.0 * math.pi

    name = 'goodin2019'

    def alpha(self, rain_rate):                                   # Eq. 5
        return self.a * rain_rate ** self.b if rain_rate > 0 else 0.0

    def p_min(self):                                              # Eq. 4
        return self.z_max_reflectivity / (math.pi * self.z_max ** 2)

    def range_sigma(self, z, rain_rate):                          # Eq. 9
        return self.range_noise_frac * np.asarray(z) * (1.0 - math.exp(-rain_rate)) ** 2

    def relative_power(self, z, rain_rate, reflectivity=None):    # Eq. 3 / 8
        rho = (self.default_reflectivity if reflectivity is None else reflectivity) / math.pi
        z = np.asarray(z, dtype=np.float64)
        return rho / np.square(z) * np.exp(-2.0 * self.alpha(rain_rate) * z)

    @property
    def sensor_sigma_extra(self):
        return extra_sensor_sigma(self.sensor_sigma_total_m, self.simulator_sigma_m)

    def apply(self, ranges, r_min, r_max, angle_min, angle_increment, rain_rate, rng,
              reflectivity=None, media=None):
        n = len(ranges)
        r = np.asarray(ranges, dtype=np.float64)
        out_r = np.full(n, math.inf, dtype=np.float64)
        out_i = np.zeros(n, dtype=np.float64)
        labels = np.full(n, LABEL_NO_RETURN, dtype=np.uint8)

        idx = np.flatnonzero(clean_hits(r, r_min, r_max))
        if idx.size == 0 and media is None:
            return out_r, out_i, labels
        z = r[idx]
        refl = None if reflectivity is None else np.asarray(reflectivity, np.float64)[idx]

        # step 2: Eq. 9 rain range noise, then the independent sensor term
        z_meas = z + rng.normal(0.0, 1.0, idx.size) * self.range_sigma(z, rain_rate)
        if self.sensor_sigma_extra > 0.0:
            z_meas = z_meas + rng.normal(0.0, self.sensor_sigma_extra, idx.size)
        # step 3: attenuated relative power at the true range
        p = self.relative_power(z, rain_rate, refl)
        if media is not None:                       # fog / dust extinction on top of rain
            p = p * np.exp(-2.0 * media.tau(idx, z))
        # step 4: threshold
        det = p >= self.p_min()

        keep = idx[det]
        out_r[keep] = np.clip(z_meas[det], r_min, r_max)
        out_i[keep] = p[det] * self.intensity_output_scale
        labels[idx] = LABEL_DROPPED
        labels[keep] = LABEL_RAIN_NOISE if rain_rate > 0 else LABEL_CLEAN
        if media is not None:                       # fog / dust echo vs the surface
            p_pub = np.full(n, -1.0)
            p_pub[keep] = p[det]
            media.merge(out_r, out_i, labels, p_pub, self.p_min(), self.intensity_output_scale)
        return out_r, out_i, labels


# =============================================================================
# alavi2025
# =============================================================================
# Alavi et al. 2025, Table 2 (rain extinction). Coefficients read from the PDF
# text layer and cross-checked against a render of page 8 and Figure 5.
#   continental  [22] Carbonneau & Wisely 1998   1.076 R^0.67  [dB/km]
#   thunderstorm [23] Shipley et al. 1974        0.16  R^0.74  (unit not printed)
#   tropical     [24] Ali 2013                   0.365 R^0.63  [dB/km]
# The thunderstorm unit is km^-1: only that reading reproduces Figure 5
# (0.16*100^0.74 = 4.8 km^-1 = 4.8e-3 m^-1 at 100 mm/h; as dB/km it would plot
# at 1.1e-3 m^-1, below the tropical curve, contradicting Sec. 4.2's ordering).
# dB/km -> 1/m uses ln(10)/10/1000, which Figure 5 also confirms.
RAIN_EXTINCTION = {
    'continental': (1.076, 0.67, DB_PER_KM_TO_PER_M),
    'thunderstorm': (0.16, 0.74, 1.0e-3),
    'tropical': (0.365, 0.63, DB_PER_KM_TO_PER_M),
}
ALPHA_OVER_BETA_RAIN = 0.60     # Alavi Eq. 11: beta_rain = alpha_rain / 0.60


def alpha_rain(rain_type, rain_rate):
    """Alavi Table 2 extinction coefficient [1/m]."""
    if rain_rate <= 0:
        return 0.0
    try:
        k, p, to_per_m = RAIN_EXTINCTION[rain_type]
    except KeyError:
        raise ValueError(f"rain_type must be one of {sorted(RAIN_EXTINCTION)}, got '{rain_type}'")
    return k * rain_rate ** p * to_per_m


def beta_rain_per_sr(alpha):
    """Volume backscatter coefficient [1/(m sr)] from Alavi Eq. 11.

    Eq. 11 gives beta = alpha / 0.60 with beta defined by Eq. 8, which has the
    same form (and units, m^-1) as Eq. 7 for alpha - i.e. an angle-integrated
    quantity, not per steradian. The lidar equation (Eq. 5/6) needs the
    per-steradian backscatter that is directly comparable with a Lambertian
    hard target's rho/pi [sr^-1], so it is divided by 4 pi here.
    """
    return alpha / ALPHA_OVER_BETA_RAIN / (4.0 * math.pi)


def overlap(r, r1, r2):
    """Bistatic overlap xi(r): 0 below R1, linear ramp, 1 above R2 (Alavi Sec. 2.1, Fig. 2)."""
    if r2 <= r1:
        return (np.asarray(r) >= r1).astype(np.float64)
    return np.clip((np.asarray(r, dtype=np.float64) - r1) / (r2 - r1), 0.0, 1.0)


def incidence_cosine(ranges, angle_min, angle_increment, window=2,
                     jump_abs_m=0.10, jump_rel=0.05):
    """|cos| of the beam's incidence angle on the surface, from a local line fit.

    A 2D scan has no surface normals, so the normal is estimated by a
    principal-axis (total least squares) line fit through the beam's own point
    and up to `window` neighbours either side that lie on the same surface
    (range within jump_abs_m + jump_rel*r). Beams with < 3 points on their
    surface get 1.0 (normal incidence, i.e. no correction). A full 360 deg
    scan wraps around.
    """
    r = np.asarray(ranges, dtype=np.float64)
    n = r.size
    ang = angle_min + angle_increment * np.arange(n)
    finite = np.isfinite(r)
    rs = np.where(finite, r, 0.0)
    x, y = rs * np.cos(ang), rs * np.sin(ang)
    wrap = abs(n * angle_increment - 2.0 * math.pi) < 2.0 * abs(angle_increment)

    cnt = np.zeros(n)
    sx = np.zeros(n)
    sy = np.zeros(n)
    sxx = np.zeros(n)
    syy = np.zeros(n)
    sxy = np.zeros(n)
    base = np.arange(n)
    for k in range(-window, window + 1):
        j = base + k
        if wrap:
            j %= n
            inb = np.ones(n, bool)
        else:
            inb = (j >= 0) & (j < n)
            j = np.clip(j, 0, n - 1)
        ok = inb & finite & finite[j] & (np.abs(rs[j] - rs) <= jump_abs_m + jump_rel * rs)
        w = ok.astype(np.float64)
        cnt += w
        sx += w * x[j]
        sy += w * y[j]
        sxx += w * x[j] ** 2
        syy += w * y[j] ** 2
        sxy += w * x[j] * y[j]

    out = np.ones(n)
    good = cnt >= 3
    if np.any(good):
        c = cnt[good]
        mx, my = sx[good] / c, sy[good] / c
        cxx = sxx[good] / c - mx ** 2
        cyy = syy[good] / c - my ** 2
        cxy = sxy[good] / c - mx * my
        theta = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)       # line direction
        nx, ny = -np.sin(theta), np.cos(theta)                  # its normal
        out[good] = np.abs(nx * np.cos(ang[good]) + ny * np.sin(ang[good]))
    return out


@dataclass
class Alavi2025:
    """Alavi et al. 2025, Sections 2.1, 2.3.2, 4.2 and 5.1.

    Extinction   alpha_rain(R)          Table 2, `rain_type` (default continental,
                                        the worst case per Sec. 4.2)
    Backscatter  beta = alpha / 0.60    Eq. 11, / (4 pi) -> per steradian (see
                                        beta_rain_per_sr)
    Hard target  P_h = cos(theta)^[opt] * (rho/pi) / z^2 * exp(-2 alpha z)
                 Eq. 1-4 with a homogeneous medium reduce to Goodin's Eq. 3.
                 The incidence factor is optional (use_incidence_angle, off).
    Soft target  Eq. 6 (after Rasshofer et al. / Hahner et al.): backscatter
                 from the rain volume between the sensor and the hard target,
                 cut off at the hard target by the Heaviside step U. Discretised
                 into range bins of width dr = c*tau/2 (tau = pulse FWHM), from
                 the overlap start R1 up to the hard-target range (or r_max if
                 the beam hits nothing). The mean normalised power of bin i is

                     Pbar_i = xi(r_i) * beta * dr * exp(-2 alpha r_i) / r_i^2

                 Why dr = c*tau/2: convolving a sin^2 pulse of total duration
                 tau_H with a uniform medium gives an equivalent length
                 c*tau_H/4 = c*FWHM/2 relative to a hard target's peak, so a bin
                 of that width carries beta*dr "reflectivity" in the same
                 normalised units as a Lambertian rho/pi target.

    STOCHASTIC BACKSCATTER (assumption - not in Alavi)
        backscatter_mode='stochastic' (default): each bin's received power is
        an independent draw  P_i = Pbar_i * Exp(1),  so a bin produces a rain
        echo with probability  P(P_i >= P_min) = exp(-P_min / Pbar_i).
        The return from a bin is the incoherent sum of many droplets with random
        positions (random phases); its intensity then follows the exponential
        (fully developed speckle) distribution with the mean given by Eq. 6.
        This keeps Eq. 6's expected power, needs no extra unknown parameters
        (a droplet-count model would need a drop-size distribution, beam
        divergence and receiver aperture - none are published for the C1),
        and yields RARE false returns when Pbar << P_min, as Goodin / Fersch
        observe for small-aperture sensors. The echo is placed uniformly within
        its bin.
        backscatter_mode='mean' uses Pbar deterministically (for analysis only:
        either every beam sees the same near-range echo, or none does).

    Echo selection (Sec. 5.1: echoes "ordered ... by intensity"). The C1 is
    single-return: echo_mode='strongest' publishes the most powerful echo above
    P_min, 'first' the nearest one. No echo above P_min -> +inf.

    Alavi adds no rain range-noise term, so a surviving hard return is labelled
    LABEL_CLEAN (sensor noise only).
    """
    rain_type: str = 'continental'
    default_reflectivity: float = 0.7
    z_max: float = 12.0
    z_max_reflectivity: float = 0.7
    pulse_width_s: float = 5.0e-9
    overlap_r1_m: float = 0.05
    overlap_r2_m: float = 1.0
    echo_mode: str = 'strongest'
    backscatter_mode: str = 'stochastic'
    use_incidence_angle: bool = False
    incidence_window: int = 2
    sensor_sigma_total_m: float = 0.012
    simulator_sigma_m: float = 0.010
    intensity_output_scale: float = 10000.0 * math.pi

    name = 'alavi2025'

    def __post_init__(self):
        if self.echo_mode not in ('strongest', 'first'):
            raise ValueError(f"echo_mode must be 'strongest' or 'first', got '{self.echo_mode}'")
        if self.backscatter_mode not in ('stochastic', 'mean'):
            raise ValueError(
                f"backscatter_mode must be 'stochastic' or 'mean', got '{self.backscatter_mode}'")
        alpha_rain(self.rain_type, 1.0)     # validates rain_type

    def alpha(self, rain_rate):
        return alpha_rain(self.rain_type, rain_rate)

    def beta(self, rain_rate):
        return beta_rain_per_sr(self.alpha(rain_rate))

    def p_min(self):
        return self.z_max_reflectivity / (math.pi * self.z_max ** 2)

    @property
    def bin_width(self):
        return C_LIGHT * self.pulse_width_s / 2.0

    @property
    def sensor_sigma_extra(self):
        return extra_sensor_sigma(self.sensor_sigma_total_m, self.simulator_sigma_m)

    def bin_centres(self, r_max):
        dr = self.bin_width
        nb = max(1, int(math.ceil((r_max - self.overlap_r1_m) / dr)))
        return self.overlap_r1_m + (np.arange(nb) + 0.5) * dr

    def mean_bin_power(self, centres, rain_rate):
        """Eq. 6 discretised: expected normalised backscatter power per bin."""
        a = self.alpha(rain_rate)
        c = np.asarray(centres, dtype=np.float64)
        return (overlap(c, self.overlap_r1_m, self.overlap_r2_m) * self.beta(rain_rate)
                * self.bin_width * np.exp(-2.0 * a * c) / np.square(c))

    def hard_power(self, z, rain_rate, reflectivity=None, cos_inc=None):
        rho = (self.default_reflectivity if reflectivity is None else reflectivity) / math.pi
        z = np.asarray(z, dtype=np.float64)
        p = rho / np.square(z) * np.exp(-2.0 * self.alpha(rain_rate) * z)
        return p if cos_inc is None else p * cos_inc

    def apply(self, ranges, r_min, r_max, angle_min, angle_increment, rain_rate, rng,
              reflectivity=None, media=None):
        n = len(ranges)
        r = np.asarray(ranges, dtype=np.float64)
        pmin = self.p_min()
        hit = clean_hits(r, r_min, r_max)
        idx = np.flatnonzero(hit)

        # --- hard target ---------------------------------------------------
        cos_inc = None
        if self.use_incidence_angle:
            cos_inc = incidence_cosine(np.where(hit, r, np.inf), angle_min,
                                       angle_increment, self.incidence_window)[idx]
        refl = None if reflectivity is None else np.asarray(reflectivity, np.float64)[idx]
        p_hard = np.full(n, -1.0)
        p_hard[idx] = self.hard_power(r[idx], rain_rate, refl, cos_inc)
        if media is not None:                       # fog / dust extinction on top of rain
            p_hard[idx] *= np.exp(-2.0 * media.tau(idx, r[idx]))
        z_meas = np.full(n, math.inf)
        z_meas[idx] = r[idx]
        if self.sensor_sigma_extra > 0.0 and idx.size:
            z_meas[idx] += rng.normal(0.0, self.sensor_sigma_extra, idx.size)

        # candidate echoes: column 0 = hard target, columns 1.. = rain bins
        cand_p = p_hard[:, None]
        cand_r = z_meas[:, None]

        # --- soft target (rain volume), Eq. 6 --------------------------------
        if rain_rate > 0.0:
            centres = self.bin_centres(r_max)
            pbar = self.mean_bin_power(centres, rain_rate)
            end = np.where(hit, r, r_max)                         # Heaviside cut-off
            inside = centres[None, :] < end[:, None]
            if self.backscatter_mode == 'stochastic':
                p_soft = pbar[None, :] * rng.exponential(1.0, (n, centres.size))
                pos = centres[None, :] + (rng.random((n, centres.size)) - 0.5) * self.bin_width
            else:
                p_soft = np.broadcast_to(pbar, (n, centres.size)).copy()
                pos = np.broadcast_to(centres, (n, centres.size)).copy()
            if media is not None:                   # fog / dust between sensor and bin
                p_soft = p_soft * np.exp(-2.0 * media.tau(np.arange(n)[:, None], centres[None, :]))
            p_soft = np.where(inside, p_soft, -1.0)
            pos = np.clip(np.minimum(pos, end[:, None] - 1e-3), r_min, r_max)
            cand_p = np.concatenate([cand_p, p_soft], axis=1)
            cand_r = np.concatenate([cand_r, pos], axis=1)

        valid = cand_p >= pmin
        any_valid = valid.any(axis=1)
        if self.echo_mode == 'strongest':
            sel = np.argmax(np.where(valid, cand_p, -np.inf), axis=1)
        else:
            sel = np.argmin(np.where(valid, cand_r, np.inf), axis=1)

        rows = np.arange(n)
        out_r = np.full(n, math.inf)
        out_i = np.zeros(n)
        out_r[any_valid] = np.clip(cand_r[rows, sel][any_valid], r_min, r_max)
        out_i[any_valid] = cand_p[rows, sel][any_valid] * self.intensity_output_scale

        labels = np.full(n, LABEL_NO_RETURN, dtype=np.uint8)
        labels[hit] = LABEL_DROPPED
        labels[any_valid & (sel == 0)] = LABEL_CLEAN
        labels[any_valid & (sel > 0)] = LABEL_RAIN_FALSE
        if media is not None:                       # fog / dust echo as one more candidate
            p_pub = np.where(any_valid, cand_p[rows, sel], -1.0)
            media.merge(out_r, out_i, labels, p_pub, pmin, self.intensity_output_scale,
                        self.echo_mode)
        return out_r, out_i, labels


# =============================================================================
# hybrid (LISA drops + alavi2025 sensor)
# =============================================================================
@dataclass
class HybridRain:
    """LISA (Kilic et al. 2021) drop physics on the alavi2025 sensor model.

    Equation numbers below are LISA's unless marked otherwise. "Alg. 1" is
    LISA's Algorithm 1 (hybrid Monte-Carlo).

    Drop-size distribution (Marshall & Palmer 1948, J. Meteor. 5:165; the
    model LISA cites as [21] but does not print constants for):
        N(D)  = N0 exp(-Lambda D),  N0 = 8000 m^-3 mm^-1,
        Lambda = 4.1 R^-0.21 mm^-1          (R in mm/h, D in mm)
    Only drops with D >= D_min (LISA Sec. 5.1: D_st = 50 um) are placed:
        N_tot = int_Dmin^inf N(D) dD = (N0 / Lambda) exp(-Lambda D_min)  [m^-3]
        D     = D_min - ln(1 - u) / Lambda,  u ~ U(0,1)            Eq. 12
    clipped at max_drop_diameter_mm (assumption: drops above ~6 mm break up;
    LISA has no clip).

    Extinction, extinction_source='dsd' (default):
        alpha = (pi/4) int_0^inf D^2 Q_ext(D) N(D) dD               Eq. 3
    LISA evaluates Q_ext with Mie theory (PyMieScatt). At 905 nm (C1 data
    sheet) the size parameter pi D / lambda exceeds 170 for every drop above
    D_min, where the Mie Q_ext has converged to its large-particle limit of 2
    (the "extinction paradox", Bohren & Huffman 1983, "Absorption and
    Scattering of Light by Small Particles"), so PyMieScatt is not needed.
    With Q_ext = 2 the integral is closed form:
        alpha = pi N0 / Lambda^3 * 1e-6  [1/m]
    The integral runs from D = 0 as in Eq. 3, because every drop attenuates,
    not only the ones placed. Starting at D_min instead changes alpha by
    < 0.1 % (test_hybrid_extinction).
    extinction_source='alavi' uses alpha_rain(rain_type, R) (Alavi Table 2)
    for sensitivity runs only; drop placement still uses Marshall-Palmer.

    Hard target (alavi2025 / Goodin Eq. 3, threshold Goodin Eq. 4):
        P_t = (rho/pi) / z^2 exp(-2 alpha z),   lost if P_t < P_min.
    LISA's P0 = rho e^{-2 alpha R} / R^2 omits the 1/pi; it is omitted from
    the drop power too, so applying it to both changes nothing.

    Drops in the beam (Alg. 1, Eq. 9-11). The beam is a cone from the sensor
    to the hard target (Heaviside cut-off: no drops behind it), or to r_max
    if the beam hits nothing (extension: LISA only processes existing points).
        D_b(r) = d0 + r tan(theta)                Alg. 1 with d0 = 0, Eq. 9
        V      = (pi/12) L (d0^2 + d0 D_b(L) + D_b(L)^2)   truncated cone;
                 d0 = 0 gives Alg. 1's (pi/3) L (D_b/2)^2
        count  ~ Poisson(N_tot V)
    (Alg. 1 instead rounds N_tot V probabilistically. Poisson is the count
    distribution for drops placed independently and uniformly in space; both
    have the same mean.)
        r: pdf proportional to the beam area (d0 + r tan theta)^2, inverted
           in closed form; with d0 = 0 it is L u^(1/3), Eq. 10 (whose text
           layer prints the exponent as -1/3, a typo: that places drops
           behind the target)
        drops with r < r_min are discarded                         Alg. 1
        P_d = (rho_d/pi) / r^2 exp(-2 alpha r) min((D / D_b(r))^2, 1) xi(r)
    rho_d = ((n-1)/(n+1))^2 = 0.0201 for water, n = 1.33: Fresnel at normal
    incidence, Eq. 11 (LISA uses this one value for every drop). xi(r) is the
    alavi2025 bistatic overlap ramp (Alavi Sec. 2.1, not in LISA), applied
    when use_overlap is true so hybrid and alavi2025 are comparable.

    Echo selection (Alg. 1, and Alavi Sec. 5.1 for the single-return C1):
    among the target and the drops with power >= P_min the strongest is
    published. A winning drop reports its own range (LABEL_RAIN_FALSE),
    which is LISA's "random scatterer replaces the target" branch. A beam
    with no candidate is +inf.

    Range noise. The calm sensor term is extra_sensor_sigma, as in goodin2019 /
    alavi2025, so calm output is clean + sensor noise. Rain adds
        sigma(P)    = dR / sqrt(2 SNR(P))                           Eq. 7
        sigma_rain  = sqrt(max(0, sigma(P_rain)^2 - sigma(P_clear)^2))
    with dR = range_accuracy_m and P_clear the same target at R = 0.
    Subtracting the clear-weather part is this package's construction
    (assumption): LISA applies the whole of Eq. 7, which in calm weather
    would double-count the sensor noise already in the simulator.
    SNR(P) = P / P_min (assumption): LISA defines SNR = P_r / P_n (Eq. 4)
    but gives no P_n, only that P_min is estimated from the data-sheet max
    range, so SNR = 1 at the detection threshold.
    dR = 0.06 m (assumption): the C1 data sheet's accuracy of +/-30 mm, read
    as a full span the way LISA Sec. 5.1 turns +/-4.5 cm into dR = 0.09 m.

    Published intensity = power * intensity_output_scale.
    """
    extinction_source: str = 'dsd'
    rain_type: str = 'continental'
    mp_n0: float = 8000.0
    mp_lambda_k: float = 4.1
    mp_lambda_p: float = -0.21
    min_drop_diameter_mm: float = 0.05
    max_drop_diameter_mm: float = 6.0
    beam_divergence_rad: float = 3.0e-3
    beam_exit_diameter_m: float = 0.0
    drop_reflectivity: float = 0.0201
    range_accuracy_m: float = 0.06
    use_overlap: bool = True
    overlap_r1_m: float = 0.05
    overlap_r2_m: float = 1.0
    default_reflectivity: float = 0.7
    z_max: float = 12.0
    z_max_reflectivity: float = 0.7
    sensor_sigma_total_m: float = 0.012
    simulator_sigma_m: float = 0.010
    intensity_output_scale: float = 10000.0 * math.pi

    name = 'hybrid'

    def __post_init__(self):
        if self.extinction_source not in ('dsd', 'alavi'):
            raise ValueError(
                f"extinction_source must be 'dsd' or 'alavi', got '{self.extinction_source}'")
        alpha_rain(self.rain_type, 1.0)     # validates rain_type

    # --- drop-size distribution -------------------------------------------
    def mp_lambda(self, rain_rate):
        """Marshall-Palmer slope Lambda [1/mm]; inf at R = 0 (no drops)."""
        if rain_rate <= 0:
            return math.inf
        return self.mp_lambda_k * rain_rate ** self.mp_lambda_p

    def drop_density(self, rain_rate):
        """N_tot [1/m^3]: drops with D >= min_drop_diameter_mm (Alg. 1 N_t / bvol)."""
        lam = self.mp_lambda(rain_rate)
        if not math.isfinite(lam):
            return 0.0
        return self.mp_n0 / lam * math.exp(-lam * self.min_drop_diameter_mm)

    def alpha_dsd(self, rain_rate):
        """Eq. 3 with Q_ext = 2, integrated from D = 0: pi N0 / Lambda^3 [1/m]."""
        lam = self.mp_lambda(rain_rate)
        if not math.isfinite(lam):
            return 0.0
        return math.pi * self.mp_n0 / lam ** 3 * 1e-6

    def alpha(self, rain_rate):
        if self.extinction_source == 'alavi':
            return alpha_rain(self.rain_type, rain_rate)
        return self.alpha_dsd(rain_rate)

    # --- sensor -------------------------------------------------------------
    def p_min(self):
        return self.z_max_reflectivity / (math.pi * self.z_max ** 2)

    @property
    def sensor_sigma_extra(self):
        return extra_sensor_sigma(self.sensor_sigma_total_m, self.simulator_sigma_m)

    def hard_power(self, z, rain_rate, reflectivity=None):
        rho = (self.default_reflectivity if reflectivity is None else reflectivity) / math.pi
        z = np.asarray(z, dtype=np.float64)
        return rho / np.square(z) * np.exp(-2.0 * self.alpha(rain_rate) * z)

    def range_sigma(self, p):
        """Eq. 7 with SNR = P / P_min (assumption)."""
        return self.range_accuracy_m / np.sqrt(2.0 * np.asarray(p, np.float64) / self.p_min())

    def rain_range_sigma(self, z, rain_rate, reflectivity=None, tau_extra=None):
        """Weather part of Eq. 7: sqrt(sigma(P_weather)^2 - sigma(P_clear)^2).

        tau_extra: fog / dust optical depth to z (fog_dust.py), which lowers the
        SNR exactly as rain extinction does. None = rain only.
        """
        p = self.hard_power(z, rain_rate, reflectivity)
        if tau_extra is not None:
            p = p * np.exp(-2.0 * tau_extra)
        s_rain = self.range_sigma(p)
        s_clear = self.range_sigma(self.hard_power(z, 0.0, reflectivity))
        return np.sqrt(np.maximum(0.0, s_rain ** 2 - s_clear ** 2))

    # --- beam geometry ------------------------------------------------------
    def beam_diameter(self, r):
        return (self.beam_exit_diameter_m
                + np.asarray(r, np.float64) * math.tan(self.beam_divergence_rad))

    def beam_volume(self, length):
        """Volume of the (truncated) cone from 0 to `length` [m^3]."""
        d0 = self.beam_exit_diameter_m
        d1 = self.beam_diameter(length)
        return math.pi / 12.0 * np.asarray(length, np.float64) * (d0 * d0 + d0 * d1 + d1 * d1)

    def sample_drop_range(self, length, u):
        """Inverse CDF of the pdf proportional to D_b(r)^2 on [0, length]."""
        t = math.tan(self.beam_divergence_rad)
        d0 = self.beam_exit_diameter_m
        if t <= 0.0:
            return length * u
        if d0 <= 0.0:
            return length * np.cbrt(u)                                  # Eq. 10
        d1 = d0 + t * length
        return (np.cbrt(d0 ** 3 + u * (d1 ** 3 - d0 ** 3)) - d0) / t

    def drop_power(self, r, d_mm, rain_rate):
        r = np.asarray(r, np.float64)
        area = np.minimum(np.square(np.asarray(d_mm) * 1e-3 / self.beam_diameter(r)), 1.0)
        p = (self.drop_reflectivity / math.pi) / np.square(r) \
            * np.exp(-2.0 * self.alpha(rain_rate) * r) * area
        if self.use_overlap:
            p = p * overlap(r, self.overlap_r1_m, self.overlap_r2_m)
        return p

    def sample_drops(self, length, rain_rate, rng):
        """Drops in each beam's cone [0, length[b]] (Alg. 1).

        Returns (beam index, range [m], diameter [mm]) per drop, before the
        r_min cut. Draw order: Poisson counts, ranges, diameters.
        """
        empty = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
        n_tot = self.drop_density(rain_rate)
        if n_tot <= 0.0:
            return empty
        length = np.asarray(length, np.float64)
        counts = rng.poisson(n_tot * self.beam_volume(length))
        total = int(counts.sum())
        if total == 0:
            return empty
        beam = np.repeat(np.arange(length.size), counts)
        r = self.sample_drop_range(length[beam], rng.random(total))
        d = np.minimum(self.min_drop_diameter_mm
                       - np.log1p(-rng.random(total)) / self.mp_lambda(rain_rate),
                       self.max_drop_diameter_mm)                      # Eq. 12, clipped
        return beam, r, d

    # --- model ----------------------------------------------------------------
    def apply(self, ranges, r_min, r_max, angle_min, angle_increment, rain_rate, rng,
              reflectivity=None, media=None):
        n = len(ranges)
        r = np.asarray(ranges, dtype=np.float64)
        pmin = self.p_min()
        hit = clean_hits(r, r_min, r_max)
        idx = np.flatnonzero(hit)
        refl = None if reflectivity is None else np.asarray(reflectivity, np.float64)[idx]

        # --- hard target: power, sensor noise, rain noise (Eq. 7) ------------
        p_hard = np.full(n, -1.0)
        z_meas = np.full(n, math.inf)
        noisy = np.zeros(n, bool)                   # weather range noise applied (media only)
        if idx.size:
            z = r[idx]
            p_hard[idx] = self.hard_power(z, rain_rate, refl)
            tau_x = None
            if media is not None:                   # fog / dust extinction on top of rain
                tau_x = media.tau(idx, z)
                p_hard[idx] *= np.exp(-2.0 * tau_x)
            zm = z.copy()
            if self.sensor_sigma_extra > 0.0:
                zm += rng.normal(0.0, self.sensor_sigma_extra, idx.size)
            if rain_rate > 0.0 or media is not None:
                sigma = self.rain_range_sigma(z, rain_rate, refl, tau_x)
                zm += rng.normal(0.0, 1.0, idx.size) * sigma
                noisy[idx] = sigma > 0.0
            z_meas[idx] = zm
        target_ok = p_hard >= pmin

        # --- drops in the beam (Alg. 1) --------------------------------------
        best_p = np.full(n, -1.0)
        best_r = np.full(n, math.inf)
        if rain_rate > 0.0:
            length = np.where(hit, r, r_max)                     # Heaviside cut-off
            beam, dr, dd = self.sample_drops(length, rain_rate, rng)
            keep = dr >= r_min
            beam, dr, dd = beam[keep], dr[keep], dd[keep]
            pd = self.drop_power(dr, dd, rain_rate)
            if media is not None:                   # fog / dust between sensor and drop
                pd = pd * np.exp(-2.0 * media.tau(beam, dr))
            ok = pd >= pmin
            beam, dr, pd = beam[ok], dr[ok], pd[ok]
            if beam.size:
                np.maximum.at(best_p, beam, pd)
                win = pd == best_p[beam]
                b_win, first = np.unique(beam[win], return_index=True)
                best_r[b_win] = dr[win][first]

        # --- single strongest echo -------------------------------------------
        drop_wins = (best_p >= pmin) & (~target_ok | (best_p > p_hard))
        target_wins = target_ok & ~drop_wins

        out_r = np.full(n, math.inf)
        out_i = np.zeros(n)
        out_r[target_wins] = np.clip(z_meas[target_wins], r_min, r_max)
        out_i[target_wins] = p_hard[target_wins] * self.intensity_output_scale
        out_r[drop_wins] = np.clip(best_r[drop_wins], r_min, r_max)
        out_i[drop_wins] = best_p[drop_wins] * self.intensity_output_scale

        labels = np.full(n, LABEL_NO_RETURN, dtype=np.uint8)
        labels[hit] = LABEL_DROPPED
        if media is None:
            labels[target_wins] = LABEL_RAIN_NOISE if rain_rate > 0 else LABEL_CLEAN
        else:
            labels[target_wins] = np.where(noisy[target_wins], LABEL_RAIN_NOISE, LABEL_CLEAN)
        labels[drop_wins] = LABEL_RAIN_FALSE
        if media is not None:                       # fog / dust echo vs target and drops
            p_pub = np.where(target_wins, p_hard, np.where(drop_wins, best_p, -1.0))
            media.merge(out_r, out_i, labels, p_pub, pmin, self.intensity_output_scale)
        return out_r, out_i, labels


# =============================================================================
MODEL_CLASSES = {'legacy': LegacyModel, 'goodin2019': Goodin2019, 'alavi2025': Alavi2025,
                 'hybrid': HybridRain}


def make_model(name, params=None):
    """Build a model by name, taking the fields it knows from a flat params dict."""
    try:
        cls = MODEL_CLASSES[str(name).lower()]
    except KeyError:
        raise ValueError(f"rain_model must be one of {MODEL_NAMES}, got '{name}'")
    return _pick(cls, params)
