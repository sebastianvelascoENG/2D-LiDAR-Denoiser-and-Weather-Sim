"""Unit tests for thesis_rain_sim.rain_models (no ROS needed).

    cd ~/ros2_ws/src/thesis_rain_sim && python3 -m pytest test/ -v
    # or: colcon test --packages-select thesis_rain_sim && colcon test-result --verbose
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from thesis_rain_sim import rain_models as rm  # noqa: E402

N = 500
ANG_MIN = 0.0
ANG_INC = 2.0 * math.pi / N
R_MIN, R_MAX = 0.1, 12.0


def synthetic_scan(seed=0, n=N):
    """Mix of real surfaces 0.3-11.9 m and no-return beams (inf), like rain_yard."""
    g = np.random.default_rng(seed)
    r = g.uniform(0.3, 11.9, n)
    r[g.random(n) < 0.2] = np.inf
    return r


# -----------------------------------------------------------------------------
# legacy: frozen copy of the ORIGINAL rain_model_node.on_scan maths, kept
# verbatim. Do not edit - it is the regression reference.
# -----------------------------------------------------------------------------
def _frozen_legacy(ranges, r_min, r_max, rain_rate, rng, p):
    n = len(ranges)
    r = np.asarray(ranges, dtype=np.float64)
    alpha = p['alpha_k'] * (rain_rate ** p['alpha_p']) if rain_rate > 0 else 0.0
    finite = np.isfinite(r)
    hit = finite & (r >= r_min) & (r < r_max - 1e-3)
    out_r = np.full(n, math.inf, dtype=np.float64)
    out_i = np.zeros(n, dtype=np.float64)
    idx = np.flatnonzero(hit)
    if idx.size:
        rr = r[idx]
        inten = p['i0'] / np.square(rr) * np.exp(-2.0 * alpha * rr)
        inten *= (1.0 + rng.normal(0.0, p['intensity_noise_frac'], idx.size))
        inten = np.clip(inten, 0.0, None)
        p_drop = p['q_scatter'] * (1.0 - np.exp(-alpha * rr)) if alpha > 0 else 0.0
        detected = (inten >= p['min_detectable_intensity']) & (rng.random(idx.size) > p_drop)
        sigma_r = np.sqrt(p['range_sigma_base_m'] ** 2
                          + (p['k_rain_range'] * rain_rate * rr / r_max) ** 2)
        meas = rr + rng.normal(0.0, sigma_r)
        meas = np.clip(meas, r_min, r_max)
        det = idx[detected]
        out_r[det] = meas[detected]
        out_i[det] = inten[detected]
    p_false = min(p['p_false_k'] * rain_rate, 0.15)
    if p_false > 0.0:
        draw = rng.random(n) < p_false
        fi = np.flatnonzero(draw)
        if fi.size:
            hi = p['false_range_max_m'] * np.ones(fi.size)
            real = np.isfinite(out_r[fi])
            hi[real] = np.minimum(hi[real], out_r[fi][real])
            lo = np.minimum(p['false_range_min_m'], hi - 1e-2)
            frange = lo + (hi - lo) * rng.random(fi.size)
            finten = np.clip(
                p['false_intensity_mean'] * (1.0 + rng.normal(0.0, p['false_intensity_frac'], fi.size)),
                0.0, None)
            out_r[fi] = frange
            out_i[fi] = finten
    return out_r.astype(np.float32), out_i.astype(np.float32)


LEGACY_PARAMS = dict(i0=10000.0, intensity_noise_frac=0.05, min_detectable_intensity=8.0,
                     alpha_k=0.01, alpha_p=0.6, q_scatter=0.6, range_sigma_base_m=0.012,
                     k_rain_range=0.02, p_false_k=0.004, false_range_min_m=0.30,
                     false_range_max_m=4.0, false_intensity_mean=6.0, false_intensity_frac=0.5)


@pytest.mark.parametrize('rain_rate', [0.0, 6.0, 25.0, 50.0])
def test_legacy_bit_for_bit(rain_rate):
    model = rm.make_model('legacy', LEGACY_PARAMS)
    rng_ref, rng_new = np.random.default_rng(1), np.random.default_rng(1)
    for k in range(200):                       # a stream of scans on one RNG, like the node
        scan = synthetic_scan(k)
        ref_r, ref_i = _frozen_legacy(scan, R_MIN, R_MAX, rain_rate, rng_ref, LEGACY_PARAMS)
        out_r, out_i, _ = model.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, rain_rate, rng_new)
        # compare the float32 payload the node publishes, bit for bit
        assert np.array_equal(out_r.astype(np.float32).view(np.uint32), ref_r.view(np.uint32))
        assert np.array_equal(out_i.astype(np.float32).view(np.uint32), ref_i.view(np.uint32))


def test_legacy_bit_for_bit_recorded_rain_yard_scan():
    scan = np.load(os.path.join(os.path.dirname(__file__), 'data', 'rain_yard_scan_clean.npy'))
    model = rm.make_model('legacy', LEGACY_PARAMS)
    rng_ref, rng_new = np.random.default_rng(1), np.random.default_rng(1)
    for _ in range(100):
        ref_r, ref_i = _frozen_legacy(scan, R_MIN, R_MAX, 25.0, rng_ref, LEGACY_PARAMS)
        out_r, out_i, _ = model.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0, rng_new)
        assert np.array_equal(out_r.astype(np.float32).view(np.uint32), ref_r.view(np.uint32))
        assert np.array_equal(out_i.astype(np.float32).view(np.uint32), ref_i.view(np.uint32))


def test_legacy_defaults_match_yaml_defaults():
    assert rm.make_model('legacy') == rm.LegacyModel(**LEGACY_PARAMS)


def test_legacy_labels():
    scan = synthetic_scan(3)
    _, _, lab = rm.make_model('legacy').apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0,
                                              np.random.default_rng(1))
    assert set(np.unique(lab)) <= {1, 2, 3, 4}
    assert np.all(lab[~np.isfinite(scan)] != rm.LABEL_CLEAN)
    _, _, lab0 = rm.make_model('legacy').apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0,
                                               np.random.default_rng(1))
    assert set(np.unique(lab0)) <= {0, 2, 4}


# -----------------------------------------------------------------------------
# goodin2019
# -----------------------------------------------------------------------------
def test_goodin_alpha():                                                  # Eq. 5
    g = rm.Goodin2019()
    assert g.alpha(6.0) == pytest.approx(0.01 * 6.0 ** 0.6)
    assert g.alpha(6.0) == pytest.approx(0.029302, rel=1e-4)
    assert g.alpha(25.0) == pytest.approx(0.068986, rel=1e-4)
    assert g.alpha(0.0) == 0.0


def test_goodin_p_min():                                                  # Eq. 4
    assert rm.Goodin2019(z_max=12.0, z_max_reflectivity=0.9).p_min() == \
        pytest.approx(0.9 / (math.pi * 144.0))
    assert rm.Goodin2019().p_min() == pytest.approx(0.7 / (math.pi * 144.0))


def test_goodin_clear_weather_max_range_is_z_max():
    """A reference-reflectivity target sits exactly on the threshold at z_max."""
    g = rm.Goodin2019(default_reflectivity=0.7, z_max_reflectivity=0.7)
    assert g.relative_power(12.0, 0.0) == pytest.approx(g.p_min())


def test_goodin_sigma():                                                  # Eq. 9
    g = rm.Goodin2019()
    for z in (1.0, 5.0, 12.0):
        assert g.range_sigma(z, 0.0) == 0.0
        assert g.range_sigma(z, 6.0) == pytest.approx(0.02 * z * (1 - math.exp(-6.0)) ** 2)
        assert g.range_sigma(z, 25.0) == pytest.approx(0.02 * z, rel=1e-9)
    assert g.range_sigma(10.0, 1.0) == pytest.approx(0.2 * (1 - math.exp(-1)) ** 2)


def test_goodin_drop_at_threshold():
    g = rm.Goodin2019(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0)
    R = 25.0
    # find the range where P_n == P_min, then test just inside / outside it
    zs = np.linspace(0.5, 11.9, 200001)
    z_star = zs[np.argmin(np.abs(g.relative_power(zs, R) - g.p_min()))]
    scan = np.array([z_star - 0.01, z_star + 0.01])
    out, _, lab = g.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0))
    assert np.isfinite(out[0]) and lab[0] == rm.LABEL_RAIN_NOISE
    assert np.isinf(out[1]) and lab[1] == rm.LABEL_DROPPED
    # 25 mm/h, rho = 0.7: attenuated max range ~ 7.3 m
    assert 7.0 < z_star < 7.6


def test_goodin_zero_rain_only_sensor_noise():
    g = rm.Goodin2019()
    scan = synthetic_scan(1)
    out, inten, lab = g.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(2))
    hit = rm.clean_hits(scan, R_MIN, R_MAX)
    assert np.all(np.isfinite(out[hit])) and np.all(np.isinf(out[~hit]))   # nothing dropped
    assert set(np.unique(lab)) <= {rm.LABEL_CLEAN, rm.LABEL_NO_RETURN}
    err = out[hit] - scan[hit]
    assert np.std(err) == pytest.approx(g.sensor_sigma_extra, rel=0.2)
    g0 = rm.Goodin2019(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0)
    out0, _, _ = g0.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(2))
    assert np.array_equal(out0[hit], scan[hit])                            # exactly zero effect


def test_goodin_no_false_returns():
    g = rm.Goodin2019()
    for R in (6.0, 25.0, 50.0):
        _, _, lab = g.apply(synthetic_scan(4), R_MIN, R_MAX, ANG_MIN, ANG_INC, R,
                            np.random.default_rng(5))
        assert not np.any(lab == rm.LABEL_RAIN_FALSE)


def test_extra_sensor_sigma_adds_in_variance():
    assert rm.extra_sensor_sigma(0.012, 0.010) == pytest.approx(math.sqrt(0.012**2 - 0.010**2))
    assert rm.extra_sensor_sigma(0.012, 0.010) == pytest.approx(0.0066332, rel=1e-4)
    assert rm.extra_sensor_sigma(0.010, 0.012) == 0.0


# -----------------------------------------------------------------------------
# alavi2025
# -----------------------------------------------------------------------------
def test_alavi_alpha_table2():
    db = math.log(10) / 1e4
    assert rm.alpha_rain('continental', 6.0) == pytest.approx(1.076 * 6 ** 0.67 * db)
    assert rm.alpha_rain('continental', 6.0) == pytest.approx(8.2297e-4, rel=1e-3)
    assert rm.alpha_rain('continental', 25.0) == pytest.approx(2.1414e-3, rel=1e-3)
    assert rm.alpha_rain('tropical', 25.0) == pytest.approx(0.365 * 25 ** 0.63 * db)
    assert rm.alpha_rain('thunderstorm', 25.0) == pytest.approx(0.16 * 25 ** 0.74 / 1000)
    for t in rm.RAIN_EXTINCTION:
        assert rm.alpha_rain(t, 0.0) == 0.0
    with pytest.raises(ValueError):
        rm.alpha_rain('drizzle', 5.0)


def test_alavi_figure5_values_at_100mm_h():
    """Alavi Fig. 5 at 100 mm/h: continental ~5.4e-3, thunderstorm ~4.8e-3, tropical ~1.5e-3 /m."""
    assert rm.alpha_rain('continental', 100.0) == pytest.approx(5.4e-3, rel=0.03)
    assert rm.alpha_rain('thunderstorm', 100.0) == pytest.approx(4.8e-3, rel=0.03)
    assert rm.alpha_rain('tropical', 100.0) == pytest.approx(1.53e-3, rel=0.03)
    # Sec. 4.2 ordering: continental > thunderstorm > tropical
    for R in (2.0, 6.0, 25.0, 100.0):
        assert rm.alpha_rain('continental', R) > rm.alpha_rain('thunderstorm', R) > \
            rm.alpha_rain('tropical', R)


def test_alavi_beta():                                                    # Eq. 11
    a = rm.alpha_rain('continental', 25.0)
    assert rm.beta_rain_per_sr(a) == pytest.approx(a / 0.60 / (4 * math.pi))


def test_alavi_p_min_and_bins():
    m = rm.Alavi2025()
    assert m.p_min() == pytest.approx(0.7 / (math.pi * 144.0))
    assert m.bin_width == pytest.approx(rm.C_LIGHT * 5e-9 / 2)
    c = m.bin_centres(12.0)
    assert c[0] == pytest.approx(0.05 + m.bin_width / 2) and c[-1] >= 12.0 - m.bin_width


def test_alavi_overlap():
    assert rm.overlap([0.0, 0.05, 0.525, 1.0, 5.0], 0.05, 1.0) == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0])


def test_alavi_zero_rain_no_effect():
    m = rm.Alavi2025(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0)
    scan = synthetic_scan(7)
    out, _, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(1))
    hit = rm.clean_hits(scan, R_MIN, R_MAX)
    assert np.array_equal(out[hit], scan[hit]) and np.all(np.isinf(out[~hit]))
    assert set(np.unique(lab)) <= {rm.LABEL_CLEAN, rm.LABEL_NO_RETURN}


def test_alavi_mean_backscatter_below_threshold_at_6_and_25():
    """With beta/(4 pi), the expected (Eq. 6) rain echo never reaches P_min at 6 or 25 mm/h."""
    m = rm.Alavi2025()
    c = m.bin_centres(12.0)
    for R in (6.0, 25.0):
        assert np.max(m.mean_bin_power(c, R)) < m.p_min()
    mm = rm.Alavi2025(backscatter_mode='mean')
    _, _, lab = mm.apply(synthetic_scan(2), R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0,
                         np.random.default_rng(0))
    assert not np.any(lab == rm.LABEL_RAIN_FALSE)


def test_alavi_stochastic_false_returns_rare_and_near():
    m = rm.Alavi2025()
    rng = np.random.default_rng(11)
    frac, ranges = [], []
    for k in range(40):
        out, _, lab = m.apply(synthetic_scan(k), R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0, rng)
        f = lab == rm.LABEL_RAIN_FALSE
        frac.append(f.mean())
        ranges.extend(out[f])
    assert 0.0 < np.mean(frac) < 0.05
    assert max(ranges) < 3.0


def test_alavi_echo_mode_first_vs_strongest():
    """A rain echo close in front of a strong target: 'first' reports the echo."""
    # low threshold (z_max 40 m) so the mean rain echo is detectable in front of the target
    kw = dict(backscatter_mode='mean', sensor_sigma_total_m=0.0, simulator_sigma_m=0.0,
              pulse_width_s=10e-9, z_max=40.0)
    scan = np.array([4.0])
    R = 100.0
    first = rm.Alavi2025(echo_mode='first', **kw)
    strong = rm.Alavi2025(echo_mode='strongest', **kw)
    c = first.bin_centres(12.0)
    assert np.max(first.mean_bin_power(c[c < 4.0], R)) >= first.p_min()   # precondition
    of, _, lf = first.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0))
    os_, _, ls = strong.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0))
    assert lf[0] == rm.LABEL_RAIN_FALSE and of[0] < 4.0
    assert ls[0] == rm.LABEL_CLEAN and os_[0] == pytest.approx(4.0)


def test_alavi_heaviside_no_echo_behind_target():
    m = rm.Alavi2025(pulse_width_s=20e-9, echo_mode='first')
    rng = np.random.default_rng(3)
    scan = np.full(N, 2.0)
    for _ in range(20):
        out, _, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 100.0, rng)
        assert np.all(out[lab == rm.LABEL_RAIN_FALSE] < 2.0)


def test_incidence_cosine():
    # a straight wall y = 2 seen from the origin: cos(incidence) = 2 / r
    ang = ANG_MIN + ANG_INC * np.arange(N)
    r = np.full(N, np.inf)
    front = (ang > math.radians(30)) & (ang < math.radians(150))
    r[front] = 2.0 / np.sin(ang[front])
    cos_inc = rm.incidence_cosine(r, ANG_MIN, ANG_INC, window=2)
    inner = front & (ang > math.radians(35)) & (ang < math.radians(145))
    assert cos_inc[inner] == pytest.approx(2.0 / r[inner], abs=1e-6)
    assert np.all(cos_inc[~front] == 1.0)


def test_incidence_reduces_power_only_when_enabled():
    scan = np.full(N, 5.0)          # circle around the sensor: normal incidence everywhere
    on = rm.Alavi2025(use_incidence_angle=True, sensor_sigma_total_m=0, simulator_sigma_m=0)
    _, i_on, _ = on.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(0))
    off = rm.Alavi2025(sensor_sigma_total_m=0, simulator_sigma_m=0)
    _, i_off, _ = off.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(0))
    assert i_on == pytest.approx(i_off, rel=1e-6)


# -----------------------------------------------------------------------------
# hybrid (LISA drops + alavi2025 sensor)
# -----------------------------------------------------------------------------
RAIN_YARD = os.path.join(os.path.dirname(__file__), 'data', 'rain_yard_scan_clean.npy')


def _mp_numeric_alpha(m, R, d_lo=0.0, d_hi=20.0, n=400001):
    """Eq. 3 with Q_ext = 2, integrated numerically (trapezoid) over [d_lo, d_hi] mm."""
    d = np.linspace(d_lo, d_hi, n)
    f = math.pi / 4.0 * d ** 2 * 2.0 * m.mp_n0 * np.exp(-m.mp_lambda(R) * d)
    return float(np.sum((f[1:] + f[:-1]) * np.diff(d)) / 2.0) * 1e-6


def test_hybrid_extinction():
    m = rm.HybridRain()
    for R in (6.0, 25.0):
        lam = 4.1 * R ** -0.21
        closed = math.pi * 8000.0 / lam ** 3 * 1e-6
        assert m.alpha(R) == pytest.approx(closed)
        assert m.alpha(R) == pytest.approx(_mp_numeric_alpha(m, R), rel=0.01)
        # integrating from D_min instead of 0 is negligible (< 0.1 %)
        assert _mp_numeric_alpha(m, R, d_lo=m.min_drop_diameter_mm) == \
            pytest.approx(m.alpha(R), rel=1e-3)
        # sanity check against Alavi Table 2 continental: same order, not equal
        alavi = rm.alpha_rain('continental', R)
        print(f'R={R:g} mm/h: alpha_dsd={m.alpha(R):.4e} /m, alavi continental={alavi:.4e} /m, '
              f'ratio {m.alpha(R) / alavi:.2f}')
        assert 0.5 < m.alpha(R) / alavi < 2.0
    assert m.alpha(0.0) == 0.0 and m.drop_density(0.0) == 0.0
    sens = rm.HybridRain(extinction_source='alavi')
    assert sens.alpha(25.0) == rm.alpha_rain('continental', 25.0)


def test_hybrid_drop_density():
    m = rm.HybridRain()
    lam = 4.1 * 25.0 ** -0.21
    assert m.drop_density(25.0) == pytest.approx(8000.0 / lam * math.exp(-lam * 0.05))


def test_hybrid_calm_is_sensor_noise_only():
    m = rm.HybridRain()
    scan = synthetic_scan(1)
    hit = rm.clean_hits(scan, R_MIN, R_MAX)
    out, _, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(2))
    assert not np.any(lab == rm.LABEL_RAIN_FALSE)
    assert set(np.unique(lab)) <= {rm.LABEL_CLEAN, rm.LABEL_NO_RETURN}
    assert np.all(np.isfinite(out[hit])) and np.all(np.isinf(out[~hit]))
    assert np.std(out[hit] - scan[hit]) == pytest.approx(m.sensor_sigma_extra, rel=0.2)
    # same RNG draws as alavi2025 in calm weather: identical output
    a_out, _, _ = rm.Alavi2025().apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0,
                                       np.random.default_rng(2))
    assert np.array_equal(out, a_out)
    m0 = rm.HybridRain(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0)
    out0, _, _ = m0.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 0.0, np.random.default_rng(2))
    assert np.array_equal(out0[hit], scan[hit])
    beam, _, _ = m.sample_drops(np.full(N, 12.0), 0.0, np.random.default_rng(0))
    assert beam.size == 0


@pytest.mark.parametrize('exit_d', [0.0, 0.004])
def test_hybrid_drop_counts_and_positions(exit_d):
    m = rm.HybridRain(beam_exit_diameter_m=exit_d)
    nb, L, R = 20000, 12.0, 25.0
    beam, r, d = m.sample_drops(np.full(nb, L), R, np.random.default_rng(4))
    lam = m.drop_density(R) * float(m.beam_volume(L))
    if exit_d == 0.0:
        assert float(m.beam_volume(L)) == pytest.approx(
            math.pi / 3.0 * L * (L * math.tan(3e-3) / 2.0) ** 2)     # Alg. 1 bvol
    counts = np.bincount(beam, minlength=nb)
    assert counts.mean() == pytest.approx(lam, abs=4.0 * math.sqrt(lam / nb))
    assert counts.var() == pytest.approx(lam, rel=0.05)                # Poisson
    assert np.all((r >= 0.0) & (r <= L))
    # positions follow the beam area: E[r] = int r A(r) dr / int A(r) dr
    t = math.tan(m.beam_divergence_rad)
    x = np.linspace(0.0, L, 100001)
    area = (exit_d + t * x) ** 2
    expect = np.sum(x * area) / np.sum(area)
    assert r.mean() == pytest.approx(expect, abs=4.0 * r.std() / math.sqrt(r.size))
    # diameters: shifted exponential with rate Lambda, clipped at 6 mm
    assert d.min() >= 0.05 and d.max() <= 6.0
    assert (d - 0.05).mean() == pytest.approx(1.0 / m.mp_lambda(R), rel=0.02)


def test_hybrid_heaviside_no_false_return_at_or_behind_target():
    m = rm.HybridRain()
    rng = np.random.default_rng(3)
    n_false = 0
    for k in range(200):
        scan = np.full(N, 2.0) if k % 2 else synthetic_scan(k)
        out, _, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 100.0, rng)
        f = (lab == rm.LABEL_RAIN_FALSE) & np.isfinite(scan)
        assert np.all(out[f] < scan[f])
        n_false += int(f.sum())
    assert n_false > 0                                          # the check was exercised


def _inject(m, drops):
    """Replace drop sampling with a fixed list of (beam, range m, diameter mm)."""
    b, r, d = (np.array(x, dtype=float) for x in zip(*drops)) if drops else ([], [], [])
    m.sample_drops = lambda length, rain_rate, rng: (np.asarray(b, np.int64),
                                                     np.asarray(r, float), np.asarray(d, float))
    return m


def test_hybrid_replacement_rule():
    kw = dict(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0)
    R = 25.0
    probe = rm.HybridRain(**kw)
    pmin = probe.p_min()
    # beam 0: target 5 m (strong). beam 1: dark target 10 m, below P_min.
    # beam 2: no surface. beam 3: target 5 m.
    scan = np.array([5.0, 10.0, np.inf, 5.0])
    refl = np.array([0.7, 0.05, 0.7, 0.7])
    p_t = probe.hard_power(scan[[0, 1]], R, refl[[0, 1]])
    assert p_t[0] >= pmin > p_t[1]                                 # preconditions

    def drop_at(power_target, r=0.25):
        """Diameter that gives a drop at range r the requested power."""
        base = probe.drop_power(r, 1e3 * probe.beam_diameter(r), R)   # area factor 1
        return 1e3 * float(probe.beam_diameter(r)) * math.sqrt(power_target / base)

    strong = drop_at(2.0 * p_t[0])               # beats the 5 m target
    weak = drop_at(0.5 * p_t[0])                 # >= P_min but below the 5 m target
    sub = drop_at(0.5 * pmin)                    # below P_min
    for dd, want in ((strong, 2.0 * p_t[0]), (weak, 0.5 * p_t[0]), (sub, 0.5 * pmin)):
        assert probe.drop_power(0.25, dd, R) == pytest.approx(want)
    assert probe.drop_power(0.25, weak, R) >= pmin
    m = _inject(rm.HybridRain(**kw), [(0, 0.25, strong), (3, 0.25, weak),
                                      (1, 0.25, sub), (2, 0.25, sub)])
    out, inten, lab = m.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0),
                       reflectivity=refl)
    assert lab[0] == rm.LABEL_RAIN_FALSE and out[0] == pytest.approx(0.25)
    assert inten[0] == pytest.approx(2.0 * p_t[0] * m.intensity_output_scale)
    assert lab[3] == rm.LABEL_RAIN_NOISE and out[3] == pytest.approx(5.0, abs=0.05)
    assert lab[1] == rm.LABEL_DROPPED and np.isinf(out[1])
    assert lab[2] == rm.LABEL_NO_RETURN and np.isinf(out[2])
    # a drop >= P_min wins when the target itself is below P_min
    m = _inject(rm.HybridRain(**kw), [(1, 0.25, weak), (2, 0.25, weak)])
    out, _, lab = m.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0),
                       reflectivity=refl)
    assert lab[1] == rm.LABEL_RAIN_FALSE and lab[2] == rm.LABEL_RAIN_FALSE
    assert out[1] == pytest.approx(0.25) and out[2] == pytest.approx(0.25)
    # drops inside r_min are discarded
    m = _inject(rm.HybridRain(**kw), [(0, 0.05, 6.0)])
    _, _, lab = m.apply(scan, R_MIN, R_MAX, 0.0, 0.01, R, np.random.default_rng(0),
                       reflectivity=refl)
    assert lab[0] == rm.LABEL_RAIN_NOISE


def test_hybrid_rain_noise():
    m = rm.HybridRain()
    z = np.array([1.0, 6.0, 11.0])
    assert np.all(m.rain_range_sigma(z, 0.0) == 0.0)
    s6, s25 = m.rain_range_sigma(z, 6.0), m.rain_range_sigma(z, 25.0)
    assert np.all(np.diff(s6) > 0) and np.all(np.diff(s25) > 0)        # grows with range
    assert np.all(s25 > s6)                                             # grows with rain
    # Eq. 7 with SNR = P / P_min
    p = m.hard_power(11.0, 25.0)
    assert m.range_sigma(p) == pytest.approx(0.06 / math.sqrt(2.0 * p / m.p_min()))
    # the applied noise has that std-dev (no drops, no sensor noise)
    q = rm.HybridRain(sensor_sigma_total_m=0.0, simulator_sigma_m=0.0, min_drop_diameter_mm=50.0)
    scan = np.full(20000, 11.0)
    out, _, lab = q.apply(scan, R_MIN, R_MAX, 0.0, 1e-4, 25.0, np.random.default_rng(6))
    assert np.all(lab == rm.LABEL_RAIN_NOISE)
    assert np.std(out - 11.0) == pytest.approx(float(q.rain_range_sigma(11.0, 25.0)), rel=0.03)


def test_hybrid_overlap_reduces_near_range_false_returns():
    scan = np.load(RAIN_YARD)
    near = {}
    for use in (True, False):
        m = rm.HybridRain(use_overlap=use)
        rng = np.random.default_rng(8)
        cnt = 0
        for _ in range(300):
            out, _, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0, rng)
            cnt += int(np.sum((lab == rm.LABEL_RAIN_FALSE) & (out < 1.0)))
        near[use] = cnt
    print(f'near-range (<1 m) false returns in 300 scans: overlap on {near[True]}, '
          f'off {near[False]}')
    assert near[False] >= near[True]
    assert near[False] > 0


def test_hybrid_yaml_and_node_defaults_match_dataclass():
    yaml = pytest.importorskip('yaml')
    here = os.path.dirname(__file__)
    with open(os.path.join(here, '..', 'config', 'weather_params.yaml')) as f:
        p = yaml.safe_load(f)['/**']['ros__parameters']
    m = rm.HybridRain()
    for fld in rm.fields(rm.HybridRain):
        assert fld.name in p, f'{fld.name} missing from weather_params.yaml'
        assert p[fld.name] == pytest.approx(getattr(m, fld.name)) \
            if isinstance(p[fld.name], float) else p[fld.name] == getattr(m, fld.name)
    with open(os.path.join(here, '..', 'scripts', 'weather_model_node.py')) as f:
        node_src = f.read()
    for fld in rm.fields(rm.HybridRain):
        assert f"'{fld.name}':" in node_src, f'{fld.name} missing from PARAM_DEFAULTS'


# -----------------------------------------------------------------------------
# all models
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('name', rm.MODEL_NAMES)
@pytest.mark.parametrize('rain_rate', [0.0, 6.0, 25.0])
def test_determinism(name, rain_rate):
    m = rm.make_model(name)
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        outs.append([m.apply(synthetic_scan(k), R_MIN, R_MAX, ANG_MIN, ANG_INC, rain_rate, rng)
                     for k in range(5)])
    for a, b in zip(*outs):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


@pytest.mark.parametrize('name', rm.MODEL_NAMES)
def test_output_shapes_and_rep117(name):
    m = rm.make_model(name)
    scan = synthetic_scan(9)
    out, inten, lab = m.apply(scan, R_MIN, R_MAX, ANG_MIN, ANG_INC, 25.0, np.random.default_rng(0))
    assert out.shape == inten.shape == lab.shape == scan.shape
    fin = np.isfinite(out)
    assert np.all((out[fin] >= R_MIN) & (out[fin] <= R_MAX))
    assert np.all(np.isposinf(out[~fin]))
    assert np.all(inten[~fin] == 0.0)
    assert np.all(fin[lab <= 1]) and np.all(fin[lab == 3]) and np.all(~fin[lab == 2])


def test_make_model_rejects_unknown():
    with pytest.raises(ValueError):
        rm.make_model('fog2030')
    with pytest.raises(ValueError):
        rm.make_model('alavi2025', {'echo_mode': 'last'})
    with pytest.raises(ValueError):
        rm.make_model('hybrid', {'extinction_source': 'mie'})
    assert isinstance(rm.make_model('hybrid'), rm.HybridRain)
