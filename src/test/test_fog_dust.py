"""Unit tests for fog and dust (thesis_rain_sim.fog_dust, dust_geometry, weather). No ROS.

    cd ~/ros2_ws/src/thesis_rain_sim && python3 -m pytest test/ -v

The reference-number tests use a second sensor (10 % target at 6 m, 20 ns pulse,
overlap R2 = 0.5 m, rho = 0.5) and pass REF_SENSOR explicitly. The
package itself runs fog and dust on the rain models' sensor (12 m at 70 %,
5 ns, R2 = 1.0 m, rho = 0.7); tests marked "rain sensor" use those defaults.
"""
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from thesis_rain_sim import fog_dust as fd  # noqa: E402
from thesis_rain_sim import rain_models as rm  # noqa: E402
from thesis_rain_sim import weather as wx  # noqa: E402
from thesis_rain_sim.dust_geometry import (  # noqa: E402
    beam_angles, cloud_chords, compose_pose, ray_box_chords, ray_circle_chords,
    yaw_from_quaternion,
)

HERE = os.path.dirname(__file__)
RMIN, RMAX = 0.05, 12.0
CLOUD = [dict(shape='circle', frame='sensor', x=2.0, y=0.0, radius=0.75)]
EVERYWHERE = [dict(shape='box', frame='sensor', x=0.0, y=0.0, size_x=1e4, size_y=1e4)]

# Reference sensor for the published-number tests (not the package's sensor)
REF_SENSOR = dict(z_max=6.0, z_max_reflectivity=0.10, default_reflectivity=0.5,
                   pulse_width_s=20e-9, overlap_r1_m=0.05, overlap_r2_m=0.5)
REF_P_MIN = 0.10 / math.pi / 36.0


def media(fog=None, dust=None, clouds=CLOUD, **params):
    return fd.WeatherMedia(fd.FogDustParams(**params), fog, dust, clouds)


def run(model, ranges, wm, rain_rate=0.0, seed=0, angles=None, wseed=1, world_pose=None):
    """One scan through `model` with fog / dust media `wm` (None = rain only)."""
    r = np.asarray(ranges, dtype=float)
    a_min, a_inc = (0.0, 2 * math.pi / r.size) if angles is None else angles
    sm = None
    if wm is not None:
        sm = wm.scan(r, rm.clean_hits(r, RMIN, RMAX), RMIN, RMAX, a_min, a_inc,
                     model.alpha(rain_rate), np.random.default_rng(wseed), world_pose)
    return model.apply(r, RMIN, RMAX, a_min, a_inc, rain_rate, np.random.default_rng(seed),
                       media=sm)


def ref_goodin(**kw):
    """goodin2019 at R = 0 is the bare sensor; with REF_SENSOR it is the reference sensor."""
    s = {k: v for k, v in REF_SENSOR.items() if k in ('z_max', 'z_max_reflectivity',
                                                        'default_reflectivity')}
    s.update(kw)
    return rm.make_model('goodin2019', s)


def ref_media(fog=None, dust=None, clouds=CLOUD, **kw):
    s = {k: REF_SENSOR[k] for k in ('pulse_width_s', 'overlap_r1_m', 'overlap_r2_m')}
    s.update(kw)
    return media(fog, dust, clouds, **s)


# -----------------------------------------------------------------------------
# dust geometry
# -----------------------------------------------------------------------------
def test_circle_head_on_and_off_axis():
    r_in, r_out = ray_circle_chords(0, 0, np.array([1.0]), np.array([0.0]), 2.0, 0.0, 0.75)
    assert r_in[0] == pytest.approx(1.25) and r_out[0] == pytest.approx(2.75)
    th = math.radians(10)
    r_in, r_out = ray_circle_chords(0, 0, np.array([math.cos(th)]), np.array([math.sin(th)]),
                                    2.0, 0.0, 0.75)
    d = 2.0 * math.sin(th)
    assert r_out[0] - r_in[0] == pytest.approx(2 * math.sqrt(0.75 ** 2 - d ** 2))


def test_circle_misses_behind_tangent_inside():
    r_in, _ = ray_circle_chords(0, 0, np.array([0.0, -1.0]), np.array([1.0, 0.0]), 2, 0, 0.75)
    assert np.isnan(r_in).all()
    r_in, _ = ray_circle_chords(0, 0.75, np.array([1.0]), np.array([0.0]), 2.0, 0.0, 0.75)
    assert np.isnan(r_in[0])                     # grazing
    r_in, r_out = ray_circle_chords(2.0, 0.0, np.array([1.0]), np.array([0.0]), 2.0, 0.0, 0.75)
    assert r_in[0] == 0.0 and r_out[0] == pytest.approx(0.75)  # sensor inside the cloud


def test_box_chords():
    dx = np.array([1.0, math.cos(math.radians(10)), 0.0, -1.0])
    dy = np.array([0.0, math.sin(math.radians(10)), 1.0, 0.0])
    r_in, r_out = ray_box_chords(0, 0, dx, dy, 2.0, 0.0, 2.0, 1.0)
    assert r_in[0] == pytest.approx(1.0) and r_out[0] == pytest.approx(3.0)
    assert r_out[1] == pytest.approx(0.5 / math.sin(math.radians(10)))
    assert np.isnan(r_in[2]) and np.isnan(r_in[3])
    r_in, r_out = ray_box_chords(0, 0.2, np.array([1.0]), np.array([0.0]), 2.0, 0.0, 2.0, 1.0)
    assert r_in[0] == pytest.approx(1.0) and r_out[0] == pytest.approx(3.0)


def test_cloud_chords_world_pose():
    # Sensor at (2, -2) facing +y (yaw 90 deg): the world circle at (2, 0) is dead ahead.
    ch = cloud_chords(CLOUD, np.array([0.0, math.pi / 2]), origin=(2.0, -2.0), yaw=math.pi / 2)
    assert ch[0, 0, 0] == pytest.approx(1.25) and np.isnan(ch[1, 0, 0])
    with pytest.raises(ValueError):
        cloud_chords([dict(shape='blob', x=0, y=0)], np.zeros(1))
    assert yaw_from_quaternion(0, 0, math.sin(0.3), math.cos(0.3)) == pytest.approx(0.6)
    x, y, yaw = compose_pose((1.0, 2.0, math.pi / 2), (-0.064, 0.0, 0.0))
    assert (x, y, yaw) == pytest.approx((1.0, 2.0 - 0.064, math.pi / 2))


# -----------------------------------------------------------------------------
# coefficients
# -----------------------------------------------------------------------------
def test_coefficients():
    assert fd.alpha_fog(100) == pytest.approx(0.0391, abs=1e-3)
    assert fd.alpha_dust(100) == pytest.approx(0.0488, abs=1e-3)
    assert fd.alpha_fog(100, 'mor') == pytest.approx(0.03)
    assert fd.beta_fog(100) == pytest.approx(0.046 / 100)
    assert fd.beta_fog(100, 'alavi_raw') == pytest.approx(3.912 / 100 / 1.44)
    assert fd.beta_dust(100, 'alavi_raw') == pytest.approx(5.38 * 100 ** -1.016)
    for v in (5, 20, 100):
        assert fd.beta_fog(v, 'alavi') == pytest.approx(fd.beta_fog(v, 'alavi_raw') / (4 * math.pi))
        assert fd.beta_dust(v) == pytest.approx(fd.beta_dust(v, 'alavi_raw') / (4 * math.pi))
    assert fd.alpha_fog(50) / fd.beta_fog(50, 'alavi') == pytest.approx(18.1, abs=0.05)
    assert fd.alpha_dust(50) / fd.beta_dust(50) == pytest.approx(12.3, abs=0.05)
    for bad in (lambda: fd.alpha_fog(0), lambda: fd.beta_fog(10, 'nope'),
                lambda: fd.beta_dust(10, 'rasshofer'),
                lambda: fd.FogDustParams(soft_integration_points=100).validate(),
                lambda: fd.FogDustParams(overlap_r1_m=1.0, overlap_r2_m=0.5).validate()):
        with pytest.raises(ValueError):
            bad()


def test_simpson_exact_for_cubic():
    x = np.linspace(0, 2, 11)
    assert (fd.simpson_weights(11) * x ** 3).sum() * 0.2 == pytest.approx(4.0)


# -----------------------------------------------------------------------------
# reference numbers (reference sensor)
# -----------------------------------------------------------------------------
TABLE = [  # V, fog max range, dust (uniform) max range, on REF_SENSOR
    (300, 5.6, 5.5), (150, 5.2, 5.1), (100, 4.9, 4.8), (50, 4.3, 4.0),
    (20, 3.2, 2.9), (10, 2.4, 2.1), (5, 1.7, 1.4),
]


@pytest.mark.parametrize('vis,fog_expected,dust_expected', TABLE)
def test_ref_max_range_table(vis, fog_expected, dust_expected):
    model = ref_goodin(default_reflectivity=0.10)
    r0 = np.arange(0.1, 6.5, 0.002)
    for wm, expected in ((ref_media(fog=vis, soft_gain=0.0), fog_expected),
                         (ref_media(dust=vis, clouds=EVERYWHERE, soft_gain=0.0), dust_expected)):
        _, _, lab = run(model, r0, wm)
        assert not np.isin(lab, (rm.LABEL_FOG_FALSE, rm.LABEL_DUST_FALSE)).any()
        assert r0[lab == rm.LABEL_CLEAN].max() == pytest.approx(expected, abs=0.1)


def _soft_peak(wm, rain_alpha=0.0, p_min=REF_P_MIN):
    uni = fd.UniformMedium(wm.fog_alpha + rain_alpha, wm.fog_beta)
    p, r = wm.soft.uniform_lookup(uni, RMIN, RMAX, np.array([RMAX]))
    return p[0] / p_min, float(wm.soft.reported_range(r, RMIN)[0])


def test_ref_sanity_magnitudes():
    ratio, rep = _soft_peak(ref_media(fog=100))
    assert ratio == pytest.approx(2.3, rel=0.10) and rep == pytest.approx(0.5, abs=0.1)
    assert _soft_peak(ref_media(fog=20))[0] == pytest.approx(10.0, rel=0.10)
    for vis in (100, 20):     # Alavi's beta as printed is ~60x Rasshofer, per sr 4.7x
        ras = _soft_peak(ref_media(fog=vis))[0]
        assert _soft_peak(ref_media(fog=vis, fog_beta_model='alavi_raw'))[0] / ras \
            == pytest.approx(60, rel=0.10)
        assert _soft_peak(ref_media(fog=vis, fog_beta_model='alavi'))[0] / ras \
            == pytest.approx(4.7, rel=0.02)


def test_ref_fog_walls_and_open_beams():
    model = ref_goodin()
    _, _, lab = run(model, [2.0, 4.0, np.inf], ref_media(fog=100, jitter_exponent_max=0.0))
    assert list(lab) == [rm.LABEL_CLEAN, rm.LABEL_CLEAN, rm.LABEL_FOG_FALSE]
    out, _, lab = run(model, [2.0, 4.0], ref_media(fog=20, jitter_exponent_max=0.0))
    assert lab[0] == rm.LABEL_CLEAN and out[0] == pytest.approx(2.0, abs=0.05)
    assert lab[1] == rm.LABEL_FOG_FALSE and out[1] == pytest.approx(0.4, abs=0.1)


@pytest.mark.parametrize('vis', [100, 20])
def test_ref_fog_false_returns_open_vs_near_wall(vis):
    model = ref_goodin()
    _, _, lab_open = run(model, np.full(1000, np.inf), ref_media(fog=vis))
    out, _, lab_wall = run(model, np.random.default_rng(0).uniform(0.3, 1.5, 1000),
                           ref_media(fog=vis))
    assert np.mean(lab_open == rm.LABEL_FOG_FALSE) > 0.95
    assert np.mean(lab_wall == rm.LABEL_FOG_FALSE) < 0.05


def test_ref_dust_front_face_and_density():
    model = ref_goodin()
    angles = np.radians(np.arange(-15, 16, 1.0))
    r_in = cloud_chords(CLOUD, angles)[:, 0, 0]
    through = ~np.isnan(r_in)
    geo = (angles[0], angles[1] - angles[0])
    for vis in (20, 5):
        for walls in (np.full(angles.size, 4.0), np.full(angles.size, np.inf)):
            out, _, lab = run(model, walls, ref_media(dust=vis, jitter_exponent_max=0.0),
                              angles=geo)
            dust = lab == rm.LABEL_DUST_FALSE
            assert dust[through].any() and not dust[~through].any()
            assert np.all(np.abs(out[dust] - r_in[dust]) < 0.5)
            if vis == 5:
                assert dust[through].all()
    # Graded: a wall 3 m away behind the cloud is seen in dust_light, lost in dust_heavy.
    one = (0.0, 0.01)
    assert run(model, [3.0], ref_media(dust=20, jitter_exponent_max=0.0), angles=one)[2][0] \
        == rm.LABEL_CLEAN
    assert run(model, [3.0], ref_media(dust=5, jitter_exponent_max=0.0), angles=one)[2][0] \
        == rm.LABEL_DUST_FALSE
    # Extinction only: the dense cloud hides a 4 m wall, a 1 m wall in front is untouched.
    assert run(model, [4.0], ref_media(dust=5, soft_gain=0.0), angles=one)[2][0] \
        == rm.LABEL_DROPPED
    assert run(model, [1.0], ref_media(dust=5), angles=one)[2][0] == rm.LABEL_CLEAN


# -----------------------------------------------------------------------------
# dust profile: media add
# -----------------------------------------------------------------------------
def test_dust_profile_merges_overlapping_clouds():
    prof = fd.DustProfile(np.array([[1.0, 3.0], [2.0, 4.0], [np.nan, np.nan]]), 0.3, 0.1)
    assert prof.tau(5.0) == pytest.approx(0.3 * 3)
    tau, beta = prof.tau_beta(np.array([0.5, 2.5, 4.5]))
    assert tau == pytest.approx([0.0, 0.3 * 1.5, 0.9]) and list(beta) == [0.0, 0.1, 0.0]


def test_fog_and_dust_optical_depths_add():
    wm = media(fog=50, dust=20)
    r = np.full(360, 5.0)
    sm = wm.scan(r, np.ones(360, bool), RMIN, RMAX, -math.pi, math.radians(1.0), 0.0,
                 np.random.default_rng(0))
    i0 = int(np.argmin(np.abs(beam_angles(-math.pi, math.radians(1.0), 360))))
    assert sm.tau(i0, 5.0) == pytest.approx(fd.alpha_fog(50) * 5 + fd.alpha_dust(20) * 1.5)
    assert sm.tau(i0 + 90, 5.0) == pytest.approx(fd.alpha_fog(50) * 5)   # misses the cloud
    assert sm.tau(np.array([i0, i0 + 90]), np.array([1.0, 1.0])) == \
        pytest.approx([fd.alpha_fog(50)] * 2)                            # before the cloud


# -----------------------------------------------------------------------------
# switches: rain / fog / dust independently
# -----------------------------------------------------------------------------
def test_off_is_none_and_rain_only_untouched():
    assert media().scan(np.ones(10), np.ones(10, bool), RMIN, RMAX, 0, 0.1, 0.0,
                        np.random.default_rng(0)) is None
    scan = np.random.default_rng(3).uniform(0.3, 11.9, 500)
    for name in ('goodin2019', 'alavi2025', 'hybrid'):
        m = rm.make_model(name)
        a = m.apply(scan, RMIN, RMAX, 0, 0.01, 25.0, np.random.default_rng(1))
        b = m.apply(scan, RMIN, RMAX, 0, 0.01, 25.0, np.random.default_rng(1), media=None)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_legacy_rejects_fog_and_dust():
    sm = media(fog=20).scan(np.ones(4), np.ones(4, bool), RMIN, RMAX, 0, 0.1, 0.0,
                            np.random.default_rng(0))
    with pytest.raises(ValueError, match='power-based'):
        rm.make_model('legacy').apply(np.ones(4), RMIN, RMAX, 0, 0.1, 0.0,
                                      np.random.default_rng(0), media=sm)


def test_dust_on_needs_clouds():
    with pytest.raises(ValueError):
        media(dust=5, clouds=[])


def test_weather_levels_and_labels():
    assert wx.visibility_for('fog', 'off') is None
    assert wx.visibility_for('fog', 'light') == 100.0 and wx.visibility_for('fog', 'heavy') == 50.0
    assert wx.visibility_for('dust', 'light') == 20.0 and wx.visibility_for('dust', 'heavy') == 5.0
    assert wx.visibility_for('fog', 'heavy', 35.0) == 35.0
    assert wx.visibility_for('fog', 'off', 35.0) is None      # an override never switches it on
    assert wx.weather_label('heavy', 'off', 'off') == 'heavy'  # rain-only labels unchanged
    assert wx.weather_label('calm', 'off', 'off') == 'calm'
    assert wx.weather_label('calm', 'heavy', 'off') == 'fog_heavy'
    assert wx.weather_label('light', 'off', 'heavy') == 'rain_light+dust_heavy'
    for r in wx.RAIN_LEVELS:
        for f in wx.FOG_LEVELS:
            for d in wx.DUST_LEVELS:
                assert wx.parse_weather_label(wx.weather_label(r, f, d)) == (r, f, d)
    assert wx.parse_weather_label('fog_light+heavy') == ('heavy', 'light', 'off')
    for bad in ('fog_thick', 'snow_light', 'fog_light+fog_heavy', 'heavy+rain_light'):
        with pytest.raises(ValueError):
            wx.parse_weather_label(bad)
    assert wx.resolve_rain('', '') == 'calm'
    assert wx.resolve_rain('', 'heavy') == 'heavy'            # deprecated alias
    assert wx.resolve_rain('light', 'light') == 'light'
    with pytest.raises(ValueError):
        wx.resolve_rain('light', 'heavy')
    with pytest.raises(ValueError):
        wx.resolve_rain('drizzle')


# -----------------------------------------------------------------------------
# combined with rain (rain sensor)
# -----------------------------------------------------------------------------
def test_rain_and_fog_extinction_add():
    m = rm.make_model('goodin2019')
    r0 = np.arange(0.5, 11.9, 0.002)
    _, _, lab = run(m, r0, media(fog=50, soft_gain=0.0), rain_rate=25.0)
    alpha = m.alpha(25.0) + fd.alpha_fog(50)
    p = 0.7 / math.pi / r0 ** 2 * np.exp(-2 * alpha * r0)
    expected = r0[p >= m.p_min()].max()
    assert r0[lab == rm.LABEL_RAIN_NOISE].max() == pytest.approx(expected, abs=0.005)


def test_rain_attenuates_the_fog_echo():
    wm = media(fog=20)
    p_min = rm.make_model('hybrid').p_min()
    dry = _soft_peak(wm, 0.0, p_min)[0]
    wet = _soft_peak(wm, rm.make_model('goodin2019').alpha(50.0), p_min)[0]
    assert wet < dry


@pytest.mark.parametrize('name', ['goodin2019', 'alavi2025', 'hybrid'])
@pytest.mark.parametrize('rain_rate', [0.0, 25.0])
def test_output_contract_with_fog_and_dust(name, rain_rate):
    m = rm.make_model(name)
    scan = np.random.default_rng(9).uniform(0.3, 11.9, 500)
    scan[::5] = np.inf
    out, inten, lab = run(m, scan, media(fog=20, dust=5), rain_rate=rain_rate)
    fin = np.isfinite(out)
    assert out.shape == inten.shape == lab.shape == scan.shape
    assert np.all((out[fin] >= RMIN) & (out[fin] <= RMAX))
    assert np.all(np.isposinf(out[~fin])) and np.all(inten[~fin] == 0.0)
    assert np.all(fin[np.isin(lab, (0, 1, 3, 5, 6))]) and np.all(~fin[np.isin(lab, (2, 4))])
    assert np.all(inten[fin] >= m.p_min() * m.intensity_output_scale * (1 - 1e-9))
    assert (lab == rm.LABEL_FOG_FALSE).any() and (lab == rm.LABEL_DUST_FALSE).any()
    assert set(np.unique(lab)) <= {0, 1, 2, 3, 4, 5, 6}


def test_rain_sensor_fog_levels():
    # fog light (V = 100 m) makes no false returns on the rain sensor: its peak is
    # below P_min. fog heavy does, on open beams; walls within ~3 m survive.
    m = rm.make_model('hybrid')
    assert _soft_peak(media(fog=100), 0.0, m.p_min())[0] < 1.0
    assert _soft_peak(media(fog=20), 0.0, m.p_min())[0] > 1.0
    _, _, lab = run(m, np.full(500, np.inf), media(fog=100))
    assert not (lab == rm.LABEL_FOG_FALSE).any()
    _, _, lab = run(m, np.full(500, np.inf), media(fog=20))
    assert (lab == rm.LABEL_FOG_FALSE).mean() > 0.95
    _, _, lab = run(m, np.full(500, 2.0), media(fog=20))
    assert (lab == rm.LABEL_RAIN_NOISE).all()          # fog lowers SNR -> weather range noise


def test_hybrid_fog_only_adds_snr_noise_only_where_fog_is():
    m = rm.make_model('hybrid')
    angles = (-math.pi, 2 * math.pi / 360)
    wm = media(dust=20, soft_gain=0.0)                  # dust only, one cloud ahead
    _, _, lab = run(m, np.full(360, 4.0), wm, angles=angles)
    through = ~np.isnan(cloud_chords(CLOUD, beam_angles(*angles, 360))[:, 0, 0])
    assert (lab[through] != rm.LABEL_CLEAN).all() and (lab[~through] == rm.LABEL_CLEAN).all()


def test_alavi_first_echo_takes_the_nearer_fog_echo():
    # fog heavy, wall at 3 m: the wall (~4.9 P_min) beats the fog echo (~2.2 P_min)
    # for the strongest echo, but the fog echo (~0.3 m) is the first one.
    scan = np.full(300, 3.0)
    first = rm.make_model('alavi2025', {'echo_mode': 'first'})
    strongest = rm.make_model('alavi2025')
    wm = media(fog=20, jitter_exponent_max=0.0)
    assert (run(first, scan, wm)[2] == rm.LABEL_FOG_FALSE).all()
    assert (run(strongest, scan, wm)[2] == rm.LABEL_CLEAN).all()


def test_rain_sensor_fog_heavy_replaces_far_walls():
    m = rm.make_model('hybrid')
    walls = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    out, _, lab = run(m, walls, media(fog=20, jitter_exponent_max=0.0))
    assert list(lab) == [1, 1, 1, 5, 5, 5]
    assert out[3:] == pytest.approx(0.32, abs=0.05)


def test_fog_presets_stay_below_the_fog_echo_threshold():
    # The fog echo is the same on every beam, so once it reaches P_min every open
    # beam becomes a ~0.3 m phantom and Nav2 cannot plan (V = 20 m: 0 % of goals
    # in Gazebo). The threshold on this sensor lies between 45 and 50 m.
    m = rm.make_model('hybrid')
    open_beams = np.full(200, np.inf)
    for level in ('light', 'heavy'):
        v = wx.visibility_for('fog', level)
        lab = run(m, open_beams, media(fog=v))[2]
        assert not (lab == rm.LABEL_FOG_FALSE).any(), level
    assert (run(m, open_beams, media(fog=45.0))[2] == rm.LABEL_FOG_FALSE).all()


def test_world_clouds_follow_the_pose_and_skip_without_it():
    world = [dict(shape='circle', frame='world', x=5.0, y=5.0, radius=0.75)]
    wm = media(dust=5, clouds=world, jitter_exponent_max=0.0)
    m = rm.make_model('hybrid')
    angles = (-math.pi, 2 * math.pi / 360)
    walls = np.full(360, np.inf)
    lab_none = run(m, walls, wm, angles=angles)[2]
    assert not (lab_none == rm.LABEL_DUST_FALSE).any()        # no pose: world clouds skipped
    lab_far = run(m, walls, wm, angles=angles, world_pose=(-5.0, -5.0, 0.0))[2]
    assert not (lab_far == rm.LABEL_DUST_FALSE).any()         # 14 m away: beyond range
    out, _, lab = run(m, walls, wm, angles=angles, world_pose=(3.0, 5.0, 0.0))
    dust = lab == rm.LABEL_DUST_FALSE
    ang = beam_angles(*angles, 360)
    assert dust.any() and np.all(np.abs(ang[dust]) < math.radians(25))
    assert np.all(np.abs(out[dust] - 1.25) < 0.6)


# -----------------------------------------------------------------------------
# reproducibility, speed, config files
# -----------------------------------------------------------------------------
def test_same_seeds_same_output():
    m = rm.make_model('hybrid')
    scan = np.random.default_rng(1).uniform(0.2, 8.0, 500)
    scan[::5] = np.inf
    a = run(m, scan, media(fog=100, dust=5), 25.0, seed=3, wseed=4)
    b = run(m, scan, media(fog=100, dust=5), 25.0, seed=3, wseed=4)
    c = run(m, scan, media(fog=100, dust=5), 25.0, seed=3, wseed=5)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], c[0])


def test_speed_robot_inside_cloud():
    # Worst case: every beam starts inside a dense dust cloud, plus fog and rain.
    m = rm.make_model('hybrid')
    inside = [dict(shape='circle', frame='sensor', x=0.3, y=0.0, radius=1.5)]
    wm = media(fog=20, dust=5, clouds=inside)
    scan = np.random.default_rng(2).uniform(0.3, 11.0, 500)
    scan[::4] = np.inf
    run(m, scan, wm, 25.0)                               # builds the lookup table
    t = []
    for k in range(5):
        t0 = time.perf_counter()
        run(m, scan, wm, 25.0, seed=k)
        t.append(time.perf_counter() - t0)
    print(f'median {1e3 * np.median(t):.1f} ms per 500-beam scan inside a dust cloud')
    assert np.median(t) < 0.02    # typically ~3 ms; per-beam integration took ~85 ms


REF_SENSOR_FD = {k: REF_SENSOR[k] for k in ('pulse_width_s', 'overlap_r1_m', 'overlap_r2_m')}


def _random_clouds(rng, k):
    clouds = []
    for _ in range(k):
        x, y = rng.uniform(-4.0, 4.0, 2)
        if rng.random() < 0.5:
            clouds.append(dict(shape='circle', frame='sensor', x=x, y=y,
                               radius=rng.uniform(0.2, 2.5)))
        else:
            clouds.append(dict(shape='box', frame='sensor', x=x, y=y,
                               size_x=rng.uniform(0.2, 3.0), size_y=rng.uniform(0.2, 3.0)))
    return clouds


@pytest.mark.parametrize('case', range(12))
def test_direct_batch_matches_direct_per_beam(case):
    # The batched soft return must reproduce the per-beam integration it replaced
    # exactly: random overlapping circles / boxes, sensor inside or outside,
    # fog on / off, rain extinction, dust soft term on / off.
    rng = np.random.default_rng(case)
    clouds = _random_clouds(rng, int(rng.integers(1, 4)))
    n = 180
    angles = beam_angles(-math.pi, 2 * math.pi / n, n)
    ch = cloud_chords(clouds, angles)
    r_end = np.where(rng.random(n) < 0.3, RMAX, rng.uniform(0.3, RMAX, n))
    r_in = np.nanmin(np.where(np.isnan(ch[:, :, 0]), np.inf, ch[:, :, 0]), axis=1)
    sel = np.flatnonzero(r_in < r_end)
    assert sel.size
    p = fd.FogDustParams(**REF_SENSOR_FD) if case % 3 == 0 else fd.FogDustParams()
    soft = fd.SoftReturnModel(p)
    fog_v = (None, 50.0, 20.0)[case % 3]
    uni = fd.UniformMedium((fd.alpha_fog(fog_v) if fog_v else 0.0) + 0.002 * (case % 2),
                           fd.beta_fog(fog_v) if fog_v else 0.0)
    dust_v = (5.0, 20.0, 1.0, 0.5)[case % 4]
    a_d, b_d = fd.alpha_dust(dust_v), fd.beta_dust(dust_v)
    scale = np.where(rng.random(sel.size) < 0.5, 1.0, 0.0) if case % 2 else np.ones(sel.size)
    batch = fd.DustBeams(sel, ch[sel], a_d, b_d)
    pb, rb, fb = soft.direct_batch(uni, batch, RMIN, RMAX, r_end[sel], scale)
    for j, i in enumerate(sel):
        prof = fd.DustProfile(ch[i], a_d, b_d)
        p1, r1, f1 = soft.direct(uni, prof, RMIN, RMAX, r_end[i], dust_beta_scale=scale[j])
        assert pb[j] == pytest.approx(p1, rel=1e-9, abs=1e-300)
        assert rb[j] == pytest.approx(r1, abs=1e-12) or (np.isnan(rb[j]) and np.isnan(r1))
        assert fb[j] == f1
        x = np.linspace(-1.0, 13.0, 57)
        assert batch.tau(np.full(x.size, j), x) == pytest.approx(prof.tau(x), rel=1e-12, abs=1e-12)
        assert batch.first_entry[j] == prof.first_entry


def test_direct_batch_falls_back_for_extreme_dust():
    # visibility 0.05 m: exp(2 alpha x) would overflow, so the per-beam path is used
    ch = cloud_chords([dict(shape='circle', frame='sensor', x=3.0, y=0.0, radius=1.0)],
                      beam_angles(-0.2, 0.01, 40))
    sel = np.flatnonzero(~np.isnan(ch[:, 0, 0]))
    soft = fd.SoftReturnModel(fd.FogDustParams())
    uni = fd.UniformMedium(fd.alpha_fog(50.0), fd.beta_fog(50.0))
    a_d, b_d = fd.alpha_dust(0.05), fd.beta_dust(0.05)
    pb, rb, fb = soft.direct_batch(uni, fd.DustBeams(sel, ch[sel], a_d, b_d), RMIN, RMAX,
                                   np.full(sel.size, RMAX), 1.0)
    for j, i in enumerate(sel):
        p1, r1, f1 = soft.direct(uni, fd.DustProfile(ch[i], a_d, b_d), RMIN, RMAX, RMAX)
        assert (pb[j], rb[j], fb[j]) == (p1, r1, f1)


def test_dust_cloud_sets_file():
    sets = fd.load_cloud_sets(os.path.join(HERE, '..', 'config', 'dust_clouds.yaml'))
    for world in ('rain_room', 'rain_yard'):
        assert world in sets and all(c['frame'] == 'world' for c in sets[world])
    assert 'sensor_ahead' in sets
    assert sets['sensor_ahead'] == [fd.check_cloud(CLOUD[0], 'x')]


def test_yaml_and_node_defaults_match_dataclass():
    yaml = pytest.importorskip('yaml')
    with open(os.path.join(HERE, '..', 'config', 'weather_params.yaml')) as f:
        p = yaml.safe_load(f)['/**']['ros__parameters']
    with open(os.path.join(HERE, '..', 'scripts', 'weather_model_node.py')) as f:
        node_src = f.read()
    d = fd.FogDustParams()
    for fld in fd.fields(fd.FogDustParams):
        assert fld.name in p, f'{fld.name} missing from weather_params.yaml'
        assert p[fld.name] == pytest.approx(getattr(d, fld.name)) \
            if isinstance(p[fld.name], float) else p[fld.name] == getattr(d, fld.name)
        assert f"'{fld.name}':" in node_src, f'{fld.name} missing from PARAM_DEFAULTS'
    for name in ('rain', 'fog', 'dust', 'fog_visibility_m', 'dust_visibility_m',
                 'dust_cloud_set'):
        assert name in p, f'{name} missing from weather_params.yaml'
