#!/usr/bin/env python3
"""plot_rain_models - sweep rain rate for every rain model, no ROS needed.

Not a ROS node. numpy + matplotlib only. Reproduces, for a 2D C1-like scan,
the two curves of Goodin et al. 2019:

  max_range_vs_rain.png    max detected range of a REAL surface vs R
                           (Goodin Fig. 4). Scan = Goodin's range-test layout:
                           targets on an arc of increasing radius, here 0.3-11.95 m.
  valid_beams_vs_rain.png  valid beams per scan vs R (Goodin Fig. 5), split into
                           real returns and rain false returns.
  alavi_backscatter.png    Alavi Eq. 6 mean rain-bin power vs range next to P_min
                           (the soft-target sanity check), R = 6 / 25 / 50 mm/h,
                           plus the hybrid (LISA) power of single drops.
  rain_model_sweep.csv     the numbers behind the plots.
  model_summary.csv        per model at 0 / 6 / 25 mm/h: alpha, max range of a
                           70 % target, real kept / dropped, false returns
                           (and how many within 1 m), rain range noise at 12 m,
                           median published intensity of real vs false returns.
  hybrid_divergence.csv    hybrid false returns etc. for beam_divergence_rad
                           1e-3 / 3e-3 / 10e-3 at 6 and 25 mm/h.

Usage:
  plot_rain_models.py [--out DIR] [--scan SCAN.npy] [--scans N] [--draws N] [--seed S]

--draws: scans per cell for the two tables (default 200). They use --scan if
given, otherwise test/data/rain_yard_scan_clean.npy from the package source.

--scan: a recorded clean scan (1-D array of ranges, inf = no return), used for
the valid-beams plot instead of the synthetic one. Default output dir:
~/rain_sim_results/rain_model_curves/
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    from thesis_rain_sim import rain_models as rm
except ImportError:                                    # running from the source tree
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from thesis_rain_sim import rain_models as rm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

N_BEAMS = 500
ANG_INC = 2.0 * math.pi / N_BEAMS
R_MIN, R_MAX = 0.1, 12.0
RATES = np.arange(0.0, 50.0 + 1e-9, 1.0)
REFLECTIVITIES = (0.07, 0.2, 0.7)       # Goodin Fig. 4 used 0.07 and 0.2; 0.7 = our default


def arc_scan():
    return np.linspace(0.3, 11.95, N_BEAMS)


def random_scan(seed):
    g = np.random.default_rng(seed)
    r = g.uniform(0.3, 11.95, N_BEAMS)
    r[g.random(N_BEAMS) < 0.2] = np.inf
    return r


def variants():
    yield 'legacy', None, rm.make_model('legacy')
    for rho in REFLECTIVITIES:
        yield 'goodin2019', rho, rm.Goodin2019(default_reflectivity=rho)
    for rho in REFLECTIVITIES:
        yield 'alavi2025', rho, rm.Alavi2025(default_reflectivity=rho)
    for rho in REFLECTIVITIES:
        yield 'hybrid', rho, rm.HybridRain(default_reflectivity=rho)


def run(model, scan, rate, rng):
    out, inten, lab = model.apply(scan, R_MIN, R_MAX, 0.0, ANG_INC, rate, rng)
    real = (lab == rm.LABEL_CLEAN) | (lab == rm.LABEL_RAIN_NOISE)
    false = lab == rm.LABEL_RAIN_FALSE
    max_real = float(np.max(scan[real])) if real.any() else 0.0
    return max_real, int(real.sum()), int(false.sum()), inten[real], inten[false]


def default_scan_path():
    here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    for cand in (os.path.join(here, '..', 'test', 'data', 'rain_yard_scan_clean.npy'),
                 os.path.join(here, '..', '..', '..', '..', 'src', 'thesis_rain_sim', 'test',
                              'data', 'rain_yard_scan_clean.npy')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('rain_yard_scan_clean.npy not found; pass --scan')


def rain_sigma_at(model, z, rate):
    """Std-dev of the RAIN range-noise term at range z [m] (sensor term excluded)."""
    if model.name == 'legacy':      # total sqrt(s0^2 + (k R z/r_max)^2); rain part k R z/r_max
        return model.k_rain_range * rate * z / R_MAX
    if model.name == 'goodin2019':
        return float(model.range_sigma(z, rate))
    if model.name == 'hybrid':
        return float(model.rain_range_sigma(z, rate))
    return 0.0                      # alavi2025 has no rain range-noise term


def scan_stats(model, scan, rate, draws, seed):
    """Per-scan means over `draws` scans of `scan`."""
    rng = np.random.default_rng(seed)
    kept, dropped, false, near = [], [], [], []
    i_real, i_false = [], []
    for _ in range(draws):
        out, inten, lab = model.apply(scan, R_MIN, R_MAX, 0.0, ANG_INC, rate, rng)
        real = (lab == rm.LABEL_CLEAN) | (lab == rm.LABEL_RAIN_NOISE)
        f = lab == rm.LABEL_RAIN_FALSE
        kept.append(real.sum())
        dropped.append((lab == rm.LABEL_DROPPED).sum())
        false.append(f.sum())
        near.append((f & (out < 1.0)).sum())
        i_real.append(inten[real])
        i_false.append(inten[f])
    i_real, i_false = np.concatenate(i_real), np.concatenate(i_false)
    return dict(kept=np.mean(kept), dropped=np.mean(dropped), false=np.mean(false),
                false_within_1m=np.mean(near), n_false_total=int(i_false.size),
                med_i_real=float(np.median(i_real)) if i_real.size else float('nan'),
                med_i_false=float(np.median(i_false)) if i_false.size else float('nan'))


def max_range_70(model, rate, draws, seed):
    """Mean over draws of the farthest real return on the 0.3-11.95 m arc (rho = 0.7)."""
    rng = np.random.default_rng(seed)
    return float(np.mean([run(model, arc_scan(), rate, rng)[0] for _ in range(draws)]))


def tables(scan, draws, seed, out_dir):
    models = [rm.make_model(n) for n in rm.MODEL_NAMES]       # defaults: rho = 0.7
    rows = []
    for m in models:
        for rate in (0.0, 6.0, 25.0):
            st = scan_stats(m, scan, rate, draws, seed)
            rows.append(dict(model=m.name, rain_mm_h=rate, alpha_per_m=m.alpha(rate),
                             max_range_70_m=max_range_70(m, rate, draws, seed),
                             rain_sigma_12m_m=rain_sigma_at(m, 12.0, rate), **st))
    with open(os.path.join(out_dir, 'model_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nmodel summary ({draws} scans per cell; max range on the arc scan)')
    print('| model | R | alpha [1/m] | max range 70 % [m] | real kept / dropped | false (<1 m) '
          '| rain sigma @12 m [m] | median I real / false |')
    for r in rows:
        print(f"| {r['model']} | {r['rain_mm_h']:g} | {r['alpha_per_m']:.3e} | "
              f"{r['max_range_70_m']:.2f} | {r['kept']:.1f} / {r['dropped']:.1f} | "
              f"{r['false']:.2f} ({r['false_within_1m']:.2f}) | {r['rain_sigma_12m_m']:.4f} | "
              f"{r['med_i_real']:.0f} / {r['med_i_false']:.0f} (n={r['n_false_total']}) |")

    drows = []
    for div in (1e-3, 3e-3, 10e-3):
        m = rm.HybridRain(beam_divergence_rad=div)
        for rate in (6.0, 25.0):
            st = scan_stats(m, scan, rate, draws, seed)
            drows.append(dict(beam_divergence_rad=div, rain_mm_h=rate,
                              drops_per_12m_beam=m.drop_density(rate) * float(m.beam_volume(R_MAX)),
                              **st))
    with open(os.path.join(out_dir, 'hybrid_divergence.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)
    print(f'\nhybrid beam-divergence sensitivity ({draws} scans per cell)')
    print('| divergence [rad] | R | drops in a 12 m beam | real kept / dropped | false (<1 m) '
          '| median I false |')
    for r in drows:
        print(f"| {r['beam_divergence_rad']:.0e} | {r['rain_mm_h']:g} | "
              f"{r['drops_per_12m_beam']:.2f} | {r['kept']:.1f} / {r['dropped']:.1f} | "
              f"{r['false']:.2f} ({r['false_within_1m']:.2f}) | {r['med_i_false']:.0f} "
              f"(n={r['n_false_total']}) |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.expanduser('~/rain_sim_results/rain_model_curves'))
    ap.add_argument('--scan', help='recorded clean scan (.npy) for the valid-beams plot')
    ap.add_argument('--scans', type=int, default=20, help='scans averaged per rain rate')
    ap.add_argument('--draws', type=int, default=200, help='scans per cell for the tables')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    beams_scan = np.load(a.scan) if a.scan else None
    rows, curves = [], {}
    for name, rho, model in variants():
        key = name if rho is None else f'{name} rho={rho}'
        c = curves.setdefault(key, {'max': [], 'real': [], 'false': []})
        for rate in RATES:
            rng = np.random.default_rng(a.seed)
            mx, re_, fa = [], [], []
            for k in range(a.scans):
                m, _, _, _, _ = run(model, arc_scan(), rate, rng)
                scan = beams_scan if beams_scan is not None else random_scan(k)
                _, nr, nf, _, _ = run(model, scan, rate, rng)
                mx.append(m)
                re_.append(nr)
                fa.append(nf)
            c['max'].append(np.mean(mx))
            c['real'].append(np.mean(re_))
            c['false'].append(np.mean(fa))
            rows.append([name, '' if rho is None else rho, rate,
                         round(np.mean(mx), 3), round(np.mean(re_), 2), round(np.mean(fa), 2)])

    with open(os.path.join(a.out, 'rain_model_sweep.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'reflectivity', 'rain_mm_h', 'max_real_range_m',
                    'real_beams_per_scan', 'rain_false_beams_per_scan'])
        w.writerows(rows)

    styles = {'legacy': ('k', '-'), 'goodin2019': ('tab:blue', None), 'alavi2025': ('tab:red', None),
              'hybrid': ('tab:green', None)}
    dashes = {0.07: ':', 0.2: '--', 0.7: '-'}

    def style(key):
        name = key.split()[0]
        col, ls = styles[name]
        if ls is None:
            ls = dashes[float(key.split('=')[1])]
        return col, ls

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, c in curves.items():
        col, ls = style(key)
        ax.plot(RATES, c['max'], color=col, ls=ls, label=key)
    ax.set_xlabel('Rain rate (mm/h)')
    ax.set_ylabel('Max detected range of a real surface (m)')
    ax.set_title('Max range vs rain rate (cf. Goodin 2019 Fig. 4)\nC1: z_max = 12 m @ 70 %')
    ax.set_ylim(0, 12.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'max_range_vs_rain.png'), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, c in curves.items():
        col, ls = style(key)
        ax.plot(RATES, c['real'], color=col, ls=ls, label=f'{key} real')
        if max(c['false']) > 0:
            ax.plot(RATES, c['false'], color=col, ls=ls, lw=0.8, marker='.', ms=3,
                    label=f'{key} rain false')
    ax.set_xlabel('Rain rate (mm/h)')
    ax.set_ylabel('Beams per scan')
    ax.set_title('Valid beams per scan vs rain rate (cf. Goodin 2019 Fig. 5)\n'
                 + ('recorded scan' if beams_scan is not None else 'synthetic scan, 80 % surfaces'))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'valid_beams_vs_rain.png'), dpi=130)

    m = rm.Alavi2025()
    cen = m.bin_centres(R_MAX)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for R, col in ((6.0, 'tab:green'), (25.0, 'tab:orange'), (50.0, 'tab:red')):
        pbar = m.mean_bin_power(cen, R)
        ax.semilogy(cen, np.maximum(pbar, 1e-12), 'o-', color=col, ms=3,
                    label=f'Eq. 6 mean bin power, R={R:g} (max {pbar.max() / m.p_min():.2f} x P_min)')
    h = rm.HybridRain()
    rr = np.linspace(R_MIN, R_MAX, 400)
    for d_mm, ls in ((0.5, '-'), (2.0, '--')):
        ax.semilogy(rr, h.drop_power(rr, d_mm, 25.0), color='tab:purple', ls=ls, lw=1.2,
                    label=f'hybrid: one {d_mm:g} mm drop, R=25 (LISA, overlap on)')
    ax.axhline(m.p_min(), color='k', ls='--', label='P_min (C1: 12 m @ 70 %)')
    for rho in REFLECTIVITIES:
        ax.semilogy(cen, m.hard_power(cen, 25.0, rho), color='grey', lw=0.8,
                    ls=dashes[rho], label=f'hard target rho={rho}, R=25')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Normalised received power P_n')
    ax.set_ylim(1e-6, 10)
    ax.set_title(f'alavi2025 soft target (beta/4pi, tau={m.pulse_width_s * 1e9:g} ns, '
                 f'dr={m.bin_width:.2f} m) and hybrid drops', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'alavi_backscatter.png'), dpi=130)

    table_scan = beams_scan if beams_scan is not None else np.load(default_scan_path())
    tables(table_scan, a.draws, a.seed, a.out)

    print(f'wrote {a.out}/{{max_range_vs_rain,valid_beams_vs_rain,alavi_backscatter}}.png, '
          f'rain_model_sweep.csv, model_summary.csv and hybrid_divergence.csv')
    for key, c in curves.items():
        i6, i25 = int(np.argmin(abs(RATES - 6))), int(np.argmin(abs(RATES - 25)))
        print(f'  {key:24s} max range  0/6/25 mm/h: {c["max"][0]:5.2f} {c["max"][i6]:5.2f} '
              f'{c["max"][i25]:5.2f} m   rain false beams/scan 6/25: '
              f'{c["false"][i6]:5.1f} {c["false"][i25]:5.1f}')


if __name__ == '__main__':
    main()
