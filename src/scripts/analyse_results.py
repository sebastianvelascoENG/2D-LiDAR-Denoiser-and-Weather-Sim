#!/usr/bin/env python3
"""analyse_results - statistics for the rain / denoising navigation experiment.

Not a ROS node. Pure numpy + scipy + csv (no pandas). Reads the
ground_truth_logger CSV (and optionally the run_missions CSV) and produces the
comparisons named in the thesis methodology:

  * paired t-test     : raw  vs  denoised, per weather   (matched by sample index)
  * independent t-test : denoised[weather]  vs  calm      (Welch)

`condition` is the run's weather label (thesis_rain_sim.weather.weather_label):
calm / light / heavy for rain-only runs, e.g. fog_heavy or rain_light+dust_heavy
otherwise. Every label other than calm is tested against calm (light and heavy
first, then the rest alphabetically).

Metrics: err_pos_m, err_yaw_rad, and scan quality (n_scan / n_denoised).

Usage:
  analyse_results.py LOG.csv [--missions MISSIONS.csv] [--out SUMMARY.csv]
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def col(rows, name, cond=None, pipe=None, numeric=True):
    out = []
    for r in rows:
        if cond is not None and r.get('condition') != cond:
            continue
        if pipe is not None and r.get('pipeline') != pipe:
            continue
        v = r.get(name, '')
        if numeric:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                out.append(x)
        else:
            out.append(v)
    return np.asarray(out, dtype=float) if numeric else out


def summarise(rows, out_path):
    groups = sorted({(r['condition'], r['pipeline']) for r in rows})
    hdr = ['condition', 'pipeline', 'n', 'err_pos_mean', 'err_pos_std',
           'err_yaw_mean', 'err_yaw_std', 'n_scan_mean', 'n_denoised_mean']
    table = [hdr]
    for cond, pipe in groups:
        ep = col(rows, 'err_pos_m', cond, pipe)
        ey = col(rows, 'err_yaw_rad', cond, pipe)
        ns = col(rows, 'n_scan', cond, pipe)
        nd = col(rows, 'n_denoised', cond, pipe)
        ns = ns[ns >= 0]
        nd = nd[nd >= 0]
        table.append([cond, pipe, len(ep),
                      _f(ep.mean() if ep.size else math.nan),
                      _f(ep.std() if ep.size else math.nan),
                      _f(ey.mean() if ey.size else math.nan),
                      _f(ey.std() if ey.size else math.nan),
                      _f(ns.mean() if ns.size else math.nan),
                      _f(nd.mean() if nd.size else math.nan)])
    _print_table(table)
    _write_csv(out_path, table)
    return table


def hypothesis_tests(rows, out_path):
    hdr = ['test', 'metric', 't', 'p', 'mean_a', 'mean_b', 'n_a', 'n_b']
    table = [hdr]
    calm = col(rows, 'err_pos_m', 'calm', 'denoised')
    if calm.size == 0:
        calm = col(rows, 'err_pos_m', 'calm', 'raw')
    present = {r['condition'] for r in rows} - {'calm'}
    for cond in [c for c in ('light', 'heavy') if c in present] + \
            sorted(present - {'light', 'heavy'}):
        raw = col(rows, 'err_pos_m', cond, 'raw')
        den = col(rows, 'err_pos_m', cond, 'denoised')
        if raw.size > 1 and den.size > 1:
            m = min(raw.size, den.size)
            t, p = stats.ttest_rel(raw[:m], den[:m])
            table.append([f'paired raw-vs-denoised [{cond}]', 'err_pos_m',
                          _f(t), _f(p), _f(raw[:m].mean()), _f(den[:m].mean()), m, m])
        if den.size > 1 and calm.size > 1:
            t, p = stats.ttest_ind(den, calm, equal_var=False)
            table.append([f'Welch denoised[{cond}]-vs-calm', 'err_pos_m',
                          _f(t), _f(p), _f(den.mean()), _f(calm.mean()), den.size, calm.size])
    if len(table) > 1:
        _print_table(table)
        _write_csv(out_path, table)
    else:
        print('(insufficient data for hypothesis tests)')
    return table


def mission_kpis(path, out_path):
    rows = read_csv(path)
    agg = defaultdict(list)
    for r in rows:
        agg[(r['condition'], r['pipeline'])].append(r)
    hdr = ['condition', 'pipeline', 'runs', 'success_rate',
           'duration_s_mean', 'travelled_m_mean', 'recoveries_mean']
    table = [hdr]
    for (cond, pipe), rs in sorted(agg.items()):
        n = len(rs)
        succ = sum(1 for r in rs if r['result'] == 'SUCCEEDED') / n
        dur = np.mean([float(r['duration_s']) for r in rs])
        trav = np.mean([float(r['travelled_m']) for r in rs])
        rec = np.mean([float(r['recoveries']) for r in rs if r['recoveries'] not in ('', '-1')] or [math.nan])
        table.append([cond, pipe, n, _f(succ), _f(dur), _f(trav), _f(rec)])
    _print_table(table)
    _write_csv(out_path, table)


def _f(x):
    return f'{x:.4f}' if isinstance(x, float) and math.isfinite(x) else ('nan' if isinstance(x, float) else x)


def _print_table(table):
    widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    for row in table:
        print('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    print()


def _write_csv(path, table):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(table)
    print(f'wrote {path}\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('log_csv')
    ap.add_argument('--missions')
    ap.add_argument('--out', default='rain_sim_summary.csv')
    a = ap.parse_args(argv)

    rows = read_csv(a.log_csv)
    print('=== per-group summary ===')
    summarise(rows, a.out)
    print('=== hypothesis tests (err_pos_m) ===')
    hypothesis_tests(rows, a.out.replace('.csv', '_ttests.csv'))
    if a.missions:
        print('=== mission KPIs ===')
        mission_kpis(a.missions, a.out.replace('.csv', '_missions.csv'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
