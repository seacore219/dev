"""
Fig. 2 panels A, B, C -- the same exact script as fig2_abc_n200.py, changed
only where the file paths differ: this reads one h_ip batch from the
hip-sweep's folder layout instead of the pristine N200 layout.

  (A), (B): normalized distributions of avalanche duration T and size S,
            for this one h_ip value. Raw data in gray, power-law fit overlaid.
  (C):      average avalanche size <S>(T) as a function of duration.

Reads: /opt/sorn/backup/<sweep-label>/h_ip_<value>/run-*/test_single/<timestamp>/common/result.h5
(sweep-label matched with a wildcard; HIP_BATCH selects which h_ip folder)

Run from /opt/sorn/delpapa/avalanches, with HIP_BATCH set, e.g.:
  HIP_BATCH=h_ip_0.02 python fig2_abc_hipsweep.py
"""

import glob
import os

import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
from pylab import *

import data_analysis as analysis
import powerlaw as pl

BACKUP_ROOT = '/opt/sorn/backup'
HIP_BATCH = os.environ.get('HIP_BATCH', 'h_ip_0.02')   # which h_ip folder to analyze
RUN_GLOB = 'run-*'
OUTPUT_DIR = os.path.join(BACKUP_ROOT, 'analysis')

THETA = 'half'
STABLE_STEPS = 3000000


def find_result_files():
    pattern = os.path.join(BACKUP_ROOT, '*', HIP_BATCH, RUN_GLOB, 'test_single', '*', 'common', 'result.h5')
    all_files = sorted(glob.glob(pattern))

    latest_per_run = {}
    for path in all_files:
        run_id = path.split(os.sep + 'test_single' + os.sep)[0].split(os.sep)[-1]
        latest_per_run[run_id] = path

    return sorted(latest_per_run.values())


def load_data(files):
    good_data = []
    for i, path in enumerate(files):
        try:
            h5 = tables.open_file(path, 'r')
            data = h5.root
            activity = data.activity[0][-STABLE_STEPS:]
            row = np.around(activity * data.c.N_e)
            h5.close()
            good_data.append(row)
            print('  [%d/%d] OK    %s' % (i + 1, len(files), path))
        except Exception as e:
            print('  [%d/%d] SKIP  %s (%s)' % (i + 1, len(files), path, e))
    return np.array(good_data)


def main():
    files = find_result_files()
    print('HIP_BATCH=%s -- found %d result.h5 files' % (HIP_BATCH, len(files)))
    if not files:
        raise SystemExit('No result.h5 files found for %s under %s' % (HIP_BATCH, BACKUP_ROOT))

    data_all = load_data(files)
    n_used = len(data_all)
    print('Using %d good files' % n_used)
    if n_used == 0:
        raise SystemExit('No readable result.h5 files -- nothing to analyze')

    a_dur, a_area = analysis.avalanches(data_all, 'N', '200', Threshold=THETA)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fig = figure(figsize=(13, 4))

    ax_a = subplot(131)
    T_x, T_inverse = np.unique(a_dur, return_inverse=True)
    T_freq = np.bincount(T_inverse)
    T_y = T_freq / float(T_freq.sum())
    plot(T_x, T_y, '.', color='gray', markersize=3, zorder=1, label='raw data')

    T_fit = pl.Fit(a_dur, discrete=True)
    pl.plot_pdf(a_dur, color='b', linewidth=1.5, zorder=2)
    T_fit.power_law.plot_pdf(color='k',
                              label=r'$\tau_t = $' + str(round(T_fit.alpha, 2)),
                              linewidth=2.0, zorder=3)
    xscale('log'); yscale('log')
    xlabel('Duration (T)')
    ylabel('f(T)')
    title('A: Duration (%s, %d runs)' % (HIP_BATCH, n_used))
    legend(loc='lower left', fontsize=8, frameon=False)
    ax_a.spines['right'].set_visible(False)
    ax_a.spines['top'].set_visible(False)

    ax_b = subplot(132)
    S_x, S_inverse = np.unique(a_area, return_inverse=True)
    S_freq = np.bincount(S_inverse)
    S_y = S_freq / float(S_freq.sum())
    plot(S_x, S_y, '.', color='gray', markersize=3, zorder=1, label='raw data')

    S_fit = pl.Fit(a_area, discrete=True)
    pl.plot_pdf(a_area, color='r', linewidth=1.5, zorder=2)
    S_fit.power_law.plot_pdf(color='k',
                              label=r'$\tau = $' + str(round(S_fit.alpha, 2)),
                              linewidth=2.0, zorder=3)
    xscale('log'); yscale('log')
    xlabel('Size (S)')
    ylabel('f(S)')
    title('B: Size (%s, %d runs)' % (HIP_BATCH, n_used))
    legend(loc='lower left', fontsize=8, frameon=False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['top'].set_visible(False)

    ax_c = subplot(133)
    unique_durations = np.unique(a_dur)
    mean_size_per_duration = np.array([
        a_area[a_dur == t].mean() for t in unique_durations
    ])
    plot(unique_durations, mean_size_per_duration, '.', color='gray',
         markersize=3, label='simulated data')

    gamma_ref = 1.3
    x_range = np.arange(1, unique_durations.max())
    y_ref = x_range ** gamma_ref
    y_ref = y_ref * (mean_size_per_duration[0] / y_ref[0])
    plot(x_range, y_ref, '--', color='k', linewidth=1.5,
         label=r'$\gamma = %.1f$ (reference)' % gamma_ref)

    xscale('log'); yscale('log')
    xlabel('Duration (T)')
    ylabel(r'$\langle S \rangle (T)$')
    title('C: Mean size vs duration (%s)' % HIP_BATCH)
    legend(loc='upper left', fontsize=8, frameon=False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['top'].set_visible(False)

    tight_layout()
    out_path = os.path.join(OUTPUT_DIR, '%s_fig2abc.pdf' % HIP_BATCH)
    savefig(out_path)
    print('Saved figure to %s' % out_path)
    print('Power-law fit: duration alpha=%.3f, size alpha=%.3f' %
          (T_fit.alpha, S_fit.alpha))


if __name__ == '__main__':
    main()