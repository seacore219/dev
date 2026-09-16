import datetime
import glob
import os
import traceback

import sys
if sys.version_info < (3, 9):
    import importlib.resources
    import importlib_resources
    importlib.resources.files = importlib_resources.files

import nolds
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.legend as mlegend
if not hasattr(mlegend.Legend, "legendHandles"):
    mlegend.Legend.legendHandles = property(lambda self: self.legend_handles)

import criticality_tumbleweed as crt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SWEEP_ROOT = os.environ.get('SWEEP_ROOT', '/data')
BATCH_GLOB = os.environ.get('BATCH_GLOB', '*/h_ip_*') 
RUN_GLOB = os.environ.get('RUN_GLOB', 'run-*')
TIMESTAMP_FMT = "%Y-%m-%d %H-%M-%S"

AV_PERC = 0.25                  # avalanche detection threshold, as a percentile of population activity

## avalanche fitting and extraction parameters
AV_FLAG = 1                 # 1 = fast (exponents + DCC). 2 = also runs KS p-value tests (slow)
AV_BM = 20                  # AV_analysis: upper limit of xmin search, burst size
AV_TM = 10                  # AV_analysis: upper limit of xmin search, duration
AV_NFACTOR_BM = 0
AV_NFACTOR_TM = 0
AV_NFACTOR_BM_TAIL = 0.8
AV_NFACTOR_TM_TAIL = 1.0
AV_EXCLUDE = True
AV_EXCLUDE_BURST, AV_EXCLUDE_DIFF_B = 50, 20
AV_EXCLUDE_TIME, AV_EXCLUDE_DIFF_T = 20, 10
BRANCHING_KMAX = 50
BRANCHING_FITFUNCS = ["exp", "complex"]
DFA_NVALS = [int(n) for n in nolds.logarithmic_r(100, 100000, 2.0)]

OUTPUT_DIR = os.path.join(SWEEP_ROOT, "criticality_analysis_output")
AV_PLOT_DIR = os.path.join(OUTPUT_DIR, "av_plots")
PER_RUN_CSV = os.path.join(OUTPUT_DIR, "criticality_summary_per_run.csv")
BATCH_CSV = os.path.join(OUTPUT_DIR, "criticality_summary_pooled_by_batch.csv")

SAVE_AV_PLOTS = True

ANALYSIS_WINDOW_START = int(os.environ.get('ANALYSIS_WINDOW_START', 3000000))
ANALYSIS_WINDOW_END = int(os.environ.get('ANALYSIS_WINDOW_END', 6000000))

# ---------------------------------------------------------------------------


def find_latest_valid_result(test_single_dir):
    candidates = []
    for name in os.listdir(test_single_dir):
        full = os.path.join(test_single_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            ts = datetime.datetime.strptime(name, TIMESTAMP_FMT)
        except ValueError:
            continue
        candidates.append((ts, full))

    candidates.sort(key=lambda x: x[0], reverse=True)  # newest first

    for ts, folder in candidates:
        h5path = os.path.join(folder, "common", "result.h5")
        if not os.path.isfile(h5path):
            continue
        try:
            with h5py.File(h5path, "r") as f:
                if "c" not in f or "N_e" not in f["c"]:
                    continue
                _ = f["c/N_e"][0]  # forces a real read, catches truncated files
            return h5path, ts
        except Exception:
            continue  # corrupt/incomplete attempt -- fall back to older one

    return None, None


def load_sim(h5path):
    with h5py.File(h5path, "r") as f:
        N_e = int(f["c/N_e"][0])
        h_ip = float(f["c/h_ip"][0])
        N_steps_total = int(f["c/N_steps"][0])
        has_raster = "Spikes" in f
        raster = f["Spikes"][0].astype(bool) if has_raster else None  # (N_e, T_saved)
        activity = f["activity"][0]  # (N_steps_total,) -- always the full run, never truncated

    a_start = max(ANALYSIS_WINDOW_START, 0)
    a_end = min(ANALYSIS_WINDOW_END, N_steps_total)
    if a_start >= a_end:
        raise ValueError(f"analysis window [{ANALYSIS_WINDOW_START}:{ANALYSIS_WINDOW_END}] "
                          f"is outside this run's N_steps={N_steps_total}")
    window_len = a_end - a_start
    activity = activity[a_start:a_end]
    raster_long_enough = False
    if raster is not None:
        T_saved = raster.shape[1]
        saved_start = N_steps_total - T_saved  # Spikes covers [saved_start, N_steps_total)
        if saved_start <= a_start and T_saved >= window_len:
            r_start = a_start - saved_start
            r_end = a_end - saved_start
            raster = raster[:, r_start:r_end]
            raster_long_enough = True
        else:
            print(f"  NOTE: raster only covers the final {T_saved} steps "
                  f"(window needs {window_len}) -- using activity instead")
            raster = None

    return h_ip, N_e, raster if raster_long_enough else None, activity, raster_long_enough


def get_run_avalanches(h5path):
    """Load one run and return its own avalanche sizes/durations, plus
    everything analyze_one_run() needs for the per-run metrics."""
    h_ip, N_e, raster, activity, raster_long_enough = load_sim(h5path)

    pop_counts = raster.sum(axis=0).astype(float) if raster_long_enough \
        else np.round(activity * N_e).astype(float)

    av_kwargs = dict(perc=AV_PERC)
    if raster_long_enough:
        av = crt.get_avalanches(raster.astype(float), **av_kwargs)
    else:
        av = crt.get_avalanches(pop_counts, ncells=N_e, **av_kwargs)

    return h_ip, N_e, raster, pop_counts, av["S"], av["T"], av["perc_threshold"], raster_long_enough


def run_av_analysis(S, T, pltname):
    return crt.AV_analysis(
        S, T, flag=AV_FLAG, verbose=False,
        plot=SAVE_AV_PLOTS, pltname=pltname, saveloc=AV_PLOT_DIR,
        bm=AV_BM, tm=AV_TM,
        nfactor_bm=AV_NFACTOR_BM, nfactor_tm=AV_NFACTOR_TM,
        nfactor_bm_tail=AV_NFACTOR_BM_TAIL, nfactor_tm_tail=AV_NFACTOR_TM_TAIL,
        exclude=AV_EXCLUDE,
        exclude_burst=AV_EXCLUDE_BURST, exclude_diff_b=AV_EXCLUDE_DIFF_B,
        exclude_time=AV_EXCLUDE_TIME, exclude_diff_t=AV_EXCLUDE_DIFF_T,
    )


def analyze_one_run(batch_label, run_label, h_ip, N_e, raster, pop_counts,
                     n_avalanches, raster_long_enough):
    row = {"batch": batch_label, "run": run_label, "h_ip": h_ip, "N_e": N_e,
           "has_full_raster": raster_long_enough, "n_avalanches": n_avalanches}

    br_input = raster.astype(float) if raster_long_enough else pop_counts
    br = crt.calculate_branching_ratio(
        br_input, k_max=BRANCHING_KMAX, name=run_label,
        fitfuncs=BRANCHING_FITFUNCS, plot_targetdir=None, lreturn_tau=0,
    )
    br_dict = {br[i]: br[i + 1] for i in range(0, len(br), 2)}
    row["branching_ratio_exp"] = br_dict.get("exp")
    row["branching_ratio_complex"] = br_dict.get("complex")

    row["dfa"] = nolds.dfa(pop_counts, nvals=DFA_NVALS)

    if raster_long_enough:
        row["branching_parameter"] = crt.branchparam(raster.astype(float))
        susc, fano = crt.population_metrics(raster.astype(float))
        row["susceptibility"] = susc
        row["fano_factor"] = fano
        activity = float(raster.mean())
        row["activity"] = activity
        row["cv"] = (susc ** 0.5) / activity if activity > 0 else None
    else:
        row["branching_parameter"] = None
        row["susceptibility"] = None
        row["fano_factor"] = None
        row["activity"] = None
        row["cv"] = None

    return row


def main():
    os.makedirs(AV_PLOT_DIR, exist_ok=True)

    per_run_rows = []
    batch_rows = []

    for batch_dir in sorted(glob.glob(os.path.join(SWEEP_ROOT, BATCH_GLOB))):
        batch_label = os.path.basename(os.path.dirname(batch_dir)) + '/' + os.path.basename(batch_dir)
        print(f"=== batch {batch_label} ===")

        batch_S, batch_T = [], []
        batch_h_ip = None

        for run_dir in sorted(glob.glob(os.path.join(batch_dir, RUN_GLOB))):
            run_label = f"{batch_label}/{os.path.basename(run_dir)}"
            test_single_dir = os.path.join(run_dir, "test_single")
            if not os.path.isdir(test_single_dir):
                print(f"  {run_label}: no test_single/ folder, skipping")
                continue

            h5path, ts = find_latest_valid_result(test_single_dir)
            if h5path is None:
                print(f"  {run_label}: no usable result.h5 in any attempt, skipping")
                continue

            print(f"  --- {run_label}  (using attempt: {ts}) ---")
            try:
                h_ip, N_e, raster, pop_counts, S, T, thr_used, raster_long_enough = \
                    get_run_avalanches(h5path)
                batch_h_ip = h_ip
                batch_S.append(S)
                batch_T.append(T)

                row = analyze_one_run(batch_label, run_label, h_ip, N_e, raster,
                                       pop_counts, len(S), raster_long_enough)
                row["av_threshold_used"] = thr_used
                per_run_rows.append(row)
                print(f"    h_ip={h_ip:.3f}  threshold={thr_used}  n_avalanches={len(S)}  "
                      f"full_raster={raster_long_enough}  "
                      f"branching_ratio_exp={row['branching_ratio_exp']}  dfa={row['dfa']}")
            except Exception:
                print(f"    FAILED on {h5path}:")
                traceback.print_exc()

        if not batch_S:
            print(f"  no usable runs in {batch_label}, skipping pooled fit\n")
            continue

        pooled_S = np.concatenate(batch_S)
        pooled_T = np.concatenate(batch_T)
        print(f"  pooling {len(batch_S)} runs -> {len(pooled_S)} avalanches total")

        try:
            safe_label = batch_label.replace('/', '_')
            av_res = run_av_analysis(pooled_S, pooled_T, pltname=safe_label + "_pooled_")
            batch_row = {
                "batch": batch_label,
                "h_ip": batch_h_ip,
                "n_runs_pooled": len(batch_S),
                "n_avalanches_pooled": len(pooled_S),
                "alpha_size_exponent": av_res.get("alpha"),
                "beta_duration_exponent": av_res.get("beta"),
                "dcc": av_res.get("df"),
                "xmin_burst": av_res.get("xmin"),
                "xmax_burst": av_res.get("xmax"),
                "xmin_dur": av_res.get("tmin"),
                "xmax_dur": av_res.get("tmax"),
                "excluded_burst_fit": av_res.get("EX_b"),
                "excluded_dur_fit": av_res.get("EX_t"),
            }
            batch_rows.append(batch_row)
            print(f"  pooled fit: alpha={batch_row['alpha_size_exponent']}  "
                  f"beta={batch_row['beta_duration_exponent']}  DCC={batch_row['dcc']}")
        except Exception:
            print(f"  FAILED pooled AV_analysis for {batch_label}:")
            traceback.print_exc()
        print()

    if per_run_rows:
        pd.DataFrame(per_run_rows).sort_values(["h_ip", "run"]).to_csv(PER_RUN_CSV, index=False)
        print(f"Saved per-run summary to {PER_RUN_CSV}")
    else:
        print("No runs analyzed successfully.")

    if batch_rows:
        pd.DataFrame(batch_rows).sort_values("h_ip").to_csv(BATCH_CSV, index=False)
        print(f"Saved pooled batch summary to {BATCH_CSV}")
    else:
        print("No batches produced a pooled fit.")


if __name__ == "__main__":
    main()