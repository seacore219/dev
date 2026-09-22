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
import scipy.optimize
import matplotlib
matplotlib.use('Agg')

import matplotlib.legend as mlegend
if not hasattr(mlegend.Legend, "legendHandles"):
    mlegend.Legend.legendHandles = property(lambda self: self.legend_handles)

import criticality_tumbleweed as crt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SWEEP_ROOT = "nrp-sweep-data"
CONFIG_NAME = "batch_0.01_0.01_0.3"    # the current parameter-combo folder under SWEEP_ROOT --
                              # change this each time you start a new combination
BATCH_GLOB = "*_h_ip_*"       # e.g. "07_29_26_h_ip_0.01"
RUN_GLOB = "h_ip_*_run*"      # e.g. "h_ip_0.01_run1", nested inside a batch
TIMESTAMP_FMT = "%Y-%m-%d %H-%M-%S"   # matches "2026-07-28 16-47-45"

USE_SPIKES_RASTER = False

AV_THRESHOLD_MODE = "perc"      # "adaptive", "perc", or "const"
ADAPTIVE_THRESHOLD_FRAC = 0.25    # only used if AV_THRESHOLD_MODE == "adaptive"
AV_PERC = 0.25             # used only if AV_THRESHOLD_MODE == "perc"
AV_CONST_THRESHOLD = 2   # used only if AV_THRESHOLD_MODE == "const"

AV_FLAG = 1                 # 1 = fast (exponents + DCC). 2 = also runs KS p-value tests (slow)
AV_BM = 20                  # AV_analysis: upper limit of xmin search, burst size
AV_TM = 10                  # AV_analysis: upper limit of xmin search, duration
AV_NFACTOR_BM = 0
AV_NFACTOR_TM = 0
AV_NFACTOR_BM_TAIL = 0.8
AV_NFACTOR_TM_TAIL = 1.0
AV_EXCLUDE = True                          # turn on the QC flags (EX_b / EX_t)
AV_EXCLUDE_BURST, AV_EXCLUDE_DIFF_B = 50, 20
AV_EXCLUDE_TIME, AV_EXCLUDE_DIFF_T = 20, 10

# don't touch yet, debugging purposes only
SIZE_FIT_RANGE = None         # (xmin, xmax) for avalanche sizes, or None
DUR_FIT_RANGE = None          # (tmin, tmax) for avalanche durations, or None

BRANCHING_KMAX = 50
BRANCHING_FITFUNCS = ["exp", "complex"]    # fitfuncs=None crashes -- must be explicit

DFA_NVALS = [int(n) for n in nolds.logarithmic_r(100, 100000, 2.0)]  # ~10 scales, not nolds' default ~65

OUTPUT_DIR = os.path.join(SWEEP_ROOT, CONFIG_NAME, "criticality_analysis_output")
AV_PLOT_DIR = os.path.join(OUTPUT_DIR, "av_plots")
PER_RUN_CSV = os.path.join(OUTPUT_DIR, "criticality_summary_per_run.csv")
BATCH_CSV = os.path.join(OUTPUT_DIR, "criticality_summary_pooled_by_batch.csv")

SAVE_AV_PLOTS = True   # set False to skip avalanche distribution plots

ANALYSIS_WINDOW_START = 2500000
ANALYSIS_WINDOW_END = 6000000

# ---------------------------------------------------------------------------


def find_latest_valid_result(test_single_dir):
    """Return (h5path, timestamp) for the newest attempt folder under
    test_single_dir whose result.h5 actually opens and looks complete, or
    (None, None) if nothing usable is found."""
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
        has_raster = USE_SPIKES_RASTER and "Spikes" in f
        raster = f["Spikes"][0].astype(bool) if has_raster else None  # (N_e, T_saved)
        activity = f["activity"][0]  # (N_steps_total,) -- always the full run, starts at step 0

    if len(activity) != N_steps_total:
        sys.exit(
            f"ERROR: {h5path} stores 'activity' with {len(activity)} entries but "
            f"c/N_steps = {N_steps_total} -- the run was probably saved with activity "
            f"subsampling (c.stats.only_last), so the analysis window would silently "
            f"pick the wrong steps. Re-run the simulation saving the full activity trace."
        )

    # activity always starts at absolute step 0, so window indices apply directly
    a_start = max(ANALYSIS_WINDOW_START, 0)
    a_end = min(ANALYSIS_WINDOW_END, N_steps_total)
    if a_start >= a_end:
        raise ValueError(f"analysis window [{ANALYSIS_WINDOW_START}:{ANALYSIS_WINDOW_END}] "
                          f"is outside this run's N_steps={N_steps_total}")
    activity = activity[a_start:a_end]

    if raster is not None:
        T_saved = raster.shape[1]
        saved_start = N_steps_total - T_saved   # Spikes covers [saved_start, N_steps_total)
        r_start = a_start - saved_start
        r_end = a_end - saved_start
        if r_start < 0 or r_end > T_saved:
            sys.exit(
                f"ERROR: USE_SPIKES_RASTER = True, but {h5path} only saved Spikes for steps "
                f"[{saved_start}:{N_steps_total}] while 'activity' is analyzed on "
                f"[{a_start}:{a_end}] -- there are not enough Spikes saved to compute "
                f"avalanches and avalanche statistics on the same window. Re-run the "
                f"simulation saving a larger raster (c.stats.only_last_spikes must cover "
                f"the analysis window), or set USE_SPIKES_RASTER = False to use the "
                f"full-length population activity instead."
            )
        raster = raster[:, r_start:r_end]

    return h_ip, N_e, raster, activity


def get_run_avalanches(h5path):
    """Load one run and return its own avalanche sizes/durations, plus
    everything analyze_one_run() needs for the per-run metrics."""
    h_ip, N_e, raster, activity = load_sim(h5path)

    if raster is not None:
        pop_counts = raster.sum(axis=0).astype(float)
    else:
        pop_counts = np.round(activity * N_e).astype(float)

    if AV_THRESHOLD_MODE == "adaptive":
        thr = max(1, int(round(pop_counts.mean() * ADAPTIVE_THRESHOLD_FRAC)))
        av_kwargs = dict(const_threshold=thr)
    elif AV_THRESHOLD_MODE == "const":
        av_kwargs = dict(const_threshold=AV_CONST_THRESHOLD)
    else:
        av_kwargs = dict(perc=AV_PERC)

    if raster is not None:
        av = crt.get_avalanches(raster.astype(float), **av_kwargs)
    else:
        av = crt.get_avalanches(pop_counts, ncells=N_e, **av_kwargs)

    return h_ip, N_e, raster, pop_counts, av["S"], av["T"], av["perc_threshold"]


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


def discrete_powerlaw_mle(data, xmin, xmax):
    """Discrete power-law exponent on a FIXED [xmin, xmax] window -- the same
    MLE that criticality_tumbleweed's tplfit() runs for each candidate xmin,
    without the automatic KS-based window search."""
    idx = np.where(np.logical_and(data >= xmin, data <= xmax))[0]
    n = np.size(data[idx])
    s = np.unique(data[idx])
    logsum = np.sum(np.log(data[idx]))
    LL = lambda x: x * logsum - n * np.log(1 / np.sum(np.power(s, -x)))
    a, _, _, _, lstatus = scipy.optimize.fmin(func=LL, x0=1.0,
                                              xtol=0.000001, ftol=0.000001,
                                              maxiter=1000,
                                              full_output=True, disp=False)
    if lstatus != 0:
        raise RuntimeError('Error: scipy.optimize.fmin not successful',
                           lstatus)
    return float(a[0]), n


def manual_av_fit(S, T, size_range, dur_range, TT=None, Sm=None):
    """Pooled avalanche fit on hand-chosen windows: tplfit's MLE estimator
    plus AV_analysis's DCC computation, with fixed xmin/xmax instead of the
    automatic search. TT/Sm (the <S>(T) curve) can be passed from an
    AV_analysis result to avoid recomputing it."""
    xmin, xmax = size_range
    tmin, tmax = dur_range
    alpha, n_size = discrete_powerlaw_mle(S, xmin, xmax)
    beta, n_dur = discrete_powerlaw_mle(T, tmin, tmax)

    if TT is None or Sm is None:
        # <S>(T), computed exactly as in AV_analysis
        TT = np.arange(1, np.max(T) + 1)
        Sm = []
        for i in np.arange(0, np.size(TT)):
            Sm.append(np.mean(S[np.where(T == TT[i])[0]]))
        Sm = np.asarray(Sm)
        Loc = np.where(Sm > 0)[0]
        TT = TT[Loc]
        Sm = Sm[Loc]

    in_win = np.where(np.logical_and(TT > tmin, TT < tmax))[0]
    fit_sigma = np.polyfit(np.log(TT[in_win]), np.log(Sm[in_win]), 1)
    sigma = (beta - 1) / (alpha - 1)

    return {"alpha": alpha, "beta": beta, "df": np.abs(sigma - fit_sigma[0]),
            "xmin": xmin, "xmax": xmax, "tmin": tmin, "tmax": tmax,
            "n_size_fit": n_size, "n_dur_fit": n_dur}


def analyze_one_run(batch_label, run_label, h_ip, N_e, raster, pop_counts, n_avalanches):
    """Per-run metrics -- everything except the pooled avalanche fit."""
    row = {"batch": batch_label, "run": run_label, "h_ip": h_ip, "N_e": N_e,
           "has_full_raster": raster is not None, "n_avalanches": n_avalanches}

    br_input = raster.astype(float) if raster is not None else pop_counts
    br = crt.calculate_branching_ratio(
        br_input, k_max=BRANCHING_KMAX, name=run_label,
        fitfuncs=BRANCHING_FITFUNCS, plot_targetdir=None, lreturn_tau=0,
    )
    br_dict = {br[i]: br[i + 1] for i in range(0, len(br), 2)}
    row["branching_ratio_exp"] = br_dict.get("exp")
    row["branching_ratio_complex"] = br_dict.get("complex")

    row["dfa"] = nolds.dfa(pop_counts, nvals=DFA_NVALS)

    if raster is not None:
        row["branching_parameter"] = crt.branchparam(raster.astype(float))  # copies via astype -- branchparam mutates its input
        susc, fano = crt.population_metrics(raster.astype(float))
        row["susceptibility"] = susc
        row["fano_factor"] = fano
        activity = float(raster.mean())               # mean firing rate per neuron per bin (old's rho())
        row["activity"] = activity
        row["cv"] = (susc ** 0.5) / activity if activity > 0 else None  # old's cv() formula
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

    for batch_dir in sorted(glob.glob(os.path.join(SWEEP_ROOT, CONFIG_NAME, BATCH_GLOB))):
        batch_label = os.path.basename(batch_dir)
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
                h_ip, N_e, raster, pop_counts, S, T, thr_used = get_run_avalanches(h5path)
                batch_h_ip = h_ip
                batch_S.append(S)
                batch_T.append(T)

                row = analyze_one_run(batch_label, run_label, h_ip, N_e, raster, pop_counts, len(S))
                row["av_threshold_used"] = thr_used
                per_run_rows.append(row)
                print(f"    h_ip={h_ip:.3f}  threshold={thr_used}  n_avalanches={len(S)}  "
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

        av_res = None
        try:
            av_res = run_av_analysis(pooled_S, pooled_T, pltname=batch_label + "_pooled_")
            batch_row = {
                "batch": batch_label,
                "fit_mode": "auto",
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
            if SIZE_FIT_RANGE is None or DUR_FIT_RANGE is None:
                del batch_row["fit_mode"]   # keep the original CSV schema
            batch_rows.append(batch_row)
            print(f"  pooled fit: alpha={batch_row['alpha_size_exponent']}  "
                  f"beta={batch_row['beta_duration_exponent']}  DCC={batch_row['dcc']}")
        except Exception:
            print(f"  FAILED pooled AV_analysis for {batch_label}:")
            traceback.print_exc()

        if SIZE_FIT_RANGE is not None and DUR_FIT_RANGE is not None:
            try:
                man = manual_av_fit(
                    pooled_S, pooled_T, SIZE_FIT_RANGE, DUR_FIT_RANGE,
                    TT=av_res.get("TT") if av_res is not None else None,
                    Sm=av_res.get("Sm") if av_res is not None else None,
                )
                batch_rows.append({
                    "batch": batch_label,
                    "fit_mode": "manual",
                    "h_ip": batch_h_ip,
                    "n_runs_pooled": len(batch_S),
                    "n_avalanches_pooled": len(pooled_S),
                    "alpha_size_exponent": man["alpha"],
                    "beta_duration_exponent": man["beta"],
                    "dcc": man["df"],
                    "xmin_burst": man["xmin"],
                    "xmax_burst": man["xmax"],
                    "xmin_dur": man["tmin"],
                    "xmax_dur": man["tmax"],
                    "excluded_burst_fit": None,
                    "excluded_dur_fit": None,
                })
                print(f"  manual fit (size {SIZE_FIT_RANGE}, dur {DUR_FIT_RANGE}): "
                      f"alpha={man['alpha']}  beta={man['beta']}  DCC={man['df']}")
            except Exception:
                print(f"  FAILED manual fixed-window fit for {batch_label}:")
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