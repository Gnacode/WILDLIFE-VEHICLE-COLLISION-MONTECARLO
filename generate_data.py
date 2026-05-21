#!/usr/bin/env python3
"""
regenerate_all_data.py
======================

Regenerates ALL CSV data and summary statistics for the corrected (Fix A)
WVC simulator model. Run this once after applying the Fix A simulator update;
it produces:

  - wvc_results.csv               (headline 20 trials x 4h x 3 modes)
  - wvc_results_spacing.csv       (7 spacings x 3 modes x 15 trials x 2h)
  - wvc_results_size.csv          (7 sizes x 3 modes x 15 trials x 2h)
  - wvc_results_detection_rate.csv (6 kappa values x 3 modes x 15 trials x 2h)
  - regeneration_summary.json     (machine-readable summary statistics)
  - regeneration_summary.txt      (human-readable summary)

Resumable: keeps a checkpoint at .regeneration_checkpoint.json so you can
Ctrl-C and re-run to resume. Delete that file to start fresh.

Usage:
  python3 regenerate_all_data.py            # run everything
  python3 regenerate_all_data.py --fresh    # delete checkpoint, run from scratch
  python3 regenerate_all_data.py --skip-sweeps  # only headline (faster)

Expected wall time on a modern laptop: 30-60 minutes total.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# --- Make sure we can find wvc_simulator.py in the script's directory ---
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

try:
    import wvc_simulator
except ImportError as e:
    print(f"FATAL: cannot import wvc_simulator.py from {HERE}", file=sys.stderr)
    print(f"       make sure this script lives in the same folder as wvc_simulator.py", file=sys.stderr)
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = HERE / ".regeneration_checkpoint.json"

HEADLINE_TRIALS = 20
HEADLINE_HOURS = 4.0

SWEEP_TRIALS = 15
SWEEP_HOURS = 2.0

SWEEPS = [
    # (label, param_name, values)
    ("spacing",        "radar_spacing",          [5, 10, 15, 20, 25, 30, 40]),
    ("size",           "size_scale",             [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    ("detection_rate", "detection_rate_per_sec", [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]),
]


# ============================================================
# Helpers
# ============================================================

def ts():
    return time.strftime("%H:%M:%S")


def welch_t(a, b):
    """Welch's t-test for unequal variances. Returns (t, df)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 0.0
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1, v2 = statistics.variance(a), statistics.variance(b)
    if v1 + v2 == 0:
        return 0.0, n1 + n2 - 2
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else 0.0
    if v1 > 0 and v2 > 0:
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    else:
        df = n1 + n2 - 2
    return t, df


def significance(t):
    a = abs(t)
    if a > 3.5: return "***"
    if a > 2.7: return "**"
    if a > 2.0: return "*"
    return "n.s."


def summarize_mode(trials, mode):
    cols = [t["collisions"] for t in trials]
    ents = [t["road_entries"] for t in trials]
    detected = [t.get("detected", 0) for t in trials]
    rates = [t["col_rate"] * 100 for t in trials]  # col_rate is fraction
    latencies = [t.get("in_range_latency_s", 0) for t in trials if t.get("detected", 0) > 0]
    return dict(
        cols=cols,
        rates=rates,
        mean_col=statistics.mean(cols),
        sd_col=statistics.stdev(cols) if len(cols) > 1 else 0,
        mean_ent=statistics.mean(ents),
        mean_rate=statistics.mean(rates),
        sd_rate=statistics.stdev(rates) if len(rates) > 1 else 0,
        det_rate=100.0 * sum(detected) / max(sum(ents), 1) if mode != "control" else 0.0,
        mean_latency=statistics.mean(latencies) if latencies else 0.0,
        sd_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    )


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            return json.load(open(CHECKPOINT_PATH))
        except Exception:
            return {"steps_done": []}
    return {"steps_done": []}


def save_checkpoint(state):
    json.dump(state, open(CHECKPOINT_PATH, "w"))


def mark_done(state, step):
    if step not in state["steps_done"]:
        state["steps_done"].append(step)
        save_checkpoint(state)


# ============================================================
# Headline experiment
# ============================================================

def run_headline(state):
    if "headline" in state["steps_done"] and Path("wvc_results.csv").exists():
        print(f"[{ts()}] Skipping headline (already done)")
        return load_headline_summary()

    print(f"[{ts()}] === HEADLINE: {HEADLINE_TRIALS} trials x {HEADLINE_HOURS}h x 3 modes ===")
    cfg = wvc_simulator.Config()
    t0 = time.time()
    results = wvc_simulator.run_monte_carlo(
        cfg,
        n_trials=HEADLINE_TRIALS,
        duration_hr=HEADLINE_HOURS,
        modes=("control", "detection", "aware"),
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"[{ts()}] Headline done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    wvc_simulator.write_csv(results, "wvc_results.csv")
    print(f"[{ts()}] Wrote wvc_results.csv")

    summary = {"wall_time_s": elapsed, "n_trials": HEADLINE_TRIALS,
               "duration_hr": HEADLINE_HOURS, "modes": {}}
    for mode in ("control", "detection", "aware"):
        summary["modes"][mode] = summarize_mode(results[mode], mode)

    for name, a, b in [
        ("ctrl_vs_det", "control", "detection"),
        ("det_vs_aware", "detection", "aware"),
        ("ctrl_vs_aware", "control", "aware"),
    ]:
        # absolute collisions
        t, df = welch_t(summary["modes"][a]["cols"], summary["modes"][b]["cols"])
        red = 100 * (summary["modes"][a]["mean_col"] - summary["modes"][b]["mean_col"]) / max(summary["modes"][a]["mean_col"], 0.01)
        # collision rate
        t_r, df_r = welch_t(summary["modes"][a]["rates"], summary["modes"][b]["rates"])
        red_r = 100 * (summary["modes"][a]["mean_rate"] - summary["modes"][b]["mean_rate"]) / max(summary["modes"][a]["mean_rate"], 0.01)
        summary[name] = dict(
            t_col=t, df_col=df, reduction_col_pct=red,
            t_rate=t_r, df_rate=df_r, reduction_rate_pct=red_r,
            sig_col=significance(t), sig_rate=significance(t_r),
        )

    mark_done(state, "headline")
    return summary


def load_headline_summary():
    # Reconstruct from the CSV (in case checkpoint is set but JSON missing)
    import csv
    rows = list(csv.DictReader(open("wvc_results.csv")))
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append({
            "collisions": int(r["collisions"]),
            "road_entries": int(r["road_entries"]),
            "detected": int(r["detected"]),
            "col_rate": float(r["col_rate"]),
            "in_range_latency_s": float(r["in_range_latency_s"]),
        })
    summary = {"wall_time_s": -1, "n_trials": HEADLINE_TRIALS,
               "duration_hr": HEADLINE_HOURS, "modes": {}}
    for mode in ("control", "detection", "aware"):
        summary["modes"][mode] = summarize_mode(by_mode.get(mode, []), mode)
    for name, a, b in [
        ("ctrl_vs_det", "control", "detection"),
        ("det_vs_aware", "detection", "aware"),
        ("ctrl_vs_aware", "control", "aware"),
    ]:
        t, df = welch_t(summary["modes"][a]["cols"], summary["modes"][b]["cols"])
        red = 100 * (summary["modes"][a]["mean_col"] - summary["modes"][b]["mean_col"]) / max(summary["modes"][a]["mean_col"], 0.01)
        t_r, df_r = welch_t(summary["modes"][a]["rates"], summary["modes"][b]["rates"])
        red_r = 100 * (summary["modes"][a]["mean_rate"] - summary["modes"][b]["mean_rate"]) / max(summary["modes"][a]["mean_rate"], 0.01)
        summary[name] = dict(
            t_col=t, df_col=df, reduction_col_pct=red,
            t_rate=t_r, df_rate=df_r, reduction_rate_pct=red_r,
            sig_col=significance(t), sig_rate=significance(t_r),
        )
    return summary


# ============================================================
# Sweeps
# ============================================================

def run_sweep(label, param, values, state):
    out_csv = f"wvc_results_{label}.csv"
    step_id = f"sweep_{label}"
    if step_id in state["steps_done"] and Path(out_csv).exists():
        print(f"[{ts()}] Skipping {label} sweep (already done)")
        return load_sweep_summary(label, param, values)

    print(f"[{ts()}] === SWEEP {label}: {len(values)} pts x {SWEEP_TRIALS} trials x {SWEEP_HOURS}h x 3 modes ===")
    cfg = wvc_simulator.Config()
    t0 = time.time()

    # Run point-by-point so we can print progress
    sweep_results = {}
    for i, val in enumerate(values):
        cfg_dict = wvc_simulator.Config().__dict__.copy()
        cfg_dict[param] = val
        cfg_local = wvc_simulator.Config(**cfg_dict)
        pt0 = time.time()
        mc = wvc_simulator.run_monte_carlo(
            cfg_local,
            n_trials=SWEEP_TRIALS,
            duration_hr=SWEEP_HOURS,
            modes=("control", "detection", "aware"),
            verbose=False,
        )
        sweep_results[val] = mc
        pt_el = time.time() - pt0
        # Quick progress summary
        det = sweep_results[val]["detection"]
        rates = [t["col_rate"] * 100 for t in det]
        print(f"[{ts()}]   point {i+1}/{len(values)}: {param}={val:>6}: det rate = {statistics.mean(rates):5.2f}% (elapsed {pt_el:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{ts()}] {label} sweep done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    wvc_simulator.write_sweep_csv(sweep_results, param, out_csv)
    print(f"[{ts()}] Wrote {out_csv}")

    summary = {"wall_time_s": elapsed, "param": param, "values": list(values), "points": {}}
    for val, mc in sweep_results.items():
        pt = {}
        for mode in ("control", "detection", "aware"):
            pt[mode] = summarize_mode(mc[mode], mode)
        # pairwise stats (rate)
        for name, a, b in [
            ("ctrl_vs_det", "control", "detection"),
            ("det_vs_aware", "detection", "aware"),
        ]:
            t_r, df_r = welch_t(pt[a]["rates"], pt[b]["rates"])
            red_r = 100 * (pt[a]["mean_rate"] - pt[b]["mean_rate"]) / max(pt[a]["mean_rate"], 0.01)
            pt[name] = dict(t_rate=t_r, df_rate=df_r, reduction_rate_pct=red_r, sig_rate=significance(t_r))
        summary["points"][str(val)] = pt

    mark_done(state, step_id)
    return summary


def load_sweep_summary(label, param, values):
    # Reconstruct from the CSV
    import csv
    out_csv = f"wvc_results_{label}.csv"
    if not Path(out_csv).exists():
        return None
    rows = list(csv.DictReader(open(out_csv)))
    by_val = {}
    for r in rows:
        v = r.get(param) or r.get("sweep_value") or r.get("value")
        try:
            v = float(v)
            if v.is_integer(): v = int(v)
        except Exception:
            pass
        by_val.setdefault(v, {}).setdefault(r["mode"], []).append({
            "collisions": int(r["collisions"]),
            "road_entries": int(r["road_entries"]),
            "detected": int(r["detected"]),
            "col_rate": float(r["col_rate"]),
            "in_range_latency_s": float(r["in_range_latency_s"]),
        })
    summary = {"wall_time_s": -1, "param": param, "values": list(values), "points": {}}
    for val in values:
        if val not in by_val:
            continue
        pt = {}
        mc = by_val[val]
        for mode in ("control", "detection", "aware"):
            pt[mode] = summarize_mode(mc.get(mode, []), mode)
        for name, a, b in [
            ("ctrl_vs_det", "control", "detection"),
            ("det_vs_aware", "detection", "aware"),
        ]:
            t_r, df_r = welch_t(pt[a]["rates"], pt[b]["rates"])
            red_r = 100 * (pt[a]["mean_rate"] - pt[b]["mean_rate"]) / max(pt[a]["mean_rate"], 0.01)
            pt[name] = dict(t_rate=t_r, df_rate=df_r, reduction_rate_pct=red_r, sig_rate=significance(t_r))
        summary["points"][str(val)] = pt
    return summary


# ============================================================
# Summary writers
# ============================================================

def write_summary(headline, sweeps):
    # JSON (machine readable, includes raw per-trial cols/rates lists)
    combined = {"headline": headline, "sweeps": sweeps,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_version": "Fix A (unified two-branch with threat-suppressed crossing)"}
    with open("regeneration_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"[{ts()}] Wrote regeneration_summary.json")

    # Human-readable text
    with open("regeneration_summary.txt", "w", encoding="utf-8") as f:
        f.write("WVC SIMULATION — REGENERATED RESULTS (Fix A model)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: corrected two-branch with threat-suppressed crossing\n\n")

        # Headline table
        f.write("HEADLINE EXPERIMENT\n")
        f.write("-" * 60 + "\n")
        n_modes = 3
        f.write(f"{HEADLINE_TRIALS} trials × {HEADLINE_HOURS}h per mode ({HEADLINE_TRIALS * n_modes} total trials)\n\n")
        f.write(f"{'Mode':12s} {'Collisions':>14s} {'Rate %':>14s} {'DetRate %':>10s} {'Latency s':>12s}\n")
        for mode in ("control", "detection", "aware"):
            d = headline["modes"][mode]
            f.write(f"{mode:12s} {d['mean_col']:6.2f} ± {d['sd_col']:5.2f}  "
                    f"{d['mean_rate']:6.2f} ± {d['sd_rate']:5.2f}  "
                    f"{d['det_rate']:9.2f}  "
                    f"{d['mean_latency']:6.3f} ± {d['sd_latency']:.3f}\n")

        f.write("\nPairwise (collision rate per road entry):\n")
        for name, label in [("ctrl_vs_det",   "Control vs Detection"),
                            ("det_vs_aware",  "Detection vs Aware"),
                            ("ctrl_vs_aware", "Control vs Aware")]:
            d = headline[name]
            f.write(f"  {label:24s}: t = {d['t_rate']:+.3f}, df = {d['df_rate']:.1f}, "
                    f"reduction = {d['reduction_rate_pct']:+.1f}%  {d['sig_rate']}\n")

        # Sweeps
        for sw_label, sw in sweeps.items():
            if not sw: continue
            f.write(f"\n\nSWEEP: {sw_label.upper()}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Parameter: {sw['param']}\n")
            f.write(f"{'value':>10s}  {'control':>20s}  {'detection':>20s}  {'aware':>20s}  {'ctrl→det red':>14s}\n")
            for val, pt in sw["points"].items():
                ctrl = pt["control"]; det = pt["detection"]; aw = pt["aware"]
                cd = pt.get("ctrl_vs_det", {})
                f.write(f"{val:>10s}  "
                        f"{ctrl['mean_rate']:6.2f} ± {ctrl['sd_rate']:4.2f}%   "
                        f"{det['mean_rate']:6.2f} ± {det['sd_rate']:4.2f}%   "
                        f"{aw['mean_rate']:6.2f} ± {aw['sd_rate']:4.2f}%   "
                        f"{cd.get('reduction_rate_pct', 0):+6.1f}% {cd.get('sig_rate', '')}\n")
    print(f"[{ts()}] Wrote regeneration_summary.txt")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="delete checkpoint and start over")
    ap.add_argument("--skip-sweeps", action="store_true", help="only run headline, not sweeps")
    args = ap.parse_args()

    if args.fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"[{ts()}] Deleted checkpoint, starting fresh")

    state = load_checkpoint()
    overall_t0 = time.time()

    print(f"[{ts()}] Working directory: {HERE}")
    print(f"[{ts()}] Model: corrected (Fix A)")
    print()

    headline = run_headline(state)

    sweeps = {}
    if not args.skip_sweeps:
        for label, param, values in SWEEPS:
            sweeps[label] = run_sweep(label, param, values, state)
    else:
        print(f"[{ts()}] Skipping sweeps (--skip-sweeps)")

    write_summary(headline, sweeps)

    elapsed = time.time() - overall_t0
    print()
    print(f"[{ts()}] === ALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min) ===")
    print()
    print("Files produced:")
    for p in ["wvc_results.csv", "wvc_results_spacing.csv", "wvc_results_size.csv",
              "wvc_results_detection_rate.csv", "regeneration_summary.json",
              "regeneration_summary.txt"]:
        if Path(p).exists():
            size_kb = Path(p).stat().st_size / 1024
            print(f"  {p}  ({size_kb:.1f} KB)")
        else:
            print(f"  {p}  (missing)")


if __name__ == "__main__":
    main()