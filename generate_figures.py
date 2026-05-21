#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for the WVC paper.

Reads the four CSV files produced by regenerate_all_data.py and produces
publication-grade PNG figures for the manuscript:

  figure_2_headline.png    Headline three-mode comparison
  figure_3_spacing.png     Radar spacing sweep
  figure_4_size.png        Animal size sweep
  figure_5_kappa.png       Baseline sensor sensitivity sweep

Style: seaborn whitegrid theme, color-blind-friendly palette, 300 DPI.
Suitable for direct insertion into the MDPI manuscript at column width
(approximately 6.5 inches wide × 5 inches tall, 4-panel grid).

Usage:
  python3 generate_figures.py                # generate all four
  python3 generate_figures.py --only 2 3     # generate only figures 2 and 3
  python3 generate_figures.py --dpi 600      # use higher DPI

Inputs (in script's directory):
  wvc_results.csv
  wvc_results_spacing.csv
  wvc_results_size.csv
  wvc_results_detection_rate.csv

Outputs (in script's directory):
  figure_2_headline.png
  figure_3_spacing.png
  figure_4_size.png
  figure_5_kappa.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# ============================================================
# Style setup
# ============================================================

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444441",
    "axes.labelcolor": "#1f2937",
    "xtick.color": "#1f2937",
    "ytick.color": "#1f2937",
    "axes.titleweight": "normal",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 100,
})

# Color-blind-friendly palette consistent with the Figure 1 architecture diagram.
# Control = neutral gray, Detection = teal-green, Aware = warm amber.
COLORS = {"control": "#5F5E5A", "detection": "#0F6E56", "aware": "#BA7517"}
MODE_ORDER = ["control", "detection", "aware"]
MODE_LABELS = {"control": "Control", "detection": "Detection", "aware": "Aware"}


# ============================================================
# Statistics helpers
# ============================================================

def welch_p(a, b):
    """Two-sided Welch's t-test p-value. Returns NaN for empty/single-element."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return p


def sig_marker(p):
    """Return *, **, ***, or '' for the conventional significance levels."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def annotate_significance(ax, x, y_top, p, fontsize=8, color="#1f2937"):
    """Draw a significance marker above (x, y_top)."""
    m = sig_marker(p)
    if not m:
        return
    ax.annotate(m, xy=(x, y_top), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize, color=color)


# ============================================================
# Figure 2 — Headline three-mode comparison
# ============================================================

def make_figure_2(out_path, dpi=300):
    """
    Four panels:
      (a) Collision rate per road entry (%) — the headline metric
      (b) Road entries per trial (throughput) — the dual mechanism
      (c) Cumulative frozen-on-road time (s) — safety mechanism
      (d) Detection rate (%) — sensor reliability check
    """
    df = pd.read_csv("wvc_results.csv")
    df["col_rate_pct"] = df["col_rate"] * 100
    df["det_rate_pct"] = df["det_rate"] * 100

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    fig.suptitle("Three-mode Monte Carlo comparison (20 trials × 4 h per mode)",
                 fontsize=11, y=0.995)

    # ---- (a) Collision rate ----
    ax = axes[0, 0]
    sns.boxplot(data=df, x="mode", y="col_rate_pct", order=MODE_ORDER,
                hue="mode", palette=COLORS, ax=ax, width=0.55,
                showfliers=False, linewidth=0.7, legend=False)
    sns.stripplot(data=df, x="mode", y="col_rate_pct", order=MODE_ORDER,
                  color="black", alpha=0.45, size=3.2, jitter=0.15, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Collision rate per road entry (%)")
    ax.set_title("(a) Collision rate")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])

    # Significance bracket Control vs Aware
    ctrl_rates = df[df["mode"] == "control"]["col_rate_pct"].values
    aware_rates = df[df["mode"] == "aware"]["col_rate_pct"].values
    p_ca = welch_p(ctrl_rates, aware_rates)
    ymax = df["col_rate_pct"].max()
    y_brack = ymax * 1.08
    ax.plot([0, 0, 2, 2], [y_brack, y_brack * 1.04, y_brack * 1.04, y_brack],
            color="#1f2937", lw=0.6)
    ax.text(1.0, y_brack * 1.06, sig_marker(p_ca) or "n.s.",
            ha="center", va="bottom", fontsize=8, color="#1f2937")
    ax.set_ylim(top=y_brack * 1.20)

    # ---- (b) Road entries (throughput) ----
    ax = axes[0, 1]
    sns.boxplot(data=df, x="mode", y="road_entries", order=MODE_ORDER,
                hue="mode", palette=COLORS, ax=ax, width=0.55,
                showfliers=False, linewidth=0.7, legend=False)
    sns.stripplot(data=df, x="mode", y="road_entries", order=MODE_ORDER,
                  color="black", alpha=0.45, size=3.2, jitter=0.15, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Road entries per trial")
    ax.set_title("(b) Road-crossing throughput")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])

    # ---- (c) Frozen-on-road time ----
    ax = axes[1, 0]
    sns.boxplot(data=df, x="mode", y="frozen_on_road_s", order=MODE_ORDER,
                hue="mode", palette=COLORS, ax=ax, width=0.55,
                showfliers=False, linewidth=0.7, legend=False)
    sns.stripplot(data=df, x="mode", y="frozen_on_road_s", order=MODE_ORDER,
                  color="black", alpha=0.45, size=3.2, jitter=0.15, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative frozen-on-road time (s)")
    ax.set_title("(c) Frozen-on-road time")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])

    # ---- (d) Detection rate ----
    ax = axes[1, 1]
    # Filter out Control (no detections by construction)
    det_df = df[df["mode"] != "control"].copy()
    sns.boxplot(data=det_df, x="mode", y="det_rate_pct",
                order=["detection", "aware"],
                hue="mode", palette={k: COLORS[k] for k in ["detection", "aware"]},
                ax=ax, width=0.5, showfliers=False, linewidth=0.7, legend=False)
    sns.stripplot(data=det_df, x="mode", y="det_rate_pct",
                  order=["detection", "aware"],
                  color="black", alpha=0.45, size=3.2, jitter=0.15, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("(d) Sensor detection rate")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([MODE_LABELS[m] for m in ["detection", "aware"]])
    ax.set_ylim(90, 102)
    # Note about Control mode using axis-relative coordinates
    ax.text(0.02, 0.05, "(Control: 0% by construction, no sensors)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, style="italic", color="#5F5E5A")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================
# Sweep figures — shared template
# ============================================================

def _sweep_summary(df, sweep_col):
    """Compute per (sweep_value, mode) summary statistics, with SEM and CI95."""
    # n_trials per point/mode is constant across the sweep — get from data
    grouped = df.groupby([sweep_col, "mode"]).agg(
        mean_rate=("col_rate", lambda x: 100 * x.mean()),
        sd_rate=("col_rate", lambda x: 100 * x.std()),
        sem_rate=("col_rate", lambda x: 100 * x.std() / np.sqrt(len(x))),
        mean_det=("det_rate", lambda x: 100 * x.mean()),
        sd_det=("det_rate", lambda x: 100 * x.std()),
        sem_det=("det_rate", lambda x: 100 * x.std() / np.sqrt(len(x))),
        mean_lat=("in_range_latency_s", "mean"),
        sd_lat=("in_range_latency_s", "std"),
        sem_lat=("in_range_latency_s", lambda x: x.std() / np.sqrt(len(x))),
        mean_ent=("road_entries", "mean"),
        sd_ent=("road_entries", "std"),
        sem_ent=("road_entries", lambda x: x.std() / np.sqrt(len(x))),
        n=("col_rate", "count"),
    ).reset_index()
    return grouped


def _plot_mean_with_sem_band(ax, x, mean, sem, color, label, marker="o", lw=1.6, ms=5, alpha_band=0.18):
    """Plot a mean line with a shaded ±1 SEM band, markers at each x."""
    x = np.asarray(x)
    mean = np.asarray(mean)
    sem = np.asarray(sem)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha_band, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=lw, marker=marker, markersize=ms, label=label)


def _ctrl_vs_det_pvalues(df, sweep_col):
    """For each sweep point, Welch's t p-value Control vs Detection (col_rate)."""
    out = []
    for val in sorted(df[sweep_col].unique()):
        ctrl = df[(df[sweep_col] == val) & (df["mode"] == "control")]["col_rate"].values
        det = df[(df[sweep_col] == val) & (df["mode"] == "detection")]["col_rate"].values
        p = welch_p(ctrl, det)
        out.append((val, p))
    return out


def _sweep_plot(csv_path, sweep_col, x_label, title_prefix, out_path,
                x_log=False, x_ticks=None, dpi=300):
    """Generic 4-panel sweep figure: collision rate, detection rate, latency, throughput."""
    df = pd.read_csv(csv_path)
    summary = _sweep_summary(df, sweep_col)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    fig.suptitle(title_prefix, fontsize=11, y=0.995)

    # ---- (a) Collision rate vs sweep ----
    ax = axes[0, 0]
    for mode in MODE_ORDER:
        d = summary[summary["mode"] == mode].sort_values(sweep_col)
        _plot_mean_with_sem_band(ax, d[sweep_col].values, d["mean_rate"].values,
                                  d["sem_rate"].values, color=COLORS[mode],
                                  label=MODE_LABELS[mode])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Collision rate per road entry (%)")
    ax.set_title("(a) Collision rate")
    ax.legend(loc="upper left" if not x_log else "upper right", framealpha=0.9, fontsize=7)
    if x_log:
        ax.set_xscale("log")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])
    ax.set_ylim(bottom=0)

    # ---- (b) Reduction by layer, with significance markers in a top row ----
    ax = axes[0, 1]
    pivot_means = summary.pivot(index=sweep_col, columns="mode", values="mean_rate").sort_index()
    reduction_det = 100 * (pivot_means["control"] - pivot_means["detection"]) / pivot_means["control"]
    reduction_aware = 100 * (pivot_means["control"] - pivot_means["aware"]) / pivot_means["control"]
    ax.plot(reduction_det.index, reduction_det.values, "o-", color=COLORS["detection"],
            label="Control → Detection", markersize=5, linewidth=1.4)
    ax.plot(reduction_aware.index, reduction_aware.values, "s-", color=COLORS["aware"],
            label="Control → Aware", markersize=5, linewidth=1.4)
    ax.axhline(0, color="#888780", linewidth=0.5, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Collision-rate reduction (%)")
    ax.set_title("(b) Reduction vs Control")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    if x_log:
        ax.set_xscale("log")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])
    # Compute pvalues for Control vs Detection at each point
    pvs = dict(_ctrl_vs_det_pvalues(df, sweep_col))
    # Place significance markers in a fixed row near the top of the chart
    ymax_data = max(reduction_det.max(), reduction_aware.max())
    ymin_data = min(reduction_det.min(), reduction_aware.min(), 0)
    y_range = ymax_data - ymin_data
    # Reserve top 12% of chart for the significance row
    chart_top = ymax_data + y_range * 0.20
    sig_y = ymax_data + y_range * 0.10
    ax.set_ylim(top=chart_top, bottom=min(0, ymin_data - y_range * 0.05))
    # Draw the significance markers as a clearly-visible top row
    for val, p in pvs.items():
        m = sig_marker(p)
        if m and val in reduction_det.index:
            ax.text(val, sig_y, m, ha="center", va="center",
                    fontsize=10, color="#1f2937", fontweight="bold")
    # Subtle horizontal line separating significance row from data
    ax.axhline(sig_y - y_range * 0.04, color="#cccccc", linewidth=0.4, linestyle=":")

    # ---- (c) Detection rate vs sweep ----
    ax = axes[1, 0]
    for mode in ["detection", "aware"]:
        d = summary[summary["mode"] == mode].sort_values(sweep_col)
        _plot_mean_with_sem_band(ax, d[sweep_col].values, d["mean_det"].values,
                                  d["sem_det"].values, color=COLORS[mode],
                                  label=MODE_LABELS[mode])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("(c) Sensor detection rate")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    if x_log:
        ax.set_xscale("log")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])
    ax.set_ylim(0, 105)

    # ---- (d) In-range latency vs sweep ----
    ax = axes[1, 1]
    for mode in ["detection", "aware"]:
        d = summary[summary["mode"] == mode].sort_values(sweep_col)
        _plot_mean_with_sem_band(ax, d[sweep_col].values, d["mean_lat"].values,
                                  d["sem_lat"].values, color=COLORS[mode],
                                  label=MODE_LABELS[mode])
    ax.set_xlabel(x_label)
    ax.set_ylabel("In-range detection latency (s)")
    ax.set_title("(d) Detection latency")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    if x_log:
        ax.set_xscale("log")
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_figure_3(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_spacing.csv",
        sweep_col="radar_spacing",
        x_label="Radar spacing (m, alternating)",
        title_prefix="Radar spacing sweep (15 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[5, 10, 15, 20, 25, 30, 40],
        dpi=dpi,
    )


def make_figure_4(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_size.csv",
        sweep_col="size_scale",
        x_label="Animal size scaling factor σ_scale",
        title_prefix="Animal size sweep (15 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        dpi=dpi,
    )


def make_figure_5(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_detection_rate.csv",
        sweep_col="detection_rate_per_sec",
        x_label="Baseline detection rate κ (s⁻¹)",
        title_prefix="Sensor sensitivity sweep (15 trials × 2 h per point)",
        out_path=out_path,
        x_log=True,
        x_ticks=[0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
        dpi=dpi,
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+", choices=[2, 3, 4, 5],
                    help="generate only specific figures (default: all)")
    ap.add_argument("--dpi", type=int, default=300, help="output DPI (default: 300)")
    args = ap.parse_args()

    only = args.only or [2, 3, 4, 5]
    print(f"Generating figures: {only} at {args.dpi} DPI")

    if 2 in only:
        make_figure_2("figure_2_headline.png", dpi=args.dpi)
    if 3 in only:
        make_figure_3("figure_3_spacing.png", dpi=args.dpi)
    if 4 in only:
        make_figure_4("figure_4_size.png", dpi=args.dpi)
    if 5 in only:
        make_figure_5("figure_5_kappa.png", dpi=args.dpi)

    print("done.")


if __name__ == "__main__":
    main()