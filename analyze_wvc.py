#!/usr/bin/env python3
"""
WVC sensor-network paper data page generator.

Reads Monte Carlo CSVs produced by wvc_simulator.py and renders an interactive
data page suitable for GitHub Pages hosting. Output is a single self-contained
index.html with embedded Plotly figures plus per-section pages for direct sharing.

Usage:
    python analyze_wvc.py                            # scans current dir, writes ./docs/
    python analyze_wvc.py --input ./data --output ./docs
    python analyze_wvc.py --paper-title "..." --authors "Thomsen, Makovetskyi"

The script auto-discovers headline (mode-only) and sweep (param-prefixed) CSVs,
falling back to backup files if the primary is missing. Re-run it whenever a
new sweep CSV is added — sections appear automatically.

Author: Generated for Thomsen & Makovetskyi WVC paper, May 2026.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as sp_stats


# ============================================================
#  Constants — visual identity
# ============================================================

MODE_COLORS = {
    "control":   "#dc2626",   # warm red
    "detection": "#ea580c",   # amber
    "aware":     "#15803d",   # forest green
}
MODE_LABELS = {
    "control":   "Control",
    "detection": "Detection",
    "aware":     "Aware",
}
MODE_ORDER = ["control", "detection", "aware"]

# Sweep parameter -> human-readable axis label + units
SWEEP_LABELS = {
    "radar_spacing":          ("Radar spacing", "m"),
    "radar_range":            ("Radar range", "m"),
    "size_scale":             ("Animal size scale (RCS factor)", "×"),
    "detection_rate_per_sec": ("Baseline detection rate κ", "s⁻¹"),
    "animal_rate_per_hr":     ("Animal arrival rate", "hr⁻¹"),
    "caution_speed_kmh":      ("Caution speed", "km/h"),
    "cruise_speed_kmh":       ("Cruise speed", "km/h"),
    "driver_reaction_s":      ("Driver reaction time", "s"),
}

# Section anchor key per sweep parameter (used in URLs)
SWEEP_SLUGS = {
    "radar_spacing":          "spacing",
    "radar_range":            "range",
    "size_scale":             "size",
    "detection_rate_per_sec": "sensitivity",
    "animal_rate_per_hr":     "arrival-rate",
    "caution_speed_kmh":      "caution-speed",
    "cruise_speed_kmh":       "cruise-speed",
    "driver_reaction_s":      "reaction-time",
}

PLOTLY_FONT = dict(family="'IBM Plex Sans', system-ui, sans-serif", size=12, color="#1f2937")
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=PLOTLY_FONT,
    title_font=dict(family="'Newsreader', Georgia, serif", size=15, color="#0f172a"),
    margin=dict(l=64, r=32, t=72, b=56),
    hoverlabel=dict(font_family="'IBM Plex Mono', monospace"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    showlegend=True,
)
PIO_CONFIG = dict(displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"], responsive=True)


# ============================================================
#  CSV discovery and loading
# ============================================================

def _normalize_dataframe(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """Strip whitespace from column names, locate and normalize the 'mode' column,
    lowercase mode values. Returns (normalized_df, warning_messages).
    """
    warnings: List[str] = []
    # Strip BOMs and whitespace from column names
    new_cols = [c.lstrip("\ufeff").strip() for c in df.columns]
    if new_cols != list(df.columns):
        df = df.rename(columns=dict(zip(df.columns, new_cols)))

    # Locate the 'mode' column case-insensitively
    mode_col = next((c for c in df.columns if c.strip().lower() == "mode"), None)
    if mode_col is None:
        warnings.append(f"{source_name}: no 'mode' column found "
                        f"(have: {list(df.columns)[:8]}{'…' if len(df.columns) > 8 else ''})")
        return df, warnings
    if mode_col != "mode":
        df = df.rename(columns={mode_col: "mode"})

    # Normalize mode values: cast to string, strip, lowercase
    df["mode"] = df["mode"].astype(str).str.strip().str.lower()

    # Validate that at least one row matches a known mode
    found = set(df["mode"].unique())
    known = set(MODE_ORDER)
    matched = found & known
    if not matched:
        warnings.append(f"{source_name}: no rows match known modes {sorted(known)}. "
                        f"Found mode values: {sorted(found)}")
    elif found - known:
        unknown = sorted(found - known)
        warnings.append(f"{source_name}: ignoring {len(unknown)} unknown mode value(s): {unknown}")

    return df, warnings


@dataclass
class HeadlineData:
    df: pd.DataFrame
    source_file: str


@dataclass
class SweepData:
    df: pd.DataFrame
    sweep_param: str
    source_file: str
    label: str
    unit: str
    slug: str


def find_headline_csv(input_dir: Path) -> Optional[HeadlineData]:
    """Find the three-mode comparison CSV. Tries primary names, then backups."""
    candidates = [
        "wvc_results.csv", "wvc_results.cs",
        "wvc_results_backup.csv", "wvc_results_backup.cs",
        "wvc_results_backup2.csv", "wvc_results_backup2.cs",
    ]
    for name in candidates:
        path = input_dir / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] could not read {path.name}: {exc}", file=sys.stderr)
            continue
        df, msgs = _normalize_dataframe(df, name)
        for m in msgs:
            print(f"[warn] {m}", file=sys.stderr)
        if "mode" not in df.columns:
            continue
        # Skip files where no rows match known modes
        if not df["mode"].isin(MODE_ORDER).any():
            print(f"[warn] {name}: skipping — no rows match known modes", file=sys.stderr)
            continue
        return HeadlineData(df=df, source_file=name)
    return None


def discover_sweeps(input_dir: Path) -> List[SweepData]:
    """Auto-discover sweep CSVs in the input directory, returning one record per sweep parameter."""
    found: Dict[str, SweepData] = {}
    candidates = list(input_dir.glob("*sweep*.cs*")) + list(input_dir.glob("*sweep*.CSV"))
    for path in candidates:
        try:
            with open(path, encoding="utf-8-sig") as fh:
                header_line = fh.readline().strip()
        except Exception as exc:
            print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
            continue
        header = [c.strip() for c in header_line.split(",")]
        if not header or header[0].lower() == "mode":
            continue
        sweep_param = header[0]
        if sweep_param not in SWEEP_LABELS:
            print(f"[warn] unknown sweep parameter '{sweep_param}' in {path.name}; skipping",
                  file=sys.stderr)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] could not read {path.name}: {exc}", file=sys.stderr)
            continue
        df, msgs = _normalize_dataframe(df, path.name)
        for m in msgs:
            print(f"[warn] {m}", file=sys.stderr)
        if sweep_param not in df.columns or "mode" not in df.columns:
            continue
        if not df["mode"].isin(MODE_ORDER).any():
            print(f"[warn] {path.name}: skipping — no rows match known modes", file=sys.stderr)
            continue
        # Keep the most recently modified file if duplicates exist
        prior = found.get(sweep_param)
        if prior is not None:
            prior_path = input_dir / prior.source_file
            if prior_path.stat().st_mtime >= path.stat().st_mtime:
                continue
        label, unit = SWEEP_LABELS[sweep_param]
        slug = SWEEP_SLUGS[sweep_param]
        found[sweep_param] = SweepData(
            df=df, sweep_param=sweep_param, source_file=path.name,
            label=label, unit=unit, slug=slug,
        )
    preferred_order = ["radar_spacing", "size_scale", "detection_rate_per_sec",
                       "radar_range", "animal_rate_per_hr", "caution_speed_kmh",
                       "cruise_speed_kmh", "driver_reaction_s"]
    return [found[p] for p in preferred_order if p in found]


# ============================================================
#  Statistics
# ============================================================

def significance_label(p: float) -> Tuple[str, str]:
    """Return (symbol, CSS class) for a p-value."""
    if p < 0.001: return "***", "sig-3"
    if p < 0.01:  return "**",  "sig-2"
    if p < 0.05:  return "*",   "sig-1"
    return "n.s.", "sig-ns"


def per_mode_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute per-mode summary statistics."""
    out = {}
    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        in_range = sub.loc[sub["in_range_latency_s"] > 0, "in_range_latency_s"]
        spawn_lat = sub.loc[sub["mean_latency_s"] > 0, "mean_latency_s"]
        out[mode] = {
            "n_trials":           int(len(sub)),
            "collisions_mean":    float(sub["collisions"].mean()),
            "collisions_std":     float(sub["collisions"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "col_rate_mean_pct":  float(sub["col_rate"].mean()) * 100,
            "col_rate_std_pct":   float(sub["col_rate"].std(ddof=1)) * 100 if len(sub) > 1 else 0.0,
            "det_rate_mean_pct":  float(sub["det_rate"].mean()) * 100,
            "det_rate_std_pct":   float(sub["det_rate"].std(ddof=1)) * 100 if len(sub) > 1 else 0.0,
            "in_range_mean":      float(in_range.mean()) if not in_range.empty else 0.0,
            "in_range_std":       float(in_range.std(ddof=1)) if len(in_range) > 1 else 0.0,
            "spawn_lat_mean":     float(spawn_lat.mean()) if not spawn_lat.empty else 0.0,
            "road_entries_mean":  float(sub["road_entries"].mean()),
            "frozen_on_road_mean":float(sub["frozen_on_road_s"].mean()),
            "awareness_pct_mean": float(sub["awareness_pct"].mean()),
        }
    return out


def pairwise_tests(df: pd.DataFrame, metric: str = "collisions") -> List[Dict]:
    """Welch t-tests on a metric across every pair of modes present."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    out = []
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            a = df.loc[df["mode"] == m1, metric].to_numpy()
            b = df.loc[df["mode"] == m2, metric].to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = sp_stats.ttest_ind(a, b, equal_var=False)
            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            reduction = 100 * (1 - mean_b / mean_a) if mean_a > 0 else 0.0
            sym, cls = significance_label(p)
            out.append({
                "m1": m1, "m2": m2,
                "mean_m1": mean_a, "mean_m2": mean_b,
                "reduction_pct": reduction,
                "t": float(t), "p": float(p),
                "sig": sym, "sig_class": cls,
            })
    return out


def sweep_table(df: pd.DataFrame, sweep_param: str) -> pd.DataFrame:
    """Per-(sweep-value × mode) summary with t-tests vs the previous mode in the layering."""
    rows = []
    values = sorted(df[sweep_param].unique())
    for v in values:
        sub_v = df[df[sweep_param] == v]
        per_mode = {}
        for mode in MODE_ORDER:
            sub = sub_v[sub_v["mode"] == mode]
            if sub.empty:
                continue
            per_mode[mode] = sub
            n = len(sub)
            rows.append({
                "value": v,
                "mode": mode,
                "n": n,
                "col_rate_mean": float(sub["col_rate"].mean()) * 100,
                "col_rate_std": float(sub["col_rate"].std(ddof=1)) * 100 if n > 1 else 0.0,
                "collisions_mean": float(sub["collisions"].mean()),
                "collisions_std": float(sub["collisions"].std(ddof=1)) if n > 1 else 0.0,
                "det_rate_mean": float(sub["det_rate"].mean()) * 100,
                "in_range_mean": float(sub.loc[sub["in_range_latency_s"] > 0, "in_range_latency_s"].mean() or 0),
            })
    return pd.DataFrame(rows)


def sweep_reduction_summary(df: pd.DataFrame, sweep_param: str) -> pd.DataFrame:
    """For each sweep value, compute reduction percentages and t-tests across the three layers."""
    rows = []
    values = sorted(df[sweep_param].unique())
    for v in values:
        sub_v = df[df[sweep_param] == v]
        per_mode = {m: sub_v[sub_v["mode"] == m]["collisions"].to_numpy() for m in MODE_ORDER if m in sub_v["mode"].unique()}
        if not per_mode:
            continue
        row = {"value": v}
        for (m1, m2) in [("control", "detection"), ("control", "aware"), ("detection", "aware")]:
            if m1 not in per_mode or m2 not in per_mode:
                continue
            a, b = per_mode[m1], per_mode[m2]
            if len(a) < 2 or len(b) < 2:
                continue
            mean_a, mean_b = a.mean(), b.mean()
            reduction = 100 * (1 - mean_b / mean_a) if mean_a > 0 else 0.0
            t, p = sp_stats.ttest_ind(a, b, equal_var=False)
            sym, cls = significance_label(p)
            key = f"{m1}_vs_{m2}"
            row[f"{key}_reduction"] = reduction
            row[f"{key}_p"] = float(p)
            row[f"{key}_sig"] = sym
            row[f"{key}_sig_class"] = cls
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
#  Plotly figure builders
# ============================================================

def _apply_layout(fig: go.Figure, height: int = 560, title: str = "") -> None:
    fig.update_layout(**PLOTLY_LAYOUT, height=height, title=title)
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False, ticks="outside")
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False, ticks="outside")


def fig_headline_overview(df: pd.DataFrame, stats: Dict) -> go.Figure:
    """Four-panel headline figure: collisions, collision rate, detection rate, in-range latency."""
    modes = [m for m in MODE_ORDER if m in stats]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>Collisions per trial</b>",
                        "<b>Collision rate per road entry (%)</b>",
                        "<b>Detection rate (%)</b>",
                        "<b>In-range detection latency (s)</b>"),
        horizontal_spacing=0.14, vertical_spacing=0.20,
    )
    for idx, m in enumerate(modes):
        sub = df[df["mode"] == m]
        show_leg = idx == 0
        # Panel 1: collisions box+strip
        fig.add_trace(go.Box(
            y=sub["collisions"], name=MODE_LABELS[m],
            marker_color=MODE_COLORS[m],
            line_color=MODE_COLORS[m],
            boxpoints="all", jitter=0.35, pointpos=-1.6,
            marker=dict(size=5, opacity=0.55),
            showlegend=show_leg, legendgroup=m,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>Collisions: %{{y}}<extra></extra>",
        ), row=1, col=1)
        # Panel 2: collision rate
        fig.add_trace(go.Box(
            y=sub["col_rate"] * 100, name=MODE_LABELS[m],
            marker_color=MODE_COLORS[m], line_color=MODE_COLORS[m],
            boxpoints="all", jitter=0.35, pointpos=-1.6,
            marker=dict(size=5, opacity=0.55),
            showlegend=False, legendgroup=m,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>Rate: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=2)
        # Panel 3: detection rate
        fig.add_trace(go.Box(
            y=sub["det_rate"] * 100, name=MODE_LABELS[m],
            marker_color=MODE_COLORS[m], line_color=MODE_COLORS[m],
            boxpoints="all", jitter=0.35, pointpos=-1.6,
            marker=dict(size=5, opacity=0.55),
            showlegend=False, legendgroup=m,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>Detection: %{{y:.2f}}%<extra></extra>",
        ), row=2, col=1)
        # Panel 4: in-range latency (only modes that detect)
        if m != "control":
            lat = sub.loc[sub["in_range_latency_s"] > 0, "in_range_latency_s"]
            fig.add_trace(go.Box(
                y=lat, name=MODE_LABELS[m],
                marker_color=MODE_COLORS[m], line_color=MODE_COLORS[m],
                boxpoints="all", jitter=0.35, pointpos=-1.6,
                marker=dict(size=5, opacity=0.55),
                showlegend=False, legendgroup=m,
                hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>Latency: %{{y:.3f}} s<extra></extra>",
            ), row=2, col=2)
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    fig.update_yaxes(rangemode="tozero", row=2, col=2)
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    _apply_layout(fig, height=720)
    fig.update_layout(title="")
    return fig


def fig_state_visits(df: pd.DataFrame, stats: Dict) -> go.Figure:
    """Stacked-bar comparison of animal state visits across modes."""
    state_keys = ["n_foraging", "n_approaching", "n_hesitating", "n_crossing", "n_frozen", "n_fleeing"]
    state_labels = ["Foraging", "Approaching", "Hesitating", "Crossing", "Frozen", "Fleeing"]
    state_palette = ["#cbd5e1", "#94a3b8", "#fcd34d", "#fb923c", "#dc2626", "#0ea5e9"]
    modes = [m for m in MODE_ORDER if m in stats]
    fig = go.Figure()
    for sk, sl, color in zip(state_keys, state_labels, state_palette):
        fig.add_trace(go.Bar(
            x=[MODE_LABELS[m] for m in modes],
            y=[float(df.loc[df["mode"] == m, sk].mean()) for m in modes],
            name=sl,
            marker=dict(color=color, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{sl}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    _apply_layout(fig, height=420, title="Mean animal-state visits per trial")
    fig.update_yaxes(title_text="Mean visits per trial")
    return fig


def fig_sweep_overview(sweep: SweepData) -> go.Figure:
    """Four-panel sweep figure."""
    df = sweep.df
    values = sorted(df[sweep.sweep_param].unique())
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    axis_label = f"{sweep.label} ({sweep.unit})"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>Collision rate per road entry (%)</b>",
                        "<b>Collision reduction by layer (%)</b>",
                        "<b>Detection rate (%)</b>",
                        "<b>In-range detection latency (s)</b>"),
        horizontal_spacing=0.13, vertical_spacing=0.20,
    )

    # ---- Panel 1: collision rate per mode with error bars ----
    for m in modes:
        means, stds = [], []
        for v in values:
            sub = df[(df["mode"] == m) & (df[sweep.sweep_param] == v)]
            means.append(float(sub["col_rate"].mean()) * 100 if not sub.empty else None)
            stds.append(float(sub["col_rate"].std(ddof=1)) * 100 if len(sub) > 1 else 0.0)
        fig.add_trace(go.Scatter(
            x=values, y=means,
            error_y=dict(type="data", array=stds, thickness=1.2, width=4),
            mode="lines+markers",
            name=MODE_LABELS[m],
            line=dict(color=MODE_COLORS[m], width=2.2),
            marker=dict(size=8, line=dict(color="white", width=1)),
            legendgroup=m, showlegend=True,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>{sweep.label}: %{{x}} {sweep.unit}<br>Rate: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=1)

    # ---- Panel 2: reduction by layer ----
    def mean_collisions(mode, v):
        sub = df[(df["mode"] == mode) & (df[sweep.sweep_param] == v)]
        return float(sub["collisions"].mean()) if not sub.empty else float("nan")

    reduction_pairs = [
        ("control", "detection", "Detection vs Control",   "#ea580c", "solid"),
        ("control", "aware",     "Aware vs Control",        "#15803d", "solid"),
        ("detection", "aware",   "Aware vs Detection (boost)", "#2563eb", "dash"),
    ]
    for m1, m2, label, color, dash in reduction_pairs:
        if m1 not in modes or m2 not in modes:
            continue
        ys = []
        for v in values:
            a, b = mean_collisions(m1, v), mean_collisions(m2, v)
            ys.append(100 * (1 - b / a) if a and a > 0 else None)
        fig.add_trace(go.Scatter(
            x=values, y=ys, mode="lines+markers",
            name=label, line=dict(color=color, width=2.2, dash=dash),
            marker=dict(size=8, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{label}</b><br>{sweep.label}: %{{x}} {sweep.unit}<br>Reduction: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af", row=1, col=2)

    # ---- Panel 3: detection rate (modes that detect) ----
    for m in modes:
        if m == "control":
            continue
        ys = []
        for v in values:
            sub = df[(df["mode"] == m) & (df[sweep.sweep_param] == v)]
            ys.append(float(sub["det_rate"].mean()) * 100 if not sub.empty else None)
        fig.add_trace(go.Scatter(
            x=values, y=ys, mode="lines+markers",
            name=MODE_LABELS[m], line=dict(color=MODE_COLORS[m], width=2.2),
            marker=dict(size=8, line=dict(color="white", width=1)),
            legendgroup=m, showlegend=False,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>{sweep.label}: %{{x}} {sweep.unit}<br>Detection: %{{y:.2f}}%<extra></extra>",
        ), row=2, col=1)
    fig.update_yaxes(range=[0, 105], row=2, col=1)

    # ---- Panel 4: in-range latency ----
    for m in modes:
        if m == "control":
            continue
        means, stds = [], []
        for v in values:
            sub = df[(df["mode"] == m) & (df[sweep.sweep_param] == v)]
            lat = sub.loc[sub["in_range_latency_s"] > 0, "in_range_latency_s"]
            means.append(float(lat.mean()) if not lat.empty else None)
            stds.append(float(lat.std(ddof=1)) if len(lat) > 1 else 0.0)
        fig.add_trace(go.Scatter(
            x=values, y=means,
            error_y=dict(type="data", array=stds, thickness=1.2, width=4),
            mode="lines+markers", name=MODE_LABELS[m],
            line=dict(color=MODE_COLORS[m], width=2.2),
            marker=dict(size=8, line=dict(color="white", width=1)),
            legendgroup=m, showlegend=False,
            hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>{sweep.label}: %{{x}} {sweep.unit}<br>Latency: %{{y:.3f}} s<extra></extra>",
        ), row=2, col=2)

    # Axis labels
    fig.update_xaxes(title_text=axis_label, row=2, col=1)
    fig.update_xaxes(title_text=axis_label, row=2, col=2)
    fig.update_xaxes(title_text=axis_label, row=1, col=1)
    fig.update_xaxes(title_text=axis_label, row=1, col=2)
    fig.update_yaxes(title_text="Collision rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Reduction (%)", row=1, col=2)
    fig.update_yaxes(title_text="Detection rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Latency (s)", row=2, col=2)

    _apply_layout(fig, height=760)
    return fig


# ============================================================
#  HTML generation
# ============================================================

INDEX_CSS = """
:root {
    --bg: #fdfcf8;
    --fg: #1f2937;
    --fg-soft: #4b5563;
    --muted: #6b7280;
    --rule: #e5e7eb;
    --rule-strong: #cbd5e1;
    --accent: #1e3a8a;
    --accent-soft: #eef2ff;
    --card: #ffffff;
    --code-bg: #f3f4f6;
    --color-control: #dc2626;
    --color-detection: #ea580c;
    --color-aware: #15803d;
    --sig-3: #047857;
    --sig-2: #65a30d;
    --sig-1: #b45309;
    --sig-ns: #94a3b8;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0b1120;
        --fg: #e2e8f0;
        --fg-soft: #cbd5e1;
        --muted: #94a3b8;
        --rule: #1f2a44;
        --rule-strong: #2c3a5c;
        --accent: #93c5fd;
        --accent-soft: #1e293b;
        --card: #111827;
        --code-bg: #1e293b;
    }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--fg);
    line-height: 1.65; max-width: 1120px;
    margin: 0 auto; padding: 3rem 1.75rem 5rem;
    -webkit-font-smoothing: antialiased;
}
header.paper-head {
    border-bottom: 1px solid var(--rule-strong);
    padding-bottom: 1.5rem; margin-bottom: 2.5rem;
}
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 0.6rem;
}
h1.paper-title {
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 500; font-size: 2.1rem; line-height: 1.18;
    letter-spacing: -0.01em; margin: 0 0 0.9rem 0;
    color: var(--fg);
}
.authors {
    color: var(--fg-soft); font-size: 1.02rem; margin: 0 0 1rem 0;
}
.authors em { font-style: italic; color: var(--muted); font-size: 0.92rem; }
.meta {
    display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem;
}
.tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; letter-spacing: 0.02em;
    background: var(--accent-soft); color: var(--accent);
    padding: 0.25rem 0.65rem; border-radius: 2px;
    border: 1px solid var(--rule);
}
nav.toc {
    position: sticky; top: 0; z-index: 5;
    background: var(--bg);
    border-bottom: 1px solid var(--rule);
    padding: 0.85rem 0; margin: -1rem 0 2.5rem 0;
    backdrop-filter: blur(8px);
}
nav.toc ul { list-style: none; margin: 0; padding: 0;
    display: flex; gap: 1.8rem; flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; letter-spacing: 0.02em; }
nav.toc a { color: var(--fg-soft); text-decoration: none; }
nav.toc a:hover { color: var(--accent); border-bottom: 1px solid var(--accent); }
section { margin-bottom: 4rem; scroll-margin-top: 4rem; }
section > h2 {
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 500; font-size: 1.65rem; line-height: 1.25;
    letter-spacing: -0.005em; margin: 0 0 0.4rem 0;
    color: var(--fg);
}
section > h2 .num {
    display: inline-block; width: 2.2rem;
    color: var(--muted); font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem; font-weight: 400; vertical-align: middle;
}
section > h2 + p.lede {
    color: var(--fg-soft); font-size: 1.02rem;
    margin: 0 0 1.4rem 2.2rem; max-width: 60ch;
}
h3 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600; font-size: 1.05rem;
    margin: 1.8rem 0 0.6rem 2.2rem;
    color: var(--fg);
}
.content { margin-left: 2.2rem; }
.summary-card {
    background: var(--card); border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    padding: 1.1rem 1.4rem; margin: 1rem 0 1.8rem 2.2rem;
    max-width: 75ch;
}
.summary-card p { margin: 0.35rem 0; }
.summary-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
}
table {
    border-collapse: collapse; width: 100%;
    margin: 1rem 0 1.4rem 0; font-size: 0.92rem;
    background: var(--card); border: 1px solid var(--rule);
}
thead th {
    text-align: left; padding: 0.55rem 0.85rem;
    background: var(--accent-soft); color: var(--accent);
    border-bottom: 1px solid var(--rule);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
}
tbody td { padding: 0.55rem 0.85rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
tbody tr:last-child td { border-bottom: 0; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; }
.mode-ctrl { color: var(--color-control); font-weight: 600; }
.mode-det  { color: var(--color-detection); font-weight: 600; }
.mode-awr  { color: var(--color-aware); font-weight: 600; }
.sig-3  { color: var(--sig-3); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.sig-2  { color: var(--sig-2); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.sig-1  { color: var(--sig-1); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.sig-ns { color: var(--sig-ns); font-family: 'IBM Plex Mono', monospace; }
.plot-wrap { margin: 1.8rem 0; }
.plot-caption {
    font-size: 0.88rem; color: var(--muted);
    margin: 0.3rem 0 1rem 0; font-style: italic;
    max-width: 78ch;
}
code, kbd {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.86em;
    background: var(--code-bg); padding: 0.08rem 0.35rem;
    border-radius: 2px;
}
pre {
    background: var(--code-bg); padding: 1rem 1.2rem;
    border-radius: 4px; overflow-x: auto;
    border-left: 3px solid var(--rule-strong);
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    line-height: 1.55;
}
.footer {
    margin-top: 5rem; padding-top: 1.8rem;
    border-top: 1px solid var(--rule);
    color: var(--muted); font-size: 0.85rem;
}
.footer a { color: var(--accent); }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid currentColor; }
a:hover { background: var(--accent-soft); }
.kpi-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 1rem; margin: 1.4rem 0 2rem 2.2rem;
}
.kpi {
    background: var(--card); border: 1px solid var(--rule);
    padding: 1rem 1.1rem; border-radius: 3px;
}
.kpi .num {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.7rem; font-weight: 500;
    color: var(--fg); line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.kpi .lab {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
    margin-top: 0.4rem;
}
@media print {
    body { max-width: none; padding: 0; }
    nav.toc { display: none; }
    section { page-break-inside: avoid; }
}
"""


def fmt(value, fmt_spec=".2f"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return format(value, fmt_spec)


def mode_class(m: str) -> str:
    return {"control": "mode-ctrl", "detection": "mode-det", "aware": "mode-awr"}.get(m, "")


def render_headline_table(stats: Dict) -> str:
    rows = []
    headers = ["Mode", "n", "Collisions/trial", "Collision rate (%)",
               "Detection rate (%)", "In-range latency (s)", "Road entries"]
    for m in MODE_ORDER:
        if m not in stats:
            continue
        s = stats[m]
        rows.append(f"""<tr>
            <td class="{mode_class(m)}">{MODE_LABELS[m]}</td>
            <td class="num">{s['n_trials']}</td>
            <td class="num">{fmt(s['collisions_mean'])} ± {fmt(s['collisions_std'])}</td>
            <td class="num">{fmt(s['col_rate_mean_pct'])} ± {fmt(s['col_rate_std_pct'])}</td>
            <td class="num">{fmt(s['det_rate_mean_pct'])} ± {fmt(s['det_rate_std_pct'])}</td>
            <td class="num">{fmt(s['in_range_mean'], '.3f')} ± {fmt(s['in_range_std'], '.3f')}</td>
            <td class="num">{fmt(s['road_entries_mean'], '.1f')}</td>
        </tr>""")
    th = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_pairwise_table(pairs: List[Dict]) -> str:
    rows = []
    for p in pairs:
        rows.append(f"""<tr>
            <td><span class="{mode_class(p['m1'])}">{MODE_LABELS[p['m1']]}</span> vs <span class="{mode_class(p['m2'])}">{MODE_LABELS[p['m2']]}</span></td>
            <td class="num">{fmt(p['mean_m1'])} → {fmt(p['mean_m2'])}</td>
            <td class="num">{fmt(p['reduction_pct'], '+.1f')}%</td>
            <td class="num">{fmt(p['t'], '+.3f')}</td>
            <td class="num">{fmt(p['p'], '.4f')}</td>
            <td class="{p['sig_class']}">{p['sig']}</td>
        </tr>""")
    return f"""<table><thead><tr>
        <th>Comparison</th><th>Mean collisions</th>
        <th>Reduction</th><th>t</th><th>p-value</th><th>Sig.</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def render_sweep_table(sweep: SweepData, table_df: pd.DataFrame, reduction_df: pd.DataFrame) -> str:
    """Render the per-value sweep summary with reduction columns."""
    headers = [sweep.label, "n", "Mode", "Coll. rate (%)", "Det. rate (%)",
               "In-range latency (s)", "Detection vs Control", "Aware vs Detection"]
    rows = []
    for v in sorted(table_df["value"].unique()):
        sub = table_df[table_df["value"] == v]
        red_row = reduction_df[reduction_df["value"] == v].squeeze() if not reduction_df.empty else None
        for i, mode_row in enumerate(sub.itertuples()):
            m = mode_row.mode
            cells = []
            if i == 0:
                cells.append(f'<td rowspan="{len(sub)}" class="num">{v:g} {sweep.unit}</td>')
                cells.append(f'<td rowspan="{len(sub)}" class="num">{int(mode_row.n)}</td>')
            cells.append(f'<td class="{mode_class(m)}">{MODE_LABELS[m]}</td>')
            cells.append(f'<td class="num">{fmt(mode_row.col_rate_mean)} ± {fmt(mode_row.col_rate_std)}</td>')
            cells.append(f'<td class="num">{fmt(mode_row.det_rate_mean)}</td>')
            cells.append(f'<td class="num">{fmt(mode_row.in_range_mean, ".3f")}</td>')
            if i == 0 and red_row is not None:
                d_vs_c = red_row.get("control_vs_detection_reduction", float("nan"))
                d_vs_c_sig = red_row.get("control_vs_detection_sig", "")
                d_vs_c_cls = red_row.get("control_vs_detection_sig_class", "")
                a_vs_d = red_row.get("detection_vs_aware_reduction", float("nan"))
                a_vs_d_sig = red_row.get("detection_vs_aware_sig", "")
                a_vs_d_cls = red_row.get("detection_vs_aware_sig_class", "")
                cells.append(f'<td rowspan="{len(sub)}" class="num">{fmt(d_vs_c, "+.1f")}% <span class="{d_vs_c_cls}">{d_vs_c_sig}</span></td>')
                cells.append(f'<td rowspan="{len(sub)}" class="num">{fmt(a_vs_d, "+.1f")}% <span class="{a_vs_d_cls}">{a_vs_d_sig}</span></td>')
            rows.append(f"<tr>{''.join(cells)}</tr>")
    th = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_kpi_row(stats: Dict, pairs: List[Dict]) -> str:
    """Render the headline KPI tiles."""
    if not stats:
        return ""
    ctrl = stats.get("control", {})
    det = stats.get("detection", {})
    aware = stats.get("aware", {})
    # Find the control-vs-detection comparison
    det_pair = next((p for p in pairs if p["m1"] == "control" and p["m2"] == "detection"), None)
    aware_pair = next((p for p in pairs if p["m1"] == "control" and p["m2"] == "aware"), None)
    kpis = []
    if ctrl:
        kpis.append((f"{ctrl['col_rate_mean_pct']:.1f}%", "Control collision rate"))
    if det_pair:
        sym = det_pair["sig"]
        kpis.append((f"{det_pair['reduction_pct']:+.1f}%", f"Detection layer reduction ({sym})"))
    if aware_pair:
        sym = aware_pair["sig"]
        kpis.append((f"{aware_pair['reduction_pct']:+.1f}%", f"Full system reduction ({sym})"))
    if det:
        kpis.append((f"{det['det_rate_mean_pct']:.1f}%", "Detection rate"))
    if det:
        kpis.append((f"{det['in_range_mean']*1000:.0f} ms", "In-range latency"))
    out = []
    for num, lab in kpis:
        out.append(f'<div class="kpi"><div class="num">{num}</div><div class="lab">{lab}</div></div>')
    return f'<div class="kpi-row">{"".join(out)}</div>'


def fig_to_div(fig: go.Figure, div_id: str, include_js: bool = False) -> str:
    """Render a Plotly figure as a stripped <div>.

    Args:
        include_js: If True, embed plotly.js source inline (use for the first figure
                    when inline_plotly is enabled). All other figures should pass False.
    """
    return pio.to_html(
        fig,
        include_plotlyjs=("inline" if include_js else False),
        full_html=False,
        div_id=div_id,
        config=PIO_CONFIG,
    )


# ============================================================
#  Top-level page assembly
# ============================================================

def _detect_plotlyjs_url() -> str:
    """Detect the plotly.js CDN URL that matches the installed plotly.py version."""
    import re
    probe = pio.to_html(go.Figure(), include_plotlyjs="cdn", full_html=False)
    m = re.search(r'(https://cdn\.plot\.ly/plotly-[\d.]+\.min\.js)', probe)
    return m.group(1) if m else "https://cdn.plot.ly/plotly-latest.min.js"


def build_index_html(
    paper_title: str,
    authors: str,
    journal: str,
    output_dir: Path,
    headline: Optional[HeadlineData],
    sweeps: List[SweepData],
    inline_plotly: bool = False,
) -> None:
    """Assemble the full index.html.

    Args:
        inline_plotly: If True, embed plotly.js source directly (~3 MB page, works offline).
                       If False, reference the matching plotly.js from cdn.plot.ly (default; ~100 KB page).
    """
    date_str = _dt.datetime.now().strftime("%d %B %Y")
    plotlyjs_url = _detect_plotlyjs_url()

    # Build the plotly.js inclusion mechanism
    if inline_plotly:
        # Embed plotly.js source directly via the first figure to_html call
        plotly_script_tag = ""
        first_fig_includes_js = True
    else:
        # CDN reference in <head>
        plotly_script_tag = f'<script src="{plotlyjs_url}" charset="utf-8"></script>'
        first_fig_includes_js = False

    # --- Build all the section bodies ---
    sections_html: List[str] = []
    nav_items: List[Tuple[str, str]] = []

    # ----- Headline -----
    total_trials = 0
    figure_counter = 0  # tracks whether the first figure has consumed the inline-JS slot

    def render_fig(fig: go.Figure, div_id: str) -> str:
        nonlocal figure_counter
        is_first = (figure_counter == 0)
        figure_counter += 1
        return fig_to_div(fig, div_id, include_js=(is_first and first_fig_includes_js))

    if headline is not None:
        df = headline.df
        stats = per_mode_stats(df)
        pairs = pairwise_tests(df, "collisions")
        total_trials += len(df)

        if not stats:
            # Defensive: shouldn't happen because find_headline_csv already filters,
            # but render a visible diagnostic so the failure mode is obvious instead of silent.
            found_modes = sorted(df["mode"].unique().tolist())
            print(f"[error] Headline data in {headline.source_file} matched 0 rows "
                  f"to known modes. Found mode values: {found_modes}", file=sys.stderr)
            nav_items.append(("Headline result", "headline"))
            sections_html.append(f"""
<section id="headline">
  <h2><span class="num">01</span> Headline three-mode comparison</h2>
  <div class="summary-card" style="border-left-color: #dc2626;">
    <p class="label" style="color: #dc2626;">Data Loading Error</p>
    <p>Could not match any rows in <code>{html.escape(headline.source_file)}</code> to known modes
    ({', '.join(MODE_ORDER)}). Found mode values: <code>{html.escape(str(found_modes))}</code>.</p>
    <p>If your simulator output uses different mode names, please rename them in the CSV or update
    <code>MODE_ORDER</code> in <code>analyze_wvc.py</code>.</p>
  </div>
</section>""")
        else:
            n_modes = len([m for m in MODE_ORDER if m in stats])
            first_n = next(iter(stats.values()))["n_trials"]

            fig_main = fig_headline_overview(df, stats)
            fig_states = fig_state_visits(df, stats)

            nav_items.append(("Headline result", "headline"))
            sections_html.append(f"""
<section id="headline">
  <h2><span class="num">01</span> Headline three-mode comparison</h2>
  <p class="lede">{first_n} trials per mode × {n_modes} modes. The two-step contrast isolates what each architectural layer contributes: <em>Detection</em> adds sensors + driver alerts, <em>Aware</em> further adds LoRa-mediated network awareness boost.</p>

  {render_kpi_row(stats, pairs)}

  <div class="content">
    <h3>Per-mode statistics</h3>
    {render_headline_table(stats)}

    <h3>Pairwise comparisons (Welch's t-test on collisions)</h3>
    {render_pairwise_table(pairs)}
  </div>

  <div class="plot-wrap">
    {render_fig(fig_main, "fig-headline-main")}
    <p class="plot-caption">Figure 1. Four-panel three-mode Monte Carlo comparison. Each point is one trial; whiskers show Tukey range. Detection rate is 0% in the Control condition by construction (no sensors active).</p>
  </div>

  <div class="plot-wrap">
    {render_fig(fig_states, "fig-headline-states")}
    <p class="plot-caption">Figure 2. Animal state visits per trial. Slow-vehicle modes increase road-entry visits (animals are not threatened by approaching vehicles) but reduce dangerous freeze-on-road events near-completely.</p>
  </div>
</section>""")

    # ----- Sweeps -----
    for i, sweep in enumerate(sweeps, start=2):
        nav_items.append((f"{sweep.label} sweep", sweep.slug))
        total_trials += len(sweep.df)
        table = sweep_table(sweep.df, sweep.sweep_param)
        reductions = sweep_reduction_summary(sweep.df, sweep.sweep_param)
        fig = fig_sweep_overview(sweep)
        per_value_trials = int(table.groupby("value")["n"].mean().mean()) if not table.empty else 0

        sweep_intro = SWEEP_PARAGRAPHS.get(sweep.sweep_param, "")
        sections_html.append(f"""
<section id="{sweep.slug}">
  <h2><span class="num">{i:02d}</span> Sensitivity sweep: {sweep.label}</h2>
  <p class="lede">{sweep_intro}</p>

  <div class="content">
    <h3>Per-value statistics</h3>
    {render_sweep_table(sweep, table, reductions)}
  </div>

  <div class="plot-wrap">
    {render_fig(fig, f"fig-sweep-{sweep.slug}")}
    <p class="plot-caption">Figure {i+1}. Sweep of <code>{sweep.sweep_param}</code> across {len(table['value'].unique())} values, {per_value_trials} trials per (mode × value). Top-left: collision rate per mode with ± 1 SD error bars. Top-right: reduction percentages by mitigation layer.</p>
  </div>
</section>""")

    # ----- Methods, Data sections (always shown) -----
    nav_items.append(("Methods", "methods"))
    nav_items.append(("Data & code", "data"))

    sources_list = []
    if headline is not None:
        sources_list.append(f"<li><code>{html.escape(headline.source_file)}</code> — headline three-mode dataset</li>")
    for s in sweeps:
        sources_list.append(f"<li><code>{html.escape(s.source_file)}</code> — sweep over <code>{s.sweep_param}</code></li>")
    sources_html = "\n".join(sources_list)

    methods_html = f"""
<section id="methods">
  <h2><span class="num">{2 + len(sweeps):02d}</span> Methods summary</h2>
  <p class="lede">All results were generated by a single Python simulator, <code>wvc_simulator.py</code>, using a behaviorally-realistic discrete-time Monte Carlo framework on a 1 km test corridor.</p>

  <div class="content">
    <h3>System architecture</h3>
    <p>Alternating-side Doppler radar nodes at 15 m spacing (50 sensors/km baseline, 67/km at 15 m), three-axis magnetometer sites every 200 m, and dynamic message signs (DMS) every 250 m. Sensors communicate over an LVDS+LoRa hybrid backbone; an awareness signal propagates between neighboring radars to boost detection sensitivity by a factor of 1.8× during 30 s windows around recent detections.</p>

    <h3>Vehicle dynamics</h3>
    <p>Treiber–Hennecke–Helbing Intelligent Driver Model (IDM) with desired gap <code>s₀</code> = 5 m, time headway <code>T</code> = 1.5 s, comfortable acceleration <code>a</code> = 2.5 m/s², comfortable deceleration <code>b</code> = 4.0 m/s², emergency braking 9.0 m/s². Driver reaction delay 1.5 s.</p>

    <h3>Animal behavior</h3>
    <p>Six-state Markov model: <code>Foraging → Approaching → Hesitating → {{Crossing | Frozen | Fleeing}} → Gone</code>. Behavioral transitions calibrated against field studies: <code>p</code>(freeze | vehicle present) = 0.10, <code>p</code>(flee | vehicle) = 0.20, <code>p</code>(cross | clear) = 0.80, <code>p</code>(freeze during cross | vehicle braking) base 0.15. Animals arriving in a slow-vehicle environment cross more often but freeze less.</p>

    <h3>Three operating modes</h3>
    <ul>
      <li><strong class="mode-ctrl">Control</strong> — no sensors, no alerts, no caution speed.</li>
      <li><strong class="mode-det">Detection</strong> — sensors active, drivers receive DMS warnings and reduce to caution speed when in any active awareness zone, no neighbor-sensor boost.</li>
      <li><strong class="mode-awr">Aware</strong> — Detection mode plus LoRa-propagated awareness boost (1.8× sensitivity on neighboring radars for 30 s after each detection).</li>
    </ul>

    <h3>Statistical analysis</h3>
    <p>All p-values from two-sided Welch's t-tests on per-trial collision counts. Significance markers: <span class="sig-3">***</span> p &lt; 0.001, <span class="sig-2">**</span> p &lt; 0.01, <span class="sig-1">*</span> p &lt; 0.05, <span class="sig-ns">n.s.</span> otherwise. Error bars and ± uncertainties throughout are one sample standard deviation (Bessel-corrected).</p>
  </div>
</section>"""

    data_html = f"""
<section id="data">
  <h2><span class="num">{3 + len(sweeps):02d}</span> Data and reproducibility</h2>
  <p class="lede">All figures and tables are generated directly from the CSVs by this repository's analysis script. Re-running the script on the raw data reproduces every number on this page byte-identical.</p>

  <div class="content">
    <h3>Source CSVs</h3>
    <ul>{sources_html}</ul>

    <h3>Reproducing the figures</h3>
    <pre>git clone &lt;this-repo&gt;
cd &lt;this-repo&gt;
python -m pip install pandas plotly scipy numpy
python analyze_wvc.py --input data --output docs</pre>

    <h3>Reproducing the raw data</h3>
    <p>The simulator <code>wvc_simulator.py</code> is fully self-contained and uses only standard scientific Python. Each dataset on this page can be regenerated:</p>
    <pre>python wvc_simulator.py --trials 20 --hours 4 --plot                # headline
python wvc_simulator.py --sweep spacing --trials 15 --hours 2       # spacing sweep
python wvc_simulator.py --sweep size --trials 15 --hours 2          # size sweep
python wvc_simulator.py --sweep detection_rate --trials 15 --hours 2 # sensitivity sweep</pre>

    <h3>Citation</h3>
    <p>If you use these results, please cite the accompanying paper:</p>
    <pre>{html.escape(authors)} ({_dt.datetime.now().year}).
    {html.escape(paper_title)}.
    Submitted to {html.escape(journal)}.</pre>
  </div>
</section>"""

    sections_html.append(methods_html)
    sections_html.append(data_html)

    # ----- TOC -----
    toc = "\n".join(f'<li><a href="#{slug}">{name}</a></li>' for name, slug in nav_items)
    toc_html = f'<nav class="toc"><ul>{toc}</ul></nav>'

    # ----- Assemble full HTML -----
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Interactive data page for the WVC sensor-network paper.">
    <title>{html.escape(paper_title)} — Data Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    {plotly_script_tag}
    <style>{INDEX_CSS}</style>
</head>
<body>
    <header class="paper-head">
        <div class="eyebrow">Supplementary data page · {date_str}</div>
        <h1 class="paper-title">{html.escape(paper_title)}</h1>
        <p class="authors">{html.escape(authors)} · <em>Submitted to {html.escape(journal)}</em></p>
        <div class="meta">
            <span class="tag">{len(sweeps) + (1 if headline else 0)} datasets</span>
            <span class="tag">{total_trials:,} trials</span>
            <span class="tag">Interactive Plotly figures</span>
            <span class="tag">Open data + code</span>
        </div>
    </header>

    {toc_html}

    {''.join(sections_html)}

    <div class="footer">
        <p>Page generated by <code>analyze_wvc.py</code> on {date_str}.
        Hover the figures for per-trial values; double-click to reset zoom.
        Click the camera icon in any figure to download as PNG.</p>
    </div>
</body>
</html>"""

    out = output_dir / "index.html"
    out.write_text(full_html, encoding="utf-8")
    print(f"✓ Wrote {out} ({out.stat().st_size:,} bytes)")


# Short prose introductions for each sweep section
SWEEP_PARAGRAPHS = {
    "radar_spacing":
        "Coverage geometry: alternating radar nodes must satisfy R_det ≥ √(s² + d_y²) "
        "to leave no detection gap at the worst-case midpoint. This sweep establishes "
        "the practical deployment-density recommendation.",
    "size_scale":
        "Multiplier applied to all sampled animal sizes; 1.0× is the default mix (15% small / 60% medium / 25% large). "
        "Lower values stress the system with smaller-RCS targets (fox/coyote class); higher values represent moose-dominant corridors.",
    "detection_rate_per_sec":
        "Baseline per-frame detection rate κ. Lower κ stresses the system with degraded sensor sensitivity "
        "(e.g. weather, vegetation occlusion, hardware aging). The cumulative detection probability is "
        "1 − exp(−κ · t_exposure), so even at low κ the system can recover detection if animals dwell long enough in range.",
    "radar_range":
        "Per-sensor detection radius. With alternating placement, this controls coverage overlap and the per-target "
        "detection-window duration.",
    "animal_rate_per_hr":
        "Poisson arrival rate. Higher values stress the system with overlapping wildlife events and probe whether the "
        "30-second awareness persistence window provides material benefit at realistic traffic densities.",
    "caution_speed_kmh":
        "Vehicle speed set by the DMS warning system when in an active awareness zone. Trades kinematic stopping margin "
        "against driver compliance.",
    "cruise_speed_kmh":
        "Baseline highway speed. Faster speeds reduce stopping margins and increase collision severity.",
    "driver_reaction_s":
        "Driver perception-and-reaction time. Standard highway design value is 1.5 s; testing 0.5–3.0 s probes the system's "
        "robustness against driver attentiveness variation.",
}


# ============================================================
#  Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive Plotly data page from WVC simulator CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path("."),
                        help="Directory containing simulator CSVs (default: current directory)")
    parser.add_argument("--output", "-o", type=Path, default=Path("./docs"),
                        help="Output directory for the data page (default: ./docs)")
    parser.add_argument("--paper-title", type=str,
                        default="Multimodal Radar–Magnetometer Sensor Network with LoRa-Mediated Awareness for Wildlife–Vehicle Collision Prevention",
                        help="Paper title shown in the page header")
    parser.add_argument("--authors", type=str,
                        default="Lars Thomsen, Sergii Makovetskyi",
                        help="Author list shown in the page header")
    parser.add_argument("--journal", type=str, default="MDPI Sustainability",
                        help="Target journal")
    parser.add_argument("--copy-csvs", action="store_true",
                        help="Copy source CSVs into <output>/data/ for direct download")
    parser.add_argument("--inline-plotly", action="store_true",
                        help="Embed plotly.js (~3 MB) directly in index.html for offline/local viewing. "
                             "Default behavior loads plotly.js from cdn.plot.ly (~100 KB page, requires internet).")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        sys.exit(f"Error: input directory {input_dir} not found.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_dir}")
    print(f"Writing to:   {output_dir}")
    print()

    headline = find_headline_csv(input_dir)
    if headline:
        mode_counts = headline.df["mode"].value_counts().to_dict()
        mode_summary = ", ".join(f"{m}: {n}" for m, n in mode_counts.items())
        print(f"  ✓ Headline data:  {headline.source_file}  ({len(headline.df)} rows  [{mode_summary}])")
    else:
        print(f"  · No headline CSV found (looked for wvc_results.csv and backups)")

    sweeps = discover_sweeps(input_dir)
    for s in sweeps:
        n_values = len(s.df[s.sweep_param].unique())
        mode_counts = s.df["mode"].value_counts().to_dict()
        mode_summary = ", ".join(f"{m}: {n}" for m, n in mode_counts.items())
        print(f"  ✓ Sweep '{s.sweep_param}': {s.source_file}  "
              f"({len(s.df)} rows, {n_values} values  [{mode_summary}])")
    if not sweeps:
        print(f"  · No sweep CSVs found (looking for *sweep*.cs[v])")
    print()

    if headline is None and not sweeps:
        sys.exit("Nothing to render. Place CSVs in the input directory and re-run.")

    if args.copy_csvs:
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        if headline:
            shutil.copy(input_dir / headline.source_file, data_dir / headline.source_file)
        for s in sweeps:
            shutil.copy(input_dir / s.source_file, data_dir / s.source_file)
        print(f"  ✓ Copied {1 + len(sweeps)} CSVs to {data_dir}")

    build_index_html(
        paper_title=args.paper_title,
        authors=args.authors,
        journal=args.journal,
        output_dir=output_dir,
        headline=headline,
        sweeps=sweeps,
        inline_plotly=args.inline_plotly,
    )

    print()
    print("Done. Open the page locally with:")
    print(f"  python -m http.server -d {output_dir}")
    print("then visit http://localhost:8000")


if __name__ == "__main__":
    main()