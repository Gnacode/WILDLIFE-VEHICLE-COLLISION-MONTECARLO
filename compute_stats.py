#!/usr/bin/env python3
"""
compute_stats.py

Companion statistical analysis to generate_data.py and generate_figures.py.
For each per-trial CSV produced by the simulation, computes:

  - Welch's t-test on per-entry collision rates  (matches figure significance markers)
  - Poisson GLM (collisions ~ mode, offset = log(entries))
      rate ratios with 95% Wald CIs; falls back to Negative Binomial GLM
      if Pearson dispersion exceeds DISPERSION_THRESHOLD = 1.5
  - Non-parametric bootstrap over trials (default 10 000 replicates) on
      the rate difference (treatment minus reference, percentage points)
  - For each sweep, a pooled GLM with mode × sweep_value interaction
      tested via likelihood-ratio test against the additive model.

Outputs companion CSVs alongside each input wvc_results_*.csv:
  wvc_stats_headline.csv                 3 rows (one per pairwise comparison)
  wvc_stats_<sweep>.csv                  one row per (sweep_value × pairwise comparison)
  wvc_stats_<sweep>_interaction.csv      one row summarising the pooled interaction test

Usage:
  python3 compute_stats.py
  python3 compute_stats.py --csv-dir /path/to/results
  python3 compute_stats.py --bootstrap-n 10000 --seed 42

Requires: numpy, pandas, scipy, statsmodels.
"""

import argparse
import os
import sys
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Sweep label (matches filename suffix) -> swept Config parameter column name
SWEEP_PARAM_COLUMNS = {
    "spacing":           "radar_spacing",
    "radar_range":       "radar_range",
    "size":              "size_scale",
    "detection_rate":    "detection_rate_per_sec",
    "awareness_beta":    "boost_factor",
    "awareness_tau":     "awareness_persist_s",
    "hesitate_dwell":    "hesitate_dwell_max",
    "freeze_mid_cross":  "p_freeze_during_cross_base",
    "traffic_volume":    "vehicles_per_dir",
    "animal_density":    "animal_rate_per_hr",
    "driver_reaction":   "driver_reaction_s",
    "cruise_speed":      "cruise_speed_kmh",
    "caution_speed":     "caution_speed_kmh",
}

# ---------------------------------------------------------------- metric specs
# Each metric defines:
#   value_col   : per-trial column name in wvc_results_*.csv used for Welch and
#                 bootstrap (the value summarised per trial)
#   count_col   : integer outcome for the GLM (None disables GLM, latency case)
#   offset_col  : exposure column whose log is used as the GLM offset
#                 (None disables offset)
#   skip_modes  : optional set of mode names to omit; e.g. latency comparisons
#                 against 'control' are not meaningful because control records
#                 no detections (no latency to report).
#   pct_scale   : True if the metric is a fraction stored as 0..1 and the
#                 reporting CSVs should label means in %; False for continuous
#                 metrics with their own units.
#
# To add a new metric, append a row here and re-run; nothing else changes.
METRICS = {
    "col_rate": {
        "value_col": "col_rate",
        "count_col": "collisions",
        "offset_col": "road_entries",
        "skip_modes": set(),
        "pct_scale": True,
        "units": "% per road entry",
    },
    "det_rate": {
        "value_col": "det_rate",
        "count_col": "detected",
        "offset_col": "animals",
        "skip_modes": {"control"},   # control records no detections by design
        "pct_scale": True,
        "units": "% per arriving animal",
    },
    "latency": {
        "value_col": "in_range_latency_s",
        "count_col": None,
        "offset_col": None,
        "skip_modes": {"control"},   # control has no detections so no latency
        "pct_scale": False,
        "units": "s (in-range detection latency)",
    },
}

# Pairwise comparisons. Constructed per-metric by pairs_for_metric() below
# from a fixed base list, skipping any pair whose reference mode is in the
# metric's skip_modes set.
DISPERSION_THRESHOLD = 1.5  # above this, switch Poisson -> Negative Binomial


# -------------------------------------------------------------------- helpers


def recover_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Recover integer per-trial collision counts from col_rate * road_entries."""
    df = df.copy()
    df["collisions"] = (df["col_rate"] * df["road_entries"]).round().astype(int)
    return df


def welch_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Welch's t-test on two rate vectors. Returns (t, Welch-Satterthwaite df, p)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    s2_a, s2_b = np.var(a, ddof=1), np.var(b, ddof=1)
    if s2_a == 0 and s2_b == 0:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n_a, n_b = len(a), len(b)
    denom = (s2_a / n_a) ** 2 / (n_a - 1) + (s2_b / n_b) ** 2 / (n_b - 1)
    df_w = ((s2_a / n_a + s2_b / n_b) ** 2 / denom) if denom > 0 else np.nan
    return float(t), float(df_w), float(p)


def fit_pairwise_glm(sub: pd.DataFrame, ref_mode: str, trt_mode: str,
                     metric: dict) -> Tuple[Optional[dict], float, str]:
    """
    Fit Poisson GLM (<count_col> ~ C(mode)) with log(<offset_col>) offset on a
    two-mode subset, for the given metric configuration. Returns
    (result_dict, dispersion, family_used). Switches to negative binomial if
    Pearson dispersion exceeds DISPERSION_THRESHOLD.

    For metrics with count_col=None (e.g. continuous latency) this function
    returns (None, NaN, 'none') and the caller should fall back to Welch +
    bootstrap only.
    """
    if metric["count_col"] is None or metric["offset_col"] is None:
        return None, np.nan, "none"

    sub = sub[sub["mode"].isin([ref_mode, trt_mode])].copy()
    if len(sub) < 4 or (sub[metric["offset_col"]] <= 0).any():
        return None, np.nan, "none"

    sub["mode_cat"]   = pd.Categorical(sub["mode"], categories=[ref_mode, trt_mode])
    sub["log_offset"] = np.log(sub[metric["offset_col"]])
    formula = f"{metric['count_col']} ~ C(mode_cat)"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.glm(formula, data=sub, family=sm.families.Poisson(),
                            offset=sub["log_offset"]).fit()

        dispersion = (float(model.pearson_chi2 / model.df_resid)
                      if model.df_resid > 0 else np.nan)
        family = "poisson"

        if not np.isnan(dispersion) and dispersion > DISPERSION_THRESHOLD:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_nb = smf.glm(formula, data=sub,
                                       family=sm.families.NegativeBinomial(alpha=1.0),
                                       offset=sub["log_offset"]).fit()
                model = model_nb
                family = "negbin"
            except Exception:
                pass  # keep Poisson result

        coef_name = next((n for n in model.params.index if trt_mode in n), None)
        if coef_name is None:
            return None, dispersion, family

        beta = float(model.params[coef_name])
        se = float(model.bse[coef_name])
        if not np.isfinite(se) or se <= 0:
            return ({"rate_ratio": float(np.exp(beta)),
                     "rr_ci_low": np.nan, "rr_ci_high": np.nan,
                     "glm_z": np.nan, "glm_p": np.nan}, dispersion, family)

        z = beta / se
        p_glm = float(stats.norm.sf(abs(z)) * 2)
        return ({"rate_ratio":  float(np.exp(beta)),
                 "rr_ci_low":   float(np.exp(beta - 1.96 * se)),
                 "rr_ci_high":  float(np.exp(beta + 1.96 * se)),
                 "glm_z":       float(z),
                 "glm_p":       p_glm},
                dispersion, family)

    except Exception:
        return None, np.nan, "failed"


def bootstrap_diff(ref_rates: np.ndarray, trt_rates: np.ndarray,
                   n: int = 10000, seed: int = 42
                   ) -> dict:
    """
    Non-parametric bootstrap over trials. Resamples each arm independently with
    replacement and reports two quantities with 95 % percentile CIs:

      diff_pp     = (trt_mean - ref_mean) * 100   ; in percentage points
      reduction   = (ref_mean - trt_mean) / ref_mean * 100   ; in %

    Returns a dict; all-NaN if either arm has fewer than 2 trials.
    """
    out = {
        "bootstrap_diff_pp":       np.nan,
        "bootstrap_ci_low_pp":     np.nan,
        "bootstrap_ci_high_pp":    np.nan,
        "reduction_pct":           np.nan,
        "reduction_ci_low_pct":    np.nan,
        "reduction_ci_high_pct":   np.nan,
    }
    if len(ref_rates) < 2 or len(trt_rates) < 2:
        return out
    rng = np.random.default_rng(seed)
    n_ref, n_trt = len(ref_rates), len(trt_rates)

    point_diff_pp = (trt_rates.mean() - ref_rates.mean()) * 100.0
    if ref_rates.mean() > 0:
        point_red = (ref_rates.mean() - trt_rates.mean()) / ref_rates.mean() * 100.0
    else:
        point_red = np.nan

    ref_idx = rng.integers(0, n_ref, size=(n, n_ref))
    trt_idx = rng.integers(0, n_trt, size=(n, n_trt))
    ref_means = ref_rates[ref_idx].mean(axis=1)
    trt_means = trt_rates[trt_idx].mean(axis=1)
    diffs_pp = (trt_means - ref_means) * 100.0
    # Guard against ref_mean == 0 in occasional replicates
    safe = ref_means > 0
    reds = np.full(n, np.nan)
    reds[safe] = (ref_means[safe] - trt_means[safe]) / ref_means[safe] * 100.0

    out.update({
        "bootstrap_diff_pp":       float(point_diff_pp),
        "bootstrap_ci_low_pp":     float(np.percentile(diffs_pp, 2.5)),
        "bootstrap_ci_high_pp":    float(np.percentile(diffs_pp, 97.5)),
        "reduction_pct":           float(point_red) if np.isfinite(point_red) else np.nan,
        "reduction_ci_low_pct":    float(np.nanpercentile(reds, 2.5))  if np.isfinite(point_red) else np.nan,
        "reduction_ci_high_pct":   float(np.nanpercentile(reds, 97.5)) if np.isfinite(point_red) else np.nan,
    })
    return out


def pairs_for_metric(metric: dict):
    """All_pairwise comparisons valid for this metric, given skip_modes."""
    base = [("control", "detection"),
            ("control", "aware"),
            ("detection", "aware")]
    skip = metric.get("skip_modes", set())
    return [(r, t) for r, t in base if r not in skip and t not in skip]


def compute_pairwise(sub: pd.DataFrame, ref: str, trt: str, metric: dict,
                     n_bootstrap: int, seed: int) -> dict:
    """Run Welch + GLM (if defined) + bootstrap on a two-mode subset
    for the given metric configuration."""
    col = metric["value_col"]
    ref_rates = sub[sub["mode"] == ref][col].values
    trt_rates = sub[sub["mode"] == trt][col].values

    welch_t, welch_df, welch_p = welch_test(ref_rates, trt_rates)
    glm_res, dispersion, family = fit_pairwise_glm(sub, ref, trt, metric)
    boot = bootstrap_diff(ref_rates, trt_rates, n=n_bootstrap, seed=seed)

    scale = 100.0 if metric["pct_scale"] else 1.0
    mean_label_suffix = "_pct" if metric["pct_scale"] else "_value"
    row = {
        "metric":         metric["value_col"],
        "comparison":     f"{ref}_vs_{trt}",
        "n_ref":          len(ref_rates),
        "n_trt":          len(trt_rates),
        f"mean_ref{mean_label_suffix}": (float(ref_rates.mean() * scale)
                                          if len(ref_rates) else np.nan),
        f"mean_trt{mean_label_suffix}": (float(trt_rates.mean() * scale)
                                          if len(trt_rates) else np.nan),
        "welch_t":        welch_t,
        "welch_df":       welch_df,
        "welch_p":        welch_p,
        "glm_family":     family,
        "glm_dispersion": dispersion,
    }
    if glm_res is not None:
        row.update(glm_res)
    else:
        row.update({"rate_ratio": np.nan, "rr_ci_low": np.nan, "rr_ci_high": np.nan,
                    "glm_z": np.nan, "glm_p": np.nan})

    row.update(boot)
    return row


def compute_interaction(df: pd.DataFrame, sweep_col: str, metric: dict) -> dict:
    """
    Pooled test for a mode × sweep_col interaction.

    For COUNT metrics (count_col + offset_col defined): Poisson GLM
        <count_col> ~ C(mode) * C(sweep_col) with log(<offset_col>) offset,
        compared against the additive model via likelihood-ratio test. Falls
        back to negative binomial if Pearson dispersion exceeds threshold.

    For CONTINUOUS metrics (count_col is None): two-way OLS ANOVA on
        <value_col> ~ C(mode) * C(sweep_col), with the F-test on the
        interaction terms reporting an effective chi-square statistic
        (F × df_inter, which has the same asymptotic null distribution
        when residuals are well-behaved).

    For both branches the returned dict has the same keys for downstream
    aggregation: interaction_chi2, interaction_df, interaction_p, dispersion,
    family. 'dispersion' is the GLM Pearson dispersion (count branch) or the
    OLS residual variance (continuous branch); 'family' is 'poisson',
    'negbin', or 'ols'.
    """
    skip = metric.get("skip_modes", set())

    # --- continuous metric path: two-way OLS ANOVA --------------------------
    if metric["count_col"] is None or metric["offset_col"] is None:
        work = df.copy()
        if skip:
            work = work[~work["mode"].isin(skip)]
        # Drop rows where the metric is undefined (e.g. control had no detection)
        work = work[np.isfinite(work[metric["value_col"]])]
        if len(work) == 0 or work["mode"].nunique() < 2 or work[sweep_col].nunique() < 2:
            return {"interaction_chi2": np.nan, "interaction_df": np.nan,
                    "interaction_p": np.nan, "dispersion": np.nan, "family": "none"}

        work = work.copy()
        work["sweep_cat"] = pd.Categorical(work[sweep_col])
        formula = f"{metric['value_col']} ~ C(mode) * sweep_cat"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.ols(formula, data=work).fit()
                aov = sm.stats.anova_lm(model, typ=2)
            # The interaction row in Type II ANOVA
            inter_idx = [i for i in aov.index if ":" in i and "mode" in i and "sweep" in i]
            if not inter_idx:
                return {"interaction_chi2": np.nan, "interaction_df": np.nan,
                        "interaction_p": np.nan, "dispersion": np.nan, "family": "ols"}
            row = aov.loc[inter_idx[0]]
            F  = float(row["F"])
            df_inter = int(row["df"])
            p  = float(row["PR(>F)"])
            # Report F × df as an "equivalent chi2" for table consistency
            return {"interaction_chi2": float(F * df_inter),
                    "interaction_df": df_inter,
                    "interaction_p": p,
                    "dispersion": float(model.mse_resid),
                    "family": "ols"}
        except Exception:
            return {"interaction_chi2": np.nan, "interaction_df": np.nan,
                    "interaction_p": np.nan, "dispersion": np.nan, "family": "failed"}

    # --- count metric path: Poisson / NB GLM --------------------------------
    work = df[df[metric["offset_col"]] > 0].copy()
    if skip:
        work = work[~work["mode"].isin(skip)]
    if len(work) == 0:
        return {"interaction_chi2": np.nan, "interaction_df": np.nan,
                "interaction_p": np.nan, "dispersion": np.nan, "family": "none"}

    work["log_offset"] = np.log(work[metric["offset_col"]])
    work["sweep_cat"] = pd.Categorical(work[sweep_col])
    full_f = f"{metric['count_col']} ~ C(mode) * sweep_cat"
    redu_f = f"{metric['count_col']} ~ C(mode) + sweep_cat"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = smf.glm(full_f, data=work, family=sm.families.Poisson(),
                           offset=work["log_offset"]).fit()
            reduced = smf.glm(redu_f, data=work, family=sm.families.Poisson(),
                              offset=work["log_offset"]).fit()

        dispersion = (float(full.pearson_chi2 / full.df_resid)
                      if full.df_resid > 0 else np.nan)
        family = "poisson"

        if not np.isnan(dispersion) and dispersion > DISPERSION_THRESHOLD:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    full = smf.glm(full_f, data=work,
                                   family=sm.families.NegativeBinomial(alpha=1.0),
                                   offset=work["log_offset"]).fit()
                    reduced = smf.glm(redu_f, data=work,
                                      family=sm.families.NegativeBinomial(alpha=1.0),
                                      offset=work["log_offset"]).fit()
                family = "negbin"
            except Exception:
                pass

        lr = 2.0 * (full.llf - reduced.llf)
        df_diff = int(reduced.df_resid - full.df_resid)
        p = float(stats.chi2.sf(lr, df_diff)) if df_diff > 0 else np.nan

        return {"interaction_chi2": float(lr), "interaction_df": df_diff,
                "interaction_p": p, "dispersion": dispersion, "family": family}

    except Exception:
        return {"interaction_chi2": np.nan, "interaction_df": np.nan,
                "interaction_p": np.nan, "dispersion": np.nan, "family": "failed"}


# -------------------------------------------------------------------- drivers


def process_headline(csv_dir: str, n_bootstrap: int, seed: int) -> None:
    path = os.path.join(csv_dir, "wvc_results.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = recover_collisions(pd.read_csv(path))
    for metric_name, metric in METRICS.items():
        pairs = pairs_for_metric(metric)
        rows = [compute_pairwise(df, ref, trt, metric, n_bootstrap, seed)
                for ref, trt in pairs]
        if not rows:
            continue
        out = pd.DataFrame(rows)
        suffix = "" if metric_name == "col_rate" else f"_{metric_name}"
        out_path = os.path.join(csv_dir, f"wvc_stats_headline{suffix}.csv")
        out.to_csv(out_path, index=False, float_format="%.6g")
        print(f"  wrote {out_path}  ({len(out)} rows, metric={metric_name})")


def process_sweep(csv_dir: str, label: str, sweep_col: str,
                  n_bootstrap: int, seed: int) -> None:
    path = os.path.join(csv_dir, f"wvc_results_{label}.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = recover_collisions(pd.read_csv(path))
    if sweep_col not in df.columns:
        print(f"[skip] sweep column '{sweep_col}' not in {path}")
        return

    for metric_name, metric in METRICS.items():
        pairs = pairs_for_metric(metric)
        if not pairs:
            continue

        # Per-cell pairwise rows
        rows = []
        for val in sorted(df[sweep_col].unique()):
            sub = df[df[sweep_col] == val]
            for ref, trt in pairs:
                row = compute_pairwise(sub, ref, trt, metric, n_bootstrap, seed)
                row[sweep_col] = val
                rows.append(row)
        out = pd.DataFrame(rows)
        cols = [sweep_col] + [c for c in out.columns if c != sweep_col]
        out = out[cols]

        suffix = "" if metric_name == "col_rate" else f"_{metric_name}"
        out_path = os.path.join(csv_dir, f"wvc_stats_{label}{suffix}.csv")
        out.to_csv(out_path, index=False, float_format="%.6g")
        print(f"  wrote {out_path}  ({len(out)} rows, metric={metric_name})")

        # Pooled interaction test
        inter = compute_interaction(df, sweep_col, metric)
        inter["sweep_label"] = label
        inter["sweep_col"]   = sweep_col
        inter["metric"]      = metric_name
        inter_df = pd.DataFrame([inter])[
            ["sweep_label", "sweep_col", "metric", "interaction_chi2",
             "interaction_df", "interaction_p", "dispersion", "family"]
        ]
        inter_path = os.path.join(
            csv_dir, f"wvc_stats_{label}{suffix}_interaction.csv")
        inter_df.to_csv(inter_path, index=False, float_format="%.6g")
        print(f"  wrote {inter_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=".",
                    help="directory containing wvc_results_*.csv files (default: cwd)")
    ap.add_argument("--bootstrap-n", type=int, default=10000,
                    help="bootstrap replicates per pairwise comparison (default 10000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="bootstrap RNG seed (default 42)")
    args = ap.parse_args()

    print(f"Reading from: {os.path.abspath(args.csv_dir)}")
    print(f"Bootstrap:    {args.bootstrap_n} replicates, seed {args.seed}")
    print(f"Dispersion threshold for NB switch: {DISPERSION_THRESHOLD}")
    print()
    print("--- Headline ---")
    process_headline(args.csv_dir, args.bootstrap_n, args.seed)
    print()
    print("--- Sweeps ---")
    for label, sweep_col in SWEEP_PARAM_COLUMNS.items():
        print(f"  sweep: {label}")
        process_sweep(args.csv_dir, label, sweep_col, args.bootstrap_n, args.seed)
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())