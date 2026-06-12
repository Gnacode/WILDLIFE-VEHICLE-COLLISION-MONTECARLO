# Wildlife-Vehicle Collision Monte Carlo Simulator

[![Paper](https://img.shields.io/badge/Paper-MDPI_Sustainability-blue)](https://doi.org/XXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Interactive Data Page](https://img.shields.io/badge/Results-Interactive_Data_Page-orange)](https://gnacode.github.io/WILDLIFE-VEHICLE-COLLISION-MONTECARLO/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20651892-1682D4.svg)](https://doi.org/10.5281/zenodo.20651892)

Monte Carlo simulation framework evaluating a **combined radar and magnetometer sensor network with LoRa-mediated awareness propagation** for mitigating wildlife-vehicle collisions (WVCs) on rural road corridors, as a model-based proof of concept.

> **S. Makovetskyi and L. Thomsen**, "A Radar-Magnetometer Sensor Network with LoRa-Mediated Awareness Propagation for Wildlife-Vehicle Collision Mitigation: A Monte Carlo Simulation-Based Proof of Concept," *MDPI Sustainability*, 2026 (submitted).

---

## Key Results

20 trials x 4 simulated hours per mode (60 trials total), 1 km corridor, under Common Random Numbers. The three-mode contrast isolates each architectural layer: **Detection** adds sensors and driver alerts; **Aware** further adds LoRa-mediated network coordination.

| Mode | Collision rate / road entry | Detection rate | In-range latency | Reduction vs Control |
|------|:-:|:-:|:-:|:-:|
| **Control** | 13.22% | 0% | — | (reference) |
| **Detection** | **7.89%** | 20.85% | 0.313 s | **40.3%** (GLM p = 0.062) |
| **Aware** | 7.89% | 20.85% | 0.313 s | 40.3% (≡ Detection) |

The headline metric is the collision rate **per road entry**, which normalises for the differing crossing throughput each mode produces. Four findings define the architecture:

- **Operating envelope (traffic-volume regime).** The strongest result: a **96.2% reduction at 1 vehicle/direction/hour** (GLM p = 0.003) and **85.7% at 2 veh/dir/hr** (p = 0.023). A point-estimate sign reversal emerges at 16 veh/dir/hr, bounding the deployment claim to rural corridors below approximately 2 veh/dir/hr.
- **Awareness layer: active but not propagating.** The LoRa boost produces a highly significant sensing-side gain (detection-rate interaction p ≈ 10⁻¹⁸) that does **not** propagate to the collision rate at default density (interaction p = 0.949). This identifies a **safety-channel bottleneck** at the alert-delivery layer, not the sensor layer.
- **Magnetometer ablation.** Magnetometer-gated vehicle-presence confirmation is statistically indistinguishable from a perfect-knowledge oracle (Welch t = +0.25 on collisions/trial, +0.38 on collision rate), confirming the realism of the vehicle-confirmation layer.
- **Robustness.** Across **13 one-dimensional sensitivity sweeps**, 12 show no detectable mode × parameter interaction (interaction p > 0.7); traffic volume is the sole exception.

## Interactive Data Page

Explore the full results, figures, and statistics:

**[Launch the Data Page](https://gnacode.github.io/WILDLIFE-VEHICLE-COLLISION-MONTECARLO/)**

The page includes:
- Headline three-mode comparison with per-mode and pairwise statistics
- Traffic-volume operating envelope
- The 13-sweep robustness table (treatment × parameter interaction p-values)
- The awareness-layer "active but not propagating" diagnosis
- The magnetometer vehicle-sensing ablation
- 300 DPI publication figures regenerated from the dataset

No installation required. Runs in any modern browser.

---

## Repository Structure

```
WILDLIFE-VEHICLE-COLLISION-MONTECARLO/
├── README.md
├── LICENSE                              # MIT
├── wvc.py                               # Monte Carlo simulation engine (headline + sweeps)
├── compute_stats.py                     # statistics pipeline (Welch, Poisson/NB GLM, bootstrap, interaction)
├── wvc_web.py                           # interactive data-page / figure generator
├── data/
│   ├── wvc_results.csv                  # headline raw per-trial output (60 trials)
│   ├── wvc_results_<param>.csv          # raw per-trial output, one file per sweep (13 sweeps)
│   ├── wvc_stats_headline.csv           # headline per-mode statistics (+ _det_rate, _latency)
│   ├── wvc_stats_<param>.csv            # per-sweep effect sizes (+ _det_rate, _latency)
│   ├── wvc_stats_<param>_interaction.csv# treatment × parameter interaction tests
│   ├── regeneration_summary.{txt,json}  # run manifest
│   └── .regeneration_checkpoint.json    # resume checkpoint
└── docs/
    ├── index.html                       # GitHub Pages data page
    ├── figure_1_road_topology.png       # sensor-network topology
    ├── figure_2_animal_model.png        # FID-draw behavioural model
    ├── figure_3_headline.png            # headline three-mode comparison
    ├── figure_4_forest_headline.png     # headline, forest-corridor configuration
    ├── figure_5_traffic_volume.png      # traffic-volume operating envelope
    ├── figure_6_animal_density.png      # animal-density sweep (awareness activation)
    └── figure_S1_spacing.png … figure_S11_radar_range.png   # 11 supplementary sweep panels
```

The 13 swept parameters are: `spacing`, `radar_range`, `size`, `detection_rate`, `awareness_beta`, `awareness_tau`, `hesitate_dwell`, `freeze_mid_cross`, `animal_density`, `traffic_volume`, `driver_reaction`, `cruise_speed`, `caution_speed`.

## Running the Simulation

### Requirements

- Python 3.10 or higher
- Standard scientific Python stack (`numpy`, `pandas`, `scipy`, `plotly`, `seaborn`)
- ~30-60 min to regenerate all sweeps; ~5 min for the headline experiment only

### Installation

```bash
git clone https://github.com/Gnacode/WILDLIFE-VEHICLE-COLLISION-MONTECARLO.git
cd WILDLIFE-VEHICLE-COLLISION-MONTECARLO
pip install numpy pandas scipy plotly seaborn
```

### Pipeline

> Note: the command-line flags below are indicative. Run each script with `--help` to confirm the exact options for your version.

**1. Generate the raw per-trial data** (headline experiment + 13 sweeps), written to `data/`:

```bash
python wvc.py --output data/                 # all experiments
python wvc.py --output data/ --headline-only # headline experiment only (~5 min)
```

The runner is checkpointed (`data/.regeneration_checkpoint.json`), so an interrupted run can resume.

**2. Compute the statistics** (`wvc_stats_*.csv`) from the raw results:

```bash
python compute_stats.py --input data/ --output data/
```

This produces, for each experiment and metric (collision rate, detection rate, latency), the per-mode/per-cell statistics and the treatment × parameter interaction tests.

**3. Build the data page and figures** into `docs/`:

```bash
python wvc_web.py --input data/ --output docs/
```

Re-running the pipeline on the raw data reproduces every number and figure on the data page.

## Model

The simulator evaluates the architecture in a discrete-time Monte Carlo framework (Δt = 0.1 s) on a 1 km test corridor.

### Animal behaviour (FID-draw)

Each animal occupies one of six behavioural states — `FORAGING → APPROACHING → HESITATING → {CROSSING | FROZEN | FLEEING} → MOVES AWAY` — but the flight decision is governed by an empirical **flight-initiation-distance (FID) draw**, not fixed transition probabilities.

| Component | Description | Parameters |
|-----------|-------------|------------|
| FID | Distance at which an animal initiates flight | Log-normal, fitted median 72 m (empirical 69.1 m), truncated at 400 m |
| Detection ceiling cap | Max distance at which a vehicle is perceived | Log-normal, median 183 m, truncated to [50, 500] m |
| Non-responders | Animals that never flee and consequently cross | Fixed fraction 0.188 (FID = 0) |
| Effective trigger | Distance that actually triggers flight | min(FID, cap) |

Distributions are calibrated to the white-tailed-deer field data of Blackwell et al. (2014). As an untuned cross-check, the emergent control-mode flee fraction (0.83) matches the field response rate (≈ 0.81). Dwell times, locomotion speeds, and the mid-crossing freeze base rate are assumed (Tier 3) and bounded by sensitivity sweeps.

### Vehicle dynamics

Treiber-Hennecke-Helbing Intelligent Driver Model (IDM): free-flow cruise v₀ = 100 km/h; under an active alert the target speed reduces to v₁ = 30 km/h after a perception-reaction lag of 1.5 s (Green 2000); maximum emergency deceleration 8.0 m/s² (dry-asphalt traction ceiling, Lorenčič 2023).

### System architecture

Alternating-side Doppler radar nodes at same-side spacing s = 15 m (7.5 m offset; 134 nodes/km across both shoulders), baseline detection radius R_det = 10 m; three-axis magnetometer sites every 200 m; dynamic message signs (DMS) every 250 m. On any detection, an awareness notification propagates over LoRa to radars within 1500 m, applying a sensitivity boost β = 1.5× for τ = 30 s, while the DMS displays a caution speed.

## Experimental Design

| Mode | Sensors | Driver alert (DMS) | LoRa awareness boost |
|------|:-:|:-:|:-:|
| **Control** | — | — | — |
| **Detection** | active | yes | — |
| **Aware** | active | yes | β = 1.5× within 1500 m for 30 s |

Modes within a trial share one random seed (Common Random Numbers) for matched-pair precision. The headline experiment uses 20 trials × 4 h per mode; each sensitivity sweep uses 15 trials × 2 h per (mode, value).

## Statistical Analysis

For each metric and pairwise mode comparison: Welch's two-sample t-test (primary cell-level test); a Poisson GLM with log-exposure offset (negative-binomial fallback when Pearson dispersion > 1.5) reporting rate ratios and 95% Wald intervals for count-based rates; a 10,000-replicate non-parametric bootstrap; and, per sweep, a pooled mode × sweep interaction test (likelihood-ratio on nested GLMs for counts; two-way OLS ANOVA for latency).

## Hardware Platform

The radar and IoT-mesh signal-processing chains underlying the sensor nodes have been developed and characterised in adjacent work (see Related Work). The architecture is intended for resource-constrained edge platforms; no independent peer-reviewed validation of this sensor combination against WVC scenarios under operational conditions has yet been conducted, and field validation is the necessary next step.

## Citation

```bibtex
@article{makovetskyi2026wvc,
  title={A Radar-Magnetometer Sensor Network with LoRa-Mediated Awareness
         Propagation for Wildlife-Vehicle Collision Mitigation:
         A Monte Carlo Simulation-Based Proof of Concept},
  author={Makovetskyi, Sergii and Thomsen, Lars},
  journal={Sustainability},
  publisher={MDPI},
  year={2026}
}
```

Please also cite the archived software release: **Zenodo v2.0.0, DOI `10.5281/zenodo.20651892`** (the version DOI for this specific release). To cite all versions, use the concept DOI shown in the *Cite all versions* box on the [Zenodo record](https://doi.org/10.5281/zenodo.20651892).

## Related Work

The sensor hardware and detection-integrity methods underlying this architecture:

> S. Makovetskyi and L. Thomsen, "Temporal Spectral Noise-Floor Adaptation for Error-Intolerant Trigger Integrity in IoT Mesh Networks," arXiv:2605.06338, 2026.

> S. Makovetskyi and L. Thomsen, "Restoring CFAR Validity for Single-Channel IoT Sensor Streams: A Monte Carlo Comparison of Five Detectors under Cortex-M0+ Constraints," arXiv:2605.16159, 2026.

## Supplementary Materials

Appendix A (Extended Methods and Results) of the manuscript contains the full coverage-geometry and FID-distribution derivations, the complete statistical battery, per-sweep robustness narratives, Supplementary Tables S1-S5, and Supplementary Figures S1-S12.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **GNACODE INC.** — [gnacode.com](https://gnacode.com)
- Lars Thomsen — Medicine Hat, Alberta, Canada
- Sergii Makovetskyi — Kharkiv National University of Radio Electronics, Ukraine
