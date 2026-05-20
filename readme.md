# Wildlife-Vehicle Collision Sensor Network:Code & Data

[![Interactive data page](https://img.shields.io/badge/Interactive-Data%20Page-2563eb?style=flat-square)](https://gnacode.github.io/WILDLIFE-VEHICLE-COLLISION-MONTECARLO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20313719-blue?style=flat-square)](https://doi.org/10.5281/zenodo.20313719)
)

Supplementary code, data, and interactive analysis for:

> **Combined Radar and Magnetometer Sensor Network with LoRa-Mediated Awareness for Wildlife–Vehicle Collision Prevention: A Monte Carlo Analysis**
>
> Lars Thomsen¹ and Sergii Makovetskyi²
> ¹ GNACODE Inc., Medicine Hat, Alberta, Canada
> ² Kharkiv National University of Radio Electronics, Kharkiv, Ukraine
>
> *Submitted to* **MDPI Sustainability**, 2026.

## Quick links

- 🌐 **Interactive data page:** <https://gnacode.github.io/WILDLIFE-VEHICLE-COLLISION-MONTECARLO/>
- 🐍 **Simulator** ([`wvc_simulator.py`](./wvc_simulator.py)) — single-file Python Monte Carlo simulator
- 📊 **Analysis pipeline** ([`analyze_wvc.py`](./analyze_wvc.py)) — generates the data page from raw CSVs
- 📁 **Raw data** ([`data/`](./data)) — per-trial CSVs reported in the paper

## What's in this repo

This repository accompanies a journal article presenting a layered animal-detection system designed to reduce wildlife–vehicle collisions on rural transport corridors. The system combines alternating-side Doppler radar nodes, three-axis magnetometers, dynamic message signs, and a LoRa-mediated coordination layer that propagates sensitivity-boost notifications between adjacent radars upon detection. System performance is evaluated through a discrete-time Monte Carlo simulation with a behaviourally-realistic six-state animal model, Intelligent Driver Model vehicle dynamics, and a three-mode experimental design that cleanly attributes collision reduction to the responsible architectural layer.

Every figure and table in the paper is generated directly from the code and data in this repository. Anyone with Python and an internet connection can reproduce the published results in approximately 30 minutes of compute.

## Headline results

| Metric | Control | Detection | Aware | Statistic |
|---|---|---|---|---|
| Collisions per trial | 5.35 ± 1.98 | 2.30 ± 1.45 | 2.30 ± 1.26 | t = 5.55, p < 0.0001 *** |
| Collision rate (per road entry) | 16.10 % | 5.08 % | 5.06 % | −57 % vs Control |
| Detection rate | 0 % | 98.9 % | 99.1 % | size-invariant |
| In-range latency | — | 299 ms | 286 ms | within kinematic budget |

The full system reduces wildlife–vehicle collisions by **57.0 % (p < 0.0001)** across 60 independent Monte Carlo trials. Sensitivity analyses across radar spacing (5–40 m), animal size (fox-class to moose-class), and sensor sensitivity (0.3–5.0 s⁻¹) establish the operating envelope and a deployment recommendation of **20–25 m alternating radar spacing**.

The full interactive results, including all sensitivity sweeps with hover-inspectable per-trial values, are at the [interactive data page](https://gnacode.github.io/WILDLIFE-VEHICLE-COLLISION-MONTECARLO/).

## Repository layout
