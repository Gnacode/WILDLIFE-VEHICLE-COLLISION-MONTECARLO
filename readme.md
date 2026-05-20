# WVC Sensor Network — Paper Data & Code

Supplementary data, simulator, and analysis pipeline for:

> **Multimodal Radar–Magnetometer Sensor Network with LoRa-Mediated Awareness for Wildlife–Vehicle Collision Prevention**
> Lars Thomsen (GNACODE Inc., Medicine Hat, Alberta, Canada) and Sergii Makovetskyi (KNURE, Kharkiv, Ukraine).
> Submitted to *MDPI Sustainability*, 2026.

🌐 **Interactive data page:** https://<your-github-username>.github.io/<repo-name>/

## What's here

| Path | What it is |
| --- | --- |
| [`wvc_simulator.py`](./wvc_simulator.py) | Single-file behaviorally-realistic Monte Carlo simulator. IDM vehicle dynamics, 6-state animal Markov model, three operating modes (control / detection / aware), 8 built-in sensitivity-sweep presets. |
| [`analyze_wvc.py`](./analyze_wvc.py) | Generates the interactive data page (`docs/index.html`) from simulator CSV outputs. Re-run after any new sweep. |
| [`data/`](./data) | Raw simulator CSVs reported in the paper. |
| [`docs/`](./docs) | Rendered data page (served via GitHub Pages). |
| [`requirements.txt`](./requirements.txt) | Python dependencies (pandas, plotly, scipy, numpy). |

## Reproducing the paper figures

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

# Re-render the data page from existing CSVs
python analyze_wvc.py --input data --output docs --copy-csvs

# Or re-run the simulator from scratch (≈ 30 min on a modern laptop)
python wvc_simulator.py --trials 20 --hours 4 --plot --csv data/wvc_results.csv
python wvc_simulator.py --sweep spacing        --trials 15 --hours 2 --csv data/sweep.csv
python wvc_simulator.py --sweep size           --trials 15 --hours 2 --csv data/sweep.csv
python wvc_simulator.py --sweep detection_rate --trials 15 --hours 2 --csv data/sweep.csv

# Then re-render
python analyze_wvc.py --input data --output docs
```

For local preview (no internet):

```bash
python analyze_wvc.py --input data --output docs --inline-plotly
python -m http.server -d docs
# Open http://localhost:8000
```

For GitHub Pages hosting, the default mode (CDN-referenced Plotly) gives a fast ~100 KB page.

## Simulator quick reference

```bash
# Default single-point Monte Carlo (10 trials × 2 h × 3 modes)
python wvc_simulator.py

# Paper-grade run (20 trials × 4 h)
python wvc_simulator.py --trials 20 --hours 4 --plot

# Custom sweep
python wvc_simulator.py --sweep spacing --values 5 10 20 40 --trials 30

# Sweep any Config field by name
python wvc_simulator.py --sweep idm_T --values 1.0 1.5 2.0 2.5
```

See `python wvc_simulator.py --help` for the full CLI.

## Built-in sweep presets

| Preset | Config field | Default values |
| --- | --- | --- |
| `spacing` | `radar_spacing` | 5, 10, 15, 20, 25, 30, 40 m |
| `range` | `radar_range` | 5, 8, 12, 15, 20, 25, 30 m |
| `size` | `size_scale` | 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0 × |
| `detection_rate` | `detection_rate_per_sec` | 0.3, 0.5, 1.0, 2.0, 3.0, 5.0 s⁻¹ |
| `rate` | `animal_rate_per_hr` | 5, 10, 15, 30, 45, 60 hr⁻¹ |
| `caution` | `caution_speed_kmh` | 20, 30, 45, 60, 80 km/h |
| `cruise` | `cruise_speed_kmh` | 60, 80, 100, 120, 140 km/h |
| `reaction` | `driver_reaction_s` | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s |

## Adding new analyses

`analyze_wvc.py` auto-discovers every `*sweep*.cs[v]` file in the input directory and adds a section for the parameter named in the first CSV column — drop a new CSV in `data/` and re-run the analysis script.

## License

Code released under MIT. Data released under CC-BY 4.0.

## Citation

If you use this code or data, please cite:

```bibtex
@article{thomsen_makovetskyi_2026_wvc,
    author  = {Thomsen, Lars and Makovetskyi, Sergii},
    title   = {Multimodal Radar--Magnetometer Sensor Network with LoRa-Mediated Awareness
               for Wildlife--Vehicle Collision Prevention: A Behaviorally-Realistic
               Monte Carlo Analysis},
    journal = {Sustainability},
    year    = {2026},
    note    = {Submitted}
}
```