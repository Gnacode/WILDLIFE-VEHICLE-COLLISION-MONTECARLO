#!/usr/bin/env python3
"""
wvc.py — Unified WVC Monte-Carlo simulation toolkit
===================================================

A single entry point for the wildlife-vehicle-collision (WVC) sensor-network
study. It merges what used to be five separate scripts into one file driven by
subcommands:

    simulate    One Monte-Carlo experiment (single point or a parameter sweep).
                The quick, interactive workhorse. (was: wvc_simulator_magmod.py)

    data        Regenerate the full data set — the headline experiment plus all
                sensitivity sweeps — writing wvc_results*.csv and a JSON/TXT
                summary. Checkpointed and resumable. (was: generate_data.py)

    stats       Compute companion statistics (Welch t-tests, Poisson/NB GLM rate
                ratios, bootstrap CIs, mode x sweep interaction tests) from the
                result CSVs, writing wvc_stats*.csv. (was: compute_stats.py)

    figures     Render publication-grade PNG figures from the result + stats
                CSVs. (was: generate_figures.py)

    ablation    Magnetometer vehicle-model ablation and geometry sweep
                (reviewer R1-04). (was: run_ablation.py)

    pipeline    Compound command: run data -> stats -> figures back to back, so
                figure generation kicks in automatically once the simulation has
                finished. Pick a subset with --stages.

Global behaviour
----------------
All inputs and outputs are resolved relative to --workdir (default: the current
directory). Every subcommand accepts it. Heavy dependencies (matplotlib,
seaborn, statsmodels) are imported lazily, so `simulate`, `data`, and `ablation`
run without them installed.

Quick reference
---------------
    # one fast experiment, all three modes, with figures
    python wvc.py simulate --trials 20 --hours 4 --plot

    # isolate the awareness benefit with a sweep, parallelised
    python wvc.py simulate --sweep spacing --jobs 8 --plot

    # full reproducible build of the paper's data + stats + figures
    python wvc.py pipeline --jobs 8

    # only redo the figures after editing a plot
    python wvc.py figures --only 3 4 --dpi 600

    # magnetometer ablation, both parts, parallelised
    python wvc.py ablation --part all --trials 30 --jobs 8

Run `python wvc.py <command> --help` for the full option list of any command.

Authors: Lars Thomsen, Sergii Makovetskyi
Affiliation: Gnacode Inc. (Canada) / KNURE (Ukraine)
License: research / non-commercial
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# matplotlib / scipy are used by the simulator's own plotting + summary helpers.
# They are optional: a `simulate`/`data` run without them still works.
try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe; this tool only ever writes PNGs
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# Default; the data stage resets this inside the working directory.
CHECKPOINT_PATH = Path(".regeneration_checkpoint.json")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False



# ============================================================
#  SIMULATION ENGINE  (from wvc_simulator_magmod.py)
# ============================================================

# ============================================================
#  CONFIGURATION
# ============================================================

@dataclass
class Config:
    """All simulation parameters. Adjust here or via Config(**overrides)."""

    # --- Road & traffic geometry ---
    road_length: float = 1000.0         # m
    road_half_width: float = 2.0        # m, half-width (lanes are 2m each side of center)
    vehicles_per_dir: int = 4
    cruise_speed_kmh: float = 100.0
    caution_speed_kmh: float = 30.0     # speed when awareness active
    driver_reaction_s: float = 1.5

    # --- Wildlife arrival process ---
    animal_rate_per_hr: float = 15.0    # Poisson rate λ
    # Size mixture (small/medium/large) — fractions and size factors
    size_mix_small: float = 0.15        # fox/coyote
    size_mix_medium: float = 0.60       # deer
    size_mix_large: float = 0.25        # elk/moose
    size_scale: float = 1.0             # multiplier on all sampled animal sizes (RCS proxy)

    # --- Animal behavior parameters ---
    forage_dwell_min: float = 2.0       # min seconds in foraging
    forage_dwell_max: float = 10.0      # max seconds in foraging
    approach_speed: float = 1.5         # m/s heading toward road
    hesitate_dwell_min: float = 0.5
    hesitate_dwell_max: float = 3.0
    cross_speed: float = 4.0            # m/s crossing
    flee_speed: float = 6.0             # m/s fleeing
    freeze_dwell_min: float = 1.0
    freeze_dwell_max: float = 4.0
    random_walk_std: float = 0.3        # m/s noise on movement
    verge_threshold: float = 3.0        # m from center where hesitation starts

    # --- Behavioural decision probabilities (Fix A formulation) ---
    # HESITATING decision uses a two-branch structure:
    #
    # Clear branch (vehicle absent or below salience threshold):
    #   p(CROSS) = p_cross_if_clear, p(FLEE) = 1 - p_cross_if_clear, p(FROZEN) = 0
    #
    # Threat branch (boolean gate fires: vehicle within close-dist AND threat > 0.3):
    #   p(FROZEN) = p_freeze_max * theta                       (0.10 * theta)
    #   p(FLEE)   = p_flee_threat_base + p_flee_threat_gain * theta   (0.30 + 0.20*theta)
    #   p(CROSS)  = residual = 1 - p(FROZEN) - p(FLEE)        (0.70 - 0.30*theta)
    #
    # At theta=0.3 (caution-speed gate threshold): (FROZEN, FLEE, CROSS) = (0.03, 0.36, 0.61)
    # At theta=1.0 (cruise-speed maximum):         (FROZEN, FLEE, CROSS) = (0.10, 0.50, 0.40)
    # Threat presence increases both freeze and flee at the expense of crossing.
    p_cross_if_clear: float = 0.80      # clear-branch crossing probability (gate model)
    p_freeze_max: float = 0.10          # max freeze probability at full threat (gate model)
    p_flee_threat_base: float = 0.30    # threat-branch baseline flee rate (gate model)
    p_flee_threat_gain: float = 0.20    # additional flee scaling with threat (gate model)
    # Mid-crossing freeze (when vehicle very close + fast during traversal):
    # RETAINED IN BOTH BEHAVIOURAL CORES.
    p_freeze_during_cross_base: float = 0.15
    vehicle_close_dist: float = 80.0    # m, threat-branch gate distance (gate model)
    vehicle_visibility: float = 150.0   # m, animal can see vehicle within this (gate fallback)

    # --- Behavioural core selector (R1-07, R1-08, R1-13, R2-2, R3-1) ---
    # "fid"  : Blackwell 2014 FID-draw core. Per-animal FID and detection ceiling
    #          drawn at spawn from empirical distributions. Non-responders never flee.
    #          Replaces the four gate transition probabilities (p_cross_if_clear,
    #          p_freeze_max, p_flee_threat_base, p_flee_threat_gain) and the
    #          vehicle_close_dist gate. Mid-cross freeze is retained.
    # "gate" : Legacy deterministic gate model (retained for backward compatibility).
    behaviour_model: str = "fid"

    # --- FID-draw core parameters (Blackwell, Seamans & DeVault 2014; PLoS ONE) ---
    # Per-encounter flight-initiation distance, drawn per animal at spawn.
    # Log-normal fit to Blackwell panel A: responder median 72 m, max ~368 m
    #   ln(72) = 4.277; sigma chosen so 99th pct of fid ~ 368 m
    fid_nonresponder_p: float = 0.188     # P(FID=0); these animals never flee
    fid_log_mu: float = 4.277             # ln(median responder FID, m)
    fid_log_sigma: float = 0.702          # spread of log-normal FID
    fid_max_m: float = 400.0              # truncation at Blackwell panel A max ~368 m + jitter pad
    fid_jitter_m: float = 4.0             # additive Gaussian smoothing on FID samples
    # Per-animal detection ceiling. FID is capped at this distance: an animal cannot
    # react to a vehicle it has not yet detected. Log-normal fit to Blackwell panel A
    # start-distance envelope: median 183 m, range 62-438 m.
    fid_detection_log_mu: float = 5.210      # ln(median detection ceiling, m)
    fid_detection_log_sigma: float = 0.499   # spread of log-normal ceiling
    fid_detection_min_m: float = 50.0        # lower clip near Blackwell observed min 62 m
    fid_detection_max_m: float = 500.0       # upper clip near Blackwell observed max 438 m + jitter pad
    fid_detection_jitter_m: float = 8.0      # additive Gaussian smoothing on ceiling

    # --- Sensor topology ---
    radar_spacing: float = 15.0         # m, alternating sides
    radar_range: float = 10.0           # m, baseline (unboosted) detection radius
    #   Under awareness boost the EFFECTIVE range becomes radar_range * boost_factor.
    #   See _process_detection() and the awareness layer documentation in Section 2.
    radar_verge_offset: float = 2.5     # m, radar position outside road
    magnetic_spacing: float = 200.0     # m
    # Detection model: per-frame Pd via exponential rate
    detection_rate_per_sec: float = 3.0  # κ for reference RCS=1.0

    # --- Magnetometer VEHICLE-sensing model (addresses reviewer R1-04) ---
    #   The radar senses ANIMALS; the magnetometer senses VEHICLES (ferromagnetic
    #   mass). A vehicle is "present-known" to the system from the instant a
    #   magnetometer site detects it until a buffer window elapses. Driver alerts
    #   are gated on that knowledge instead of being issued corridor-wide blind.
    #   vehicle_model = "perfect"      -> system has perfect vehicle knowledge
    #                                     (original behaviour; every vehicle alertable)
    #   vehicle_model = "magnetometer" -> alerts gated on magnetometer presence
    vehicle_model: str = "perfect"        # "perfect" | "magnetometer"
    mag_sensor_offset: float = 0.75       # m, sensor set-back from road EDGE     # PROVISIONAL
    mag_envelope_m: float = 3.0           # m, full-Pd closest-approach radius     # PROVISIONAL
    mag_pd_r50: float = 4.2               # m, single-pass 50% detection range     # PROVISIONAL
    mag_pd_k: float = 2.2                 # 1/m, logistic roll-off sharpness       # PROVISIONAL
    mag_buffer_s: float = 10.0            # s, presence hold (must exceed spacing/speed)
    mag_timing_jitter_s: float = 0.08     # s, 1-sigma detection-timestamp jitter  # PROVISIONAL

    # --- Awareness / LoRa propagation ---
    awareness_range_m: float = 1500.0   # corridor-wide
    awareness_persist_s: float = 30.0   # min duration after last detection
    lora_delay_ms: float = 200.0
    boost_factor: float = 1.5           # detection sensitivity boost on neighbors

    # --- IDM vehicle dynamics ---
    idm_s0: float = 5.0
    idm_T: float = 1.5
    idm_a_max: float = 2.5
    idm_b_comf: float = 4.0
    idm_a_emergency: float = 8.0   # traction-limited (a=mu*g, mu~0.81 dry asphalt); matches provenance spec
    idm_delta: int = 4
    warning_stop_buffer: float = 30.0   # m short of warning to stop

    # --- Simulation control ---
    dt: float = 0.1                     # s
    seed: Optional[int] = None


# ============================================================
#  ENUMS
# ============================================================

class AnimalState(IntEnum):
    FORAGING = 0
    APPROACHING = 1
    HESITATING = 2
    CROSSING = 3
    FROZEN = 4
    FLEEING = 5
    GONE = 6     # terminal


# ============================================================
#  ENTITIES (dataclasses for clarity, dict-of-arrays for speed if needed later)
# ============================================================

@dataclass
class Animal:
    x: float            # longitudinal position along road [0, road_length]
    y: float            # lateral position; |y|<road_half_width = on road
    side: int           # +1=south start, -1=north start
    size: float         # RCS factor; influences detection probability
    state: AnimalState
    state_changed_at: float
    spawned_at: float
    dwell_target: float = 0.0
    detected: bool = False
    detected_at: float = -1.0
    first_in_range_at: float = -1.0  # when animal first entered any radar range
    entered_road: bool = False
    dead: bool = False
    # --- FID-draw core (Blackwell 2014); only used when behaviour_model == "fid" ---
    fid_distance: float = -1.0        # m, per-animal FID; 0 = non-responder
    fid_detection_cap: float = -1.0   # m, max distance at which animal can react


@dataclass
class Vehicle:
    x: float
    direction: int      # +1 or -1
    speed: float        # m/s
    alerted: bool = False
    alert_reaction_at: float = 0.0
    crashed: bool = False
    crashed_at: float = 0.0
    last_acc: float = 0.0
    # --- magnetometer vehicle-sensing state ---
    y: float = 0.0              # lateral lane-centre position [m]
    mag_last_seen: float = -1e9  # s, last magnetometer detection time
    mag_known: bool = False      # currently present-known to the system
    mag_ever: bool = False       # detected by >=1 site at least once (coverage stat)


@dataclass
class Radar:
    x: float
    side: int           # +1=south, -1=north
    boost_until: float = 0.0


@dataclass
class MagSite:
    """Magnetometer site sensing vehicle passages (ferromagnetic mass)."""
    x: float
    side: int           # +1=south edge, -1=north edge


@dataclass
class TrialStats:
    total_animals: int = 0
    detected: int = 0
    road_entries: int = 0
    collisions: int = 0
    warnings_issued: int = 0
    vehicles_alerted: int = 0
    mag_known_frames: int = 0       # vehicle-frames present-known to the system
    mag_total_frames: int = 0       # total vehicle-frames (denominator)
    mag_gated_out_frames: int = 0   # alert-eligible frames suppressed (vehicle not known)
    detection_latencies: List[float] = field(default_factory=list)        # spawn -> detected
    in_range_latencies: List[float] = field(default_factory=list)         # first_in_range -> detected
    awareness_seconds: float = 0.0
    state_visits: dict = field(default_factory=dict)
    frozen_on_road_seconds: float = 0.0


# ============================================================
#  CORE SIMULATION
# ============================================================

class WVCSimulation:
    """A single Monte Carlo trial. Independent random number stream.

    Modes:
      'control'   No sensors deployed; pure baseline (no detection, no DMS).

      'detection' Sensors deployed. Each radar detects independently at its
                  baseline sensitivity. On detection, the DMS alerts drivers
                  and the IDM responds (caution speed + hard-brake for road
                  animals). Sensors do NOT cooperate — no neighbour boost.

      'aware'     Same driver/IDM behaviour as 'detection', PLUS network
                  awareness: any detection LoRa-broadcasts a sensitivity
                  boost to all sensors within R_a, lasting τ_persist seconds.
                  Boosted neighbours detect subsequent animals — and animals
                  in coverage gaps — earlier, giving drivers a longer span
                  to react. The boost is the only mechanism that distinguishes
                  'aware' from 'detection'.
    """

    VALID_MODES = ("control", "detection", "aware")

    def __init__(self, config: Config, mode: str = "aware"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.cfg = config
        self.mode = mode
        # Behavior toggles derived from mode
        self.detection_enabled = (mode != "control")
        self.awareness_enabled = (mode == "aware")
        # Back-compat shim for any code that still reads warnings_on
        self.warnings_on = self.awareness_enabled

        self.rng = np.random.default_rng(config.seed)
        self.time = 0.0
        self.spawn_acc = 0.0
        self.awareness_active = False
        self.awareness_until = 0.0

        self.radars: List[Radar] = []
        self.mag_sites: List[MagSite] = []
        self.vehicles: List[Vehicle] = []
        self.animals: List[Animal] = []
        self.stats = TrialStats()
        self.stats.state_visits = {s: 0 for s in AnimalState}

        self._init_sensors()
        self._init_vehicles()

    # ----- Initialization -----

    def _init_sensors(self):
        c = self.cfg
        side = -1   # start north
        x = 0.0
        while x <= c.road_length + 1e-6:
            self.radars.append(Radar(x=x, side=side))
            side *= -1
            x += c.radar_spacing
        # Magnetometer sites along the corridor, alternating sides
        side = -1
        x = 0.0
        while x <= c.road_length + 1e-6:
            self.mag_sites.append(MagSite(x=x, side=side))
            side *= -1
            x += c.magnetic_spacing

    def _init_vehicles(self):
        c = self.cfg
        cruise = c.cruise_speed_kmh / 3.6
        for _ in range(c.vehicles_per_dir):
            for direction in (+1, -1):
                x0 = self.rng.uniform(-50, c.road_length + 50)
                # lane centre: dir +1 in south lane (y=-1), dir -1 in north lane (y=+1)
                # (matches the lane bands used in _check_collisions)
                y = -1.0 if direction == +1 else 1.0
                self.vehicles.append(Vehicle(x=x0, direction=direction, speed=cruise, y=y,
                                             mag_last_seen=0.0, mag_known=True))

    # ----- Spawn -----

    def _spawn_animal(self):
        c = self.cfg
        x = self.rng.uniform(0, c.road_length)
        side = int(self.rng.choice([-1, +1]))
        # Size from three-component mixture (multiplied by size_scale for sweeps)
        r = self.rng.random()
        if r < c.size_mix_small:
            size = self.rng.uniform(0.25, 0.55)
        elif r < c.size_mix_small + c.size_mix_medium:
            size = self.rng.uniform(0.7, 1.2)
        else:
            size = self.rng.uniform(1.4, 2.3)
        size *= c.size_scale

        y_start = -14.0 if side == -1 else 14.0
        a = Animal(
            x=x, y=y_start, side=side, size=size,
            state=AnimalState.FORAGING,
            state_changed_at=self.time,
            spawned_at=self.time,
            dwell_target=self.rng.uniform(c.forage_dwell_min, c.forage_dwell_max),
        )
        # --- FID-draw core: per-animal flight-initiation distance and detection ceiling
        if c.behaviour_model == "fid":
            # Non-responder with prob fid_nonresponder_p; FID = 0 -> never flees
            if self.rng.random() < c.fid_nonresponder_p:
                a.fid_distance = 0.0
            else:
                # Log-normal draw with additive Gaussian jitter; clipped to Blackwell range
                fid = float(self.rng.lognormal(c.fid_log_mu, c.fid_log_sigma))
                if c.fid_jitter_m > 0:
                    fid += float(self.rng.normal(0.0, c.fid_jitter_m))
                a.fid_distance = float(np.clip(fid, 0.0, c.fid_max_m))
            # Detection ceiling — caps FID at distance the animal can detect a vehicle
            cap = float(self.rng.lognormal(c.fid_detection_log_mu, c.fid_detection_log_sigma))
            if c.fid_detection_jitter_m > 0:
                cap += float(self.rng.normal(0.0, c.fid_detection_jitter_m))
            a.fid_detection_cap = float(np.clip(cap, c.fid_detection_min_m, c.fid_detection_max_m))
        self.animals.append(a)
        self.stats.total_animals += 1
        self.stats.state_visits[AnimalState.FORAGING] += 1

    # ----- Animal behavior (state machine) -----

    def _closest_vehicle(self, a: Animal) -> Tuple[Optional[Vehicle], float]:
        """Return (closest vehicle within visibility, distance), or (None, inf)."""
        c = self.cfg
        best, best_d = None, c.vehicle_visibility
        for v in self.vehicles:
            d = abs(v.x - a.x)
            if d < best_d:
                best, best_d = v, d
        return best, best_d if best is not None else float("inf")

    def _vehicle_threat_factor(self, v: Optional[Vehicle]) -> float:
        """0..1 — how threatening a vehicle is to animals (proportional to speed)."""
        if v is None:
            return 0.0
        cruise = self.cfg.cruise_speed_kmh / 3.6
        return min(1.0, v.speed / cruise)

    def _set_state(self, a: Animal, new_state: AnimalState):
        c = self.cfg
        a.state = new_state
        a.state_changed_at = self.time
        self.stats.state_visits[new_state] = self.stats.state_visits.get(new_state, 0) + 1
        # Set dwell time for states that have one
        if new_state == AnimalState.HESITATING:
            a.dwell_target = self.rng.uniform(c.hesitate_dwell_min, c.hesitate_dwell_max)
        elif new_state == AnimalState.FROZEN:
            a.dwell_target = self.rng.uniform(c.freeze_dwell_min, c.freeze_dwell_max)
        elif new_state == AnimalState.FORAGING:
            a.dwell_target = self.rng.uniform(c.forage_dwell_min, c.forage_dwell_max)
        else:
            a.dwell_target = 0.0

    def _fid_triggers_flee(self, a: Animal, v_near, v_dist) -> bool:
        """True if a vehicle is within the animal's effective FID (FID-draw core only).

        Effective FID is min(fid_distance, fid_detection_cap). Non-responders
        (fid_distance == 0) never trigger.
        """
        if a.fid_distance <= 0.0:
            return False
        if v_near is None:
            return False
        effective = a.fid_distance
        if a.fid_detection_cap > 0 and a.fid_detection_cap < effective:
            effective = a.fid_detection_cap
        return v_dist <= effective

    def _update_animal_state(self, a: Animal):
        c = self.cfg
        elapsed = self.time - a.state_changed_at
        v_near, v_dist = self._closest_vehicle(a)
        threat = self._vehicle_threat_factor(v_near)
        # "Vehicle close" means within close-dist AND moving (threat > 0.3) — gate model
        vehicle_threatens = (v_near is not None and v_dist < c.vehicle_close_dist and threat > 0.3)

        # --- FID-draw core: pre-state-machine flee trigger ---
        # Under the Blackwell FID-draw model, an animal flees whenever any approaching
        # vehicle reaches its drawn FID (capped by its detection ceiling). This applies
        # in any pre-crossing state. Non-responders (fid_distance == 0) never trigger.
        if c.behaviour_model == "fid" and a.state in (
                AnimalState.FORAGING, AnimalState.APPROACHING, AnimalState.HESITATING):
            if self._fid_triggers_flee(a, v_near, v_dist):
                self._set_state(a, AnimalState.FLEEING)
                return

        if a.state == AnimalState.FORAGING:
            if elapsed > a.dwell_target:
                self._set_state(a, AnimalState.APPROACHING)

        elif a.state == AnimalState.APPROACHING:
            if abs(a.y) <= c.verge_threshold:
                self._set_state(a, AnimalState.HESITATING)
            elif c.behaviour_model == "gate" and vehicle_threatens and self.rng.random() < 0.05 * threat * c.dt:
                # Gate-model probabilistic flee on approach due to nearby fast vehicle.
                # (Under "fid", the deterministic FID trigger above handles approach-time flight.)
                self._set_state(a, AnimalState.FLEEING)

        elif a.state == AnimalState.HESITATING:
            if elapsed > a.dwell_target:
                # Decision time.
                if c.behaviour_model == "fid":
                    # Under FID-draw: if we reach dwell expiry without an FID trigger
                    # firing upstream, no vehicle has reached this animal's FID during
                    # its hesitation window. The animal therefore commits to crossing.
                    # (Non-responders always reach this branch.)
                    self._set_state(a, AnimalState.CROSSING)
                else:
                    # Legacy gate-model decision (R1-07/R1-08/R1-13 superseded path).
                    # Clear branch (vehicle absent or below salience threshold):
                    #   p(CROSS) = 0.80, p(FLEE) = 0.20, p(FROZEN) = 0
                    # Threat branch (boolean gate fires, theta >= 0.3):
                    #   p(FROZEN) = p_freeze_max * theta            (0.10*theta)
                    #   p(FLEE)   = p_flee_threat_base + p_flee_threat_gain * theta
                    #                                                (0.30 + 0.20*theta)
                    #   p(CROSS)  = residual
                    if vehicle_threatens:
                        p_frozen = c.p_freeze_max * threat
                        p_fleeing = c.p_flee_threat_base + c.p_flee_threat_gain * threat
                        r = self.rng.random()
                        if r < p_frozen:
                            self._set_state(a, AnimalState.FROZEN)
                        elif r < p_frozen + p_fleeing:
                            self._set_state(a, AnimalState.FLEEING)
                        else:
                            self._set_state(a, AnimalState.CROSSING)
                    else:
                        if self.rng.random() < c.p_cross_if_clear:
                            self._set_state(a, AnimalState.CROSSING)
                        else:
                            self._set_state(a, AnimalState.FLEEING)

        elif a.state == AnimalState.CROSSING:
            # Check freeze trigger: vehicle very close + fast
            if (v_near is not None and v_dist < 40 and threat > 0.5
                    and self.rng.random() < c.p_freeze_during_cross_base * threat * c.dt):
                self._set_state(a, AnimalState.FROZEN)
            # Check if reached far side
            if (a.side == -1 and a.y > c.verge_threshold + 1) or \
               (a.side == +1 and a.y < -(c.verge_threshold + 1)):
                self._set_state(a, AnimalState.GONE)

        elif a.state == AnimalState.FROZEN:
            self.stats.frozen_on_road_seconds += c.dt
            if elapsed > a.dwell_target:
                # 50/50 cross or flee
                if self.rng.random() < 0.5:
                    self._set_state(a, AnimalState.CROSSING)
                else:
                    self._set_state(a, AnimalState.FLEEING)

        elif a.state == AnimalState.FLEEING:
            # Escaped to verge?
            if (a.side == -1 and a.y < -5) or (a.side == +1 and a.y > 5):
                self._set_state(a, AnimalState.GONE)

    def _move_animal(self, a: Animal):
        c = self.cfg
        dt = c.dt
        noise_y = self.rng.normal(0, c.random_walk_std)
        noise_x = self.rng.normal(0, c.random_walk_std)

        if a.state == AnimalState.FORAGING:
            drift_y = -np.sign(a.y) * 0.1  # weak drift away from road
            a.y += (drift_y + noise_y * 0.5) * dt
            a.x += noise_x * dt
        elif a.state == AnimalState.APPROACHING:
            sign_to_road = -np.sign(a.y) if abs(a.y) > 0.1 else 0
            a.y += (sign_to_road * c.approach_speed + noise_y * 0.4) * dt
            a.x += noise_x * 0.3 * dt
        elif a.state == AnimalState.HESITATING:
            a.y += noise_y * 0.3 * dt
            a.x += noise_x * dt
        elif a.state == AnimalState.CROSSING:
            sign = +1 if a.side == -1 else -1
            a.y += (sign * c.cross_speed + noise_y * 0.2) * dt
            a.x += noise_x * 0.2 * dt
        elif a.state == AnimalState.FROZEN:
            a.x += noise_x * 0.1 * dt  # tiny tremor
        elif a.state == AnimalState.FLEEING:
            sign = -1 if a.side == -1 else +1
            a.y += (sign * c.flee_speed + noise_y * 0.2) * dt

        # Track road entry
        on_road_now = abs(a.y) < c.road_half_width
        if on_road_now and not a.entered_road:
            a.entered_road = True
            self.stats.road_entries += 1

    # ----- Detection -----

    def _process_detection(self):
        c = self.cfg
        # In CONTROL mode, no sensors are deployed — true unmitigated baseline
        if not self.detection_enabled:
            return
        for r in self.radars:
            radar_y = -c.radar_verge_offset if r.side == -1 else c.radar_verge_offset
            # Awareness boost expands the EFFECTIVE detection range rather than
            # multiplying the per-frame detection rate. Physically this corresponds
            # to elevated transmit power on the Doppler RF front-end: by the radar
            # equation R_max scales with P_t^(1/4), so a modest power increase yields
            # a measurable range extension while leaving per-frame detection rate at
            # its baseline saturated value once the target is in range.
            boost = c.boost_factor if self.time < r.boost_until else 1.0
            effective_range = c.radar_range * boost
            range2 = effective_range ** 2

            for a in self.animals:
                if a.detected or a.dead or a.state == AnimalState.GONE:
                    continue
                dx = r.x - a.x
                dy = radar_y - a.y
                if dx*dx + dy*dy > range2:
                    continue
                # Animal is in range of at least one radar — record first occurrence
                if a.first_in_range_at < 0:
                    a.first_in_range_at = self.time
                size_factor = min(2.0, 0.4 + a.size * 0.6)
                rate = c.detection_rate_per_sec * size_factor   # κ NOT boosted; only range is
                p_frame = 1 - np.exp(-rate * c.dt)
                if self.rng.random() < p_frame:
                    a.detected = True
                    a.detected_at = self.time
                    self.stats.detected += 1
                    self.stats.detection_latencies.append(self.time - a.spawned_at)
                    self.stats.in_range_latencies.append(self.time - a.first_in_range_at)
                    self.stats.warnings_issued += 1
                    # Driver alert fires on ANY successful detection (both 'detection'
                    # and 'aware' modes). The DMS lights up and vehicles will react.
                    self.awareness_active = True
                    self.awareness_until = self.time + c.awareness_persist_s
                    # Only 'aware' mode propagates the LoRa-broadcast sensitivity boost
                    # to neighbouring sensors — this is the *network awareness* layer.
                    # Its kinematic benefit is earlier detection of subsequent animals
                    # (and earlier detection at the edge of coverage gaps), giving the
                    # driver a longer span to act.
                    if self.awareness_enabled:
                        for r2 in self.radars:
                            if r2 is r:
                                continue
                            if abs(r2.x - r.x) <= c.awareness_range_m:
                                r2.boost_until = max(r2.boost_until,
                                                     self.time + c.awareness_persist_s)

        # Decay awareness when no dangerous animals remain AND persist time elapsed
        if self.awareness_active and self.time >= self.awareness_until:
            still_dangerous = any(
                a.detected and a.state in (
                    AnimalState.APPROACHING, AnimalState.HESITATING,
                    AnimalState.CROSSING, AnimalState.FROZEN
                ) and not a.dead
                for a in self.animals
            )
            if not still_dangerous:
                self.awareness_active = False

    # ----- Magnetometer vehicle sensing (R1-04) -----

    def _mag_pd(self, r: float) -> float:
        """Single-pass detection probability vs closest-approach distance r [m]."""
        c = self.cfg
        pd = 1.0 / (1.0 + np.exp(c.mag_pd_k * (r - c.mag_pd_r50)))
        if r <= c.mag_envelope_m:
            pd = max(pd, 0.999)
        return float(min(1.0, max(0.0, pd)))

    def _mag_closest_approach(self, v: Vehicle, site: MagSite) -> float:
        """Lateral closest-approach distance between a magnetometer and a vehicle path."""
        c = self.cfg
        site_y = -(c.road_half_width + c.mag_sensor_offset) if site.side == +1 \
            else (c.road_half_width + c.mag_sensor_offset)
        return abs(v.y - site_y)

    def _update_vehicle_sensing(self, prev_x: dict):
        """Detect vehicles as they pass magnetometer sites; maintain presence state.

        In 'perfect' mode the system is omniscient about vehicles (original
        behaviour). In 'magnetometer' mode a vehicle is present-known only while
        within the buffer of its last successful magnetometer detection.
        """
        c = self.cfg
        if c.vehicle_model == "perfect":
            for v in self.vehicles:
                v.mag_known = True
                v.mag_ever = True
            return
        now = self.time
        for v in self.vehicles:
            px = prev_x.get(id(v), v.x)
            for s in self.mag_sites:
                crossed = (v.direction == +1 and px < s.x <= v.x) or \
                          (v.direction == -1 and px > s.x >= v.x)
                if crossed:
                    r = self._mag_closest_approach(v, s)
                    if self.rng.random() < self._mag_pd(r):
                        v.mag_last_seen = now  # (timestamp jitter irrelevant to presence)
                        v.mag_ever = True
            v.mag_known = (now - v.mag_last_seen) <= c.mag_buffer_s

    # ----- Vehicles (IDM) -----

    def _update_vehicles(self):
        c = self.cfg
        dt = c.dt
        cruise = c.cruise_speed_kmh / 3.6
        caution = c.caution_speed_kmh / 3.6
        now = self.time

        # Driver alerts and IDM hard-brake fire whenever the system has detection
        # capability (both 'detection' and 'aware' modes). The difference between
        # those two modes is how *quickly* detection happens, not whether drivers
        # are told.
        awareness_alerts = self.awareness_active and self.detection_enabled

        # Build list of currently-dangerous animals (on road or just about to be).
        # Hard-brake virtual obstacles apply whenever the system is detecting.
        immediate_dangers = []
        if self.detection_enabled:
            for a in self.animals:
                if not a.detected or a.dead or a.state == AnimalState.GONE:
                    continue
                if a.state in (AnimalState.CROSSING, AnimalState.FROZEN):
                    immediate_dangers.append(a.x)
                elif a.state == AnimalState.HESITATING and abs(a.y) < 2.5:
                    immediate_dangers.append(a.x)

        for v in self.vehicles:
            # Alert is gated on whether the system KNOWS the vehicle is present.
            # In 'perfect' mode mag_known is always True (original corridor-wide
            # behaviour); in 'magnetometer' mode only present-known vehicles alert.
            self.stats.mag_total_frames += 1
            if v.mag_known:
                self.stats.mag_known_frames += 1
            can_alert = awareness_alerts and v.mag_known
            if awareness_alerts and not v.mag_known:
                self.stats.mag_gated_out_frames += 1
            if can_alert and not v.alerted:
                v.alerted = True
                v.alert_reaction_at = now + c.driver_reaction_s
                self.stats.vehicles_alerted += 1
            elif not can_alert and v.alerted:
                v.alerted = False

            # Find lead vehicle in same direction
            lead_dist = np.inf
            lead_speed = cruise
            for o in self.vehicles:
                if o is v or o.direction != v.direction:
                    continue
                d = (o.x - v.x) * v.direction
                if 0 < d < lead_dist:
                    lead_dist = d
                    lead_speed = o.speed

            # Cruise target — drops to caution when awareness active
            if v.alerted and now >= v.alert_reaction_at:
                cruise_target = min(cruise, caution)
            else:
                cruise_target = cruise

            # Hard-brake for imminent danger: nearest dangerous animal ahead
            if v.alerted and now >= v.alert_reaction_at and immediate_dangers:
                # find nearest danger ahead of vehicle
                nearest_danger_dist = np.inf
                for danger_x in immediate_dangers:
                    d = (danger_x - v.x) * v.direction
                    if 0 < d < nearest_danger_dist:
                        nearest_danger_dist = d
                if nearest_danger_dist < np.inf:
                    # treat as virtual stopped obstacle with safety buffer
                    virtual_dist = nearest_danger_dist - c.warning_stop_buffer
                    if virtual_dist < lead_dist:
                        lead_dist = max(0.5, virtual_dist)
                        lead_speed = 0.0

            # IDM acceleration
            if lead_dist == np.inf:
                acc = c.idm_a_max * (1 - (v.speed / cruise_target) ** c.idm_delta)
            else:
                dv = v.speed - lead_speed
                s_star = c.idm_s0 + max(
                    0,
                    v.speed * c.idm_T + (v.speed * dv) / (2 * np.sqrt(c.idm_a_max * c.idm_b_comf))
                )
                acc = c.idm_a_max * (
                    1 - (v.speed / cruise_target) ** c.idm_delta - (s_star / lead_dist) ** 2
                )
            acc = max(acc, -c.idm_a_emergency)
            v.speed = max(0.0, v.speed + acc * dt)
            v.last_acc = acc
            if v.crashed:
                v.speed *= 0.7
            v.x += v.direction * v.speed * dt

            # Wraparound
            if v.direction == 1 and v.x > c.road_length + 80:
                v.x = -50
                v.alerted = False
                v.speed = cruise
                v.mag_last_seen = now; v.mag_known = True
            elif v.direction == -1 and v.x < -80:
                v.x = c.road_length + 50
                v.alerted = False
                v.speed = cruise
                v.mag_last_seen = now; v.mag_known = True

    # ----- Collisions -----

    def _check_collisions(self):
        c = self.cfg
        for v in self.vehicles:
            if v.crashed:
                if self.time - v.crashed_at > 1.5:
                    v.crashed = False
                continue
            for a in self.animals:
                if a.dead or a.state == AnimalState.GONE:
                    continue
                if abs(a.y) >= c.road_half_width:
                    continue
                if abs(v.x - a.x) > 4:
                    continue
                # Lane match
                if v.direction == 1 and -2.2 < a.y < 0.2:
                    a.dead = True
                    v.crashed = True
                    v.crashed_at = self.time
                    self.stats.collisions += 1
                    break
                if v.direction == -1 and -0.2 < a.y < 2.2:
                    a.dead = True
                    v.crashed = True
                    v.crashed_at = self.time
                    self.stats.collisions += 1
                    break

    # ----- Step -----

    def step(self):
        c = self.cfg
        self.time += c.dt
        if self.awareness_active:
            self.stats.awareness_seconds += c.dt

        # Poisson spawn
        self.spawn_acc += (c.animal_rate_per_hr / 3600) * c.dt
        while self.spawn_acc >= 1:
            self._spawn_animal()
            self.spawn_acc -= 1

        # Update animals
        for a in self.animals:
            if a.state == AnimalState.GONE or a.dead:
                continue
            self._update_animal_state(a)
            self._move_animal(a)

        # Garbage collect terminal animals
        self.animals = [a for a in self.animals
                        if not (a.state == AnimalState.GONE
                                or (a.dead and self.time - a.spawned_at > 10))]

        self._process_detection()
        prev_x = {id(v): v.x for v in self.vehicles}
        self._update_vehicles()
        self._update_vehicle_sensing(prev_x)
        self._check_collisions()

    def run(self, duration_s: float, verbose: bool = False):
        n_steps = int(duration_s / self.cfg.dt)
        for i in range(n_steps):
            self.step()


# ============================================================
#  MONTE CARLO HARNESS
# ============================================================

# Common Random Numbers (CRN): all three modes within a single trial use the SAME
# seed, so animal arrival times, sizes, FID draws, and other world-state random
# decisions are identical across modes. This is the standard variance-reduction
# design for pairwise policy comparison in stochastic simulation: any difference in
# collision rate between Control / Detection / Aware then reflects the policy, not
# a different stochastic world. Note that worlds diverge once the policies start
# making different decisions (different alert timing -> different vehicle dynamics),
# but the matched-pair structure is preserved through the pre-policy random draws.
_MODE_OFFSETS = {"control": 0, "detection": 0, "aware": 0}


def _simulate_trial(cfg_dict: dict, mode: str, trial: int, duration_s: float) -> dict:
    """Run ONE trial and build its result dict. Module-level so it is picklable
    for multiprocessing; used by both the serial and parallel harnesses so their
    output is identical. Seed is derived deterministically from (trial, mode)."""
    seed = trial * 10000 + _MODE_OFFSETS.get(mode, 0)
    trial_cfg = Config(**{**cfg_dict, "seed": seed})
    sim = WVCSimulation(trial_cfg, mode=mode)
    t0 = time.time()
    sim.run(duration_s)
    elapsed = time.time() - t0
    s = sim.stats
    mean_lat = float(np.mean(s.detection_latencies)) if s.detection_latencies else 0.0
    mean_in_range = float(np.mean(s.in_range_latencies)) if s.in_range_latencies else 0.0
    return {
        "mode": mode,
        "trial": trial,
        "animals": s.total_animals,
        "detected": s.detected,
        "det_rate": s.detected / max(1, s.total_animals),
        "road_entries": s.road_entries,
        "collisions": s.collisions,
        "col_rate": s.collisions / max(1, s.road_entries),
        "warnings": s.warnings_issued,
        "vehicles_alerted": s.vehicles_alerted,
        "mean_latency_s": mean_lat,
        "in_range_latency_s": mean_in_range,
        "awareness_pct": 100 * s.awareness_seconds / duration_s,
        "veh_present_known_pct": 100 * s.mag_known_frames / max(1, s.mag_total_frames),
        "veh_coverage_pct": 100 * sum(1 for v in sim.vehicles if v.mag_ever) / max(1, len(sim.vehicles)),
        "veh_gated_out_pct": 100 * s.mag_gated_out_frames / max(1, s.mag_total_frames),
        "frozen_on_road_s": s.frozen_on_road_seconds,
        "n_foraging": s.state_visits.get(AnimalState.FORAGING, 0),
        "n_approaching": s.state_visits.get(AnimalState.APPROACHING, 0),
        "n_hesitating": s.state_visits.get(AnimalState.HESITATING, 0),
        "n_crossing": s.state_visits.get(AnimalState.CROSSING, 0),
        "n_frozen": s.state_visits.get(AnimalState.FROZEN, 0),
        "n_fleeing": s.state_visits.get(AnimalState.FLEEING, 0),
        "wall_seconds": elapsed,
    }


def _trial_star(args):
    """Pool.imap helper: unpack (cfg_dict, mode, trial, duration_s)."""
    return _simulate_trial(*args)


def run_monte_carlo_parallel(
    config: Config,
    n_trials: int = 10,
    duration_hr: float = 2.0,
    modes: Tuple[str, ...] = ("control", "detection", "aware"),
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Same contract and identical numbers as run_monte_carlo, but distributes
    trials across n_jobs worker processes. n_jobs=None -> all cores; n_jobs<=1
    falls back to the serial harness (no pool overhead)."""
    import multiprocessing as _mp
    if n_jobs is None:
        n_jobs = _mp.cpu_count()
    if n_jobs <= 1 or _mp.cpu_count() <= 1:
        return run_monte_carlo(config, n_trials, duration_hr, modes, verbose=verbose)

    duration_s = duration_hr * 3600
    cfg_dict = dict(config.__dict__)
    for mode in modes:
        if mode not in WVCSimulation.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {WVCSimulation.VALID_MODES}")
    tasks = [(cfg_dict, mode, trial, duration_s)
             for mode in modes for trial in range(n_trials)]
    results = {m: [None] * n_trials for m in modes}
    if verbose:
        print(f"  [parallel] {len(tasks)} trials across {n_jobs} workers "
              f"({len(modes)} modes x {n_trials} trials x {duration_hr} h)")
    with _mp.Pool(processes=n_jobs) as pool:
        for res in pool.imap_unordered(_trial_star, tasks, chunksize=1):
            results[res["mode"]][res["trial"]] = res
    return results


def run_monte_carlo(
    config: Config,
    n_trials: int = 10,
    duration_hr: float = 2.0,
    modes: Tuple[str, ...] = ("control", "detection", "aware"),
    verbose: bool = True,
) -> dict:
    """Run independent trials under each mode and return raw per-trial results.

    Modes:
      'control'   — no system; pure baseline
      'detection' — sensors detect, but no driver-facing alert
      'aware'     — full system with corridor-wide DMS caution speed
    """
    duration_s = duration_hr * 3600
    results = {m: [] for m in modes}

    for mode in modes:
        if mode not in WVCSimulation.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {WVCSimulation.VALID_MODES}")
        if verbose:
            print(f"\n=== Mode: {mode.upper()}  ({n_trials} trials × {duration_hr} h) ===")
        for trial in range(n_trials):
            res = _simulate_trial(config.__dict__, mode, trial, duration_s)
            results[mode].append(res)
            if verbose:
                print(f"  trial {trial+1:2d}: "
                      f"N={res['animals']:3d}  "
                      f"det={res['detected']:3d} ({100*res['det_rate']:5.1f}%)  "
                      f"road={res['road_entries']:3d}  "
                      f"coll={res['collisions']:2d} ({100*res['col_rate']:5.2f}%)  "
                      f"aware={res['awareness_pct']:5.1f}%  "
                      f"frozen={res['frozen_on_road_s']:5.1f}s  "
                      f"[{res['wall_seconds']:.1f}s]")
    return results


# ============================================================
#  ANALYSIS / REPORTING
# ============================================================

def summarize(results: dict, modes=("control", "detection", "aware")) -> None:
    """Print mean ± std table. Reductions are reported relative to 'control' when present."""
    width_per_col = 22
    table_width = 24 + width_per_col * len(modes) + (width_per_col if "control" in modes and len(modes) > 1 else 0)
    print("\n" + "=" * table_width)
    header = f"{'Metric':<24}"
    for m in modes:
        header += f"{m.upper() + ' (mean ± σ)':>{width_per_col}}"
    if "control" in modes and len(modes) > 1:
        header += f"{'vs control':>{width_per_col}}"
    print(header)
    print("-" * table_width)

    metrics = [
        ("Animals / trial",       "animals",            "{:.1f}"),
        ("Detection rate",        "det_rate",           "{:.2%}"),
        ("Spawn->detect (s)",     "mean_latency_s",     "{:.2f}"),
        ("In-range->detect (s)",  "in_range_latency_s", "{:.3f}"),
        ("Road entries",          "road_entries",       "{:.1f}"),
        ("Collisions / trial",    "collisions",         "{:.2f}"),
        ("Collision rate",        "col_rate",           "{:.2%}"),
        ("Awareness % time",      "awareness_pct",      "{:.1f}"),
        ("Frozen-on-road (s)",    "frozen_on_road_s",   "{:.1f}"),
        ("Freeze events",         "n_frozen",           "{:.1f}"),
    ]
    means = {}
    for label, key, fmt in metrics:
        line = f"{label:<24}"
        for m in modes:
            vals = np.array([r[key] for r in results[m]])
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            means.setdefault(m, {})[key] = mean
            line += f"  {fmt.format(mean):>10} ± {fmt.format(std):>6}"
        # Reduction column (relative to control, for collision-related metrics only)
        if "control" in modes and len(modes) > 1 and key in ("collisions", "col_rate"):
            base = means["control"][key]
            # Report reduction for the LAST non-control mode (typically 'aware')
            non_control = [m for m in modes if m != "control"]
            target_m = non_control[-1] if non_control else None
            if target_m is not None and base > 0:
                red = 100 * (1 - means[target_m][key] / base)
                line += f"     {target_m.upper():>5}: {red:>+5.1f}%"
            else:
                line += f"     {'n/a':>11}"
        print(line)
    print("=" * table_width)

    # Pairwise Welch's t-tests on collisions
    if HAS_SCIPY and len(modes) >= 2:
        print("\nPairwise Welch's t-tests on collisions / trial:")
        for i, m1 in enumerate(modes):
            for m2 in modes[i+1:]:
                c1 = [r["collisions"] for r in results[m1]]
                c2 = [r["collisions"] for r in results[m2]]
                try:
                    t, p = sp_stats.ttest_ind(c1, c2, equal_var=False)
                    sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
                    print(f"  {m1:>10} vs {m2:<10}   t = {t:+6.3f}   p = {p:.4g}{sig}")
                except Exception as e:
                    print(f"  {m1:>10} vs {m2:<10}   (test failed: {e})")


def write_csv(results: dict, path: str) -> None:
    rows = []
    keys = None
    for mode, trials in results.items():
        for r in trials:
            # r already contains 'mode'; ensure dict key wins (use mode from dict key as source of truth)
            row = {**r, "mode": mode}
            rows.append(row)
            if keys is None:
                # Place 'mode' first for readability
                rest = [k for k in row.keys() if k != "mode"]
                keys = ["mode"] + rest
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-trial data written to: {path}")


# ============================================================
#  SENSITIVITY SWEEP
# ============================================================

# Presets map a short CLI keyword to (config_field_name, default_value_list).
# Users can also pass any Config field name directly with --sweep <field> --values v1 v2 ...
SWEEP_PRESETS = {
    "spacing":        ("radar_spacing",          [5, 10, 15, 20, 25, 30, 40]),
    "range":          ("radar_range",            [5, 8, 12, 15, 20, 25, 30]),
    "size":           ("size_scale",             [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    "detection_rate": ("detection_rate_per_sec", [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]),
    "rate":           ("animal_rate_per_hr",     [5, 10, 15, 30, 45, 60]),
    "caution":        ("caution_speed_kmh",      [20, 30, 45, 60, 80]),
    "cruise":         ("cruise_speed_kmh",       [60, 80, 100, 120, 140]),
    "reaction":       ("driver_reaction_s",      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
}


def run_sensitivity_sweep(
    base_config: Config,
    sweep_param: str,
    sweep_values: List[float],
    n_trials: int = 10,
    duration_hr: float = 2.0,
    modes: Tuple[str, ...] = ("control", "detection", "aware"),
    verbose: bool = True,
) -> dict:
    """Run a parameter sweep, holding all other config fields constant.

    Returns a nested dict: results[sweep_value][mode] -> list of per-trial result dicts.
    """
    if sweep_param not in base_config.__dict__:
        raise ValueError(f"Config has no field '{sweep_param}'. Valid fields include: "
                         + ", ".join(sorted(base_config.__dict__.keys())))
    sweep_results = {}
    n_points = len(sweep_values)
    overall_t0 = time.time()
    for i, val in enumerate(sweep_values):
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# Sweep point {i+1}/{n_points}:  {sweep_param} = {val}")
            print(f"{'#' * 70}")
        cfg_dict = {**base_config.__dict__, sweep_param: val}
        cfg = Config(**cfg_dict)
        mc_results = run_monte_carlo(cfg, n_trials=n_trials, duration_hr=duration_hr,
                                     modes=modes, verbose=False)
        sweep_results[val] = mc_results
        if verbose:
            for m in modes:
                trials = mc_results[m]
                c_mean = float(np.mean([r['collisions'] for r in trials]))
                c_std = float(np.std([r['collisions'] for r in trials], ddof=1)) if len(trials) > 1 else 0.0
                r_mean = float(np.mean([r['col_rate'] for r in trials]))
                print(f"  {m:>10}:  {c_mean:5.2f} ± {c_std:4.2f} collisions   {100*r_mean:5.2f}% rate")
    if verbose:
        print(f"\nSweep wall time: {time.time() - overall_t0:.1f} s "
              f"({n_points} points × {len(modes)} modes × {n_trials} trials × {duration_hr} h sim)")
    return sweep_results


def summarize_sweep(results: dict, sweep_param: str,
                    modes: Tuple[str, ...] = ("control", "detection", "aware")) -> None:
    """Print a per-point summary table with collision rates and reductions."""
    print(f"\n{'=' * 100}")
    print(f"SENSITIVITY SWEEP RESULTS:  {sweep_param}")
    print(f"{'=' * 100}")
    # Header
    line = f"{sweep_param:>14}"
    for m in modes:
        line += f"  {m.upper() + ' col-rate':>18}"
    if "control" in modes and "detection" in modes:
        line += f"  {'det-vs-ctrl':>12}"
    if "detection" in modes and "aware" in modes:
        line += f"  {'aware-vs-det':>12}"
    if "control" in modes and "aware" in modes:
        line += f"  {'aware-vs-ctrl':>13}"
    print(line)
    print("-" * 100)

    for val in sorted(results.keys()):
        line = f"{val:>14.4g}"
        means_rate = {}
        for m in modes:
            trials = results[val][m]
            rates = [100 * r['col_rate'] for r in trials]
            mean = float(np.mean(rates))
            std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
            means_rate[m] = mean
            line += f"  {mean:6.2f}% ± {std:5.2f}%"
        def red(a, b):
            if means_rate.get(a, 0) <= 0:
                return "    n/a   "
            r = 100 * (1 - means_rate[b] / means_rate[a])
            return f"  {r:+7.1f}%  "
        if "control" in modes and "detection" in modes:
            line += red("control", "detection")
        if "detection" in modes and "aware" in modes:
            line += red("detection", "aware")
        if "control" in modes and "aware" in modes:
            line += red("control", "aware")
        print(line)
    print("=" * 100)


def write_sweep_csv(results: dict, sweep_param: str, path: str) -> None:
    """Write per-trial sweep results to CSV with the swept parameter value as a column."""
    rows = []
    keys = None
    for val in sorted(results.keys()):
        for mode, trials in results[val].items():
            for r in trials:
                row = {sweep_param: val, **r, "mode": mode}
                rows.append(row)
                if keys is None:
                    other = [k for k in row.keys() if k not in (sweep_param, "mode")]
                    keys = [sweep_param, "mode"] + other
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSweep data written to: {path}")


def plot_sweep(results: dict, sweep_param: str, out_path: str,
               modes: Tuple[str, ...] = ("control", "detection", "aware")) -> None:
    """4-panel figure: collision rate, reduction by layer, detection rate, in-range latency."""
    if not HAS_PLT:
        print("matplotlib not available; skipping plots")
        return

    mode_colors = {"control": "#f85149", "detection": "#f9c513", "aware": "#56d364"}
    mode_labels = {"control": "CONTROL", "detection": "DETECTION", "aware": "AWARE"}
    values = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Sensitivity sweep — parameter: {sweep_param}",
                 fontsize=12, fontweight="bold")

    # --- Panel 1: collision rate vs swept parameter ---
    ax = axes[0, 0]
    for m in modes:
        means, stds = [], []
        for v in values:
            rates = [100 * r['col_rate'] for r in results[v][m]]
            means.append(float(np.mean(rates)))
            stds.append(float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0)
        ax.errorbar(values, means, yerr=stds, label=mode_labels[m],
                    color=mode_colors[m], marker='o', capsize=4, linewidth=2, alpha=0.85)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Collision rate by mode")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 2: reduction (aware vs control, aware vs detection, detection vs control) ---
    ax = axes[0, 1]

    def mean_collisions(v, m):
        return float(np.mean([r['collisions'] for r in results[v][m]]))

    if "control" in modes and "detection" in modes:
        d_vs_c = []
        for v in values:
            c, d = mean_collisions(v, "control"), mean_collisions(v, "detection")
            d_vs_c.append(100 * (1 - d / c) if c > 0 else 0.0)
        ax.plot(values, d_vs_c, label="DETECTION vs CONTROL",
                color="#f9c513", marker='s', linewidth=2)
    if "control" in modes and "aware" in modes:
        a_vs_c = []
        for v in values:
            c, a = mean_collisions(v, "control"), mean_collisions(v, "aware")
            a_vs_c.append(100 * (1 - a / c) if c > 0 else 0.0)
        ax.plot(values, a_vs_c, label="AWARE vs CONTROL",
                color="#56d364", marker='o', linewidth=2)
    if "detection" in modes and "aware" in modes:
        a_vs_d = []
        for v in values:
            d, a = mean_collisions(v, "detection"), mean_collisions(v, "aware")
            a_vs_d.append(100 * (1 - a / d) if d > 0 else 0.0)
        ax.plot(values, a_vs_d, label="AWARE vs DETECTION (boost effect)",
                color="#58a6ff", marker='^', linewidth=2, linestyle='--')
    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Collision reduction (%)")
    ax.set_title("Reduction by layer")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: detection rate vs swept parameter ---
    ax = axes[1, 0]
    for m in modes:
        if m == "control":
            continue
        means = []
        for v in values:
            rates = [100 * r['det_rate'] for r in results[v][m]]
            means.append(float(np.mean(rates)))
        ax.plot(values, means, label=mode_labels[m],
                color=mode_colors[m], marker='o', linewidth=2)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Detection rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    # --- Panel 4: in-range latency vs swept parameter ---
    ax = axes[1, 1]
    for m in modes:
        if m == "control":
            continue
        means, stds = [], []
        for v in values:
            lats = [r['in_range_latency_s'] for r in results[v][m] if r['in_range_latency_s'] > 0]
            if lats:
                means.append(float(np.mean(lats)))
                stds.append(float(np.std(lats, ddof=1)) if len(lats) > 1 else 0.0)
            else:
                means.append(0.0); stds.append(0.0)
        ax.errorbar(values, means, yerr=stds, label=mode_labels[m],
                    color=mode_colors[m], marker='o', capsize=4, linewidth=2)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("In-range detection latency (s)")
    ax.set_title("Sensor latency (true response time)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Figures written to: {out_path}")


# ============================================================
#  OPTIONAL PLOTTING (single-point Monte Carlo)
# ============================================================

def plot_results(results: dict, out_path: str = "wvc_figures.png") -> None:
    if not HAS_PLT:
        print("matplotlib not available; skipping plots")
        return

    # Stable order and consistent color/label per mode
    mode_order = [m for m in ("control", "detection", "aware") if m in results]
    mode_colors = {"control": "#f85149", "detection": "#f9c513", "aware": "#56d364"}
    mode_labels = {"control": "CONTROL", "detection": "DETECTION", "aware": "AWARE"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("WVC Monte Carlo — three-way comparison: control, detection-only, full system",
                 fontsize=11, fontweight="bold")

    # ----- Box plot: collisions per trial -----
    ax = axes[0, 0]
    data = [[r["collisions"] for r in results[m]] for m in mode_order]
    positions = list(range(1, len(mode_order) + 1))
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch, m in zip(bp["boxes"], mode_order):
        patch.set_facecolor(mode_colors[m]); patch.set_alpha(0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels([mode_labels[m] for m in mode_order])
    ax.set_ylabel("Collisions per trial")
    ax.set_title("Collision count by mode")
    ax.grid(axis="y", alpha=0.3)

    # ----- Per-trial collision rate, grouped bars -----
    ax = axes[0, 1]
    n_trials = max(len(results[m]) for m in mode_order)
    x = np.arange(n_trials)
    width = 0.8 / len(mode_order)
    for i, m in enumerate(mode_order):
        rates = [100 * r["col_rate"] for r in results[m]]
        offset = (i - (len(mode_order) - 1) / 2) * width
        ax.bar(x + offset, rates, width=width, label=mode_labels[m],
               color=mode_colors[m], alpha=0.75)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Per-trial collision rate")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ----- Detection latency distributions for modes with detection -----
    ax = axes[1, 0]
    plotted_any = False
    for m in mode_order:
        if m == "control":
            continue  # no detection in control
        latencies = [r["in_range_latency_s"] for r in results[m]]
        if latencies:
            ax.hist(latencies, bins=12, alpha=0.55, edgecolor="black",
                    label=f"{mode_labels[m]} (n={len(latencies)})",
                    color=mode_colors[m])
            plotted_any = True
    if plotted_any:
        ax.set_xlabel("Mean in-range detection latency (s)")
        ax.set_ylabel("Frequency")
        ax.set_title("True sensor latency (in-range → detected)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No detection in control mode\n(by definition)",
                ha="center", va="center", transform=ax.transAxes, color="#7d8590")
        ax.set_xticks([]); ax.set_yticks([])

    # ----- Animal behavior state composition -----
    ax = axes[1, 1]
    state_keys = ["n_foraging", "n_approaching", "n_hesitating",
                  "n_crossing", "n_frozen", "n_fleeing"]
    state_labels = ["Forage", "Approach", "Hesitate", "Cross", "Frozen", "Flee"]
    state_x = np.arange(len(state_keys))
    bw = 0.8 / len(mode_order)
    for i, m in enumerate(mode_order):
        means = [np.mean([r[k] for r in results[m]]) for k in state_keys]
        off = (i - (len(mode_order) - 1) / 2) * bw
        ax.bar(state_x + off, means, width=bw, label=mode_labels[m],
               color=mode_colors[m], alpha=0.75)
    ax.set_xticks(state_x); ax.set_xticklabels(state_labels, rotation=20)
    ax.set_ylabel("Mean state visits per trial")
    ax.set_title("Animal behavior composition")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Figures written to: {out_path}")



# ============================================================
#  DATA REGENERATION  (from generate_data.py)
# ============================================================

HEADLINE_TRIALS = 20
HEADLINE_HOURS = 4.0

SWEEP_TRIALS = 15
SWEEP_HOURS = 2.0

SWEEPS = [
    ("spacing",            "radar_spacing",             [5, 10, 15, 20, 25, 30, 40]),
    ("radar_range",        "radar_range",               [8, 10, 12, 15, 20]),
    ("size",               "size_scale",                [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    ("detection_rate",     "detection_rate_per_sec",    [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]),
    ("awareness_beta",     "boost_factor",              [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
    ("awareness_tau",      "awareness_persist_s",       [10, 30, 60, 120]),
    ("hesitate_dwell",     "hesitate_dwell_max",        [1.0, 2.0, 3.0, 5.0, 8.0]),
    ("freeze_mid_cross",   "p_freeze_during_cross_base",[0.0, 0.05, 0.15, 0.30, 0.50]),
    ("traffic_volume",     "vehicles_per_dir",          [1, 2, 4, 8, 16]),
    # Animal arrival density: tests whether the LoRa awareness boost activates as
    # λτ approaches 1 (i.e. when subsequent animals routinely arrive inside the
    # boost window). With τ=30 s default these values span λτ from 0.042 to 2.0.
    ("animal_density",     "animal_rate_per_hr",        [5, 15, 30, 60, 120, 240]),
    ("driver_reaction",    "driver_reaction_s",         [0.5, 1.0, 1.5, 2.5, 4.0]),
    ("cruise_speed",       "cruise_speed_kmh",          [60, 80, 100, 120]),
    ("caution_speed",      "caution_speed_kmh",         [30, 45, 60, 80, 100]),
]

# Number of worker processes (overridden by --jobs in main()). 1 = serial.
JOBS = 1


def _mc(cfg, n_trials, duration_hr, modes, verbose=False):
    """Dispatch to the parallel or serial Monte Carlo harness based on JOBS."""
    if JOBS and JOBS > 1:
        return run_monte_carlo_parallel(
            cfg, n_trials=n_trials, duration_hr=duration_hr,
            modes=modes, n_jobs=JOBS, verbose=verbose)
    return run_monte_carlo(
        cfg, n_trials=n_trials, duration_hr=duration_hr,
        modes=modes, verbose=verbose)


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
    cfg = Config()
    t0 = time.time()
    results = _mc(cfg, HEADLINE_TRIALS, HEADLINE_HOURS,
                  ("control", "detection", "aware"))
    elapsed = time.time() - t0
    print(f"[{ts()}] Headline done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    write_csv(results, "wvc_results.csv")
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
    t0 = time.time()

    # Control ignores every swept parameter (no sensors), so its trials are
    # identical at every point. Compute the control arm ONCE and reuse it,
    # instead of redundantly re-running it per sweep point.
    control_trials = _mc(Config(), SWEEP_TRIALS, SWEEP_HOURS,
                         ("control",))["control"]

    # Run point-by-point (detection + aware only) so we can print progress
    sweep_results = {}
    for i, val in enumerate(values):
        cfg_dict = Config().__dict__.copy()
        cfg_dict[param] = val
        cfg_local = Config(**cfg_dict)
        pt0 = time.time()
        mc = _mc(cfg_local, SWEEP_TRIALS, SWEEP_HOURS, ("detection", "aware"))
        mc["control"] = control_trials
        sweep_results[val] = mc
        pt_el = time.time() - pt0
        # Quick progress summary
        det = sweep_results[val]["detection"]
        rates = [t["col_rate"] * 100 for t in det]
        print(f"[{ts()}]   point {i+1}/{len(values)}: {param}={val:>6}: det rate = {statistics.mean(rates):5.2f}% (elapsed {pt_el:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{ts()}] {label} sweep done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    write_sweep_csv(sweep_results, param, out_csv)
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
#  STATISTICS  (from compute_stats.py)
# ============================================================

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


# ============================================================
#  FIGURES  (from generate_figures.py)
# ============================================================

# Palette / ordering constants (consistent with the Figure 1 diagram).
# Color-blind-friendly palette consistent with the Figure 1 architecture diagram.
# Control = neutral gray, Detection = teal-green, Aware = warm amber.
COLORS = {"control": "#5F5E5A", "detection": "#0F6E56", "aware": "#BA7517"}
MODE_ORDER = ["control", "detection", "aware"]
MODE_LABELS = {"control": "Control", "detection": "Detection", "aware": "Aware"}

# Offset dash patterns: Detection's dashes fall in Aware's gaps and vice versa,
# so when the two curves overlap pixel-for-pixel (typical of the safety-channel
# bottleneck) both colours remain visible as an alternating green/orange interleave.
LINESTYLES = {
    "control":   "-",
    "detection": (0, (4, 4)),    # dash starts at offset 0
    "aware":     (4, (4, 4)),    # dash starts at offset 4 -- exactly in detection's gap
}


def _setup_figure_style():
    """Apply the publication style (seaborn whitegrid + serif rcParams)."""
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
    ax.set_ylim(0, 100)
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


def _plot_mean_with_sem_band(ax, x, mean, sem, color, label, marker="o", lw=1.6, ms=5, alpha_band=0.18, linestyle="-"):
    """Plot a mean line with a shaded ±1 SEM band, markers at each x."""
    x = np.asarray(x)
    mean = np.asarray(mean)
    sem = np.asarray(sem)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha_band, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=lw, marker=marker, markersize=ms, label=label, linestyle=linestyle)


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
                x_log=False, x_ticks=None, dpi=300, show_ci_band=True):
    """Generic 4-panel sweep figure: collision rate, detection rate, latency, throughput.

    If show_ci_band=True and a companion wvc_stats_<sweep>.csv exists (produced by
    compute_stats.py), panel (b) overlays 95% bootstrap CIs as faint shaded bands
    behind the reduction-vs-control lines.
    """
    df = pd.read_csv(csv_path)
    summary = _sweep_summary(df, sweep_col)

    # Locate companion stats CSV (e.g. wvc_results_spacing.csv -> wvc_stats_spacing.csv)
    stats_path = csv_path.replace("wvc_results_", "wvc_stats_") if "wvc_results_" in csv_path else None
    stats_df = pd.read_csv(stats_path) if (stats_path and os.path.exists(stats_path)) else None

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    # Trial count in the title is read from the data, not a hardcoded literal.
    _n = int(summary["n"].max()) if ("n" in summary.columns and len(summary)) else None
    _title = title_prefix.replace("40 trials", f"{_n} trials") if _n else title_prefix
    fig.suptitle(_title, fontsize=11, y=0.995)

    # ---- (a) Collision rate vs sweep ----
    ax = axes[0, 0]
    for mode in MODE_ORDER:                 # control, detection, aware (legend reads naturally)
        d = summary[summary["mode"] == mode].sort_values(sweep_col)
        _plot_mean_with_sem_band(ax, d[sweep_col].values, d["mean_rate"].values,
                                  d["sem_rate"].values, color=COLORS[mode],
                                  label=MODE_LABELS[mode], linestyle=LINESTYLES[mode])
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
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction_det = 100 * (pivot_means["control"] - pivot_means["detection"]) / pivot_means["control"]
        reduction_aware = 100 * (pivot_means["control"] - pivot_means["aware"]) / pivot_means["control"]
    # Guard: at any point where control collisions are zero the reduction is undefined
    # (0/0). Mask those so the panel still renders instead of crashing on set_ylim.
    reduction_det = reduction_det.replace([np.inf, -np.inf], np.nan)
    reduction_aware = reduction_aware.replace([np.inf, -np.inf], np.nan)

    # CI bands from companion stats CSV (drawn first, lower zorder)
    if show_ci_band and stats_df is not None:
        for comp_name, color_key in [("control_vs_detection", "detection"),
                                     ("control_vs_aware",     "aware")]:
            sel = stats_df[stats_df["comparison"] == comp_name].sort_values(sweep_col)
            if sel.empty:
                continue
            xs = sel[sweep_col].values
            lo = sel["reduction_ci_low_pct"].values
            hi = sel["reduction_ci_high_pct"].values
            mask = np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                ax.fill_between(xs[mask], lo[mask], hi[mask],
                                alpha=0.15, color=COLORS[color_key],
                                linewidth=0, zorder=1)

    ax.plot(reduction_det.index, reduction_det.values, "o", color=COLORS["detection"],
            label="Control → Detection", markersize=5, linewidth=1.6, zorder=4,
            linestyle=LINESTYLES["detection"])
    ax.plot(reduction_aware.index, reduction_aware.values, "s", color=COLORS["aware"],
            label="Control → Aware", markersize=5, linewidth=1.6, zorder=3,
            linestyle=LINESTYLES["aware"])
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
    _finite = np.concatenate([reduction_det.values, reduction_aware.values])
    _finite = _finite[np.isfinite(_finite)]
    if _finite.size:
        ymax_data = float(_finite.max())
        ymin_data = min(float(_finite.min()), 0.0)
        y_range = (ymax_data - ymin_data) or 1.0
        # Reserve top 12% of chart for the significance row
        chart_top = ymax_data + y_range * 0.20
        sig_y = ymax_data + y_range * 0.10
        ax.set_ylim(top=chart_top, bottom=min(0, ymin_data - y_range * 0.05))
    else:
        # No finite reductions (e.g. zero control collisions at tiny N): autoscale.
        y_range = 1.0
        sig_y = ax.get_ylim()[1]
        # An all-NaN series registers no x-data, which breaks a log x-axis tick
        # locator; give it positive bounds from the (positive) sweep values.
        idx = np.asarray(reduction_det.index, dtype=float)
        if x_log and idx.size:
            ax.set_xlim(idx.min() * 0.9, idx.max() * 1.1)
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
                                  label=MODE_LABELS[mode], linestyle=LINESTYLES[mode])
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
                                  label=MODE_LABELS[mode], linestyle=LINESTYLES[mode])
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
        title_prefix="Radar spacing sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[5, 10, 15, 20, 25, 30, 40],
        dpi=dpi,
    )


def make_figure_4(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_size.csv",
        sweep_col="size_scale",
        x_label="Animal size scaling factor σ_scale",
        title_prefix="Animal size sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        dpi=dpi,
    )


def make_figure_5(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_detection_rate.csv",
        sweep_col="detection_rate_per_sec",
        x_label="Baseline detection rate κ (s⁻¹)",
        title_prefix="Sensor sensitivity sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_log=True,
        x_ticks=[0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
        dpi=dpi,
    )


def make_figure_6(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_awareness_beta.csv",
        sweep_col="boost_factor",
        x_label="Neighbor sensitivity boost β (×)",
        title_prefix="Awareness boost sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        show_ci_band=False,  # bit-identical across cells under CRN seeds
        dpi=dpi,
    )


def make_figure_7(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_awareness_tau.csv",
        sweep_col="awareness_persist_s",
        x_label="Awareness persistence τ (s)",
        title_prefix="Awareness persistence sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[10, 30, 60, 120],
        dpi=dpi,
    )


def make_figure_8(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_hesitate_dwell.csv",
        sweep_col="hesitate_dwell_max",
        x_label="Hesitation dwell maximum (s)",
        title_prefix="Hesitation dwell sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[1.0, 2.0, 3.0, 5.0, 8.0],
        dpi=dpi,
    )


def make_figure_9(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_freeze_mid_cross.csv",
        sweep_col="p_freeze_during_cross_base",
        x_label="Mid-crossing freeze probability base",
        title_prefix="Mid-crossing freeze sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[0.0, 0.05, 0.15, 0.30, 0.50],
        show_ci_band=False,  # bit-identical across cells under CRN seeds
        dpi=dpi,
    )


def make_figure_10(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_traffic_volume.csv",
        sweep_col="vehicles_per_dir",
        x_label="Traffic volume (vehicles per direction)",
        title_prefix="Traffic volume sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_log=True,
        x_ticks=[1, 2, 4, 8, 16],
        dpi=dpi,
    )


def make_figure_11(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_driver_reaction.csv",
        sweep_col="driver_reaction_s",
        x_label="Driver reaction time (s)",
        title_prefix="Driver reaction sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[0.5, 1.0, 1.5, 2.5, 4.0],
        dpi=dpi,
    )


def make_figure_12(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_cruise_speed.csv",
        sweep_col="cruise_speed_kmh",
        x_label="Cruise speed (km/h)",
        title_prefix="Cruise speed sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[60, 80, 100, 120],
        dpi=dpi,
    )


def make_figure_13(out_path, dpi=300):
    _sweep_plot(
        csv_path="wvc_results_caution_speed.csv",
        sweep_col="caution_speed_kmh",
        x_label="Caution speed (km/h, compliance proxy)",
        title_prefix="Caution speed / compliance sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[30, 45, 60, 80, 100],
        dpi=dpi,
    )


def make_figure_14_forest(out_path, dpi=300):
    """Forest plot of headline rate ratios with 95% CIs (Poisson/NB GLM).

    Reads wvc_stats_headline.csv (from compute_stats.py) and renders the three
    pairwise comparisons (control vs detection, control vs aware, detection vs
    aware) as rate ratios with horizontal error bars on a log x-axis.
    """
    stats_path = "wvc_stats_headline.csv"
    if not os.path.exists(stats_path):
        print(f"  [skip] {stats_path} not found — run compute_stats.py first")
        return

    df = pd.read_csv(stats_path)

    # Display order: control->det, control->aware, det->aware (top to bottom)
    desired = ["control_vs_detection", "control_vs_aware", "detection_vs_aware"]
    df = df.set_index("comparison").reindex(desired).reset_index()
    labels = {
        "control_vs_detection": "Control → Detection",
        "control_vs_aware":     "Control → Aware",
        "detection_vs_aware":   "Detection → Aware",
    }
    row_color = {
        "control_vs_detection": COLORS["detection"],
        "control_vs_aware":     COLORS["aware"],
        "detection_vs_aware":   "#7c5895",
    }

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    fig.suptitle("Headline rate ratios with 95 % CIs (Poisson/NB GLM)",
                 fontsize=11, y=0.99)

    ys = np.arange(len(df))[::-1]  # top row = first comparison
    for y, (_, r) in zip(ys, df.iterrows()):
        c = row_color[r["comparison"]]
        rr = r["rate_ratio"]
        lo = r["rr_ci_low"]
        hi = r["rr_ci_high"]
        if np.isfinite(rr):
            ax.errorbar([rr], [y], xerr=[[rr - lo], [hi - rr]],
                        fmt="o", color=c, ecolor=c, capsize=4,
                        markersize=7, linewidth=1.4)
            # annotation on the right: "RR = X.XX [lo, hi], p = ..."
            p_str = ("p < 0.001" if r["glm_p"] < 0.001
                     else f"p = {r['glm_p']:.3f}")
            label = f"RR = {rr:.2f}  [{lo:.2f}, {hi:.2f}]   {p_str}"
            ax.text(1.02, y, label, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8, color="#1f2937")

    ax.axvline(1.0, color="#888780", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Rate ratio  (treatment / reference)")
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[c] for c in df["comparison"]])
    ax.set_ylim(-0.6, len(df) - 0.4)

    # Sensible x-range covering all CIs
    finite_lo = df["rr_ci_low"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_hi = df["rr_ci_high"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_lo) and len(finite_hi):
        ax.set_xlim(min(0.3, finite_lo.min() * 0.85),
                    max(3.0, finite_hi.max() * 1.15))

    # Clean tick labels (decimal, not scientific) for readability on log axis
    import matplotlib.ticker as mtick
    tick_locs = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    xlim = ax.get_xlim()
    visible_ticks = [t for t in tick_locs if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(visible_ticks)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(mtick.NullLocator())

    # Reserve right side for the inline RR/p annotations
    plt.subplots_adjust(left=0.18, right=0.55, top=0.88, bottom=0.20)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_figure_15_radar_range(out_path, dpi=300):
    """Radar detection-range sweep (baseline R_det varied 8..20 m).

    Characterises the hardware design space. Under the β-as-range-multiplier
    model, the effective detection radius is R_det × boost during awareness;
    this sweep maps how the system performs as the baseline range varies,
    holding β at its default 1.5×.
    """
    _sweep_plot(
        csv_path="wvc_results_radar_range.csv",
        sweep_col="radar_range",
        x_label="Baseline radar detection range R_det (m)",
        title_prefix="Radar range sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_ticks=[8, 10, 12, 15, 20],
        dpi=dpi,
    )


def make_figure_16_animal_density(out_path, dpi=300):
    """Animal arrival density sweep (λ varied 5..240 animals/hr).

    Tests whether the LoRa awareness boost mechanism activates as the
    expected number of subsequent animals per boost window (λτ) approaches
    or exceeds 1. At default τ=30s the swept range covers λτ from 0.042
    (extreme low density, where boost has nothing to act on) to 2.0
    (multiple animals per boost window).
    """
    _sweep_plot(
        csv_path="wvc_results_animal_density.csv",
        sweep_col="animal_rate_per_hr",
        x_label="Animal arrival rate λ (animals/hr)",
        title_prefix="Animal density sweep (40 trials × 2 h per point)",
        out_path=out_path,
        x_log=True,
        x_ticks=[5, 15, 30, 60, 120, 240],
        dpi=dpi,
    )


# ============================================================
#  ABLATION (reviewer R1-04) — magnetometer vehicle model
# ============================================================

def _abl_stat(rows, key):
    a = np.array([r[key] for r in rows], float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

def _abl_welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2 or len(b) < 2: return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/len(a) + vb/len(b))
    return float("nan") if se == 0 else (a.mean()-b.mean())/se

def main_ablation(n=10, hr=2.0):
    print(f"\n{'='*78}\nPART 1 - VEHICLE-MODEL ABLATION   ({n} trials x {hr} h, deployment geometry)\n{'='*78}")
    cfgP = Config(vehicle_model="perfect")
    cfgM = Config(vehicle_model="magnetometer")
    ctrl = _mc(cfgP, n, hr, ("control",))["control"]
    P = _mc(cfgP, n, hr, ("detection","aware"))
    Mr = _mc(cfgM, n, hr, ("detection","aware"))

    groups = [("control",          ctrl),
              ("detection PERFECT", P["detection"]),
              ("detection MAGNETO", Mr["detection"]),
              ("aware     PERFECT", P["aware"]),
              ("aware     MAGNETO", Mr["aware"])]

    print(f"\n{'group':20s} {'coll/trial':>11} {'col_rate%':>10} {'entries':>8} "
          f"{'present-known%':>14} {'coverage%':>10}")
    for name, rows in groups:
        c_m,c_s = _abl_stat(rows,"collisions")
        cr_m,_  = _abl_stat(rows,"col_rate")
        e_m,_   = _abl_stat(rows,"road_entries")
        pk_m,_  = _abl_stat(rows,"veh_present_known_pct")
        cov_m,_ = _abl_stat(rows,"veh_coverage_pct")
        print(f"{name:20s} {c_m:5.2f}\u00b1{c_s:4.2f} {100*cr_m:9.2f} {e_m:8.1f} "
              f"{pk_m:13.1f} {cov_m:9.1f}")

    print(f"\n  Key comparison - AWARE: perfect vs magnetometer vehicle knowledge")
    for key,lab in [("collisions","collisions/trial"),("col_rate","collision rate")]:
        ap=[r[key] for r in P["aware"]]; am=[r[key] for r in Mr["aware"]]
        t=_abl_welch(ap,am)
        print(f"    {lab:18s}: perfect={np.mean(ap):.4f}  magneto={np.mean(am):.4f}  "
              f"Welch t={t:+.2f}  -> {'indistinguishable' if abs(t)<2 else 'DIFFERENT'}")

def geometry_sweep(n=10, hr=2.0):
    print(f"\n{'='*78}\nPART 2 - GEOMETRY SWEEP (aware + magnetometer, {n} trials x {hr} h)\n{'='*78}")
    print(f"{'offset(m)':>9} {'present-known%':>14} {'coverage%':>10} "
          f"{'coll/trial':>11} {'col_rate%':>10}")
    base = _mc(Config(vehicle_model="perfect"), n, hr, ("aware",))["aware"]
    cb,_ = _abl_stat(base,"collisions"); crb,_ = _abl_stat(base,"col_rate")
    print(f"{'perfect':>9} {100.0:13.1f} {100.0:9.1f} {cb:5.2f}{'':6} {100*crb:9.2f}")
    for off in (0.75, 1.5, 2.5, 4.0):
        cfg = Config(vehicle_model="magnetometer", mag_sensor_offset=off)
        rows = _mc(cfg, n, hr, ("aware",))["aware"]
        pk,_  = _abl_stat(rows,"veh_present_known_pct")
        cov,_ = _abl_stat(rows,"veh_coverage_pct")
        c,cs  = _abl_stat(rows,"collisions")
        cr,_  = _abl_stat(rows,"col_rate")
        print(f"{off:9.2f} {pk:13.1f} {cov:9.1f} {c:5.2f}\u00b1{cs:4.2f} {100*cr:9.2f}")


# ============================================================
#  STAGE RUNNERS (shared by standalone subcommands and pipeline)
# ============================================================

def _stage_data(jobs=1, fresh=False, skip_sweeps=False,
                headline_trials=None, headline_hours=None,
                sweep_trials=None, sweep_hours=None):
    """Run the headline experiment + sweeps, write CSVs and the summary.
    Mirrors the old generate_data.py main()."""
    global JOBS, HEADLINE_TRIALS, HEADLINE_HOURS, SWEEP_TRIALS, SWEEP_HOURS, CHECKPOINT_PATH
    JOBS = jobs
    if headline_trials is not None: HEADLINE_TRIALS = headline_trials
    if headline_hours is not None: HEADLINE_HOURS = headline_hours
    if sweep_trials   is not None: SWEEP_TRIALS   = sweep_trials
    if sweep_hours    is not None: SWEEP_HOURS    = sweep_hours
    CHECKPOINT_PATH = Path(".regeneration_checkpoint.json")

    print(f"[{ts()}] jobs={JOBS}  headline={HEADLINE_TRIALS}x{HEADLINE_HOURS}h  "
          f"sweeps={SWEEP_TRIALS}x{SWEEP_HOURS}h")

    if fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"[{ts()}] Deleted checkpoint, starting fresh")

    state = load_checkpoint()
    overall_t0 = time.time()
    print(f"[{ts()}] Working directory: {Path('.').resolve()}")
    print(f"[{ts()}] Model: corrected (Fix A)")
    print()

    headline = run_headline(state)
    sweeps = {}
    if not skip_sweeps:
        for label, param, values in SWEEPS:
            sweeps[label] = run_sweep(label, param, values, state)
    else:
        print(f"[{ts()}] Skipping sweeps (--skip-sweeps)")

    write_summary(headline, sweeps)

    elapsed = time.time() - overall_t0
    print()
    print(f"[{ts()}] === DATA STAGE DONE in {elapsed:.1f}s ({elapsed/60:.1f} min) ===")
    print()
    print("Files produced:")
    for p in ["wvc_results.csv", "wvc_results_spacing.csv", "wvc_results_size.csv",
              "wvc_results_detection_rate.csv", "regeneration_summary.json",
              "regeneration_summary.txt"]:
        if Path(p).exists():
            print(f"  {p}  ({Path(p).stat().st_size/1024:.1f} KB)")
        else:
            print(f"  {p}  (missing)")


def _stage_stats(csv_dir=".", bootstrap_n=10000, seed=42):
    """Compute the companion statistics CSVs. Mirrors the old compute_stats.py main().
    Imports statsmodels/scipy/pandas lazily."""
    global sm, smf, stats, pd
    import pandas as pd                       # noqa: F401
    from scipy import stats                   # noqa: F401
    import statsmodels.api as sm              # noqa: F401
    import statsmodels.formula.api as smf     # noqa: F401

    print(f"Reading from: {os.path.abspath(csv_dir)}")
    print(f"Bootstrap:    {bootstrap_n} replicates, seed {seed}")
    print(f"Dispersion threshold for NB switch: {DISPERSION_THRESHOLD}")
    print()
    print("--- Headline ---")
    process_headline(csv_dir, bootstrap_n, seed)
    print()
    print("--- Sweeps ---")
    for label, sweep_col in SWEEP_PARAM_COLUMNS.items():
        print(f"  sweep: {label}")
        process_sweep(csv_dir, label, sweep_col, bootstrap_n, seed)
    print()
    print("Done.")


def _stage_figures(only=None, dpi=300):
    """Render the publication figures. Mirrors the old generate_figures.py main().
    Imports matplotlib/seaborn/pandas/scipy lazily."""
    global plt, sns, pd, stats
    import matplotlib.pyplot as plt           # noqa: F811  (backend already set to Agg at import)
    import seaborn as sns                      # noqa: F401
    import pandas as pd                        # noqa: F811
    from scipy import stats                    # noqa: F811
    _setup_figure_style()

    ALL_FIGS = ["3", "4", "5", "6",
                "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
    if only:
        only = [f.upper() for f in only]
        unknown = [f for f in only if f not in ALL_FIGS]
        if unknown:
            raise SystemExit(f"unknown figure identifier(s): {unknown}. Valid: {ALL_FIGS}")
    else:
        only = ALL_FIGS

    print(f"Generating figures: {only} at {dpi} DPI")
    if "3"  in only: make_figure_2("figure_3_headline.png", dpi=dpi)
    if "4"  in only: make_figure_14_forest("figure_4_forest_headline.png", dpi=dpi)
    if "5"  in only: make_figure_10("figure_5_traffic_volume.png", dpi=dpi)
    if "6"  in only: make_figure_16_animal_density("figure_6_animal_density.png", dpi=dpi)
    if "S1" in only: make_figure_3("figure_S1_spacing.png", dpi=dpi)
    if "S2" in only: make_figure_4("figure_S2_size.png", dpi=dpi)
    if "S3" in only: make_figure_5("figure_S3_kappa.png", dpi=dpi)
    if "S4" in only: make_figure_6("figure_S4_awareness_beta.png", dpi=dpi)
    if "S5" in only: make_figure_7("figure_S5_awareness_tau.png", dpi=dpi)
    if "S6" in only: make_figure_8("figure_S6_hesitate_dwell.png", dpi=dpi)
    if "S7" in only: make_figure_9("figure_S7_freeze_mid_cross.png", dpi=dpi)
    if "S8" in only: make_figure_11("figure_S8_driver_reaction.png", dpi=dpi)
    if "S9" in only: make_figure_12("figure_S9_cruise_speed.png", dpi=dpi)
    if "S10" in only: make_figure_13("figure_S10_caution_speed.png", dpi=dpi)
    if "S11" in only: make_figure_15_radar_range("figure_S11_radar_range.png", dpi=dpi)
    print("done.")


# ============================================================
#  SUBCOMMAND HANDLERS
# ============================================================

def cmd_simulate(args):
    """One Monte-Carlo experiment: single point (default) or a parameter sweep."""
    global JOBS
    JOBS = args.jobs
    cfg = Config(
        animal_rate_per_hr=args.rate,
        cruise_speed_kmh=args.cruise,
        caution_speed_kmh=args.caution,
        driver_reaction_s=args.reaction,
        radar_spacing=args.radar_spacing,
        radar_range=args.radar_range,
        size_scale=args.size_scale,
        vehicle_model=args.vehicle_model,
    )

    print("=" * 60)
    print(" WVC Monte Carlo simulation")
    print("=" * 60)
    print(f"  Road length:        {cfg.road_length} m")
    print(f"  Vehicles / dir:     {cfg.vehicles_per_dir} @ {cfg.cruise_speed_kmh} km/h")
    print(f"  Caution speed:      {cfg.caution_speed_kmh} km/h (when awareness active)")
    print(f"  Animal arrival:     lambda = {cfg.animal_rate_per_hr} /hr (Poisson)")
    print(f"  Radar topology:     spacing {cfg.radar_spacing} m, range {cfg.radar_range} m, alternating sides")
    print(f"  Awareness corridor: {cfg.awareness_range_m} m, persist {cfg.awareness_persist_s} s")
    print(f"  Driver reaction:    {cfg.driver_reaction_s} s")
    print(f"  Vehicle model:      {cfg.vehicle_model}")
    print(f"  Time step:          {cfg.dt} s")
    print(f"  Modes to run:       {', '.join(args.modes)}")
    print(f"  Worker processes:   {JOBS}")

    modes = tuple(args.modes)

    if args.sweep:
        if args.sweep in SWEEP_PRESETS:
            sweep_param, default_values = SWEEP_PRESETS[args.sweep]
        else:
            sweep_param, default_values = args.sweep, None
        values = args.values if args.values is not None else default_values
        if values is None:
            print(f"\nERROR: no preset for sweep '{args.sweep}' and no --values provided.")
            print(f"       Either use a preset name ({', '.join(SWEEP_PRESETS.keys())})")
            print(f"       or pass explicit values, e.g. --sweep {args.sweep} --values 1 2 4 8")
            return
        print(f"  Sweep:              {args.sweep} ({sweep_param}) over {values}")
        print(f"  Total trials:       {len(values)} x {len(modes)} x {args.trials} = "
              f"{len(values) * len(modes) * args.trials}\n")
        sweep_results = run_sensitivity_sweep(
            cfg, sweep_param, values,
            n_trials=args.trials, duration_hr=args.hours, modes=modes)
        summarize_sweep(sweep_results, sweep_param, modes=modes)
        sweep_csv = args.csv.replace(".csv", f"_sweep_{args.sweep}.csv")
        write_sweep_csv(sweep_results, sweep_param, sweep_csv)
        if args.plot:
            plot_sweep(sweep_results, sweep_param,
                       args.csv.replace(".csv", f"_sweep_{args.sweep}.png"), modes=modes)
        return

    results = _mc(cfg, args.trials, args.hours, modes, verbose=True)
    summarize(results, modes=modes)
    write_csv(results, args.csv)
    if args.plot:
        plot_results(results, out_path=args.csv.replace(".csv", ".png"))


def cmd_data(args):
    _stage_data(jobs=args.jobs, fresh=args.fresh, skip_sweeps=args.skip_sweeps,
                headline_trials=args.headline_trials, headline_hours=args.headline_hours,
                sweep_trials=args.sweep_trials, sweep_hours=args.sweep_hours)


def cmd_stats(args):
    _stage_stats(csv_dir=".", bootstrap_n=args.bootstrap_n, seed=args.seed)


def cmd_figures(args):
    _stage_figures(only=args.only, dpi=args.dpi)


def cmd_ablation(args):
    global JOBS
    JOBS = args.jobs
    if args.part in ("all", "1"):
        main_ablation(args.trials, args.hours)
    if args.part in ("all", "2"):
        geometry_sweep(args.trials, args.hours)


def cmd_pipeline(args):
    stages = args.stages or ["data", "stats", "figures"]
    print(f"[pipeline] stages: {' -> '.join(stages)}  (jobs={args.jobs})\n")
    t0 = time.time()
    if "data" in stages:
        print("\n########## STAGE 1/3: DATA ##########")
        _stage_data(jobs=args.jobs, fresh=args.fresh, skip_sweeps=args.skip_sweeps,
                    headline_trials=args.headline_trials, headline_hours=args.headline_hours,
                    sweep_trials=args.sweep_trials, sweep_hours=args.sweep_hours)
    if "stats" in stages:
        print("\n########## STAGE 2/3: STATS ##########")
        _stage_stats(csv_dir=".", bootstrap_n=args.bootstrap_n, seed=args.seed)
    if "figures" in stages:
        print("\n########## STAGE 3/3: FIGURES ##########")
        _stage_figures(only=args.only, dpi=args.dpi)
    print(f"\n[pipeline] all requested stages done in {time.time()-t0:.1f}s")


# ============================================================
#  CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="wvc.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=".",
                   help="directory for all inputs/outputs (default: current dir). "
                        "Applied before any command runs.")
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)

    # ---- simulate ----
    s = sub.add_parser("simulate", help="one Monte-Carlo experiment (single point or sweep)",
                       description="Run a single Monte-Carlo experiment. Without --sweep this is "
                                   "one operating point across the chosen modes; with --sweep it "
                                   "varies one parameter and reports all modes at each value.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--trials", type=int, default=10, help="trials per mode (default 10)")
    s.add_argument("--hours", type=float, default=2.0, help="hours per trial (default 2.0)")
    s.add_argument("--modes", nargs="+", default=["control", "detection", "aware"],
                   choices=["control", "detection", "aware"],
                   help="modes to compare (default: all three)")
    s.add_argument("--jobs", type=int, default=1,
                   help="worker processes (default 1 = serial; e.g. 8 to use cores)")
    s.add_argument("--rate", type=float, default=15.0, help="animal arrival rate per hour")
    s.add_argument("--cruise", type=float, default=100.0, help="vehicle cruise speed (km/h)")
    s.add_argument("--caution", type=float, default=30.0, help="caution speed when alerted (km/h)")
    s.add_argument("--reaction", type=float, default=1.5, help="driver reaction time (s)")
    s.add_argument("--radar-spacing", type=float, default=15.0, help="radar spacing (m)")
    s.add_argument("--radar-range", type=float, default=10.0, help="radar detection range (m)")
    s.add_argument("--size-scale", type=float, default=1.0, help="RCS-proxy multiplier on animal size")
    s.add_argument("--vehicle-model", default="perfect", choices=["perfect", "magnetometer"],
                   help="vehicle-knowledge model (default perfect)")
    s.add_argument("--csv", default="wvc_results.csv", help="CSV output path")
    s.add_argument("--plot", action="store_true", help="also generate matplotlib figures")
    s.add_argument("--sweep", default=None,
                   help="sweep a parameter. Presets: " + ", ".join(SWEEP_PRESETS.keys())
                        + ". Or any Config field name (e.g. idm_T).")
    s.add_argument("--values", type=float, nargs="+", default=None,
                   help="values to sweep (overrides preset). Example: --values 5 10 15 20")
    s.set_defaults(func=cmd_simulate)

    # ---- data ----
    d = sub.add_parser("data", aliases=["generate-data"],
                       help="regenerate all result CSVs (headline + sweeps), checkpointed",
                       description="Regenerate the full data set: headline experiment plus every "
                                   "sensitivity sweep. Resumable via .regeneration_checkpoint.json.")
    d.add_argument("--jobs", type=int, default=1, help="worker processes (default 1 = serial)")
    d.add_argument("--fresh", action="store_true", help="delete checkpoint and start over")
    d.add_argument("--skip-sweeps", action="store_true", help="only run the headline experiment")
    d.add_argument("--headline-trials", type=int, default=None, help="trials for the headline run")
    d.add_argument("--headline-hours", type=float, default=None, help="hours per headline trial")
    d.add_argument("--sweep-trials", type=int, default=None, help="trials per sweep point")
    d.add_argument("--sweep-hours", type=float, default=None, help="hours per sweep trial")
    d.set_defaults(func=cmd_data)

    # ---- stats ----
    st = sub.add_parser("stats", help="compute statistics CSVs from result CSVs",
                        description="Welch t-tests, Poisson/NB GLM rate ratios, bootstrap CIs and "
                                    "mode x sweep interaction tests. Reads wvc_results*.csv, writes "
                                    "wvc_stats*.csv. Requires statsmodels.")
    st.add_argument("--bootstrap-n", type=int, default=10000,
                    help="bootstrap replicates per comparison (default 10000)")
    st.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed (default 42)")
    st.set_defaults(func=cmd_stats)

    # ---- figures ----
    f = sub.add_parser("figures", help="render publication PNG figures from CSVs",
                       description="Render the manuscript figures. Main figures are 3-6, "
                                   "supplementary are S1-S11. Identifiers are case-insensitive.")
    f.add_argument("--only", nargs="+", default=None, metavar="FIG",
                   help="render only these figures, e.g. --only 3 5 S1 S11 (default: all)")
    f.add_argument("--dpi", type=int, default=300, help="output DPI (default 300)")
    f.set_defaults(func=cmd_figures)

    # ---- ablation ----
    a = sub.add_parser("ablation", help="magnetometer vehicle-model ablation + geometry sweep",
                       description="Reviewer R1-04: does magnetometer-derived vehicle knowledge "
                                   "reproduce perfect knowledge (part 1), and where does the "
                                   "geometry break (part 2)?")
    a.add_argument("--part", default="all", choices=["all", "1", "2"],
                   help="1 = ablation, 2 = geometry sweep, all = both (default)")
    a.add_argument("--trials", type=int, default=10, help="trials per cell (default 10)")
    a.add_argument("--hours", type=float, default=2.0, help="hours per trial (default 2.0)")
    a.add_argument("--jobs", type=int, default=1, help="worker processes (default 1 = serial)")
    a.set_defaults(func=cmd_ablation)

    # ---- pipeline ----
    pl = sub.add_parser("pipeline", help="compound: data -> stats -> figures in sequence",
                        description="Run the full reproduction pipeline. Figure generation kicks in "
                                    "automatically once the data and stats stages finish. Use "
                                    "--stages to run a subset.")
    pl.add_argument("--stages", nargs="+", choices=["data", "stats", "figures"], default=None,
                    help="stages to run, in order (default: data stats figures)")
    pl.add_argument("--jobs", type=int, default=1, help="worker processes for the data stage")
    pl.add_argument("--fresh", action="store_true", help="delete data checkpoint and start over")
    pl.add_argument("--skip-sweeps", action="store_true", help="data stage: headline only")
    pl.add_argument("--headline-trials", type=int, default=None)
    pl.add_argument("--headline-hours", type=float, default=None)
    pl.add_argument("--sweep-trials", type=int, default=None)
    pl.add_argument("--sweep-hours", type=float, default=None)
    pl.add_argument("--bootstrap-n", type=int, default=10000, help="stats stage bootstrap replicates")
    pl.add_argument("--seed", type=int, default=42, help="stats stage bootstrap seed")
    pl.add_argument("--only", nargs="+", default=None, help="figures stage: render only these")
    pl.add_argument("--dpi", type=int, default=300, help="figures stage DPI")
    pl.set_defaults(func=cmd_pipeline)

    return p


def _force_utf8_stdio():
    """Make stdout/stderr UTF-8 so the summary tables (which contain σ, ±, ×, →)
    print on any platform. On Windows the default console/pipe encoding is often
    cp1252, which has no 'σ' and raises UnicodeEncodeError mid-run; reconfiguring
    to UTF-8 (with errors='replace' as a last resort) avoids that."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already wrapped, or not a reconfigurable stream


def main(argv=None):
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        parser.error(f"--workdir does not exist or is not a directory: {workdir}")
    os.chdir(workdir)
    args.func(args)


if __name__ == "__main__":
    main()