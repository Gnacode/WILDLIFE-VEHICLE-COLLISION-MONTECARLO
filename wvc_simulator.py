"""
WVC Monte Carlo Simulation Framework
=====================================

A behaviorally-realistic Monte Carlo simulation of a radar-magnetometer
sensor network for wildlife-vehicle collision (WVC) prevention.

Three operating modes are compared in a single experiment:

  control    No sensors deployed at all (true unmitigated baseline).
             Animals approach the road and cross/freeze/flee by their own
             state machine; vehicles cruise at full speed throughout.

  detection  Sensors deployed and detecting animals independently at their
             baseline sensitivity. Drivers are alerted on every detection
             via the DMS and the IDM responds (caution speed + hard-brake).
             Sensors do not cooperate — no LoRa-mediated sensitivity boost.

  aware      Same driver/IDM response as 'detection', PLUS network awareness:
             any detection LoRa-broadcasts a sensitivity boost to every
             sensor within R_a for τ_persist seconds. Boosted neighbours
             catch subsequent animals (and coverage-gap animals) earlier,
             giving drivers a longer span to act. The boost is the only
             mechanism that distinguishes 'aware' from 'detection'.

Models:
  - Stochastic animal behavior (6-state Markov-like state machine)
  - Vehicle dynamics via Intelligent Driver Model (Treiber et al. 2000)
  - Radar detection with size-dependent probability and coverage geometry
  - LoRa-mediated corridor-wide awareness propagation
  - Driver response via caution speed when DMS active

Authors: Lars Thomsen, Sergii Makovetskyi
Affiliation: Gnacode Inc. (Canada) / KNURE (Ukraine)
License: research / non-commercial

Usage:
  python wvc_simulator.py                                 # default 10 × 2 h, all 3 modes
  python wvc_simulator.py --trials 20 --hours 4 --plot    # paper-grade single-point run
  python wvc_simulator.py --modes control aware           # just the headline contrast
  python wvc_simulator.py --modes detection aware         # isolate awareness benefit

Sensitivity sweeps (vary one parameter, all three modes at each point):
  python wvc_simulator.py --sweep spacing --plot          # radar spacing 5..40 m
  python wvc_simulator.py --sweep range --plot            # radar detection range 5..30 m
  python wvc_simulator.py --sweep size --plot             # animal RCS scale 0.25..3.0
  python wvc_simulator.py --sweep detection_rate --plot   # baseline κ 0.3..5.0 /sec
  python wvc_simulator.py --sweep rate --plot             # arrival λ 5..60 /hr
  python wvc_simulator.py --sweep caution --plot          # caution speed 20..80 km/h
  python wvc_simulator.py --sweep cruise --plot           # cruise speed 60..140 km/h
  python wvc_simulator.py --sweep reaction --plot         # driver reaction 0.5..3.0 s

  # Custom values for any sweep:
  python wvc_simulator.py --sweep spacing --values 5 10 20 40 --plot
"""

from __future__ import annotations
import argparse
import csv
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

    # --- Behavioral decision probabilities ---
    # At HESITATING decision point with vehicle close:
    p_freeze_if_vehicle: float = 0.10   # rare freeze at road edge
    p_flee_if_vehicle: float = 0.20     # cautious retreat
    # else 70% brave-cross
    # When no vehicle close at HESITATING:
    p_cross_if_clear: float = 0.80
    # else 20% flee
    # Freeze during CROSSING when vehicle suddenly approaches:
    p_freeze_during_cross_base: float = 0.15
    vehicle_close_dist: float = 80.0    # m, "close" threshold
    vehicle_visibility: float = 150.0   # m, animal can see vehicle within this

    # --- Sensor topology ---
    radar_spacing: float = 15.0         # m, alternating sides
    radar_range: float = 15.0           # m, detection radius
    radar_verge_offset: float = 2.5     # m, radar position outside road
    magnetic_spacing: float = 200.0     # m
    # Detection model: per-frame Pd via exponential rate
    detection_rate_per_sec: float = 3.0  # κ for reference RCS=1.0

    # --- Awareness / LoRa propagation ---
    awareness_range_m: float = 1500.0   # corridor-wide
    awareness_persist_s: float = 30.0   # min duration after last detection
    lora_delay_ms: float = 200.0
    boost_factor: float = 1.8           # detection sensitivity boost on neighbors

    # --- IDM vehicle dynamics ---
    idm_s0: float = 5.0
    idm_T: float = 1.5
    idm_a_max: float = 2.5
    idm_b_comf: float = 4.0
    idm_a_emergency: float = 9.0
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


@dataclass
class Radar:
    x: float
    side: int           # +1=south, -1=north
    boost_until: float = 0.0


@dataclass
class TrialStats:
    total_animals: int = 0
    detected: int = 0
    road_entries: int = 0
    collisions: int = 0
    warnings_issued: int = 0
    vehicles_alerted: int = 0
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

    def _init_vehicles(self):
        c = self.cfg
        cruise = c.cruise_speed_kmh / 3.6
        for _ in range(c.vehicles_per_dir):
            for direction in (+1, -1):
                x0 = self.rng.uniform(-50, c.road_length + 50)
                self.vehicles.append(Vehicle(x=x0, direction=direction, speed=cruise))

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

    def _update_animal_state(self, a: Animal):
        c = self.cfg
        elapsed = self.time - a.state_changed_at
        v_near, v_dist = self._closest_vehicle(a)
        threat = self._vehicle_threat_factor(v_near)
        # "Vehicle close" means within close-dist AND moving (threat > 0.3)
        vehicle_threatens = (v_near is not None and v_dist < c.vehicle_close_dist and threat > 0.3)

        if a.state == AnimalState.FORAGING:
            if elapsed > a.dwell_target:
                self._set_state(a, AnimalState.APPROACHING)

        elif a.state == AnimalState.APPROACHING:
            if abs(a.y) <= c.verge_threshold:
                self._set_state(a, AnimalState.HESITATING)
            elif vehicle_threatens and self.rng.random() < 0.05 * threat * c.dt:
                # Occasionally flee on approach due to nearby fast vehicle
                self._set_state(a, AnimalState.FLEEING)

        elif a.state == AnimalState.HESITATING:
            if elapsed > a.dwell_target:
                # Decision time
                if vehicle_threatens:
                    r = self.rng.random()
                    if r < c.p_freeze_if_vehicle * threat:
                        self._set_state(a, AnimalState.FROZEN)
                    elif r < (c.p_freeze_if_vehicle + c.p_flee_if_vehicle) * threat:
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
            boost = c.boost_factor if self.time < r.boost_until else 1.0
            range2 = c.radar_range ** 2

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
                rate = c.detection_rate_per_sec * size_factor * boost
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
            # Corridor-wide alert from awareness state
            if awareness_alerts and not v.alerted:
                v.alerted = True
                v.alert_reaction_at = now + c.driver_reaction_s
                self.stats.vehicles_alerted += 1
            elif not awareness_alerts and v.alerted:
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
            elif v.direction == -1 and v.x < -80:
                v.x = c.road_length + 50
                v.alerted = False
                v.speed = cruise

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
        self._update_vehicles()
        self._check_collisions()

    def run(self, duration_s: float, verbose: bool = False):
        n_steps = int(duration_s / self.cfg.dt)
        for i in range(n_steps):
            self.step()


# ============================================================
#  MONTE CARLO HARNESS
# ============================================================

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
    # Distinct seed offsets so each mode draws an independent RNG stream
    mode_offsets = {"control": 0, "detection": 1, "aware": 2}

    for mode in modes:
        if mode not in WVCSimulation.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {WVCSimulation.VALID_MODES}")
        offset = mode_offsets.get(mode, 0)
        if verbose:
            print(f"\n=== Mode: {mode.upper()}  ({n_trials} trials × {duration_hr} h) ===")
        for trial in range(n_trials):
            seed = trial * 10000 + offset
            trial_cfg = Config(**{**config.__dict__, "seed": seed})
            sim = WVCSimulation(trial_cfg, mode=mode)
            t0 = time.time()
            sim.run(duration_s)
            elapsed = time.time() - t0
            s = sim.stats
            mean_lat = float(np.mean(s.detection_latencies)) if s.detection_latencies else 0.0
            mean_in_range = float(np.mean(s.in_range_latencies)) if s.in_range_latencies else 0.0
            res = {
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
                "frozen_on_road_s": s.frozen_on_road_seconds,
                "n_foraging": s.state_visits.get(AnimalState.FORAGING, 0),
                "n_approaching": s.state_visits.get(AnimalState.APPROACHING, 0),
                "n_hesitating": s.state_visits.get(AnimalState.HESITATING, 0),
                "n_crossing": s.state_visits.get(AnimalState.CROSSING, 0),
                "n_frozen": s.state_visits.get(AnimalState.FROZEN, 0),
                "n_fleeing": s.state_visits.get(AnimalState.FLEEING, 0),
                "wall_seconds": elapsed,
            }
            results[mode].append(res)
            if verbose:
                print(f"  trial {trial+1:2d}: "
                      f"N={s.total_animals:3d}  "
                      f"det={s.detected:3d} ({100*res['det_rate']:5.1f}%)  "
                      f"road={s.road_entries:3d}  "
                      f"coll={s.collisions:2d} ({100*res['col_rate']:5.2f}%)  "
                      f"aware={res['awareness_pct']:5.1f}%  "
                      f"frozen={s.frozen_on_road_seconds:5.1f}s  "
                      f"[{elapsed:.1f}s]")
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
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WVC Monte Carlo Simulator — radar-magnetometer fusion with corridor awareness"
    )
    parser.add_argument("--trials", type=int, default=10, help="Trials per mode (default 10)")
    parser.add_argument("--hours", type=float, default=2.0, help="Hours per trial (default 2.0)")
    parser.add_argument(
        "--modes", nargs="+",
        default=["control", "detection", "aware"],
        choices=["control", "detection", "aware"],
        help="Modes to compare (default: all three). "
             "'control' = no system; 'detection' = sensors detect but no alert; "
             "'aware' = full system with DMS caution speed.",
    )
    parser.add_argument("--rate", type=float, default=15.0, help="Animal arrival rate per hour")
    parser.add_argument("--cruise", type=float, default=100.0, help="Vehicle cruise speed (km/h)")
    parser.add_argument("--caution", type=float, default=30.0, help="Caution speed when alerted (km/h)")
    parser.add_argument("--reaction", type=float, default=1.5, help="Driver reaction time (s)")
    parser.add_argument("--radar-spacing", type=float, default=15.0, help="Radar spacing (m)")
    parser.add_argument("--radar-range", type=float, default=15.0, help="Radar detection range (m)")
    parser.add_argument("--size-scale", type=float, default=1.0,
                        help="Multiplier on all sampled animal sizes (RCS proxy). Default 1.0.")
    parser.add_argument("--csv", type=str, default="wvc_results.csv", help="CSV output path")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    parser.add_argument(
        "--sweep", type=str, default=None,
        help="Run sensitivity sweep on a parameter. Preset names: "
             + ", ".join(SWEEP_PRESETS.keys())
             + ". Or pass any Config field name (e.g. 'idm_T').",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="Values to sweep (overrides preset defaults). Example: --values 5 10 15 20",
    )
    args = parser.parse_args()

    cfg = Config(
        animal_rate_per_hr=args.rate,
        cruise_speed_kmh=args.cruise,
        caution_speed_kmh=args.caution,
        driver_reaction_s=args.reaction,
        radar_spacing=args.radar_spacing,
        radar_range=args.radar_range,
        size_scale=args.size_scale,
    )

    print("=" * 60)
    print(" WVC Monte Carlo simulation")
    print("=" * 60)
    print(f"  Road length:        {cfg.road_length} m")
    print(f"  Vehicles / dir:     {cfg.vehicles_per_dir} @ {cfg.cruise_speed_kmh} km/h")
    print(f"  Caution speed:      {cfg.caution_speed_kmh} km/h (when awareness active)")
    print(f"  Animal arrival:     λ = {cfg.animal_rate_per_hr} /hr (Poisson)")
    print(f"  Radar topology:     spacing {cfg.radar_spacing} m, range {cfg.radar_range} m, alternating sides")
    print(f"  Awareness corridor: {cfg.awareness_range_m} m, persist {cfg.awareness_persist_s} s")
    print(f"  Driver reaction:    {cfg.driver_reaction_s} s")
    print(f"  Time step:          {cfg.dt} s")
    print(f"  Modes to run:       {', '.join(args.modes)}")

    modes = tuple(args.modes)

    # ----- Sweep mode -----
    if args.sweep:
        # Resolve preset or direct config field name
        if args.sweep in SWEEP_PRESETS:
            sweep_param, default_values = SWEEP_PRESETS[args.sweep]
        else:
            sweep_param = args.sweep
            default_values = None
        values = args.values if args.values is not None else default_values
        if values is None:
            print(f"\nERROR: no preset for sweep '{args.sweep}' and no --values provided.")
            print(f"       Either use a preset name ({', '.join(SWEEP_PRESETS.keys())})")
            print(f"       or pass explicit values, e.g. --sweep {args.sweep} --values 1 2 4 8")
            return
        print(f"  Sweep:              {args.sweep} ({sweep_param}) over {values}")
        print(f"  Total trials:       {len(values)} × {len(modes)} × {args.trials} = "
              f"{len(values) * len(modes) * args.trials}\n")
        sweep_results = run_sensitivity_sweep(
            cfg, sweep_param, values,
            n_trials=args.trials, duration_hr=args.hours, modes=modes,
        )
        summarize_sweep(sweep_results, sweep_param, modes=modes)
        sweep_csv = args.csv.replace(".csv", f"_sweep_{args.sweep}.csv")
        write_sweep_csv(sweep_results, sweep_param, sweep_csv)
        if args.plot:
            sweep_png = args.csv.replace(".csv", f"_sweep_{args.sweep}.png")
            plot_sweep(sweep_results, sweep_param, sweep_png, modes=modes)
        return

    # ----- Single-point Monte Carlo (default) -----
    results = run_monte_carlo(cfg, n_trials=args.trials, duration_hr=args.hours, modes=modes)
    summarize(results, modes=modes)
    write_csv(results, args.csv)
    if args.plot:
        plot_results(results, out_path=args.csv.replace(".csv", ".png"))


if __name__ == "__main__":
    main()