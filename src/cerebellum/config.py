"""Central configuration for the contextual-inference cerebellar model.

All parameters that were scattered as literals across the original Colab
notebook are collected here so that simulations are reproducible and the
The parameter table can be generated directly from this file.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Config:
    # --- explicitly selectable simulation implementation ---
    simulation_variant: str = "corrected_closed_loop"
    # --- time base ---
    T: int = 2500            # number of steps
    dt: float = 0.01         # s

    # --- gravity schedule (fraction of Earth g) ---
    g_earth: float = 1.0
    g_micro: float = 0.15
    micro_start_frac: float = 1 / 3   # microgravity spans middle third
    micro_end_frac: float = 2 / 3

    # --- sensory noise (std) ---
    sigma_canal: float = 0.06
    sigma_oto_base: float = 0.06     # otolith std at 1 g; scales as base/g
    sigma_vis: float = 0.08
    sigma_not: float = 0.05          # NOT/DTN channel noise
    tau_not: float = 0.15            # NOT/DTN low-pass time constant (s)

    # --- process noise (diag of Q) ---
    q_omega: float = 1e-4
    q_a: float = 2e-4
    q_obj: float = 8e-4

    # --- forward model (visual mapping A = A_slow + A_fast) ---
    A0: tuple = (1.0, 0.9, 1.0)
    A_fast0: tuple = (0.8, 0.7, 0.8)
    forward_initialization: str = "corrected"  # corrected | legacy
    forward_rule: str = "canonical_two_rate"  # canonical_two_rate | legacy_tracking
    retention_fast: float = 0.98
    retention_slow: float = 0.9995
    enable_fast: bool = True
    enable_slow: bool = True
    context_gating: bool = True
    eta_fast: float = 3e-3           # fast cerebellar adaptation rate
    eta_slow: float = 3e-4           # slow consolidation rate

    # --- efference / self-motion drive ---
    B: tuple = (0.8, 0.6, 0.0)
    consistent_efference: bool = True  # align process model with generative
                                       # efference (removes innovation confound)

    # --- otolith reliability tracking ---
    eta_sigma: float = 5e-4
    sigma_o_min: float = 0.02
    sigma_o_max: float = 0.30

    # --- context inference ---
    context_method: str = "leaky_llr"  # clipped leaky LLR accumulator, not an exact HMM
    eta_p: float = 5e-3              # energy-rule sensitivity (original)
    tau_p: float = 0.995            # leak / hysteresis
    # llr-rule hypotheses (assumed otolith std under each context)
    sigma_oto_earth_hyp: float = 0.06
    sigma_oto_micro_hyp: float = 0.40
    llr_leak: float = 0.98          # leak on accumulated log-odds
    llr_clip: float = 8.0           # bound on |log-odds|
    initial_context_log_odds: float = -4.0
    oto_variance_policy: str = "belief_weighted"  # belief_weighted | fixed | oracle | adaptive_legacy
    fixed_oto_std: float = 0.06

    # --- eye movements ---
    g_okr0: float = 0.6
    eta_g: float = 5e-4
    g_okr_min: float = 0.1
    g_okr_max: float = 1.5
    slip_target: float = 0.05
    # vestibulo-ocular reflex (added; distinct from OKR pursuit/optokinetic)
    use_vor: bool = True
    g_vor0: float = 1.0             # VOR gain, near unity
    eta_vor: float = 1e-3           # flocculus retinal-slip calibration rate
    g_vor_min: float = 0.3
    g_vor_max: float = 1.2
    tau_eye: float = 1.0
    eye_compensation: bool = True
    force_zero_eye: bool = False
    legacy_visual_nonlinearity: bool = False

    # --- trials / detection ---
    n_trials: int = 30
    n_events_per_epoch: int = 3
    event_len_s: float = 0.8
    detect_threshold: float = 0.2

    seed: int = 7
    bootstrap_samples: int = 2000
    bootstrap_alpha: float = 0.05

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.T) * self.dt

    @property
    def micro_start(self) -> int:
        return int(self.T * self.micro_start_frac)

    @property
    def micro_end(self) -> int:
        return int(self.T * self.micro_end_frac)

    @property
    def event_len(self) -> int:
        return int(self.event_len_s / self.dt)

    def gravity(self) -> np.ndarray:
        g = np.ones(self.T) * self.g_earth
        g[self.micro_start:self.micro_end] = self.g_micro
        return g

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.q_omega, self.q_a, self.q_obj])

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
