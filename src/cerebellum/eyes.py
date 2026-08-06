"""Adaptive eye movements: VOR + optokinetic (OKR) gaze stabilization.

The previous version modelled only an OKR-like retinal-slip reflex and
called it the gaze-stabilization system, which was correctly noted to
conflate pursuit/optokinetic control with the vestibulo-ocular reflex (VOR).

Here the two are separated:

* **VOR** produces compensatory eye velocity opposite to head angular velocity
  (canal-driven), with gain near unity, and its gain is recalibrated by
  retinal slip -- the flocculus mechanism the reviewer highlighted.
* **OKR** nulls residual wide-field retinal slip (NOT/DTN-driven), with a
  slower homeostatic gain adaptation.

Setting ``cfg.use_vor = False`` recovers the original OKR-only behaviour.
"""
from __future__ import annotations
import numpy as np


class EyePlant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.g_okr = cfg.g_okr0
        self.g_vor = cfg.g_vor0
        self.eye_vel = 0.0

    def commands(self, retinal_slip: float, head_omega: float):
        """Return VOR and OKR commands evaluated at the current sample."""
        vor = -self.g_vor * head_omega if self.cfg.use_vor else 0.0
        okr = self.g_okr * retinal_slip
        return float(vor), float(okr)

    def advance(self, retinal_slip: float, head_omega: float):
        """Use commands at t to advance eye velocity to t+1.

        The returned tuple contains ``(eye_next, vor_t, okr_t)``.  Keeping the
        command and plant transition separate makes the closed-loop timing
        explicit and prevents a current sensory sample from changing itself.
        """
        cfg = self.cfg
        vor_cmd, okr_cmd = self.commands(retinal_slip, head_omega)
        alpha = min(1.0, cfg.dt / max(cfg.tau_eye, cfg.dt))
        self.eye_vel += alpha * (vor_cmd + okr_cmd - self.eye_vel)
        self.g_okr = float(np.clip(
            self.g_okr + cfg.eta_g * (abs(retinal_slip) - cfg.slip_target),
            cfg.g_okr_min, cfg.g_okr_max))
        if cfg.use_vor:
            self.g_vor = float(np.clip(
                self.g_vor + cfg.eta_vor * retinal_slip * np.sign(head_omega),
                cfg.g_vor_min, cfg.g_vor_max))
        return float(self.eye_vel), vor_cmd, okr_cmd

    def step(self, retinal_slip: float, head_omega: float) -> float:
        """One eye-velocity update.

        retinal_slip : low-passed NOT/DTN retinal slip estimate
        head_omega   : canal-derived head angular velocity (for VOR)
        """
        eye_next, _, _ = self.advance(retinal_slip, head_omega)
        return eye_next
