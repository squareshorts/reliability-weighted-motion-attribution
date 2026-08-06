"""Gravity-context evidence accumulation.

Three update rules are provided:

* ``energy`` -- the *original* leaky accumulator from the previous model version,
  which integrates raw square of prediction errors (H * x_err). Because ``innov^2`` is strictly positive under
  every context, this ramps monotonically toward 1 even under Earth gravity,
  producing an artifact (P(micro) saturating *before*
  microgravity onset).

* ``llr_fixed`` -- an intermediate rule that accumulates the log-likelihood
  ratio using *fixed* assumed otolith variances (the previous Eq. 10, with
  added exponential smoothing to emulate lambda memory).  It is a knife-edge: because it ignores the
  filter's state uncertainty, ordinary Earth innovations can trip it.

* ``leaky_llr`` (the corrected default; ``llr`` is a compatibility alias) --
  accumulates the log-likelihood ratio between
  the two contexts evaluated against the *innovation covariance*
  ``S_c = H P H^T + R_c``.  Because it accounts for the current state
  uncertainty, it is robust: P(micro) stays low under Earth and rises only
  when the otolith innovation variance genuinely exceeds the Earth
  expectation.  This realises the reviewer's point that gravity context is a
  central estimate in the vestibular-nuclei/cerebellar loop (a function of
  prediction-error statistics), not a quantity read directly off the otoliths.

The corrected rule is a clipped leaky log-likelihood accumulator.  It is not
the exact recursion of a two-state hidden Markov model:

``L_t = clip(lambda * L_(t-1) + ell_t, -L_max, L_max); p_t = sigmoid(L_t)``.
"""
from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class ContextEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.method = cfg.context_method
        self.p = 0.0
        self.L = cfg.initial_context_log_odds
        self.rE = cfg.sigma_oto_earth_hyp ** 2   # assumed otolith var, Earth
        self.rM = cfg.sigma_oto_micro_hyp ** 2   # assumed otolith var, micro
        # constants for the fixed-variance rule
        self._base_fixed = 0.5 * np.log(self.rE / self.rM)
        self._slope_fixed = 0.5 * (1.0 / self.rE - 1.0 / self.rM)

    def update(self, innov_o: float, Saa: float = 0.0) -> float:
        """Advance one step.

        innov_o : otolith innovation (measurement - prediction)
        Saa     : state contribution to innovation variance, H P H^T
                  (used by the robust ``llr`` rule)
        """
        cfg = self.cfg
        if self.method == "energy":
            self.p = float(np.clip(cfg.tau_p * self.p + cfg.eta_p * innov_o ** 2, 0, 1))
        elif self.method == "llr_fixed":
            ell = self._base_fixed + self._slope_fixed * innov_o ** 2
            self.L = np.clip(cfg.llr_leak * self.L + ell, -cfg.llr_clip, cfg.llr_clip)
            self.p = float(_sigmoid(self.L))
        elif self.method in ("llr", "leaky_llr"):
            SE = Saa + self.rE
            SM = Saa + self.rM
            ell = 0.5 * np.log(SE / SM) + 0.5 * innov_o ** 2 * (1.0 / SE - 1.0 / SM)
            self.L = np.clip(cfg.llr_leak * self.L + ell, -cfg.llr_clip, cfg.llr_clip)
            self.p = float(_sigmoid(self.L))
        else:
            raise ValueError(f"unknown context_method: {self.method}")
        return self.p
