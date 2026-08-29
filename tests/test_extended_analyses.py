"""Lightweight invariants for the extended analyses."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SCRIPTS_ROOT, SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cerebellum.config import Config  # noqa: E402
from model_variants import canonical_core_run, run_model, single_timescale_mapping  # noqa: E402


def test_instrumented_canonical_is_bitwise_equivalent():
    cfg = Config()
    instrumented = run_model(cfg, 0, "A_full_canonical")
    core = canonical_core_run(cfg, 0)
    assert np.array_equal(instrumented["x_hat"], core["x_hat"])
    assert np.array_equal(instrumented["p_hist"], core["p_hist"])


def test_paired_seed_streams_and_events_are_preserved():
    cfg = Config()
    full = run_model(cfg, 1, "A_full_canonical")
    fixed = run_model(cfg, 1, "B_fixed_context")
    assert full["rng_seed"] == fixed["rng_seed"] == 101
    assert np.array_equal(full["object_motion"], fixed["object_motion"])
    assert np.array_equal(full["canal_standard_normal"], fixed["canal_standard_normal"])
    assert np.array_equal(full["otolith_standard_normal"], fixed["otolith_standard_normal"])
    assert np.array_equal(full["visual_standard_normal"], fixed["visual_standard_normal"])


def test_baseline_reduced_and_extrapolative_noise_mapping():
    cfg = Config()
    assert np.isclose(cfg.sigma_oto_base / 1.0, cfg.sigma_oto_earth_hyp)
    assert np.isclose(cfg.sigma_oto_base / 0.15, cfg.sigma_oto_micro_hyp)
    assert np.isclose(cfg.sigma_oto_base / 0.10, 0.60)
    assert cfg.sigma_oto_base / 0.10 > cfg.sigma_oto_micro_hyp


def test_context_variance_mapping_uses_previous_belief_mixture():
    cfg = Config()
    for p in (0.0, 0.25, 0.5, 1.0):
        variance = (1 - p) * cfg.sigma_oto_earth_hyp**2 + p * cfg.sigma_oto_micro_hyp**2
        assert cfg.sigma_oto_earth_hyp <= np.sqrt(variance) <= cfg.sigma_oto_micro_hyp
    assert np.isclose(np.sqrt(cfg.sigma_oto_earth_hyp**2), 0.06)
    assert np.isclose(np.sqrt(cfg.sigma_oto_micro_hyp**2), 0.40)


def test_threshold_definitions_are_declared_fractions_of_event_amplitude():
    thresholds = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    expected_percent = np.array([25.0, 37.5, 50.0, 62.5, 75.0])
    assert np.array_equal(100.0 * thresholds / 0.40, expected_percent)


def test_within_realization_percentage_precedes_averaging():
    baseline = np.array([1.0, 2.0])
    reduced = np.array([2.0, 3.0])
    within = np.mean(100.0 * (reduced - baseline) / baseline)
    ratio_of_means = 100.0 * (reduced.mean() - baseline.mean()) / baseline.mean()
    assert np.isclose(within, 75.0)
    assert not np.isclose(within, ratio_of_means)


def test_recovery_epoch_extraction_is_exact_and_nonoverlapping():
    cfg = Config()
    baseline = np.arange(0, cfg.micro_start)
    reduced = np.arange(cfg.micro_start, cfg.micro_end)
    recovery = np.arange(cfg.micro_end, cfg.T)
    assert baseline[-1] + 1 == reduced[0]
    assert reduced[-1] + 1 == recovery[0]
    assert len(np.unique(np.concatenate([baseline, reduced, recovery]))) == cfg.T


def test_single_timescale_mapping_preserves_impulse_and_one_step_retention():
    cfg = Config()
    mapped = single_timescale_mapping(cfg)
    eta = mapped["eta_single"]
    rho = mapped["retention_single"]
    assert np.isclose(eta, cfg.eta_fast + cfg.eta_slow)
    assert np.isclose(
        eta * rho,
        cfg.eta_fast * cfg.retention_fast + cfg.eta_slow * cfg.retention_slow,
    )
