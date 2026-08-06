"""Unit and invariant tests for the corrected closed-loop implementation."""
import numpy as np
import pytest

from cerebellum.config import Config
from cerebellum import model as M, stimulus as stim
from cerebellum.experiment import run_trial


def small_cfg(**kwargs):
    base = dict(T=600, n_trials=2, event_len_s=0.4,
                n_events_per_epoch=2, context_method="leaky_llr")
    base.update(kwargs)
    return Config(**base)


def test_initial_effective_forward_model_equals_A0_for_both_contexts():
    cfg = small_cfg(forward_initialization="corrected")
    assert M.assert_initial_forward_model(cfg)
    parts = M.initial_forward_models(cfg)
    for suffix in ("e", "m"):
        assert np.array_equal(parts[f"A_fast_{suffix}"], np.zeros((1, 3)))
        assert np.allclose(parts[f"A_slow_{suffix}"], np.array([cfg.A0]))


def test_legacy_initialization_is_explicitly_not_A0():
    cfg = small_cfg(forward_initialization="legacy")
    with pytest.raises(AssertionError):
        M.assert_initial_forward_model(cfg)


def test_eye_compensation_algebra_and_zero_eye_identity():
    cfg = small_cfg()
    v = stim.random_object_events(cfg, np.random.default_rng(1))
    out = M.simulate_closed_loop(cfg, np.random.default_rng(2), v, trial_id=4)
    assert np.allclose(out["y_vis_comp"], out["y_vis_raw"] + out["eye_vel"])
    zcfg = small_cfg(force_zero_eye=True)
    zero = M.simulate_closed_loop(zcfg, np.random.default_rng(2), v, trial_id=4)
    assert np.array_equal(zero["eye_vel"], np.zeros(cfg.T))
    assert np.allclose(zero["y_vis_comp"], zero["y_vis_raw"])


def test_corrected_trial_provenance_never_reuses_another_eye_trace():
    cfg = small_cfg()
    out = run_trial(cfg, "corrected_closed_loop", 7)
    assert out["trial_id"] == out["object_trial_id"] == out["eye_trial_id"] == 7
    assert out["object_motion"].shape == out["eye_vel"].shape == (cfg.T,)


def test_per_trial_diagnostic_ids_match_and_legacy_is_labelled_external():
    cfg = small_cfg()
    per = run_trial(cfg, "per_trial_eye_trace", 3)
    legacy = run_trial(cfg, "legacy_fixed_eye_trace", 3)
    assert per["object_trial_id"] == per["eye_trial_id"] == 3
    assert legacy["object_trial_id"] == 3 and legacy["eye_trial_id"] == -1


def test_nonoracle_variance_does_not_depend_on_hidden_true_context():
    cfg = small_cfg(oto_variance_policy="belief_weighted")
    a = M._otolith_variance(cfg, 0.25, true_sigma=0.06, sigma_adaptive=0.1)
    b = M._otolith_variance(cfg, 0.25, true_sigma=99.0, sigma_adaptive=0.1)
    assert a == b
    oracle = small_cfg(oto_variance_policy="oracle")
    assert M._otolith_variance(oracle, 0.25, 0.06, 0.1) != M._otolith_variance(oracle, 0.25, 0.4, 0.1)


def test_determinism_and_seeded_event_schedules():
    cfg = small_cfg()
    v = stim.random_object_events(cfg, np.random.default_rng(10))
    one = M.simulate_closed_loop(cfg, np.random.default_rng(11), v, trial_id=1)
    two = M.simulate_closed_loop(cfg, np.random.default_rng(11), v, trial_id=1)
    assert np.array_equal(one["x_hat"], two["x_hat"])
    v2 = stim.random_object_events(cfg, np.random.default_rng(12))
    assert not np.array_equal(v, v2)


def test_epoch_and_event_masks_are_valid():
    cfg = Config()
    epochs = stim.epoch_indices(cfg)
    assert [len(epochs[k]) for k in ("pre", "micro", "post")] == [833, 833, 834]
    v = stim.random_object_events(cfg, np.random.default_rng(1))
    present = np.abs(v) > 1e-3
    free = ~present
    assert not np.any(present & free)
    assert np.all(present | free)


@pytest.mark.parametrize("variant", ["legacy_fixed_eye_trace", "per_trial_eye_trace", "corrected_closed_loop"])
def test_all_variants_smoke_without_nans_or_empty_epochs(variant):
    cfg = small_cfg()
    out = run_trial(cfg, variant, 0)
    assert np.isfinite(out["x_hat"]).all()
    for idx in stim.epoch_indices(cfg).values():
        assert len(idx) > 0
