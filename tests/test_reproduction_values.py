import json
import csv
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "aggregate"

def test_canonical_effect_sizes():
    with open(RESULTS / "03_paired_effect_sizes.csv") as f:
        rows = list(csv.DictReader(f))
    
    # "angular RMSE change: 4.37%; object RMSE change: 48.77%; false-positive RMS change: 47.61%;"
    angular = next(r for r in rows if r["metric"] == "angular_rmse")
    object_rmse = next(r for r in rows if r["metric"] == "object_rmse")
    fp = next(r for r in rows if r["metric"] == "false_positive_rms")
    
    assert round(float(angular["mean_percentage_change"]), 2) == 4.37
    assert round(float(object_rmse["mean_percentage_change"]), 2) == 48.77
    assert round(float(fp["mean_percentage_change"]), 2) == 47.61

def test_reliability_sweep():
    # reliability-sweep slope: 0.04135
    with open(RESULTS / "04_reliability_sweep_seed_slopes.csv") as f:
        rows = list(csv.DictReader(f))
    assert round(float(rows[0]["mean_slope"]), 5) == 0.04135

def test_eye_signal():
    # 112 robust and 12 attenuated eye-signal settings; worst eye-signal condition: 39/50 positive seeds
    with open(ROOT / "figures" / "main" / "Figure_5A_eye_uncertainty_conditions.csv") as f:
        rows = list(csv.DictReader(f))
    
    assert len(rows) == 124
    robust = sum(1 for r in rows if r["status"] == "ROBUST")
    attenuated = sum(1 for r in rows if r["status"] == "ATTENUATED")
    reversals = sum(1 for r in rows if r["status"] == "REVERSAL")
    
    assert robust == 112
    assert attenuated == 12
    assert reversals == 0
    
    # worst eye-signal condition: 39/50 positive seeds
    min_positive_seeds = min(int(r["n_predicted"]) for r in rows)
    assert min_positive_seeds == 39

def test_global_sensitivity():
    # 992/1000 positive means, 979/1000 robust positive, one robust reversal, zero failures
    with open(RESULTS / "sensitivity_results.csv") as f:
        rows = list(csv.DictReader(f))
    
    assert len(rows) == 1000
    
    positive_means = sum(1 for r in rows if float(r["mean_effect"]) > 0)
    robust_positive = sum(1 for r in rows if float(r["ci_lower"]) > 0)
    robust_reversal = sum(1 for r in rows if float(r["ci_upper"]) < 0)
    failures = sum(1 for r in rows if r.get("failed", "false").lower() == "true")
    
    assert positive_means == 992
    assert robust_positive == 979
    assert robust_reversal == 1
    assert failures == 0
    
def test_prcc():
    # PRCC: baseline otolith noise approximately +0.932, canal noise approximately -0.843
    with open(RESULTS / "importance_analysis.json") as f:
        data = json.load(f)
        
    main_effects = data["main_effects"]
    sigma_oto_base = main_effects["sigma_oto_base"]["prcc"]
    sigma_canal = main_effects["sigma_canal"]["prcc"]
    
    assert abs(sigma_oto_base - 0.932) < 0.01
    assert abs(sigma_canal - -0.843) < 0.01
