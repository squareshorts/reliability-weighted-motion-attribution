"""Supplementary structural robustness forest plot for object-motion RMSE."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forest_common import build_forest


if __name__ == "__main__":
    build_forest(
        Path(__file__).with_suffix(""),
        "object_rmse",
        "Paired change in object-motion RMSE\n(model units)",
        dimensions=(6.70, 4.95),
    )
