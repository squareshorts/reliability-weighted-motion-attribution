"""Final structural robustness forest plot for false-positive RMS."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forest_common import build_forest


if __name__ == "__main__":
    build_forest(
        Path(__file__).with_suffix(""),
        "false_positive_rms",
        "Paired change in false-positive world-motion RMS\n(model units)",
    )
