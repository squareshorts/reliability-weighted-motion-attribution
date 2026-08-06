import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("c:/work/cerebellum").resolve()
FINAL = ROOT / "analysis_modvis_completion" / "05_figures" / "final"
CSV_PATH = ROOT / "analysis_modvis_completion/limitations_closure/corrective_rerun/01_eye_signal_uncertainty/eye_uncertainty_raw_results.csv"

def generate_figure_s5():
    df = pd.read_csv(CSV_PATH)
    combined = df[df['type'] == 'combined']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Just a placeholder plot for S5
    ax.scatter(combined['val'], combined['mean_effect'])
    ax.set_ylabel("Mean Effect")
    ax.set_title("Combined Perturbations (Figure S5)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(FINAL / "Figure_S5_eye_signal_combined_grid.pdf")
    plt.savefig(FINAL / "Figure_S5_eye_signal_combined_grid.png")
    
if __name__ == "__main__":
    generate_figure_s5()
