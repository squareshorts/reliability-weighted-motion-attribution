import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("c:/work/cerebellum").resolve()
FINAL = ROOT / "analysis_modvis_completion" / "05_figures" / "final"
CSV_PATH = ROOT / "analysis_modvis_completion/limitations_closure/corrective_rerun/01_eye_signal_uncertainty/eye_uncertainty_raw_results.csv"

def generate_figure_s4():
    df = pd.read_csv(CSV_PATH)
    single_perturbations = df[df['type'].isin(['noise', 'gain', 'delay', 'bias'])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Just a placeholder plot for S4
    ax.bar(single_perturbations['type'] + "=" + single_perturbations['val'].astype(str), single_perturbations['mean_effect'])
    ax.set_ylabel("Mean Effect")
    ax.set_title("Single Perturbations (Figure S4)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(FINAL / "Figure_S4_eye_signal_single_perturbations.pdf")
    plt.savefig(FINAL / "Figure_S4_eye_signal_single_perturbations.png")
    
if __name__ == "__main__":
    generate_figure_s4()
