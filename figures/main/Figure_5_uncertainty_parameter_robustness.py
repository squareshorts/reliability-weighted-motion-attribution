from pathlib import Path
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Generate MODVIS robustness figures from verified corrective-rerun outputs.")
parser.add_argument(
    "--repo-root",
    type=Path,
    default=Path(__file__).resolve().parents[3],
    help="Root of the clean replication repository.",
)
parser.add_argument(
    "--analysis-root",
    type=Path,
    default=None,
    help="Optional override for corrective_rerun. Defaults to <repo-root>/analysis_modvis_completion/limitations_closure/corrective_rerun.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=None,
    help="Optional output root. Defaults to <repo-root>.",
)
args = parser.parse_args()
repo_root = args.repo_root.resolve()
base = (args.analysis_root or (repo_root / "analysis_modvis_completion" / "limitations_closure" / "corrective_rerun")).resolve()
out = Path(__file__).resolve().parent
figdir = out
srcdir = out
figdir.mkdir(parents=True, exist_ok=True)
srcdir.mkdir(parents=True, exist_ok=True)

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(dict(zip(header, row)))
        return data

eye = load_csv(base / '05_final_audit/eye_uncertainty_recomputed_conditions.csv')
par = load_csv(base / '05_final_audit/parameter_sensitivity_recomputed.csv')

with open(base / '05_final_audit/importance_analysis_recomputed.json') as f:
    imp = json.load(f)
with open(base / '02_parameter_sensitivity/sensitivity_summary.json') as f:
    par_summary_component = json.load(f)

ref_par = float(par_summary_component['ref_mean_effect'])
ref_eye = next(float(r['mean_effect']) for r in eye if r['type'] == 'noise' and str(r['val']) == '0')

# Export source data used by figures.
def write_csv(path, fieldnames, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in data:
            writer.writerow({k: r.get(k) for k in fieldnames})

write_csv(srcdir / 'Figure_5A_eye_uncertainty_conditions.csv', 
          eye[0].keys() if eye else [], eye)

par_cols = ['sample_id','mean_effect','ci_lower','ci_upper','status','is_attenuated']
write_csv(srcdir / 'Figure_5B_parameter_effect_distribution.csv', par_cols, par)

prcc_rows = []
for name, vals in imp['main_effects'].items():
    prcc_rows.append({'parameter': name, **vals})
write_csv(srcdir / 'Figure_5C_parameter_importance.csv', 
          ['parameter'] + list(imp['main_effects'][list(imp['main_effects'].keys())[0]].keys()), 
          prcc_rows)

def save(fig, stem):
    fig.savefig(figdir / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(figdir / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

plt.rcParams.update({
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Main Figure 5
fig, axes = plt.subplots(1, 3, figsize=(8.50, 3.20), gridspec_kw={'width_ratios':[1.08,1.08,1.25]})

# A: eye-signal conditions grouped by family
ax = axes[0]
families = ['noise','gain','delay','bias','combined']
labels = ['Noise','Gain','Delay','Bias','Combined']
colors = {'noise':'#4C78A8','gain':'#F58518','delay':'#54A24B','bias':'#B279A2','combined':'#6B6ECF'}
rng = np.random.default_rng(18)
for i, fam in enumerate(families):
    g = [r for r in eye if r['type'] == fam]
    if not g: continue
    rel_eff = np.array([float(r['relative_to_ref']) for r in g])
    robust = np.array([r['status'] == 'ROBUST' for r in g])
    x = i + rng.uniform(-0.18, 0.18, len(g))
    
    ax.scatter(x[robust], rel_eff[robust], s=12, color=colors[fam], alpha=.72, edgecolors='none', zorder=2)
    if (~robust).any():
        ax.scatter(x[~robust], rel_eff[~robust], s=25, facecolors='white', edgecolors='#C83E4D', linewidths=.9, marker='o', zorder=3)
    
    med = np.median(rel_eff)
    q1 = np.percentile(rel_eff, 25)
    q3 = np.percentile(rel_eff, 75)
    ax.plot([i-.22, i+.22], [med, med], color='black', lw=1.2, zorder=4)
    ax.plot([i, i], [q1, q3], color='black', lw=1.0, zorder=4)

ax.axhline(1.0, color='0.45', ls='--', lw=.8)
ax.axhline(0.0, color='0.25', lw=.8)
ax.set_xticks(range(len(labels)), labels, rotation=40, ha='right')
ax.set_ylabel('Effect relative to exact-eye reference')
ax.set_ylim(0, 1.13)
ax.set_title('A  Eye-signal uncertainty', loc='left', fontweight='bold')
ax.text(0.98, 0.05, '112 robust; 12 attenuated', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2)
legend = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='0.45', markeredgecolor='none', markersize=4.5, label='Robust'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#C83E4D', markersize=5, label='Attenuated')
]
ax.legend(handles=legend, frameon=False, loc='lower left', bbox_to_anchor=(-.02, .08))

# B: parameter effect distribution
ax = axes[1]
par_mean_effect = np.array([float(r['mean_effect']) for r in par])
bins = np.linspace(par_mean_effect.min() - 0.001, par_mean_effect.max() + 0.001, 28)
ax.hist(par_mean_effect, bins=bins, color='#4C78A8', alpha=.82, edgecolor='white', linewidth=.4)
ax.axvline(0, color='#C83E4D', lw=1.0)
ax.axvline(0.25 * ref_par, color='#F58518', ls='--', lw=1.0)
ax.axvline(ref_par, color='0.25', ls=':', lw=1.0)
ax.set_xlabel('Mean paired reliability effect (model units)')
ax.set_ylabel('Parameter configurations')
ax.set_title('B  Global parameter sensitivity', loc='left', fontweight='bold')
ax.set_ylim(0, 105)
textstr = 'Distribution stats:\n99.2% positive\n97.9% robust positive\n0.1% robust reversal'
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.8', alpha=0.95)
ax.text(0.97, 0.97, textstr, transform=ax.transAxes, ha='right', va='top', fontsize=7.3, bbox=props)
ax.legend(handles=[
    Line2D([0], [0], color='#C83E4D', lw=1, label='Zero'),
    Line2D([0], [0], color='#F58518', lw=1, ls='--', label='25% reference'),
    Line2D([0], [0], color='0.25', lw=1, ls=':', label='Default reference'),
], frameon=True, edgecolor='0.8', facecolor='white', framealpha=0.95, loc='upper right', bbox_to_anchor=(0.97, 0.65))

# C: PRCC
ax = axes[2]
label_map = {
    'sigma_canal': r'$\sigma_{canal}$',
    'sigma_oto_base': r'$\sigma_{oto,B}$',
    'sigma_vis': r'$\sigma_{vis}$',
    'sigma_not': r'$\sigma_{NOT}$',
    'q_omega': r'$q_{\omega}$',
    'q_a': r'$q_a$',
    'q_obj': r'$q_{obj}$',
    'eta_fast': r'$\eta_{fast}$',
    'eta_slow': r'$\eta_{slow}$',
    'tau_not': r'$\tau_{NOT}$',
    'B_omega': r'$B_{\omega}$',
    'B_a': r'$B_a$',
}
prcc_sorted = sorted(prcc_rows, key=lambda r: abs(float(r['prcc'])))
y = np.arange(len(prcc_sorted))
prcc_vals = np.array([float(r['prcc']) for r in prcc_sorted])
barcolors = np.where(prcc_vals >= 0, '#4C78A8', '#F58518')
ax.barh(y, prcc_vals, color=barcolors, height=.72)
ax.axvline(0, color='0.25', lw=.8)
ax.set_yticks(y, [label_map.get(r['parameter'], r['parameter']) for r in prcc_sorted])
ax.tick_params(axis='y', pad=10)
ax.set_xlim(-1, 1)
ax.set_xlabel('Partial rank correlation')
ax.set_title('C  Parameter influence', loc='left', fontweight='bold')
ax.grid(axis='x', alpha=.2, lw=.5)

fig.subplots_adjust(wspace=.80, left=.09, right=.98, bottom=.22, top=.86)
save(fig, 'Figure_5_uncertainty_parameter_robustness')

# Supplementary Figure S4: separate eye perturbations
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
settings = [
    ('noise', 'Eye-signal noise (% RMS eye velocity)', 'A'),
    ('gain', 'Eye-signal gain', 'B'),
    ('delay', 'Eye-signal delay (ms)', 'C'),
    ('bias', 'Eye-signal bias (% peak eye velocity)', 'D'),
]
for ax, (fam, xlab, panel) in zip(axes.ravel(), settings):
    g = [r for r in eye if r['type'] == fam]
    if not g: continue
    g_sorted = sorted(g, key=lambda r: float(r['val']))
    x = np.array([float(r['val']) for r in g_sorted])
    y_vals = np.array([float(r['relative_to_ref']) for r in g_sorted])
    lo = np.array([float(r['ci_lower']) for r in g_sorted]) / ref_eye
    hi = np.array([float(r['ci_upper']) for r in g_sorted]) / ref_eye
    
    ax.errorbar(x, y_vals, yerr=[y_vals - lo, hi - y_vals], fmt='o-', capsize=2.2, lw=1.2, ms=4, color=colors[fam])
    ax.axhline(1, color='0.45', ls='--', lw=.8)
    ax.axhline(0, color='0.25', lw=.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel('Effect relative to exact-eye reference')
    ax.set_title(panel, loc='left', fontweight='bold')
    ax.set_ylim(0, 1.13)
    ax.grid(axis='y', alpha=.2, lw=.5)

fig.subplots_adjust(wspace=.33, hspace=.42, left=.10, right=.98, bottom=.10, top=.96)
save(fig, 'Figure_S4_eye_signal_single_perturbations')

# Supplementary Figure S5: combined grid heatmaps
import re
comb = [r for r in eye if r['type'] == 'combined']
for r in comb:
    match = re.search(r'N([0-9.]+)_G([0-9.]+)_D([0-9.]+)', r['val'])
    if match:
        r['noise_pct'] = float(match.group(1))
        r['gain_factor'] = float(match.group(2))
        r['delay_ms'] = float(match.group(3))

# export combined eye grid csv
write_csv(srcdir / 'Figure_S5_combined_eye_grid.csv', 
          list(comb[0].keys()) if comb else [], comb)

noise_levels = [0, 10, 20, 40]
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), sharex=True, sharey=True)
for ax, n, panel in zip(axes.ravel(), noise_levels, list('ABCD')):
    g = [r for r in comb if r.get('noise_pct') == n]
    if not g: continue
    
    gain_factors = sorted(list(set(r['gain_factor'] for r in g)), reverse=True)
    delay_ms = sorted(list(set(r['delay_ms'] for r in g)))
    
    # build pivot grids
    piv = np.zeros((len(gain_factors), len(delay_ms)))
    stat = np.empty((len(gain_factors), len(delay_ms)), dtype=object)
    
    for r in g:
        iy = gain_factors.index(r['gain_factor'])
        ix = delay_ms.index(r['delay_ms'])
        piv[iy, ix] = float(r['relative_to_ref'])
        stat[iy, ix] = r['status']
        
    im = ax.imshow(piv, aspect='auto', vmin=.3, vmax=1.05, cmap='viridis', origin='upper')
    ax.set_xticks(range(len(delay_ms)), [f'{x:g}' for x in delay_ms])
    ax.set_yticks(range(len(gain_factors)), [f'{x:.1f}' for x in gain_factors])
    ax.set_title(f'{panel}  Noise = {n}% RMS', loc='left', fontweight='bold')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Gain')
    
    for iy in range(len(gain_factors)):
        for ix in range(len(delay_ms)):
            val = piv[iy, ix]
            txt = f'{val:.2f}'
            color = 'white' if val < .65 else 'black'
            ax.text(ix, iy, txt, ha='center', va='center', fontsize=7, color=color)
            if stat[iy, ix] == 'ATTENUATED':
                ax.scatter(ix, iy, s=115, facecolors='none', edgecolors='#D62728', linewidths=1.1)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.03)
cbar.set_label('Effect relative to exact-eye reference')
fig.subplots_adjust(wspace=.20, hspace=.28, left=.09, right=.88, bottom=.10, top=.96)
save(fig, 'Figure_S5_eye_signal_combined_grid')

print(f'Created figures in {figdir}')
print(f'Created source data in {srcdir}')
