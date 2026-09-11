from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

eye = read_csv(HERE / 'Figure_5A_eye_uncertainty_conditions.csv')
par = read_csv(HERE / 'Figure_5B_parameter_effect_distribution.csv')
prcc_rows = read_csv(HERE / 'Figure_5C_parameter_importance.csv')
ref_par = next(float(r['mean_effect']) for r in eye if r['type'] == 'noise' and str(r['val']) == '0')

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

fig, axes = plt.subplots(1, 3, figsize=(8.50, 3.20), gridspec_kw={'width_ratios':[1.08,1.08,1.25]})

ax = axes[0]
families = ['noise','gain','delay','bias','combined']
labels = ['Noise','Gain','Delay','Bias','Combined']
colors = {'noise':'#4C78A8','gain':'#F58518','delay':'#54A24B','bias':'#B279A2','combined':'#6B6ECF'}
rng = np.random.default_rng(18)
for i, fam in enumerate(families):
    g = [r for r in eye if r['type'] == fam]
    if not g:
        continue
    rel_eff = np.array([float(r['relative_to_ref']) for r in g])
    robust = np.array([r['status'] == 'ROBUST' for r in g])
    x = i + rng.uniform(-0.18, 0.18, len(g))
    ax.scatter(x[robust], rel_eff[robust], s=12, color=colors[fam], alpha=.72, edgecolors='none', zorder=2)
    if (~robust).any():
        ax.scatter(x[~robust], rel_eff[~robust], s=25, facecolors='white', edgecolors='#C83E4D', linewidths=.9, zorder=3)
    med = np.median(rel_eff)
    q1, q3 = np.percentile(rel_eff, [25, 75])
    ax.plot([i-.22, i+.22], [med, med], color='black', lw=1.2, zorder=4)
    ax.plot([i, i], [q1, q3], color='black', lw=1.0, zorder=4)
ax.axhline(1.0, color='0.45', ls='--', lw=.8)
ax.axhline(0.0, color='0.25', lw=.8)
ax.set_xticks(range(len(labels)), labels, rotation=40, ha='right')
ax.set_ylabel('Effect relative to exact-eye reference')
ax.set_ylim(0, 1.13)
ax.set_title('A  Eye-signal uncertainty', loc='left', fontweight='bold')
ax.text(0.98, 0.05, '112 robust; 12 attenuated', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2)
ax.legend(handles=[
    Line2D([0], [0], marker='o', color='none', markerfacecolor='0.45', markeredgecolor='none', markersize=4.5, label='Robust'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#C83E4D', markersize=5, label='Attenuated')
], frameon=False, loc='lower left', bbox_to_anchor=(-.02, .08))

ax = axes[1]
par_mean_effect = np.array([float(r['mean_effect']) for r in par])
bins = np.linspace(par_mean_effect.min()-0.001, par_mean_effect.max()+0.001, 28)
ax.hist(par_mean_effect, bins=bins, color='#4C78A8', alpha=.82, edgecolor='white', linewidth=.4)
ax.axvline(0, color='#C83E4D', lw=1.0)
ax.axvline(0.25 * ref_par, color='#F58518', ls='--', lw=1.0)
ax.axvline(ref_par, color='0.25', ls=':', lw=1.0)
ax.set_xlabel('Mean paired reliability effect (model units)')
ax.set_ylabel('Parameter configurations')
ax.set_title('B  Global parameter sensitivity', loc='left', fontweight='bold')
ax.set_ylim(0, 105)
# Numerical distribution statistics are reported in the caption rather than duplicated in the panel.
ax.legend(handles=[
    Line2D([0], [0], color='#C83E4D', lw=1, label='Zero'),
    Line2D([0], [0], color='#F58518', lw=1, ls='--', label='25% reference'),
    Line2D([0], [0], color='0.25', lw=1, ls=':', label='Default reference'),
], frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.99), borderaxespad=0.0,
   handlelength=2.0, handletextpad=0.6, labelspacing=0.45)

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
ax.barh(y, prcc_vals, color=np.where(prcc_vals >= 0, '#4C78A8', '#F58518'), height=.72)
ax.axvline(0, color='0.25', lw=.8)
ax.set_yticks(y, [label_map.get(r['parameter'], r['parameter']) for r in prcc_sorted])
ax.tick_params(axis='y', pad=10)
ax.set_xlim(-1, 1)
ax.set_xlabel('Partial rank correlation')
ax.set_title('C  Parameter influence', loc='left', fontweight='bold')
ax.grid(axis='x', alpha=.2, lw=.5)

fig.subplots_adjust(wspace=.80, left=.09, right=.98, bottom=.22, top=.86)
fig.savefig(HERE / 'Figure_5_uncertainty_parameter_robustness.pdf', bbox_inches='tight')
fig.savefig(HERE / 'Figure_5_uncertainty_parameter_robustness.png', dpi=600, bbox_inches='tight')
plt.close(fig)
