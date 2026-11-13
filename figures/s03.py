"""
Supplementary Figure 3: Absolute two-sided classification leakage.
Plots monthly contributions to the annual epoch-change flux in declining-ice
and darkening regions for both monthly ice albedo and fixed 0.54 contrast.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data_file = ROOT / 'source_data' / 'SourceData_Supplementary_Figure3.csv'
d = pd.read_csv(data_file)

# Compute annual sums
a = d.groupby('variant')[['residual_in_ext', 'concentration_in_drk', 'net_reassignment_to_darkening']].sum() / 6.0

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
f, axs = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True, layout='constrained')

for ax, v, title in zip(axs, ['data_contrast', 'fixed_054'], ['a  Monthly ice albedo − 0.06', 'b  Fixed contrast: 0.54']):
    s = d[d.variant == v].sort_values('month')
    ax.plot(s.month, s.residual_in_ext / 6.0, 'o-', color='#0072B2', label='Residual in declining-ice region')
    ax.plot(s.month, s.concentration_in_drk / 6.0, '^-', color='#D55E00', label='Concentration in darkening region')
    ax.set_xticks(range(4, 10), ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
    ax.set_title(title, loc='left', fontsize=10)
    ax.axhline(0, color='0.6', lw=0.7)
    ax.grid(axis='y', alpha=0.2)
    r = a.loc[v]
    ax.text(0.03, 0.97,
            f"Annual sums (W m$^{{-2}}$)\nResidual: {r.residual_in_ext:.5f}\nConcentration: {r.concentration_in_drk:.5f}\nNet to darkening: {r.net_reassignment_to_darkening:.5f}",
            transform=ax.transAxes, va='top', fontsize=9)
    ax.set_ylim(-0.0008, 0.014)

axs[0].set_ylabel('Monthly contribution to annual flux change (W m$^{-2}$)')
f.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', fontsize=9)

out_png = ROOT / 'figures' / 'fs03.png'
out_pdf = ROOT / 'figures' / 'fs03.pdf'
f.savefig(out_png, dpi=600, bbox_inches='tight')
f.savefig(out_pdf, bbox_inches='tight')
print(f"Saved {out_png} and {out_pdf}")
