"""Main Figure 2: observed SIAF decomposition with distributional realizations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
SD = R.parent / 'source_data'
(R / 'figures').mkdir(exist_ok=True)

d = pd.read_csv(SD / 'SourceData_Figure2.csv')
boot = pd.read_csv(SD / 'SourceData_Figure2_Bootstrap.csv')

print(d[['period', 'rs_method', 'SIAF_tot_global', 'sigma2_tot_global']].to_string(index=False))

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
})

fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 4.7), gridspec_kw={'width_ratios': [2.1, 1]})
rng = np.random.default_rng(20261125)
labels = []

SPECS = [
    (-0.17, 'tot', 'total', '#111111', 'o', 'Total'),
    (0.0, 'ext', 'area_loss', '#3A6BAF', 's', 'Area loss'),
    (0.17, 'drk', 'darkening', '#D95F1A', '^', 'Darkening'),
]

for i, (method, period, boot_period) in enumerate([
    ('CERES', 'CMIP6', '1982-2014'),
    ('CERES', 'Obs-era', '1982-2024'),
    ('Kernel', 'CMIP6', '1982-2014'),
    ('Kernel', 'Obs-era', '1982-2024'),
]):
    r = d[(d.rs_method == method) & (d.period == period)].iloc[0]
    bsub = boot[(boot.rs_method == method) & (boot.period == boot_period)]
    labels.append(('CERES APRP' if method == 'CERES' else 'ERA5 kernel') + '\n' + boot_period.replace('-', '\u2013'))
    is_hist = (boot_period == '1982-2014')
    ddof = 1 if is_hist else 0
    alpha = 0.022 if is_hist else 0.10

    for off, col, bcol, c, mark, leg in SPECS:
        draws = bsub[bcol].to_numpy(float)
        draws = draws[np.isfinite(draws)]
        np.testing.assert_allclose(2 * draws.std(ddof=ddof), float(r[f'sigma2_{col}_global']), atol=1e-4, rtol=0)
        y_jit = i + off + rng.uniform(-0.055, 0.055, size=len(draws))
        a.scatter(draws, y_jit, s=5, color=c, alpha=alpha, edgecolors='none', zorder=2, rasterized=True)
        val = float(r[f'SIAF_{col}_global'])
        err = float(r[f'sigma2_{col}_global'])
        a.errorbar(val, i + off, xerr=err, fmt='none', ecolor='white',
                   elinewidth=3.4, capsize=5.8, capthick=2.6, zorder=4)
        a.errorbar(val, i + off, xerr=err, fmt=mark, color=c, ecolor=c,
                   ms=6.2, markeredgecolor='white', markeredgewidth=0.9,
                   elinewidth=1.8, capsize=4.8, capthick=1.5, zorder=5,
                   label=leg if i == 0 else None)

    b.barh(i, r.ext_pct, color='#3A6BAF', height=0.65)
    b.barh(i, r.drk_pct, left=r.ext_pct, color='#D95F1A', height=0.65)
    b.text(r.ext_pct / 2, i, f'{r.ext_pct:.0f}%', va='center', ha='center', color='white', weight='bold')
    b.text(r.ext_pct + r.drk_pct / 2, i, f'{r.drk_pct:.0f}%', va='center', ha='center', color='white', weight='bold')

a.set_yticks(range(4), labels)
b.set_yticks(range(4), [''] * 4)
for ax in (a, b):
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.set_axisbelow(True)

a.set_xlabel('Annual SIAF (W m$^{-2}$ K$^{-1}$)')
a.set_title('a  Observed feedback', loc='left')
a.set_xlim(0, 0.345)
a.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
b.set_title('b  Component shares', loc='left')
b.set_xlabel('Share of total (%)')
b.set_xlim(0, 100)
b.set_xticks([0, 25, 50, 75, 100])

fig.legend(*a.get_legend_handles_labels(), loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.55, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(R / 'figures/f02.png', dpi=300)
fig.savefig(R / 'figures/f02.pdf')
plt.close(fig)
print('wrote', R / 'figures/f02.png')
