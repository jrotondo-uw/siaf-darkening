from pathlib import Path
"""Threshold sensitivity: dot-and-interval plots with threshold-specific bootstrap clouds."""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
D = pd.read_csv(R / 'siaf_pixelwise_summary.csv')
boot_df = pd.read_csv(R.parent / 'source_data' / 'SourceData_Supplementary_Figure1_Bootstrap.csv')

plt.rcParams.update({'font.size': 11.2, 'pdf.fonttype': 42})
fig, axs = plt.subplots(4, 2, figsize=(7.2, 9), gridspec_kw={'width_ratios': [1.5, 1]}, layout='constrained')
rng = np.random.default_rng(20261125)
xmax = 0.30

for j, (period, method, title) in enumerate([
    ('CMIP6', 'CERES', '1982–2014 · CERES APRP'),
    ('CMIP6', 'Kernel', '1982–2014 · ERA5 kernel'),
    ('Obs-era', 'CERES', '1982–2024 · CERES APRP'),
    ('Obs-era', 'Kernel', '1982–2024 · ERA5 kernel'),
]):
    sub = D[D.period.eq(period) & D.rs_method.eq(method)].sort_values('pval')
    sub = sub[np.isclose(sub.pval, .01) | np.isclose(sub.pval, .05) | np.isclose(sub.pval, .1)]
    pvals = [0.01, 0.05, 0.10]
    for k, (c, label, color, marker) in enumerate([
        ('tot', 'Total', '#333333', 'o'),
        ('ext', 'Area loss', '#3A6BAF', 's'),
        ('drk', 'Darkening', '#D95F1A', '^'),
    ]):
        y = np.arange(3) + (k - 1) * .18
        for p_idx, pv in enumerate(pvals):
            bsub = boot_df[(boot_df.rs_method == method) & (boot_df.period == period) & np.isclose(boot_df.pval, pv)]
            draws = bsub[f'SIAF_{c}_global'].to_numpy(float)
            assert np.isclose(2 * np.nanstd(draws, ddof=0), float(sub.iloc[p_idx][f'sigma2_{c}_global']), atol=1e-5)
            xmax = max(xmax, float(np.nanmax(draws)) + 0.01)
            axs[j, 0].scatter(draws, y[p_idx] + rng.uniform(-0.052, 0.052, len(draws)),
                              s=5, color=color, alpha=0.075, edgecolors='none', zorder=2, rasterized=True)
        axs[j, 0].errorbar(sub[f'SIAF_{c}_global'], y, xerr=sub[f'sigma2_{c}_global'],
                           fmt=marker, color=color, markeredgecolor='white', markeredgewidth=0.5,
                           ms=4.5, capsize=2, lw=1.2, label=label, zorder=4)
        if c != 'tot':
            axs[j, 1].scatter(100 * sub[f'SIAF_{c}_global'] / sub.SIAF_tot_global, y, c=color, marker=marker, s=20)
    for col in range(2):
        ax = axs[j, col]
        ax.set_yticks(range(3), ['p < 0.01', 'p < 0.05', 'p < 0.10'] if col == 0 else [])
        ax.set_ylim(2.5, -.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
        ax.set_title(f'{chr(97 + 2 * j + col)}  ' + (title if col == 0 else 'Component share'), loc='left', fontsize=11.2)
    axs[j, 1].set_xlim(0, 100)

for j in range(4):
    axs[j, 0].set_xlim(0, xmax)

h_, l_ = axs[0, 0].get_legend_handles_labels()
h_.append(Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555',
                 markeredgecolor='none', markersize=4.5, alpha=0.55, label='Pairs bootstrap'))
l_.append('Pairs bootstrap')
fig.legend(h_, l_, loc='outside upper center', ncol=4, fontsize=11.2, frameon=False)
axs[-1, 0].set_xlabel('SIAF (W m$^{-2}$ K$^{-1}$)')
axs[-1, 1].set_xlabel('Percent of total')
(R / 'figures').mkdir(exist_ok=True)
fig.savefig(R / 'figures/s01.png', dpi=300, bbox_inches='tight', facecolor='white')
