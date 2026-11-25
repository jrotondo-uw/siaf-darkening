from pathlib import Path
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
OUT = R / 'figures'
OUT.mkdir(exist_ok=True)
BOOT_CSV = R.parent / 'source_data' / 'SourceData_Figure3_Bootstrap.csv'
boot_df = pd.read_csv(BOOT_CSV)

plt.rcParams.update({'font.size': 15.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
d = pd.read_csv(R / 'siaf_pixelwise_summary_monthly.csv')
d = d[np.isclose(d.pval, .05)]
months = ['April', 'May', 'June', 'July', 'August', 'September']
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
rng = np.random.default_rng(20261125)
ymin, ymax = 0.0, 0.105

for ax, (method, period) in zip(axs.flat, [(m, p) for m in ['CERES', 'Kernel'] for p in ['CMIP6', 'Obs-era']]):
    s = d.query('rs_method==@method and period==@period').set_index('month').loc[months]
    bsub = boot_df[(boot_df.rs_method == method) & (boot_df.period == period) & np.isclose(boot_df.pval, 0.05)]
    for m_idx, mo in enumerate(months):
        dm = bsub.loc[bsub.month == mo, 'SIAF_tot_global'].to_numpy(float) / 6.0
        dm = dm[np.isfinite(dm)]
        assert np.isclose(2 * np.std(dm, ddof=0), float(s.sigma2_tot_global.iloc[m_idx]) / 6.0, atol=1e-5)
        ymin = min(ymin, float(dm.min()) - 0.004)
        ymax = max(ymax, float(dm.max()) + 0.005)
        ax.scatter(-0.19 + m_idx + rng.uniform(-0.065, 0.065, len(dm)), dm,
                   s=6, color='#555555', alpha=0.12, edgecolors='none', zorder=2, rasterized=True)
    for k, label, col, off, mk in [('tot', 'Total', '#222222', -.19, 'o'),
                                   ('ext', 'Area loss', '#3A6BAF', 0, 's'),
                                   ('drk', 'Darkening', '#D95F1A', .19, '^')]:
        y = s[f'SIAF_{k}_global'] / 6
        x = np.arange(6) + off
        if k == 'tot':
            ax.errorbar(x, y, yerr=s.sigma2_tot_global / 6, fmt='none', ecolor=col,
                        capsize=3, lw=1.2, capthick=1.1, zorder=4)
        ax.plot(x, y, ls='none', marker=mk, color=col, markeredgecolor='white',
                markeredgewidth=0.5, label=label, markersize=6.2, zorder=5)
    ax.set_title(('CERES APRP' if method == 'CERES' else 'ERA5 kernel') + ' · ' +
                 ('1982–2014' if period == 'CMIP6' else '1982–2024'), fontsize=15.5, loc='left')
    ax.axhline(0, color='.5', lw=.6)
    ax.grid(axis='y', color='.9')
    ax.set_axisbelow(True)
    ax.set_xticks(range(6), [m[:3] for m in months])
    ax.tick_params(labelsize=15.5)

for ax in axs.flat:
    ax.set_ylim(ymin, ymax)

fig.supylabel('Contribution to annual SIAF (W m$^{-2}$ K$^{-1}$)', x=0.015)
for i, ax in enumerate(axs.flat):
    ax.text(-.1, 1.12, chr(97 + i), transform=ax.transAxes, weight='bold', fontsize=15.5)

h, l = axs[0, 0].get_legend_handles_labels()
h.append(Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555',
                markeredgecolor='none', markersize=4.5, alpha=0.55, label='Pairs bootstrap'))
l.append('Pairs bootstrap')
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=14.0, columnspacing=1.2)
fig.subplots_adjust(top=.86, bottom=.08, left=.135, right=.99, hspace=.30, wspace=.12)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'f03.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
