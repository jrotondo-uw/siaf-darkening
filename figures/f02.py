#!/usr/bin/env python
"""
Figure 1 (revision) — pan-Arctic SIAF partitioned into area loss and darkening.

Reviewer 2 read this figure as a model-observation comparison, which it never
was: both bars in each pair are observational, evaluated over two periods. The
submitted legend said "(CMIP6)" and "(Obs-era)", which is what caused it.

Changes:
  * every period label now says "Observations"; the legend carries components
    only, with the hatch explained as the period
  * an explicit statement on the panel that no model output appears here
  * individual bootstrap replicates overlaid on each total (editor requirement
    for a mean shown with error bars)
  * larger type, 300 dpi
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CSV_IN   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
BOOT_NPZ = '/glade/derecho/scratch/jrotondo/rev_out/fig1_bootstrap_replicates.npz'
OUT_DIR  = '/glade/derecho/scratch/jrotondo/paper_revision_20261024/figures'
os.makedirs(OUT_DIR, exist_ok=True)
PLOT_OUT = os.path.join(OUT_DIR, 'f01.png')
DPI = 300

matplotlib.rcParams.update({
    'font.size':       13,
    'axes.labelsize':  14.5,
    'xtick.labelsize': 13,
    'ytick.labelsize': 12.5,
    'legend.fontsize': 12,
})

COL_EXT, COL_DRK = '#3A6BAF', '#D95F1A'
COL_PT = '#2b2b2b'
ALPHA = {'CMIP6': 0.60, 'Obs-era': 1.00}
HATCH = {'CMIP6': '////', 'Obs-era': ''}
PERIOD_LABEL = {'CMIP6': 'Observations\n1982–2014',
                'Obs-era': 'Observations\n1982–2024'}

methods = [('CERES', 'CERES APRP\n(Donohoe et al., 2020)'),
           ('Kernel', 'ERA5 radiative kernel\n(Huang & Huang, 2023)')]
periods = ['CMIP6', 'Obs-era']

df = pd.read_csv(CSV_IN)
df = df[np.isclose(df['pval'], 0.05)].copy()

boot = {}
if os.path.exists(BOOT_NPZ):
    z = np.load(BOOT_NPZ)
    boot = {k: z[k] for k in z.files}
    print('bootstrap replicates:', {k: v.shape for k, v in boot.items()})

n = len(methods)
x = np.arange(n)
w = 0.32
offs = {'CMIP6': -(w / 2 + 0.035), 'Obs-era': (w / 2 + 0.035)}
rng = np.random.default_rng(3)

fig, ax = plt.subplots(figsize=(10.5, 7.4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ymax = 0.0
for period in periods:
    a, hat = ALPHA[period], HATCH[period]
    for i, (rs, _) in enumerate(methods):
        row = df[(df.rs_method == rs) & (df.period == period)]
        if row.empty:
            print(f'  WARNING: no row for {rs}/{period}')
            continue
        tot = float(row.SIAF_tot_global.iloc[0])
        ext = float(row.SIAF_ext_global.iloc[0])
        drk = float(row.SIAF_drk_global.iloc[0])
        err = float(row.sigma2_tot_global.iloc[0])
        xp = x[i] + offs[period]

        ax.bar(xp, ext, w, color=COL_EXT, alpha=a, hatch=hat,
               edgecolor='white', linewidth=0.6, zorder=3)
        ax.bar(xp, drk, w, bottom=ext, color=COL_DRK, alpha=a, hatch=hat,
               edgecolor='white', linewidth=0.6, zorder=3)
        ax.errorbar(xp, tot, yerr=err, fmt='none', color='#222222',
                    capsize=6, linewidth=1.8, capthick=1.8, zorder=5)

        reps = boot.get(f'{rs}_{period}')
        if reps is not None:
            vals = reps[:, 0]
            vals = vals[np.isfinite(vals)]
            if vals.size > 300:
                vals = rng.choice(vals, 300, replace=False)
            # strip beside the bar, so the in-bar percentages stay legible
            jit = rng.uniform(-w * 0.11, w * 0.11, size=vals.size)
            ax.scatter(xp + w * 0.66 + jit, vals, s=7, facecolor=COL_PT,
                       edgecolor='none', alpha=0.32, zorder=6, rasterized=True)
            ymax = max(ymax, float(np.nanmax(vals)))

        for val, base, col in ((ext, 0.0, COL_EXT), (drk, ext, COL_DRK)):
            if val > 0.004:
                ax.text(xp - w * 0.24, base + val / 2, f'{val / tot * 100:.0f}%',
                        ha='center', va='center', color='white',
                        fontsize=12.5, fontweight='bold', zorder=7)

        ax.text(xp - w * 0.08, tot + err + 0.006, PERIOD_LABEL[period],
                ha='center', va='bottom', color='#333333', fontsize=11.5)
        ymax = max(ymax, tot + err)

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in methods])
ax.set_ylabel('Pan-Arctic annual SIAF  (W m$^{-2}$ K$^{-1}$)')
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(0, ymax * 1.30)
ax.yaxis.grid(True, alpha=0.35, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax.set_title('Satellite observations only: both bars in each pair are '
             'observational estimates,\nevaluated over two periods with the '
             'same method. No model output appears in this figure.',
             fontsize=12.5, color='#333333', pad=14, loc='left')

handles = [
    mpatches.Patch(facecolor=COL_EXT, edgecolor='white',
                   label=r'SIAF$_{\mathrm{area}}$ (sea ice area loss)'),
    mpatches.Patch(facecolor=COL_DRK, edgecolor='white',
                   label=r'SIAF$_{\mathrm{drk}}$ (darkening of surviving ice)'),
    mpatches.Patch(facecolor='#BBBBBB', hatch='////', edgecolor='white',
                   label='Hatched: 1982–2014   Solid: 1982–2024'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_PT,
           markeredgecolor='none', markersize=5, alpha=0.5,
           label='Bootstrap replicates (n = 1000)'),
]
ax.legend(handles=handles, loc='upper right', framealpha=0.95,
          handlelength=1.6, borderpad=0.7)

fig.tight_layout()
fig.savefig(PLOT_OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'-> {PLOT_OUT}  (dpi={DPI})')
