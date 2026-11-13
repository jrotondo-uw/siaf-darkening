#!/usr/bin/env python
"""
plot_sial_darkening.py
======================
Loads sial_darkening_ts.csv and plots monthly albedo trends
on darkening pixels (1982–2024).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/sial_darkening_ts.csv'
output_dir = '/glade/work/jrotondo/ncc_paper/plots/'
os.makedirs(output_dir, exist_ok=True)

PVAL   = 0.05
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']
COLORS = {'April': '#e41a1c', 'May': '#ff7f00', 'June': '#4daf4a',
          'July': '#377eb8', 'August': '#984ea3', 'September': '#a65628'}

plt.rcParams.update({'font.size': 18, 'axes.linewidth': 1.1})

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, index_col='year')

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor('white')
ax.grid(True, alpha=0.25, linestyle='--', color='#cccccc', zorder=0)

for month in MONTHS:
    col  = df[month].dropna()
    if len(col) < 5:
        print(f"  [!] Skipping {month} — only {len(col)} years")
        continue
    yrs  = col.index.values.astype(float)
    vals = col.values
    sl, ic, r, p, _ = stats.linregress(yrs, vals)
    ax.scatter(yrs, vals, color=COLORS[month], s=20, alpha=0.6, zorder=3)
    ax.plot(yrs, sl * yrs + ic, color=COLORS[month],
            lw=2.0, linestyle='-' if p < PVAL else '--', zorder=4,
            label=f"{month:<10} {sl*10:+.4f} dec⁻¹   R²={r*r:.2f}"
                  f"   {'*' if p < PVAL else 'ns'}")

ax.set_xlabel('Year', fontsize=18)
ax.set_ylabel(r'Mean $\alpha_{\mathrm{drk}}$', fontsize=18)
ax.set_xlim(1981, 2025)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.tick_params(labelsize=18)

# NOTE: when using prop=, put 'size' inside the dict — a separate fontsize=
# kwarg is silently ignored once prop= is provided.
ax.legend(loc='upper right', framealpha=0.95,
          title='Month          slope              fit   sig',
          title_fontsize=18,
          prop={'family': 'monospace', 'size': 14})

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
fpath = os.path.join(output_dir, 'sial_darkening_all_months_full.png')
plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {fpath}")