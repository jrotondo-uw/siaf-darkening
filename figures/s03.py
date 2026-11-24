"""Supplementary Figure 3 — absolute two-sided classification leakage, W m-2.

Reads leakage_flux.csv (written by compute_leakage_flux.py); draws only.
Replaces the earlier relative-share version, kept as s03_relative_superseded.py.

Font sizes follow the convention of the rest of the set: the figure is drawn
10 in wide and scaled to the 5.147 in column, so 15.5 pt renders at 8.0 pt.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
OUT = R / 'figures'
OUT.mkdir(exist_ok=True)

FS = 15.5
plt.rcParams.update({'font.size': FS, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

BLUE, ORANGE = '#3A6BAF', '#D95F1A'
d = pd.read_csv(R / 'leakage_flux.csv')
months = [m[:3] for m in d.month]
x = range(len(d))

PANELS = [('mon', 'Monthly ice albedo $-$ 0.06'),
          ('fix', 'Fixed contrast: 0.54')]

fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

for i, (ax, (tag, title)) in enumerate(zip(axs, PANELS)):
    ax.plot(x, d[f'blue_{tag}'], marker='o', ms=7, lw=2.0, color=BLUE)
    ax.plot(x, d[f'orange_{tag}'], marker='^', ms=8, lw=2.0, color=ORANGE)
    ax.axhline(0, color='.6', lw=.8)
    ax.set_xticks(list(x), months)
    ax.set_title(f'{chr(97 + i)}  {title}', loc='left', fontsize=FS)
    ax.grid(axis='y', color='.92')
    ax.set_axisbelow(True)

    res = d[f'blue_{tag}'].sum()
    sic = d[f'orange_{tag}'].sum()
    ax.text(.02, .97,
            f'Annual sums (W m$^{{-2}}$)\nResidual: {res:.5f}\n'
            f'Concentration: {sic:.5f}\nNet to darkening: {res - sic:.5f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=FS - 1.5,
            linespacing=1.35)

# headroom for the four-line annotation, measured from the larger panel
ymax = max(d.blue_mon.max(), d.orange_mon.max())
ymin = min(0., d.blue_fix.min(), d.orange_fix.min())
axs[0].set_ylim(ymin - .05 * ymax, ymax * 1.78)

axs[0].set_ylabel('Monthly contribution to annual\nflux change (W m$^{-2}$)')

fig.legend(handles=[plt.Line2D([], [], color=BLUE, marker='o', ms=7, lw=2.0,
                               label='Residual in declining-ice region'),
                    plt.Line2D([], [], color=ORANGE, marker='^', ms=8, lw=2.0,
                               label='Concentration in darkening region')],
           loc='lower center', ncol=2, frameon=False, fontsize=FS,
           bbox_to_anchor=(.5, -.02))
fig.subplots_adjust(top=.92, bottom=.22, left=.13, right=.98, wspace=.10)

for ext in ['png', 'pdf']:
    fig.savefig(OUT / f's03.{ext}', dpi=300, bbox_inches='tight',
                facecolor='white')
plt.close(fig)
print('wrote', OUT / 's03.png')
