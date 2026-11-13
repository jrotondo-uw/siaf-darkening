# %%
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

PKL_PATH = ('/glade/work/jrotondo/ncc_paper/plots/'
            'extended_data_figs/plot_data_covariance.pkl')
OUT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/extended_data_figs/'

# ── load ──────────────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as fh:
    d = pickle.load(fh)

monthly_res_ext  = d['monthly_res_ext']
monthly_dalb_drk = d['monthly_dalb_drk']
monthly_dalb_ext = d['monthly_dalb_ext']
monthly_dalb_sic = d['monthly_dalb_sic']
monthly_dalb_res = d['monthly_dalb_res']
monthly_meta     = d['monthly_meta']
MONTHS           = d['MONTHS']

# ── style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'legend.title_fontsize': 6,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.2, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

COL_EXT = '#3A7DC9'
COL_RES = '#D95F1A'
MONTH_COLORS = {4:'#2166AC', 5:'#4393C3', 6:'#92C5DE',
                7:'#F4A582', 8:'#D6604D', 9:'#B2182B'}

# ── figure ────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(6.30, 2.76),
    gridspec_kw={'wspace': 0.42, 'left': 0.09, 'right': 0.98,
                 'bottom': 0.18, 'top': 0.88}, dpi=300,
)

# panel (a) — KDE of residual darkening
x_kde = np.linspace(-0.38, 0.28, 600)
for mn in sorted(monthly_res_ext):
    res = monthly_res_ext[mn][np.isfinite(monthly_res_ext[mn])]
    col = MONTH_COLORS[mn]
    kde = gaussian_kde(res, bw_method='silverman')
    y   = kde(x_kde)
    ax.plot(x_kde, y, color=col, lw=1.4,
            label=f"{MONTHS[mn][:3]}  (med={np.median(res):+.3f})")
    ax.fill_between(x_kde, y, where=(x_kde < 0), color=col, alpha=0.10, lw=0)
    if mn in monthly_dalb_drk:
        drk = monthly_dalb_drk[mn][np.isfinite(monthly_dalb_drk[mn])]
        if len(drk) > 50:
            ax.plot(x_kde, gaussian_kde(drk, bw_method='silverman')(x_kde),
                    color=col, lw=0.7, ls=':', alpha=0.55)
    pct = (res < 0).mean() * 100
    print(f"{MONTHS[mn]}: {pct:.0f}% negative residual  |  "
          f"α_ice={monthly_meta[mn]['alpha_ice']:.3f}  "
          f"α_ocn={monthly_meta[mn]['alpha_ocn']:.3f}  "
          f"n_ext={monthly_meta[mn]['n_ext_px']:,}")

ax.axvline(0, color='0.3', lw=0.7, ls='--', zorder=5)
ax.set_xlabel(
    r'Residual darkening in area pixels,'
    '\n'
    r'$\Delta\alpha_\mathrm{res}=\Delta\mathrm{SIAL}'
    r'-(\bar{\alpha}_\mathrm{ice}-\bar{\alpha}_\mathrm{ocn})\,\Delta f$',
    labelpad=3)
ax.set_ylabel('Probability density')
ax.set_title('a)', loc='left', pad=4)
ax.set_xlim(-0.38, 0.28)
ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
leg = ax.legend(loc='upper left', frameon=False, title='Month (median)',
                handlelength=1.2, handletextpad=0.4, labelspacing=0.25)
leg.get_title().set_fontsize(6)

# panel (b) — stacked bar decomposition
mn_list  = sorted(monthly_dalb_ext)
x_pos    = np.arange(len(mn_list))
tot_vals = np.array([monthly_dalb_ext[m] for m in mn_list])
sic_vals = np.array([monthly_dalb_sic[m] for m in mn_list])
res_vals = np.array([monthly_dalb_res[m] for m in mn_list])

ax2.bar(x_pos, sic_vals, width=0.55, color=COL_EXT, zorder=3,
        label=r'SIC-driven  $[(\bar{\alpha}_\mathrm{ice}-\bar{\alpha}_\mathrm{ocn})\cdot\Delta f]$')
ax2.bar(x_pos, res_vals,  width=0.55, bottom=sic_vals, color=COL_RES, zorder=3,
        label=r'Residual  $[\Delta\mathrm{SIAL}-\mathrm{SIC\text{-}driven}]$')

for i, (sv, rv, tv) in enumerate(zip(sic_vals, res_vals, tot_vals)):
    if tv != 0 and np.isfinite(rv) and abs(rv / tv) > 0.02:
        ax2.text(x_pos[i], sv + rv / 2, f'{rv/tv*100:.0f}%',
                 ha='center', va='center', fontsize=5.5,
                 color='white', fontweight='bold', zorder=6)

ax2.axhline(0, color='0.6', lw=0.5, zorder=1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([MONTHS[m][:3] for m in mn_list])
ax2.set_ylabel(r'Mean $\Delta$SIAL in area pixels')
ax2.set_title('b)', loc='left', pad=4)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.legend(loc='lower left', frameon=False,
           handlelength=1.0, handletextpad=0.4, labelspacing=0.3)

# ── save ──────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    out = f'{OUT_DIR}/fe_covariance.{ext}'
    fig.savefig(out)
    print(f'Saved → {out}')
plt.show()