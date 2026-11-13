# %%
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

PKL_PATH = '/glade/work/jrotondo/ncc_paper/plots/plot_data_sic_trend_maps.pkl'
OUT_PATH = '/glade/work/jrotondo/ncc_paper/plots/extended_data_fig_unified_masks'

# ── load ──────────────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as fh:
    d = pickle.load(fh)

sig_maps        = d['sig_maps']
meta            = d['meta']
lons            = d['lons']
lats            = d['lats']
EXPERIMENTS     = d['EXPERIMENTS']
MONTHS          = d['MONTHS']
PVAL_THRESHOLDS = d['PVAL_THRESHOLDS']

# ── style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 7,
    'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# Updated colors: Blue for significant extent loss, Orange for darkening
COLOR_SIG    = '#2980b9'  # Deep Blue
COLOR_NONSIG = '#e67e22'  # Vibrant Orange
COLOR_OCEAN  = '#eaf4fb'
COLOR_LAND   = '#e8e6e1'

proj          = ccrs.NorthPolarStereo(central_longitude=-45)
data_crs      = ccrs.PlateCarree()
cmap_discrete = mcolors.ListedColormap([COLOR_NONSIG, COLOR_SIG])

def make_circle_boundary(ax, n=100):
    theta = np.linspace(0, 2 * np.pi, n)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

# ── layout ────────────────────────────────────────────────────────────
# 13 rows: 6 obs + 1 spacer + 6 cmip6

MONTHS = MONTHS[4:6]
gs = GridSpec(5, 3, figure=plt.figure(figsize=(7.2, 8.4)))

fig = plt.gcf()

obs_axes   = np.array([[fig.add_subplot(gs[r,   c], projection=proj) for c in range(3)] for r in range(2)])
cmip6_axes = np.array([[fig.add_subplot(gs[r+3, c], projection=proj) for c in range(3)] for r in range(2)])
exp_axes   = [obs_axes, cmip6_axes]

# ── draw ──────────────────────────────────────────────────────────────
for exp_idx, (exp_name, exp_label, _) in enumerate(EXPERIMENTS):
    axes_block = exp_axes[exp_idx]

    axes_block[0, 1].text(0.5, 1.22, exp_label,
                          transform=axes_block[0, 1].transAxes,
                          ha='center', va='bottom', fontsize=9, fontweight='bold')

    for r_idx, mname in enumerate(MONTHS):
        for c_idx, pthresh in enumerate(PVAL_THRESHOLDS):
            ax  = axes_block[r_idx, c_idx]
            sig = sig_maps[(exp_name, mname, pthresh)]
            m   = meta[(exp_name, mname, pthresh)]

            ax.set_facecolor(COLOR_OCEAN)
            make_circle_boundary(ax)
            ax.set_extent([-180, 180, 60, 90], crs=data_crs)

            ax.pcolormesh(lons, lats, sig, transform=data_crs,
                          cmap=cmap_discrete, vmin=0, vmax=1,
                          shading='auto', zorder=2)
            ax.add_feature(cfeature.LAND, facecolor=COLOR_LAND, zorder=3)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#777777',
                           linewidth=0.2, zorder=4)

            ax.text(0.97, 0.03, f"{m['pct_drk']:.0f}% drk",
                    transform=ax.transAxes, fontsize=9, ha='right',
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=COLOR_NONSIG, alpha=0.8, lw=0))

            if c_idx == 0:
                ax.text(-0.12, 0.5, mname, transform=ax.transAxes,
                        fontsize=9, fontweight='bold',
                        va='center', ha='right', rotation=0)
            if r_idx == 0:
                ax.set_title(f'p < {pthresh:.2f}', fontsize=9,
                             fontweight='bold', pad=4)

# ── legend ────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=COLOR_SIG,    label='Area Loss (significant negative SIC trend)'),
    mpatches.Patch(facecolor=COLOR_NONSIG, label='Darkening (complement of significant retreat)'),
    mpatches.Patch(facecolor=COLOR_OCEAN,  label='Open Ocean (SIC < 15%)'),
    mpatches.Patch(facecolor=COLOR_LAND,   label='Land'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.01))

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT_PATH}.{ext}', dpi=300, bbox_inches='tight')
    print(f'Saved → {OUT_PATH}.{ext}')
plt.show()