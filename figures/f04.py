#!/usr/bin/env python
"""
plot_siaf_spatial_both.py  (updated: fontsize=18, non-overlapping colorbars)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as mgridspec
import os
from scipy.interpolate import griddata

matplotlib.rcParams.update({
    'font.size':          18,
    'axes.titlesize':     18,
    'axes.labelsize':     18,
    'xtick.labelsize':    18,
    'ytick.labelsize':    18,
    'legend.fontsize':    18,
    'figure.titlesize':   18,
})

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('WARNING: cartopy not installed.')

from pyproj import Transformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
NPZ_CERES = os.path.join(OUT_DIR, 'siaf_spatial_boot_ceres.npz')
NPZ_HUANG = os.path.join(OUT_DIR, 'siaf_spatial_boot_huang.npz')
PLOT_STEM = os.path.join(OUT_DIR, 'siaf_spatial_both')

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

ROW_DEFS = [
    ('ext', r'SIAF$_\mathregular{area}$', '#3A6BAF'),
    ('drk', r'SIAF$_\mathregular{drk}$', '#D95F1A'),
    ('tot', r'SIAF$_\mathregular{tot}$', '#6A3FA5'),
]
CMAP_VAL   = 'YlOrRd'
CMAP_SIGMA = 'Blues'

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading NPZ files...")
for p in [NPZ_CERES, NPZ_HUANG]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Cannot find {p}")

ceres_npz  = np.load(NPZ_CERES)
huang_npz  = np.load(NPZ_HUANG)
psn_xx     = ceres_npz['psn_xx']
psn_yy     = ceres_npz['psn_yy']
DOMAIN_PSN = ceres_npz['domain'].astype(bool)
psn_x      = psn_xx[0, :]
psn_y      = psn_yy[:, 0]

# ── REPAIR SPARSE ARTIFACTS ───────────────────────────────────────────────────
def fill_sparse_map(arr, domain_mask):
    valid = ~np.isnan(arr) & domain_mask
    if not valid.any():
        return arr
    y_v, x_v = np.where(valid)
    y_q, x_q = np.where(domain_mask)
    filled    = griddata((y_v, x_v), arr[valid], (y_q, x_q), method='nearest')
    out       = np.full_like(arr, np.nan)
    out[y_q, x_q] = filled
    return out

# ── CARTOPY / IMSHOW HELPERS ──────────────────────────────────────────────────
def make_psn_crs():
    if not HAS_CARTOPY:
        return None
    globe = ccrs.Globe(semimajor_axis=6378273., semiminor_axis=6356889.449)
    return ccrs.Stereographic(central_latitude=90., central_longitude=-45.,
                               true_scale_latitude=70., globe=globe)

PSN_CRS = make_psn_crs()

_dx = float(psn_x[1] - psn_x[0]);  _dy = float(psn_y[1] - psn_y[0])
_x0 = float(psn_x[0])  - abs(_dx) / 2
_x1 = float(psn_x[-1]) + abs(_dx) / 2
_y0 = min(float(psn_y[0]), float(psn_y[-1])) - abs(_dy) / 2
_y1 = max(float(psn_y[0]), float(psn_y[-1])) + abs(_dy) / 2
_origin = 'upper' if _dy < 0 else 'lower'

_tfm  = Transformer.from_crs('EPSG:4326', PSN25_PROJ4, always_xy=True)
_lons = np.linspace(-180, 180, 361)
_xs, _ys = _tfm.transform(_lons, np.full_like(_lons, 57.))
_rim  = max(np.abs(_xs).max(), np.abs(_ys).max()) * 1.04
_ext  = [-_rim, _rim, -_rim, _rim]

def _add_map_features(ax):
    ax.add_feature(cfeature.LAND, facecolor='#CCCCCC', zorder=6, edgecolor='none')
    ax.coastlines(resolution='50m', linewidth=0.5, color='#333333', zorder=7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.35,
                      color='#AAAAAA', linestyle='--', zorder=8)
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))

def _make_ax(fig, gs_cell):
    if HAS_CARTOPY:
        ax = fig.add_subplot(gs_cell, projection=PSN_CRS)
        ax.set_extent(_ext, crs=PSN_CRS)
        _add_map_features(ax)
    else:
        ax = fig.add_subplot(gs_cell)
        ax.set_aspect('equal')
    return ax

def _plot_field(ax, arr, cmap, vmin, vmax):
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad(alpha=0.0)
    disp = np.where(DOMAIN_PSN, arr, np.nan).astype(np.float32)
    if HAS_CARTOPY:
        im = matplotlib.axes.Axes.imshow(
            ax, disp, origin=_origin,
            extent=[_x0, _x1, _y0, _y1],
            cmap=cmap_obj, vmin=vmin, vmax=vmax,
            interpolation='nearest', zorder=5)
    else:
        im = ax.imshow(disp, origin=_origin, cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
    return im

# ── FIGURE BUILDER ────────────────────────────────────────────────────────────
def make_figure(c_maps, h_maps, period, pval_str, out_path):

    vmax_val = max(
        max(np.nanpercentile(c_maps[f'mean_{t}'][DOMAIN_PSN], 98),
            np.nanpercentile(h_maps[f'mean_{t}'][DOMAIN_PSN], 98))
        for t in ('ext', 'drk', 'tot'))
    vmax_val = max(vmax_val, 0.05)

    avg_sigs = []
    for t in ('ext', 'drk', 'tot'):
        avg = np.where(DOMAIN_PSN,
                       (c_maps[f'sigma2_{t}'] + h_maps[f'sigma2_{t}']) / 2.,
                       np.nan)
        avg_sigs.append(np.nanpercentile(avg[DOMAIN_PSN], 98))
    vmax_sig = max(max(avg_sigs), 0.01)

    # Layout:  [CERES] [Huang] [pad_cb1] [cbar1] [pad_ticks] [sigma] [pad_cb2] [cbar2]
    # We add an extra padding column (index 4) to give the cbar1 tick labels room to breathe
    fig = plt.figure(figsize=(24, 20), facecolor='#F4F4F4')
    gs  = mgridspec.GridSpec(
        3, 8, figure=fig,
        left=0.05, right=0.96, top=0.93, bottom=0.03,
        hspace=0.06, wspace=0.04,
        width_ratios=[1, 1, 0.04, 0.06, 0.12, 1, 0.04, 0.06])
    #                m1 m2  pd1   cb1   pd2   m3  pd3   cb2
    # cols:          0  1   2     3     4     5   6     7

    last_im_val = last_im_sig = None

    for row, (tag, row_label, accent) in enumerate(ROW_DEFS):
        ax_c   = _make_ax(fig, gs[row, 0])
        ax_h   = _make_ax(fig, gs[row, 1])
        ax_sig = _make_ax(fig, gs[row, 5])  # Shifted to col 5

        avg_sig = np.where(
            DOMAIN_PSN,
            (c_maps[f'sigma2_{tag}'] + h_maps[f'sigma2_{tag}']) / 2.,
            np.nan).astype(np.float32)

        last_im_val = _plot_field(ax_c,   c_maps[f'mean_{tag}'], CMAP_VAL,   0, vmax_val)
        _           = _plot_field(ax_h,   h_maps[f'mean_{tag}'], CMAP_VAL,   0, vmax_val)
        last_im_sig = _plot_field(ax_sig, avg_sig,               CMAP_SIGMA, 0, vmax_sig)

        ax_c.text(-0.11, 0.5, row_label,
                  transform=ax_c.transAxes,
                  fontsize=18, fontweight='bold', color=accent,
                  va='center', ha='right',
                  rotation=90, rotation_mode='anchor')

        if row == 0:
            ax_c.set_title('CERES APRP\n(Donohoe et al., 2020)',
                           fontsize=18, fontweight='bold', pad=10)
            ax_h.set_title('ERA5 Radiative Kernel\n(Huang & Huang, 2023)',
                           fontsize=18, fontweight='bold', pad=10)
            ax_sig.set_title('Avg 2σ bootstrap\nuncertainty',
                             fontsize=18, fontweight='bold', pad=10)

    # Colorbars mapped to the correct columns
    cb_val = fig.colorbar(last_im_val, cax=fig.add_subplot(gs[:, 3]))
    cb_sig = fig.colorbar(last_im_sig, cax=fig.add_subplot(gs[:, 7])) # Shifted to col 7
    cb_val.set_label('W m⁻² K⁻¹', fontsize=18, labelpad=10)
    cb_sig.set_label('W m⁻² K⁻¹', fontsize=18, labelpad=10)
    cb_val.ax.tick_params(labelsize=18)
    cb_sig.ax.tick_params(labelsize=18)

    clean_pval = pval_str.lstrip('0') or '0'
    '''fig.suptitle(
        f'SIAF Spatial Maps  |  {period}  |  p < 0.{clean_pval}\n'
        'Left: CERES     Center: ERA5 Kernel     '
        'Right: Avg 2σ = (σ_CERES + σ_Huang) / 2',
        fontsize=18, fontweight='bold', y=0.965)'''

    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out_path}')

# ── DISCOVER PERIODS / PVALS AND PLOT ─────────────────────────────────────────
prefixes = set()
for key in ceres_npz.files:
    if '_mean_tot' in key:
        prefixes.add(key.replace('_mean_tot', ''))

print(f"\nFound {len(prefixes)} period/pval combination(s).")

for prefix in sorted(prefixes):
    if not prefix.startswith('Obs-era_'): continue
    parts  = prefix.split('_')
    period = parts[0]
    pval_s = parts[1]

    c_maps = {}
    h_maps = {}
    for tag in ('tot', 'ext', 'drk'):
        c_maps[f'mean_{tag}']   = fill_sparse_map(
            ceres_npz[f'{prefix}_mean_{tag}'],   DOMAIN_PSN)
        c_maps[f'sigma2_{tag}'] = fill_sparse_map(
            ceres_npz[f'{prefix}_sigma2_{tag}'], DOMAIN_PSN)
        h_maps[f'mean_{tag}']   = fill_sparse_map(
            huang_npz[f'{prefix}_mean_{tag}'],   DOMAIN_PSN)
        h_maps[f'sigma2_{tag}'] = fill_sparse_map(
            huang_npz[f'{prefix}_sigma2_{tag}'], DOMAIN_PSN)

    make_figure(c_maps, h_maps, period, pval_s,
                f'{PLOT_STEM}_{period}_p{pval_s}.png')

print("Done!")