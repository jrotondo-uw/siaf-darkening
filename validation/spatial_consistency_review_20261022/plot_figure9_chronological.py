# %% [markdown]
# # 20-Panel In-Situ Validation: CLARA-A3 and APP-x
# Load pre-computed data from `save_validation_data.py` and render the figure.

# %%
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Font size constants ────────────────────────────────────────────────────────
FS_TITLE  = 14   # panel title (letter + name)
FS_LABEL  = 13   # axis labels
FS_TICK   = 12   # tick labels
FS_STATS  = 11   # stats text box
FS_LEGEND = 11   # legend entries
FS_CB     = 11   # colorbar label & ticks
FS_NOTE   = 11   # italic site notes

plt.rcParams.update({
    'font.size': FS_LABEL,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.1,
    'grid.alpha': 0.3,
})

# %% [markdown]
# ## Load pickle

# %%
PKL_PATH = "/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/validation_fallback_corrected_SIC.pkl"

with open(PKL_PATH, 'rb') as f:
    D = pickle.load(f)

# Unpack DataFrames
df_tower_c   = D['df_tower_c'];   df_tower_a   = D['df_tower_a']
df_transect_c= D['df_transect_c'];df_transect_a= D['df_transect_a']
df_heli_c    = D['df_heli_c'];    df_heli_a    = D['df_heli_a']
df_summit_c  = D['df_summit_c'];  df_summit_a  = D['df_summit_a']
df_raven_c   = D['df_raven_c'];   df_raven_a   = D['df_raven_a']
df_drone_c   = D['df_drone_c'];   df_drone_a   = D['df_drone_a']
df_arise_c   = D['df_arise_c'];   df_arise_a   = D['df_arise_a']
df_arcsix_c  = D['df_arcsix_c'];  df_arcsix_a  = D['df_arcsix_a']

# Unpack arrays
x_tower_c  = D['x_tower_c'];  y_tower_c  = D['y_tower_c'];  doy_tower_c = D['doy_tower_c']
x_tower_a  = D['x_tower_a'];  y_tower_a  = D['y_tower_a'];  doy_tower_a = D['doy_tower_a']

# Unpack stats
stats_tower_c   = D['stats_tower_c'];   stats_tower_a   = D['stats_tower_a']
stats_transect_c= D['stats_transect_c'];stats_transect_a= D['stats_transect_a']
stats_heli_c    = D['stats_heli_c'];    stats_heli_a    = D['stats_heli_a']
stats_summit_c  = D['stats_summit_c'];  stats_summit_a  = D['stats_summit_a']
stats_raven_c   = D['stats_raven_c'];   stats_raven_a   = D['stats_raven_a']
stats_drone_c   = D['stats_drone_c'];   stats_drone_a   = D['stats_drone_a']
stats_arise_c   = D['stats_arise_c'];   stats_arise_a   = D['stats_arise_a']
stats_arcsix_c  = D['stats_arcsix_c'];  stats_arcsix_a  = D['stats_arcsix_a']

# Unpack metadata
MAX_SZA         = D['MAX_SZA']
MAX_DELTA_SZA   = D['MAX_DELTA_SZA']
MAX_DIST_HELI   = D['MAX_DIST_HELI']
MAX_DIST_ARISE  = D['MAX_DIST_ARISE']
MAX_DIST_ARCSIX = D['MAX_DIST_ARCSIX']
SUMMIT_LAT      = D['SUMMIT_LAT'];  SUMMIT_LON = D['SUMMIT_LON']
RAVEN_LAT       = D['RAVEN_LAT'];   RAVEN_LON  = D['RAVEN_LON']
SURFACE_COLORS  = D['SURFACE_COLORS']
SURFACE_PRIORITY= D['SURFACE_PRIORITY']
valid_surf      = D['valid_surf']
output_dir      = D['output_dir']

# %% [markdown]
# ## Rebuild surface-type legend patches

# %%
surf_patches = [
    mpatches.Patch(color=SURFACE_COLORS.get(s, '#7f7f7f'), label=s.title())
    for s in valid_surf
]

surf_patches_a = [
    mpatches.Patch(color=SURFACE_COLORS.get(s, '#7f7f7f'), label=s.title())
    for s in SURFACE_PRIORITY + ['unknown']
    if len(df_transect_a) and s in df_transect_a['surface_type'].values
]

# %% [markdown]
# ## Drawing helpers

# %%
def _stats_text(s):
    return (f"r = {s['r']:.3f}\nbias = {s['bias']:+.3f}\n"
            f"RMSE = {s['rmse']:.3f}\nn = {s['n']}")


def _add_trend(ax, x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan
    r = float(np.corrcoef(x[valid], y[valid])[0, 1])
    if r < 0.5:
        return np.nan, np.nan
    z  = np.polyfit(x[valid], y[valid], 1)
    xf = np.linspace(float(x[valid].min()), float(x[valid].max()), 200)
    ax.plot(xf, np.polyval(z, xf), 'r-', lw=1.8, zorder=4,
            label=f'y={z[0]:.2f}x{z[1]:+.2f}')
    return z[0], z[1]


def _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel,
              xlim=(0, 1), ylim=(0, 1)):
    lo = max(xlim[0], ylim[0]); hi = min(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.45, zorder=2, label='1:1')
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.97, _stats_text(s_d), transform=ax.transAxes,
            fontsize=FS_STATS, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, lw=0.5))
    ax.set_title(f'{panel_label}  {title}', loc='left', fontsize=FS_TITLE, pad=4)


def draw_panel(ax, x, y, c_vals, cmap, norm, s_d,
               panel_label, title, xlabel, ylabel,
               colorbar_label=None, xlim=(0, 1), ylim=(0, 1)):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sc = ax.scatter(x, y, c=c_vals, cmap=cmap, norm=norm,
                    s=38, edgecolors='k', linewidth=0.4, alpha=0.75, zorder=3)
    _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.8, handlelength=1.4)
    if colorbar_label is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label(colorbar_label, fontsize=FS_CB)
        cb.ax.tick_params(labelsize=FS_CB)
    return sc


def draw_surface_panel(ax, x, y, surface_col, surf_patches_,
                        s_d, panel_label, title, xlabel, ylabel):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    c_vals = [SURFACE_COLORS.get(s, '#7f7f7f') for s in surface_col]
    ax.scatter(x, y, c=c_vals, s=38, edgecolors='k',
               linewidth=0.4, alpha=0.75, zorder=3)
    z0, z1 = _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel)
    handles = surf_patches_[:]
    if np.isfinite(z0):
        handles = [Line2D([0], [0], color='r', lw=1.8,
                          label=f'y={z0:.2f}x{z1:+.2f}')] + handles
    handles += [Line2D([0], [0], color='k', ls='--', lw=1.2,
                       alpha=0.45, label='1:1')]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc='lower right',
              framealpha=0.8, ncol=2, handlelength=1.1)


def draw_greenland_panel(ax, x, y, doy, s_d,
                          panel_label, title, xlabel, ylabel, site_label=''):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sc = ax.scatter(x, y, c=doy, cmap='viridis',
                    norm=plt.Normalize(vmin=1, vmax=366),
                    s=38, edgecolors='k', linewidth=0.4, alpha=0.8, zorder=3)
    _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel,
              xlim=(0, 1), ylim=(0, 1))
    if site_label:
        ax.text(0.03, 0.03, f'{site_label}\nPentad mean match\n(no SIC blend)',
                transform=ax.transAxes, fontsize=FS_NOTE, va='bottom', ha='left',
                style='italic', color='#444444')
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.8, handlelength=1.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75, aspect=16)
    cb.set_label('Day of Year', fontsize=FS_CB)
    cb.ax.tick_params(labelsize=FS_CB)


def no_data_panel(ax, panel_label, title, reason='No data available'):
    ax.text(0.5, 0.5, reason, ha='center', va='center',
            transform=ax.transAxes, fontsize=FS_LABEL, color='gray', style='italic')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{panel_label}  {title}', loc='left', fontsize=FS_TITLE, pad=4)

# %% [markdown]
# ## Create figure

# %%
fig, axes = plt.subplots(6, 4, figsize=(26, 39),
                          gridspec_kw={'hspace': 0.52, 'wspace': 0.36})


_chrono_axes = axes.copy()
axes[4, 0] = _chrono_axes[0, 0]
axes[4, 1] = _chrono_axes[0, 1]
axes[5, 0] = _chrono_axes[0, 2]
axes[5, 1] = _chrono_axes[0, 3]
axes[4, 2] = _chrono_axes[1, 0]
axes[4, 3] = _chrono_axes[1, 1]
axes[0, 3] = _chrono_axes[1, 2]
axes[1, 3] = _chrono_axes[1, 3]
axes[2, 1] = _chrono_axes[2, 0]
axes[3, 1] = _chrono_axes[2, 1]
axes[0, 0] = _chrono_axes[2, 2]
axes[1, 0] = _chrono_axes[2, 3]
axes[0, 1] = _chrono_axes[3, 0]
axes[1, 1] = _chrono_axes[3, 1]
axes[0, 2] = _chrono_axes[3, 2]
axes[1, 2] = _chrono_axes[3, 3]
axes[2, 0] = _chrono_axes[4, 0]
axes[3, 0] = _chrono_axes[4, 1]
axes[2, 3] = _chrono_axes[4, 2]
axes[3, 3] = _chrono_axes[4, 3]
axes[2, 2] = _chrono_axes[5, 0]
axes[3, 2] = _chrono_axes[5, 1]
axes[5,2] = _chrono_axes[5,2]
axes[5,3] = _chrono_axes[5,3]

CLARA_Y = 'CLARA-A3 Albedo'
APPX_Y  = 'APP-x Albedo'

# %% [markdown]
# ### Row 0 — CLARA-A3: Tower | Transects | Helicopter | Summit

# %%
# A) Tower CLARA-A3
ax = axes[0, 0]
if len(x_tower_c) > 0:
    draw_panel(ax, x_tower_c, y_tower_c, doy_tower_c,
               'viridis', plt.Normalize(1, 366), stats_tower_c,
               'A)', 'MOSAiC Tower (CLARA-A3)',
               'Tower Albedo\n(SZA-matched × SIC blend)', CLARA_Y,
               colorbar_label='Day of Year')
else:
    no_data_panel(ax, 'A)', 'MOSAiC Tower (CLARA-A3)')

# B) Transects CLARA-A3
ax = axes[0, 1]
if len(df_transect_c) > 0:
    draw_surface_panel(ax,
        df_transect_c['transect_albedo'].values,
        df_transect_c['sat_albedo'].values,
        df_transect_c['surface_type'].values,
        surf_patches, stats_transect_c,
        'B)', f'MOSAiC Transects (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Transect Albedo', CLARA_Y)
else:
    no_data_panel(ax, 'B)', f'MOSAiC Transects (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# C) Helicopter CLARA-A3
ax = axes[0, 2]
if len(df_heli_c) > 0 and 'heli_albedo_corr' in df_heli_c.columns:
    df_ch   = df_heli_c[df_heli_c['heli_albedo_corr'].notna()].copy()
    norm_ch = plt.Normalize(vmin=0, vmax=MAX_DIST_HELI)
    sc_ch   = None
    for instr, mkr in {'helicopter': 'o', 'HELiPOD': '^'}.items():
        sub = df_ch[df_ch['instrument'] == instr]
        if sub.empty: continue
        sc_ch = ax.scatter(sub['heli_albedo_corr'], sub['sat_albedo'],
                           c=sub['mean_dist_km'], cmap='cividis', norm=norm_ch,
                           s=50, marker=mkr, edgecolors='k', linewidth=0.4,
                           alpha=0.75, zorder=3, label=f'{instr} (n={len(sub)})')
    _add_trend(ax, df_ch['heli_albedo_corr'].values, df_ch['sat_albedo'].values)
    _frame_ax(ax, stats_heli_c, 'C)',
              'MOSAiC Heli+HELiPOD\n(CLARA-A3, linear SZA corr.)',
              'Helicopter/HELiPOD Albedo\n(SZA-corrected)', CLARA_Y)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.8)
    if sc_ch is not None:
        cb = fig.colorbar(sc_ch, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label('Dist. to sat. pixel (km)', fontsize=FS_CB)
        cb.ax.tick_params(labelsize=FS_CB)
else:
    no_data_panel(ax, 'C)', 'MOSAiC Heli+HELiPOD (CLARA-A3)')

# D) Summit CLARA-A3
ax = axes[0, 3]
if len(df_summit_c) > 0:
    draw_greenland_panel(ax,
        df_summit_c['tower_albedo'].values, df_summit_c['sat_albedo'].values,
        df_summit_c['doy'].values, stats_summit_c,
        'D)', 'Summit Greenland (CLARA-A3, 2012–2013)',
        'Tower Albedo (pentad mean)', CLARA_Y,
        site_label=f'Summit ({SUMMIT_LAT:.2f}°N, {abs(SUMMIT_LON):.2f}°W)')
else:
    no_data_panel(ax, 'D)', 'Summit Greenland (CLARA-A3, 2012–2013)')

# %% [markdown]
# ### Row 1 — APP-x: Tower | Transects | Helicopter | Summit

# %%
# E) Tower APP-x
ax = axes[1, 0]
if len(x_tower_a) > 0:
    draw_panel(ax, x_tower_a, y_tower_a, doy_tower_a,
               'viridis', plt.Normalize(1, 366), stats_tower_a,
               'E)', 'MOSAiC Tower (APP-x)',
               'Tower Albedo\n(SZA-matched × SIC blend)', APPX_Y,
               colorbar_label='Day of Year')
else:
    no_data_panel(ax, 'E)', 'MOSAiC Tower (APP-x)')

# F) Transects APP-x
ax = axes[1, 1]
if len(df_transect_a) > 0:
    draw_surface_panel(ax,
        df_transect_a['transect_albedo'].values,
        df_transect_a['sat_albedo'].values,
        df_transect_a['surface_type'].values,
        surf_patches_a, stats_transect_a,
        'F)', f'MOSAiC Transects (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Transect Albedo', APPX_Y)
else:
    no_data_panel(ax, 'F)', f'MOSAiC Transects (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# G) Helicopter APP-x
ax = axes[1, 2]
if len(df_heli_a) > 0 and 'heli_albedo_corr' in df_heli_a.columns:
    df_ah   = df_heli_a[df_heli_a['heli_albedo_corr'].notna()].copy()
    norm_ah = plt.Normalize(vmin=0, vmax=MAX_DIST_HELI)
    sc_ah   = None
    for instr, mkr in {'helicopter': 'o', 'HELiPOD': '^'}.items():
        sub = df_ah[df_ah['instrument'] == instr]
        if sub.empty: continue
        sc_ah = ax.scatter(sub['heli_albedo_corr'], sub['sat_albedo'],
                           c=sub['mean_dist_km'], cmap='cividis', norm=norm_ah,
                           s=50, marker=mkr, edgecolors='k', linewidth=0.4,
                           alpha=0.75, zorder=3, label=f'{instr} (n={len(sub)})')
    _add_trend(ax, df_ah['heli_albedo_corr'].values, df_ah['sat_albedo'].values)
    _frame_ax(ax, stats_heli_a, 'G)',
              'MOSAiC Heli+HELiPOD\n(APP-x, linear SZA corr.)',
              'Helicopter/HELiPOD Albedo\n(SZA-corrected)', APPX_Y)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.8)
    if sc_ah is not None:
        cb = fig.colorbar(sc_ah, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label('Dist. to sat. pixel (km)', fontsize=FS_CB)
        cb.ax.tick_params(labelsize=FS_CB)
else:
    no_data_panel(ax, 'G)', 'MOSAiC Heli+HELiPOD (APP-x)')

# H) Summit APP-x
ax = axes[1, 3]
if len(df_summit_a) > 0:
    draw_greenland_panel(ax,
        df_summit_a['tower_albedo'].values, df_summit_a['sat_albedo'].values,
        df_summit_a['doy'].values, stats_summit_a,
        'H)', 'Summit Greenland (APP-x, 2012–2013)',
        'Tower Albedo (pentad mean)', APPX_Y,
        site_label=f'Summit ({SUMMIT_LAT:.2f}°N, {abs(SUMMIT_LON):.2f}°W)')
else:
    no_data_panel(ax, 'H)', 'Summit Greenland (APP-x, 2012–2013)')

# %% [markdown]
# ### Row 2 — CLARA-A3: Drone | ARISE | ARCSIX | Raven

# %%
# I) Drone CLARA-A3
ax = axes[2, 0]
if len(df_drone_c) > 0:
    alt_vals = df_drone_c['alt_mean_w'].values
    draw_panel(ax,
        df_drone_c['drone_albedo'].values, df_drone_c['sat_albedo'].values,
        alt_vals, 'plasma',
        plt.Normalize(vmin=0, vmax=float(np.nanpercentile(alt_vals, 95))),
        stats_drone_c,
        'I)', f'MOSAiC Drone (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Drone Albedo\n(altitude-weighted)', CLARA_Y,
        colorbar_label='Alt-weighted mean AGL (m)')
else:
    no_data_panel(ax, 'I)', f'MOSAiC Drone (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# J) ARISE CLARA-A3
ax = axes[2, 1]
if len(df_arise_c) > 0:
    draw_panel(ax,
        df_arise_c['aircraft_albedo'].values, df_arise_c['sat_albedo'].values,
        df_arise_c['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARISE), stats_arise_c,
        'J)', f'NASA ARISE 2014 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', CLARA_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'J)', f'NASA ARISE 2014 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# K) ARCSIX CLARA-A3
ax = axes[2, 2]
if len(df_arcsix_c) > 0:
    draw_panel(ax,
        df_arcsix_c['aircraft_albedo'].values, df_arcsix_c['sat_albedo'].values,
        df_arcsix_c['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARCSIX), stats_arcsix_c,
        'K)', f'NASA ARCSIX 2024 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', CLARA_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'K)', f'NASA ARCSIX 2024 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# L) Raven CLARA-A3
ax = axes[2, 3]
if len(df_raven_c) > 0:
    draw_greenland_panel(ax,
        df_raven_c['tower_albedo'].values, df_raven_c['sat_albedo'].values,
        df_raven_c['doy'].values, stats_raven_c,
        'L)', 'Raven / SLEIGH (CLARA-A3, 2024)',
        'SLEIGH Tower Albedo (pentad mean)', CLARA_Y,
        site_label=f'Raven ({RAVEN_LAT:.2f}°N, {abs(RAVEN_LON):.2f}°W)')
else:
    no_data_panel(ax, 'L)', 'Raven / SLEIGH (CLARA-A3, 2024)')

# %% [markdown]
# ### Row 3 — APP-x: Drone | ARISE | ARCSIX | Raven

# %%
# M) Drone APP-x
ax = axes[3, 0]
if len(df_drone_a) > 0:
    alt_vals_a = df_drone_a['alt_mean_w'].values
    draw_panel(ax,
        df_drone_a['drone_albedo'].values, df_drone_a['sat_albedo'].values,
        alt_vals_a, 'plasma',
        plt.Normalize(vmin=0, vmax=float(np.nanpercentile(alt_vals_a, 95))),
        stats_drone_a,
        'M)', f'MOSAiC Drone (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Drone Albedo\n(altitude-weighted)', APPX_Y,
        colorbar_label='Alt-weighted mean AGL (m)')
else:
    no_data_panel(ax, 'M)', f'MOSAiC Drone (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# N) ARISE APP-x
ax = axes[3, 1]
if len(df_arise_a) > 0:
    draw_panel(ax,
        df_arise_a['aircraft_albedo'].values, df_arise_a['sat_albedo'].values,
        df_arise_a['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARISE), stats_arise_a,
        'N)', f'NASA ARISE 2014 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', APPX_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'N)', f'NASA ARISE 2014 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# O) ARCSIX APP-x
ax = axes[3, 2]
if len(df_arcsix_a) > 0:
    draw_panel(ax,
        df_arcsix_a['aircraft_albedo'].values, df_arcsix_a['sat_albedo'].values,
        df_arcsix_a['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARCSIX), stats_arcsix_a,
        'O)', f'NASA ARCSIX 2024 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', APPX_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'O)', f'NASA ARCSIX 2024 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# P) Raven APP-x
ax = axes[3, 3]
if len(df_raven_a) > 0:
    draw_greenland_panel(ax,
        df_raven_a['tower_albedo'].values, df_raven_a['sat_albedo'].values,
        df_raven_a['doy'].values, stats_raven_a,
        'P)', 'Raven / SLEIGH (APP-x, 2024)',
        'SLEIGH Tower Albedo (pentad mean)', APPX_Y,
        site_label=f'Raven ({RAVEN_LAT:.2f}°N, {abs(RAVEN_LON):.2f}°W)')
else:
    no_data_panel(ax, 'P)', 'Raven / SLEIGH (APP-x, 2024)')

# %% [markdown]
# ## Save figure

# %%
import os

# %% [markdown]
# ### Row 4 - the two pre-2012 drifting ice stations (added in revision)

# %%
import pandas as _pd

_VB = "/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/fallback"

def _load(fn, tcol):
    d = _pd.read_csv(fn, parse_dates=[tcol])
    d["_doy"] = d[tcol].dt.dayofyear
    return d

def _stats(d):
    b = d["sat"] - d["insitu"]
    return {"r": float(np.corrcoef(d["insitu"], d["sat"])[0, 1]),
            "bias": float(b.mean()), "rmse": float(np.sqrt((b ** 2).mean())),
            "n": int(len(d))}

_sh_c = _load(f"{_VB}/sheba_surface_flux/sheba_clara_matchups.csv", "pentad")
_sh_a = _load(f"{_VB}/sheba_surface_flux/sheba_appx_matchups.csv",  "date")
_ta_c = _load(f"{_VB}/tara_surface_flux/tara_broadband_clara_matchups.csv",   "pentad")
_ta_a = _load(f"{_VB}/tara_surface_flux/tara_broadband_appx_matchups.csv",    "date")

# Q) SHEBA CLARA-A3
draw_greenland_panel(axes[4, 0], _sh_c["insitu"].values, _sh_c["sat"].values,
    _sh_c["_doy"].values, _stats(_sh_c), 'Q)',
    'SHEBA 1997-98 (CLARA-A3)', 'SHEBA Albedo (pentad mean)', CLARA_Y,
    site_label='')

# R) SHEBA APP-x
draw_greenland_panel(axes[4, 1], _sh_a["insitu"].values, _sh_a["sat"].values,
    _sh_a["_doy"].values, _stats(_sh_a), 'R)',
    'SHEBA 1997-98 (APP-x)', 'SHEBA Albedo (daily mean)', APPX_Y,
    site_label='')

# S) Tara CLARA-A3 - broadband radiometer albedo
draw_greenland_panel(axes[4, 2], _ta_c["insitu"].values, _ta_c["sat"].values,
    _ta_c["_doy"].values, _stats(_ta_c), 'S)',
    'Tara 2007 (CLARA-A3)', 'Tara Broadband Albedo (pentad mean)', CLARA_Y,
    site_label='')

# T) Tara APP-x
draw_greenland_panel(axes[4, 3], _ta_a["insitu"].values, _ta_a["sat"].values,
    _ta_a["_doy"].values, _stats(_ta_a), 'T)',
    'Tara 2007 (APP-x)', 'Tara Broadband Albedo (daily mean)', APPX_Y,
    site_label='')


# correct site notes for the new row (draw_greenland_panel's default text is for
# the Greenland towers and does not apply here)
_NOTES = {
    (4, 0): 'SHEBA (75.2-80.4$\\degree$N)\nPentad mean match\n(SIC blended)',
    (4, 1): 'SHEBA (75.2-80.4$\\degree$N)\nDaily match\n(SIC blended)',
    (4, 2): 'Tara 2007\n' + ('SIC assumed 100%' if np.allclose(_ta_c.fsic, 1) else 'SIC blended'),
    (4, 3): 'Tara 2007\n' + ('SIC assumed 100%' if np.allclose(_ta_a.fsic, 1) else 'SIC blended'),
}
for _rc, _txt in _NOTES.items():
    axes[_rc].text(0.03, 0.03, _txt, transform=axes[_rc].transAxes,
                   fontsize=FS_NOTE, va='bottom', ha='left', style='italic',
                   color='#444444')

assert (_ta_c.albedo_type == 'broadband').all() and (_ta_a.albedo_type == 'broadband').all()

# Perovich survey-line validation added as a distinct platform.
_pc = _load('/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/fallback/sheba_perovich_transects/perovich_clara_matchups.csv', 'pentad')
_pa = _load('/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/fallback/sheba_perovich_transects/perovich_appx_matchups.csv', 'date')
for _col, _d, _product, _letter, _xlabel in [
    (0, _pc, CLARA_Y, 'U)', 'Survey mean albedo (pentad)'),
    (1, _pa, APPX_Y, 'V)', 'Survey mean albedo (survey date)')]:
    if len(_d):
        draw_greenland_panel(axes[5, _col], _d.insitu, _d.sat, _d._doy, _stats(_d),
            _letter, 'SHEBA Perovich transects\n' + _product.replace(' Albedo',''), _xlabel, _product)
    else:
        no_data_panel(axes[5, _col], _letter, 'SHEBA Perovich transects')
axes[5,2].axis('off'); axes[5,3].axis('off')
_chrono_axes[0, 0].set_title('A) SHEBA surface flux (1997–98)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[0, 1].set_title('B) SHEBA surface flux (1997–98)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[0, 2].set_title('C) SHEBA transects (1998)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[0, 3].set_title('D) SHEBA transects (1998)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[1, 0].set_title('E) Tara (2007)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[1, 1].set_title('F) Tara (2007)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[1, 2].set_title('G) Summit (2012–13)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[1, 3].set_title('H) Summit (2012–13)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[2, 0].set_title('I) ARISE (2014)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[2, 1].set_title('J) ARISE (2014)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[2, 2].set_title('K) MOSAiC tower (2020)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[2, 3].set_title('L) MOSAiC tower (2020)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[3, 0].set_title('M) MOSAiC transects (2020)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[3, 1].set_title('N) MOSAiC transects (2020)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[3, 2].set_title('O) MOSAiC helicopter / HELiPOD (2020)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[3, 3].set_title('P) MOSAiC helicopter / HELiPOD (2020)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[4, 0].set_title('Q) MOSAiC drone (2020)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[4, 1].set_title('R) MOSAiC drone (2020)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[4, 2].set_title('S) Raven / SLEIGH (2024)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[4, 3].set_title('T) Raven / SLEIGH (2024)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[5, 0].set_title('U) ARCSIX (2024)\nCLARA-A3',loc='left',fontsize=FS_TITLE,pad=4)
_chrono_axes[5, 1].set_title('V) ARCSIX (2024)\nAPP-x',loc='left',fontsize=FS_TITLE,pad=4)
fig.suptitle('Figure 9 — In-situ albedo validation', fontsize=24, y=.94)

outpath = '/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/Figure_9_chronological.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"Saved → {outpath}")
fig.show()

# %% [markdown]
# ## Summary table

# %%
all_stats_list = [
    ('Q', 'SHEBA 1997-98', 'CLARA-A3', _stats(_sh_c)),
    ('R', 'SHEBA 1997-98', 'APP-x', _stats(_sh_a)),
    ('S', 'Tara 2007 broadband', 'CLARA-A3', _stats(_ta_c)),
    ('T', 'Tara 2007 broadband', 'APP-x', _stats(_ta_a)),
    ('A', 'MOSAiC Tower (SZA-matched, SIC blend)',              'CLARA-A3', stats_tower_c),
    ('B', f'MOSAiC Transects (|ΔSZA|≤{MAX_DELTA_SZA}°)',       'CLARA-A3', stats_transect_c),
    ('C', 'MOSAiC Helicopter+HELiPOD (linear SZA corr.)',       'CLARA-A3', stats_heli_c),
    ('D', 'Summit Greenland 2012–13 (pentad mean)',              'CLARA-A3', stats_summit_c),
    ('E', 'MOSAiC Tower (SZA-matched, SIC blend)',              'APP-x',    stats_tower_a),
    ('F', f'MOSAiC Transects (|ΔSZA|≤{MAX_DELTA_SZA}°)',       'APP-x',    stats_transect_a),
    ('G', 'MOSAiC Helicopter+HELiPOD (linear SZA corr.)',       'APP-x',    stats_heli_a),
    ('H', 'Summit Greenland 2012–13 (pentad mean)',              'APP-x',    stats_summit_a),
    ('I', f'HELiX Drone (|ΔSZA|≤{MAX_DELTA_SZA}°)',            'CLARA-A3', stats_drone_c),
    ('J', f'ARISE 2014 nobs_snow>0 (|ΔSZA|≤{MAX_DELTA_SZA}°)','CLARA-A3', stats_arise_c),
    ('K', f'ARCSIX 2024 (|ΔSZA|≤{MAX_DELTA_SZA}°)',            'CLARA-A3', stats_arcsix_c),
    ('L', 'Raven Greenland 2024 SLEIGH (pentad mean)',           'CLARA-A3', stats_raven_c),
    ('M', f'HELiX Drone (|ΔSZA|≤{MAX_DELTA_SZA}°)',            'APP-x',    stats_drone_a),
    ('N', f'ARISE 2014 (|ΔSZA|≤{MAX_DELTA_SZA}°)',             'APP-x',    stats_arise_a),
    ('O', f'ARCSIX 2024 (|ΔSZA|≤{MAX_DELTA_SZA}°)',            'APP-x',    stats_arcsix_a),
    ('P', 'Raven Greenland 2024 SLEIGH (pentad mean)',           'APP-x',    stats_raven_a),
]
print(f"{'Pan':<4}  {'Dataset':<42}  {'Product':<10}  {'r':>7}  {'bias':>8}  {'RMSE':>8}  {'n':>6}")
print("-" * 90)
for label, name, product, s in all_stats_list:
    r_s    = f"{s['r']:7.3f}"     if np.isfinite(s['r'])    else '    nan'
    bias_s = f"{s['bias']:+8.4f}" if np.isfinite(s['bias']) else '     nan'
    rmse_s = f"{s['rmse']:8.4f}"  if np.isfinite(s['rmse']) else '     nan'
    print(f"  {label:<3}  {name:<42}  {product:<10}  {r_s}  {bias_s}  {rmse_s}  {s['n']:>6d}")
fig.savefig('/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/Figure_9_chronological.pdf', dpi=200, bbox_inches='tight')
# Figure 9 in the revised manuscript; retain legacy f06 for existing references.
print('Saved separate Figure 9 with Perovich transects')

all_stats_list.extend([("C","SHEBA transects","CLARA-A3",_stats(_pc)),("D","SHEBA transects","APP-x",_stats(_pa))])
_pd.DataFrame([dict(dataset=name,product=product,**st) for _,name,product,st in all_stats_list]).to_csv("/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022/Figure_9_chronological_statistics.csv",index=False)
