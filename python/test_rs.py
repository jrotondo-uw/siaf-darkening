#!/usr/bin/env python
"""
rs_ceres_comparison.py
----------------------
Computes RS from old and new CERES files, plots per-month difference
maps and scalar bar chart, and prints a summary table.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import glob, os

# =============================================================================
# CONFIG
# =============================================================================

CERES_OLD  = ('/glade/work/skygale/_variables/albedo/'
              'CERES_EBAF_Edition4.2.1_200003-202504.nc')
CERES_NEW  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
              'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
PLOT_DIR   = '/glade/work/jrotondo/ncc_paper/plots/RS_ceres_comparison/'
os.makedirs(PLOT_DIR, exist_ok=True)

MONTH_NAMES  = {4:'April', 5:'May', 6:'June',
                7:'July',  8:'August', 9:'September'}
MONTH_NUMS   = [4, 5, 6, 7, 8, 9]
N_BOOT_RS    = 100
CERES_RESAMP = 9
SEED         = 42

# =============================================================================
# HELPERS
# =============================================================================

def make_circle_boundary(ax, n=100):
    theta = np.linspace(0, 2*np.pi, n)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

def smooth_field(data_2d, sigma=0.7):
    valid   = np.isfinite(data_2d)
    tmp     = np.where(valid, data_2d, 0.)
    w       = gaussian_filter(valid.astype(float), sigma=sigma)
    blurred = gaussian_filter(tmp, sigma=sigma)
    out     = np.where(w > 0, blurred / w, np.nan)
    return np.where(valid, out, np.nan)

def prep(data_2d, lon_1d, lat_1d, apply_smooth=True):
    d = data_2d.copy()
    if apply_smooth:
        d = smooth_field(d, sigma=0.7)
    data_cyc, lon_cyc = add_cyclic_point(d, coord=lon_1d)
    lon2d, lat2d      = np.meshgrid(lon_cyc, lat_1d)
    return data_cyc, lon2d, lat2d

def arctic_wmean(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid])
            if valid.any() else np.nan)

def invert_T07_2d(S, SW_up, SW_d_sfc, SW_u_sfc):
    S_s  = S.where(S > 1.)
    Sd_s = SW_d_sfc.where(SW_d_sfc > 1.)
    a    = (SW_u_sfc / Sd_s).clip(0., 1.)
    p    = SW_d_sfc / S_s
    q    = SW_up    / S_s
    den  = q + p - 2.*p*a
    R    = ((q - p*a) / den.where(np.abs(den) > 1e-6)).clip(0., 0.99)
    A    = (1. - p*(1.-a*R) /
            (1.-R).where((1.-R) > 0.01)).clip(0., 0.99)
    return R, A, a

def compute_RS_central(ceres_file, seed=SEED):
    """
    Bootstrap RS — builds its own lat/lon grid from the file itself
    so it works regardless of spatial extent or resolution.
    Returns RS_cent, RS_scal, file_lat, file_lon.
    """
    ds = xr.open_dataset(ceres_file)
    ds = ds.sel(time=(ds.time.dt.year >= 2000) &
                     (ds.time.dt.year <= 2024))

    file_lat       = ds['lat'].values
    file_lon       = ds['lon'].values
    flon2d, flat2d = np.meshgrid(file_lon, file_lat)

    times   = pd.DatetimeIndex(ds.time.values)
    df      = pd.DataFrame({'year': times.year, 'month': times.month},
                           index=np.arange(len(times)))
    years   = np.arange(2000, 2024)
    rng_loc = np.random.default_rng(seed)
    RS_cent = {}
    RS_scal = {}

    for month_num in MONTH_NUMS:
        boot_maps = []; boot_scalars = []
        for _ in range(N_BOOT_RS):
            yr_sel  = rng_loc.choice(years, size=CERES_RESAMP, replace=True)
            idx_sel = df[(df['year'].isin(yr_sel)) &
                         (df['month'] == month_num)].index.tolist()
            if not idx_sel: continue
            sel    = ds.isel(time=idx_sel)
            S_b    = sel['solar_mon'].mean('time')
            SWup_b = sel['toa_sw_all_mon'].mean('time')
            SWds_b = sel['sfc_sw_down_all_mon'].mean('time')
            SWus_b = sel['sfc_sw_up_all_mon'].mean('time')
            SWup_c = sel['toa_sw_clr_c_mon'].mean('time')
            SWds_c = sel['sfc_sw_down_clr_c_mon'].mean('time')
            SWus_c = sel['sfc_sw_up_clr_c_mon'].mean('time')
            R_all, A_all, alpha = invert_T07_2d(S_b, SWup_b, SWds_b, SWus_b)
            R_clr, _,     _     = invert_T07_2d(S_b, SWup_c, SWds_c, SWus_c)
            R_cloud = (R_all - R_clr).clip(0., 0.99)
            S_safe  = S_b.where(S_b > 1.)
            RS_b    = (S_safe * (1.-A_all) * (1.-R_cloud)**2
                       / (1.-alpha*R_cloud)**2).values
            boot_maps.append(RS_b)
            boot_scalars.append(arctic_wmean(
                np.where(flat2d >= 60., RS_b, np.nan), file_lat))

        RS_cent[month_num] = np.nanmean(np.stack(boot_maps, axis=0), axis=0)
        RS_scal[month_num] = float(np.nanmean(boot_scalars))
        print(f"  Month {month_num}: Arctic mean RS = {RS_scal[month_num]:.2f}",
              flush=True)

    ds.close()
    return RS_cent, RS_scal, file_lat, file_lon

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("Computing RS from OLD CERES file...", flush=True)
    RS_old, RS_scal_old, lat_old, lon_old = compute_RS_central(CERES_OLD)
    lon2d_old, lat2d_old = np.meshgrid(lon_old, lat_old)

    print("\nComputing RS from NEW CERES file...", flush=True)
    RS_new, RS_scal_new, lat_new, lon_new = compute_RS_central(CERES_NEW)

    # ── Regrid new RS onto old grid for differencing ──────────────────────────

    print("\nRegridding new RS onto old grid...", flush=True)

    RS_new_on_old = {}
    for month_num in MONTH_NUMS:
        interp = RegularGridInterpolator(
            (lat_new, lon_new), RS_new[month_num],
            method='linear', bounds_error=False, fill_value=np.nan)
        pts = np.column_stack([lat2d_old.ravel(), lon2d_old.ravel()])
        RS_new_on_old[month_num] = interp(pts).reshape(lat2d_old.shape)

    # ── PLOT 1: 6-panel difference maps ──────────────────────────────────────

    print("\nPlotting difference maps...", flush=True)

    proj_p   = ccrs.NorthPolarStereo(central_longitude=-35)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 11),
        subplot_kw={'projection': proj_p},
        facecolor='white'
    )
    fig.patch.set_facecolor('white')
    fig.suptitle('RS difference: NEW − OLD CERES  (W m⁻² per unit α)',
                 fontsize=13, fontweight='bold', y=0.97)
    axes = axes.ravel()

    for i, month_num in enumerate(MONTH_NUMS):
        ax    = axes[i]
        mname = MONTH_NAMES[month_num]
        diff  = RS_new_on_old[month_num] - RS_old[month_num]

        ax.set_facecolor('#cce5f0')
        make_circle_boundary(ax)
        ax.set_extent([-180, 180, 60, 90], crs=data_crs)

        plot_raw  = np.where(lat2d_old >= 60., diff, np.nan)
        data_cyc, lon2d_c, lat2d_c = prep(plot_raw, lon_old, lat_old,
                                           apply_smooth=True)
        data_plot = np.where(lat2d_c >= 60., data_cyc, np.nan)

        finite_vals = plot_raw[np.isfinite(plot_raw)]
        vabs = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 1.
        vabs = max(vabs, 1.0)

        im = ax.pcolormesh(lon2d_c, lat2d_c, data_plot,
                           transform=data_crs, cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs,
                           shading='auto', zorder=2)
        ax.add_feature(cfeature.LAND,      facecolor='#d0cfc4', zorder=3)
        ax.add_feature(cfeature.COASTLINE, edgecolor='#555555',
                       linewidth=0.5, zorder=4)
        gl = ax.gridlines(crs=data_crs, draw_labels=False,
                          linewidth=0.4, color='#888888',
                          alpha=0.6, linestyle='--', zorder=5)
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator([60, 70, 80])

        mean_diff = arctic_wmean(plot_raw, lat_old)
        ax.set_title(f"{mname}\nΔ Arctic mean = {mean_diff:+.2f} W m⁻² per unit α",
                     color='black', fontsize=9.5, pad=4)

        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                            pad=0.04, fraction=0.046, shrink=0.85)
        cbar.ax.tick_params(labelsize=7, colors='black')
        cbar.set_label('W m⁻² per unit α', color='black', fontsize=8)
        cbar.outline.set_edgecolor('#aaaaaa')
        cbar.outline.set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(PLOT_DIR, 'RS_diff_new_minus_old_6panel.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fpath}", flush=True)

    # ── PLOT 2: scalar bar chart ──────────────────────────────────────────────

    print("Plotting scalar comparison...", flush=True)

    months_labels = [MONTH_NAMES[m] for m in MONTH_NUMS]
    old_vals  = [RS_scal_old[m] for m in MONTH_NUMS]
    new_vals  = [RS_scal_new[m] for m in MONTH_NUMS]
    diff_vals = [RS_scal_new[m] - RS_scal_old[m] for m in MONTH_NUMS]

    x = np.arange(len(MONTH_NUMS))
    w = 0.30

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.bar(x - w/2, old_vals, width=w, label='Old CERES',
           color='#4393c3', alpha=0.85)
    ax.bar(x + w/2, new_vals, width=w, label='New CERES',
           color='#d6604d', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(months_labels)
    ax.set_ylabel('Arctic mean RS  (W m⁻² per unit α)', fontsize=11)
    ax.set_title('Arctic mean Radiative Sensitivity: Old vs New CERES',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax2 = axes[1]
    colors = ['#d6604d' if v > 0 else '#4393c3' for v in diff_vals]
    ax2.bar(x, diff_vals, width=0.45, color=colors, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(months_labels)
    ax2.set_ylabel('Difference (New − Old)  (W m⁻² per unit α)', fontsize=11)
    ax2.set_title('RS difference per month', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for xi, dv in zip(x, diff_vals):
        ax2.text(xi, dv + (0.3 if dv >= 0 else -0.5),
                 f'{dv:+.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fpath2 = os.path.join(PLOT_DIR, 'RS_scalar_comparison_bar.png')
    fig.savefig(fpath2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fpath2}", flush=True)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print(f"{'Month':<12} {'Old RS':>10} {'New RS':>10} "
          f"{'Diff':>10} {'Diff %':>10}")
    print("-" * 60)
    for m in MONTH_NUMS:
        o = RS_scal_old[m]; n = RS_scal_new[m]; d = n - o
        print(f"{MONTH_NAMES[m]:<12} {o:>10.3f} {n:>10.3f} "
              f"{d:>+10.3f} {d/o*100:>+9.2f}%")
    print("=" * 60)
    print("DONE", flush=True)