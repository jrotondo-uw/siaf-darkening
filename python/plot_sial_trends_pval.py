'''#!/usr/bin/env python
"""
plot_sial_trends_pval.py
------------------------
SIAL & SIC trends by month (AMJJAS), 6-panel figure.
Pixel classification uses per-pixel OLS SIC trend from the new caches:
  Line 1 (blue)  : SIAL on EXTENT pixels  — significant negative SIC trend (p < PVAL)
  Line 2 (green) : SIAL on DARKENING pixels — stable ice (not significant, SIC_max >= 15%)
  Line 3 (red)   : SIC  on DARKENING pixels — tracks SIC on stable ice domain
Albedo: CLARA-A3 + APP-x averaged (nanmean).

Updated to loop through both observational and CMIP6 experiments.
Trends displayed per decade.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
from netCDF4 import Dataset as NC4Dataset
import glob, os, re, gc

# =============================================================================
# CONFIG
# =============================================================================

albedo_dir = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir   = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
sic_dir    = '/glade/derecho/scratch/jrotondo/SIC_data/'
output_dir = '/glade/work/jrotondo/ncc_paper/plots/'
os.makedirs(output_dir, exist_ok=True)

PVAL              = 0.05
SIC_SLOPE_THRESHOLD = -0.001  # -1% per decade (SIC as 0-1 fraction = -0.001/yr)
SIC_ICE_THRESHOLD = 0.15
PENTAD_DAYS       = 5

# We load pentads for the maximum possible range once, then slice by experiment
years_to_process  = list(range(1982, 2025)) 

MONTHS_TO_ANALYZE = {
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
}

EXPERIMENTS = {
    'obs': {
        'label': 'Observational Era (1982-2024)',
        'years': list(range(1982, 2025)),
        'cache': '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
    },
    'cmip6': {
        'label': 'CMIP6 Historical (1982-2014)',
        'years': list(range(1982, 2015)),
        'cache': '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
    }
}

FSIZE = 14
plt.rcParams.update({
    'font.size':       FSIZE,
    'axes.titlesize':  FSIZE,
    'axes.labelsize':  FSIZE,
    'xtick.labelsize': FSIZE - 2,
    'ytick.labelsize': FSIZE - 2,
    'legend.fontsize': 11,
    'font.family':     'sans-serif',
})

print("=" * 80)
print(f"SIAL & SIC TRENDS — magnitude classification (SIC slope <= {SIC_SLOPE_THRESHOLD*10:.3f} per decade)")
print("=" * 80)

# =============================================================================
# HELPERS
# =============================================================================

def get_pentad_from_doy(doy):
    return ((doy - 1) // 5) + 1

def trend_line(years, values):
    slope, intercept, r, p, _ = stats.linregress(years, values)
    return slope, intercept, r**2, p, slope * np.array(years) + intercept

def parse_date_appx(fname):
    base = os.path.basename(fname)
    try:
        return datetime.strptime(base.split('_d')[1][:8], "%Y%m%d")
    except Exception:
        return None

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

# =============================================================================
# STEP 1: INITIALIZE GRIDS
# =============================================================================

print("\nStep 1: Initializing Grids...")

sample_sial_f = sorted(glob.glob(f"{albedo_dir}/*.nc"))[0]
with xr.open_dataset(sample_sial_f) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

sample_sic = sorted(glob.glob(f"{sic_dir}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values; sic_y = ds['y'].values
sic_xx, sic_yy = np.meshgrid(sic_x, sic_y)
transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
sic_lon2d_psn, sic_lat2d_psn = transformer.transform(sic_xx, sic_yy)

sic_xyz  = latlon_to_xyz(sic_lat2d_psn.ravel(), sic_lon2d_psn.ravel())
sial_xyz = latlon_to_xyz(sial_lat2d.ravel(),    sial_lon2d.ravel())
sic_tree = cKDTree(sic_xyz)
_, idx_sic_to_sial = sic_tree.query(sial_xyz, workers=-1)

def regrid_sic_to_sial(sic_2d):
    return sic_2d.ravel()[idx_sic_to_sial].reshape(sial_shape)

month_to_pentads = {}
for month in range(1, 13):
    seen = []
    for day in range(1, 32):
        try:
            p = get_pentad_from_doy(datetime(2020, month, day).timetuple().tm_yday)
            if p not in seen: seen.append(p)
        except ValueError: continue
    month_to_pentads[month] = seen

# =============================================================================
# STEP 2: LOAD GLOBAL RAW PENTADS (1982-2024)
# =============================================================================

print("\nStep 2: Loading global pentad data (this only runs once)...")

# --- SIC PENTADS ---
sic_pentad_dict = {}
for year in years_to_process:
    print(f"  SIC {year}...", end='\r', flush=True)
    for f in sorted(glob.glob(f"{sic_dir}/sic_psn25_{year}*.nc")):
        try:
            with xr.open_dataset(f) as ds:
                t   = pd.Timestamp(ds['time'].values[0])
                raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                if np.nanmax(raw) > 1.5: raw /= 100.0
                raw = np.where(np.isfinite(raw), raw, np.nan)
            pentad = get_pentad_from_doy(t.timetuple().tm_yday)
            sic_pentad_dict.setdefault((t.year, pentad), []).append(regrid_sic_to_sial(raw))
        except Exception: continue

sic_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in sic_pentad_dict.items()}
del sic_pentad_dict; gc.collect()

# --- CLARA PENTADS ---
clara_pentad_dict = {}
for f in sorted(glob.glob(f"{albedo_dir}/*.nc")):
    try:
        dt = datetime.strptime(os.path.basename(f)[5:13], "%Y%m%d")
        if dt.year not in years_to_process: continue
        pentad = get_pentad_from_doy(dt.timetuple().tm_yday)
        with xr.open_dataset(f) as ds:
            snow = ds['black_sky_albedo_snow_mean'].squeeze() / 100.0
            reg  = ds['black_sky_albedo_mean'].squeeze()      / 100.0
            alb  = xr.where(~np.isnan(snow), snow, reg).values
        clara_pentad_dict.setdefault((dt.year, pentad), []).append(alb)
    except Exception: continue

clara_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in clara_pentad_dict.items()}
del clara_pentad_dict; gc.collect()

# --- APP-X PENTADS ---
sample_appx_f = sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx_f, 'r') as ds:
    alat = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]), np.nan).astype(float)
    alon = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
appx_lat2d, appx_lon2d = (np.meshgrid(alon, alat) if alat.ndim == 1 else (alat, alon))
appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
_, idx_appx_to_sial = cKDTree(sial_xyz).query(appx_xyz, workers=-1)

def regrid_appx_to_sial(appx_2d):
    flat  = appx_2d.ravel()
    accum = np.zeros(np.prod(sial_shape)); count = np.zeros(np.prod(sial_shape))
    valid = np.isfinite(flat)
    np.add.at(accum, idx_appx_to_sial[valid], flat[valid])
    np.add.at(count, idx_appx_to_sial[valid], 1)
    out = np.full(np.prod(sial_shape), np.nan)
    m   = count > 0; out[m] = accum[m] / count[m]
    return out.reshape(sial_shape)

best_by_date = {}
for f in sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True)):
    dt = parse_date_appx(f)
    if dt is None or dt.year not in years_to_process: continue
    obs_str = dt.strftime('%Y%m%d')
    m_cdate = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
    cdate   = m_cdate.group(1) if m_cdate else '00000000'
    if obs_str not in best_by_date or cdate > best_by_date[obs_str][1]:
        best_by_date[obs_str] = (f, cdate, dt)

appx_pentad_dict = {}
for obs_str, (f, cdate, dt) in sorted(best_by_date.items()):
    try:
        with NC4Dataset(f, 'r') as ds:
            alb = ds.variables['cdr_surface_albedo'][0]
            if hasattr(alb, 'filled'): alb = alb.filled(np.nan)
            alb = alb.astype(float); alb[alb == appx_fill] = np.nan
            alb[(alb < 0) | (alb > 1)] = np.nan
        regridded = regrid_appx_to_sial(alb)
        appx_pentad_dict.setdefault((dt.year, get_pentad_from_doy(dt.timetuple().tm_yday)), []).append(regridded)
    except Exception: continue

appx_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in appx_pentad_dict.items()}
del appx_pentad_dict; gc.collect()

# --- COMBINE ALBEDO ---
# Primary: CLARA-A3. Fill remaining NaNs with APP-x.
sial_pentad_combined = {}
for key in set(clara_pentad_mean.keys()) | set(appx_pentad_mean.keys()):
    c, a = clara_pentad_mean.get(key), appx_pentad_mean.get(key)
    if c is not None and a is not None:
        sial_pentad_combined[key] = np.where(np.isfinite(c), c, a)
    else:
        sial_pentad_combined[key] = c if c is not None else a

print(f"\n  Globally cached {len(sial_pentad_combined)} combined albedo pentads and {len(sic_pentad_mean)} SIC pentads.")

# =============================================================================
# EXPERIMENT LOOP
# =============================================================================

for exp_key, exp in EXPERIMENTS.items():
    print("\n" + "=" * 80)
    print(f"PROCESSING EXPERIMENT: {exp['label']}")
    print("=" * 80)
    
    sens_cache_dir = exp['cache']
    years_to_use   = exp['years']
    
    if not os.path.exists(sens_cache_dir):
        print(f"  [!] Missing cache directory: {sens_cache_dir}")
        continue

    arctic_60N_sial = np.load(f"{sens_cache_dir}/arctic_60N_sial.npy")

    slope_maps, pval_maps, sic_max_maps = {}, {}, {}
    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        slope_maps[month_name] = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{month_name}.npy")
        pval_maps[month_name]  = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{month_name}.npy")
        sic_early = np.load(f"{sens_cache_dir}/sic_early_sial_{month_name}.npy")
        sic_late  = np.load(f"{sens_cache_dir}/sic_late_sial_{month_name}.npy")
        sic_max_maps[month_name] = np.fmax(sic_early, sic_late)

    results = {}

    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        pentads = month_to_pentads[month_num]
        slope   = slope_maps[month_name]
        pval    = pval_maps[month_name]
        sic_max = sic_max_maps[month_name]

        # Valid domain constraints
        valid   = arctic_60N_sial & np.isfinite(slope) & np.isfinite(pval)
        has_ice = valid & (sic_max >= SIC_ICE_THRESHOLD)

        # Pixel classification: magnitude threshold (-1% SIC per decade)
        extent_mask    = has_ice & (slope <= SIC_SLOPE_THRESHOLD)
        darkening_mask = has_ice & ~extent_mask

        sial_extent, sial_darkening, sic_extent, sic_darkening = {}, {}, {}, {}

        for year in years_to_use:
            ext_vals, drk_sial_vals, ext_sic_vals, drk_sic_vals = [], [], [], []

            for pentad in pentads:
                key = (year, pentad)
                if key not in sial_pentad_combined or key not in sic_pentad_mean:
                    continue
                sial = sial_pentad_combined[key]
                sic  = sic_pentad_mean[key]

                m_ext = extent_mask & np.isfinite(sial)
                if m_ext.sum() > 0: ext_vals.append(np.nanmean(sial[m_ext]))

                m_drk_sial = darkening_mask & np.isfinite(sial)
                if m_drk_sial.sum() > 0: drk_sial_vals.append(np.nanmean(sial[m_drk_sial]))

                m_ext_sic = extent_mask & np.isfinite(sic)
                if m_ext_sic.sum() > 0: ext_sic_vals.append(np.nanmean(sic[m_ext_sic]))

                m_drk_sic = darkening_mask & np.isfinite(sic)
                if m_drk_sic.sum() > 0: drk_sic_vals.append(np.nanmean(sic[m_drk_sic]))

            if ext_vals:      sial_extent[year]    = np.mean(ext_vals)
            if drk_sial_vals: sial_darkening[year] = np.mean(drk_sial_vals)
            if ext_sic_vals:  sic_extent[year]     = np.mean(ext_sic_vals)
            if drk_sic_vals:  sic_darkening[year]  = np.mean(drk_sic_vals)

        results[month_name] = {
            'sial_extent':    sial_extent,
            'sial_darkening': sial_darkening,
            'sic_extent':     sic_extent,
            'sic_darkening':  sic_darkening,
            'n_extent':       int(extent_mask.sum()),
            'n_darkening':    int(darkening_mask.sum()),
        }

    # --- PLOTTING ---
    C_EXT  = '#2166ac' 
    C_DARK = '#4dac26' 
    C_SIC  = '#d62728' 

    C_SIC_EXT = '#d6740a'  # orange — SIC on extent pixels

    line_styles = {
        'sial_extent':    dict(color=C_EXT,     marker='o', ms=4, lw=1.8, label='Surface albedo — extent pixels (SIC trend ≤ −1% dec⁻¹)'),
        'sial_darkening': dict(color=C_DARK,    marker='s', ms=4, lw=1.8, label='Surface albedo — darkening pixels (stable ice)'),
        'sic_extent':     dict(color=C_SIC_EXT, marker='D', ms=4, lw=1.8, label='Sea ice concentration — extent pixels'),
        'sic_darkening':  dict(color=C_SIC,     marker='^', ms=4, lw=1.8, label='Sea ice concentration — darkening pixels'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, facecolor='white')
    axes = axes.flatten()
    fig.patch.set_facecolor('white')

    for idx, (month_name, month_num) in enumerate(MONTHS_TO_ANALYZE.items()):
        ax  = axes[idx]
        res = results[month_name]
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.25, linestyle='--', color='#cccccc', zorder=0)

        for key, style in line_styles.items():
            d = res[key]
            if not d: continue
            yrs  = np.array(sorted(d.keys()))
            vals = np.array([d[y] for y in yrs])
            if len(yrs) < 5: continue

            sl, ic, r2, p, fitted = trend_line(yrs, vals)
            is_sig      = p < 0.05
            # Significant but trivially small SIC change (<0.5% per decade)
            # — keep solid line but dim to flag as physically marginal
            is_meaningful = abs(sl * 10) >= 0.005  # 0.5% per decade in 0-1 units
            if is_sig and is_meaningful:
                pt_alph, ln_alph, ls = 0.70, 1.00, '-'
            elif is_sig and not is_meaningful:
                pt_alph, ln_alph, ls = 0.35, 0.45, '-'
            else:
                pt_alph, ln_alph, ls = 0.20, 0.30, '--'

            ax.scatter(yrs, vals, marker=style['marker'], color=style['color'], s=style['ms']**2, alpha=pt_alph, zorder=3, label=style['label'])
            ax.plot(yrs, fitted, color=style['color'], linewidth=style['lw'], linestyle=ls, alpha=ln_alph, zorder=4)

            sig_marker = '*' if is_sig else ''
            ann_color  = style['color'] if is_sig else '#aaaaaa'
            line_idx   = list(line_styles.keys()).index(key)
            # Display slope per decade (multiply per-year slope by 10)
            ax.text(0.97, 0.97 - line_idx * 0.08, f"{sig_marker}{sl*10:+.5f} dec\u207b\u00b9  R\u00b2={r2:.2f}  p={p:.3f}",
                    transform=ax.transAxes, fontsize=9, color=ann_color, fontweight='bold', va='top', ha='right',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.15', lw=0), zorder=5)

        ax.text(0.03, 0.97, f"Extent: {res['n_extent']:,} px  |  Dark: {res['n_darkening']:,} px",
                transform=ax.transAxes, fontsize=8, va='top', ha='left', color='#444444',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.15', lw=0), zorder=5)

        ax.set_title(month_name, fontweight='bold', pad=4)
        ax.set_xlim(years_to_use[0] - 1, years_to_use[-1] + 1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

        if idx >= 3: ax.set_xlabel('Year', fontweight='bold')
        if idx % 3 == 0: ax.set_ylabel('Albedo / Sea Ice Concentration', fontweight='bold')

        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen: seen.add(l); h2.append(h); l2.append(l)

    h2 += [
        plt.Line2D([0], [0], color='#555555', lw=1.8, linestyle='-',  alpha=1.0),
        plt.Line2D([0], [0], color='#555555', lw=1.8, linestyle='-',  alpha=0.45),
        plt.Line2D([0], [0], color='#555555', lw=1.8, linestyle='--', alpha=0.30),
    ]
    l2 += [
        'Solid (full) = significant & ≥0.5% dec⁻¹',
        'Solid (dim)  = significant but <0.5% dec⁻¹ (physically marginal)',
        'Dashed       = not significant (p ≥ 0.05)',
    ]

    fig.legend(h2, l2, loc='lower center', ncol=2, fontsize=11, frameon=True, framealpha=0.95,
               edgecolor='#cccccc', fancybox=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.5, handleheight=0.9, columnspacing=1.5, title=exp['label'], title_fontsize=12)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.12, wspace=0.25, hspace=0.20)

    fpath = os.path.join(output_dir, f'sial_sic_trends_pval_6panel_{exp_key}.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Plot: {fpath}")

    # --- SUMMARY TABLE ---
    print("\n  " + "-" * 88)
    hdr = (f"  {'Month':<12} {'Series':<35} {'Mean':>7} {'Slope/dec':>11} "
           f"{'Total':>9} {'R2':>6} {'p':>8} {'Sig':>5}")
    print(hdr)
    print("  " + "-" * 88)

    for month_name in MONTHS_TO_ANALYZE:
        res = results[month_name]
        for key, label in [
            ('sial_extent',    'Surface albedo (extent pixels)'),
            ('sial_darkening', 'Surface albedo (darkening pixels)'),
            ('sic_extent',     'Sea ice conc. (extent pixels)'),
            ('sic_darkening',  'Sea ice conc. (darkening pixels)'),
        ]:
            d = res[key]
            if not d: continue
            yrs  = np.array(sorted(d.keys()))
            vals = np.array([d[y] for y in yrs])
            if len(yrs) < 5: continue
            sl, _, r2, p, _ = trend_line(yrs, vals)
            sig   = 'YES' if p < 0.05 else 'no'
            total = sl * (yrs[-1] - yrs[0])
            # Display slope per decade (multiply per-year slope by 10)
            print(f"  {month_name:<12} {label:<35} "
                  f"{np.mean(vals):7.4f} {sl*10:+11.6f} "
                  f"{total:+9.4f} {r2:6.3f} {p:8.4f} {sig:>5}")
        print("  " + "-" * 88)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import xarray as xr
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
import glob, os, re

# ── CONFIG ───────────────────────────────────────────────────────────────────
sens_cache_dir = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
albedo_dir     = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir       = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
output_dir     = '/glade/work/jrotondo/ncc_paper/plots/'
os.makedirs(output_dir, exist_ok=True)

SIC_SLOPE_THRESH = -0.001
SIC_ICE_THRESH   = 0.15
PVAL             = 0.05
MIN_CLARA_PX     = 500

MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']
MONTHS_NUM = {
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
}
COLORS = {
    'April':     '#e41a1c',
    'May':       '#ff7f00',
    'June':      '#4daf4a',
    'July':      '#377eb8',
    'August':    '#984ea3',
    'September': '#a65628',
}

# ── BUILD GRIDS & REGRID INDEX ───────────────────────────────────────────────
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])

print("Building grids and regrid index...")

sample_sial = sorted(glob.glob(f"{albedo_dir}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape
sial_xyz   = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
sial_tree  = cKDTree(sial_xyz)

sample_appx = sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill    = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if appx_lat_raw.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
appx_shape = appx_lat2d.shape
appx_xyz   = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())

_, idx_appx_to_sial = sial_tree.query(appx_xyz, workers=-1)

def regrid_appx_to_sial(appx_2d):
    flat  = appx_2d.ravel()
    accum = np.zeros(np.prod(sial_shape))
    count = np.zeros(np.prod(sial_shape))
    valid = np.isfinite(flat)
    np.add.at(accum, idx_appx_to_sial[valid], flat[valid])
    np.add.at(count, idx_appx_to_sial[valid], 1)
    out    = np.full(np.prod(sial_shape), np.nan)
    m      = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(sial_shape)

arctic_mask = np.load(f"{sens_cache_dir}/arctic_60N_sial.npy")
print(f"  SIAL grid: {sial_shape}  |  APP-x grid: {appx_shape}")

# ── HELPER: load raw CLARA for a given year/month ────────────────────────────
def load_clara_raw(year, month_num):
    sial_files = [f for f in sorted(glob.glob(f"{albedo_dir}/*.nc"))
                  if str(year) in f]
    if not sial_files:
        return None
    try:
        with xr.open_mfdataset(sial_files, combine='by_coords') as ds:
            snow_alb   = ds['black_sky_albedo_snow_mean'] / 100.0
            reg_alb    = ds['black_sky_albedo_mean'] / 100.0
            sial_raw   = xr.where(~np.isnan(snow_alb), snow_alb, reg_alb)
            month_data = sial_raw.sel(
                time=sial_raw['time.month'] == month_num
            ).mean(dim='time').values
        if month_data.shape == sial_shape:
            return month_data
    except Exception as e:
        print(f"  [!] CLARA raw load failed {year}: {e}")
    return None

# ── DIAGNOSTIC: CLARA valid darkening pixel counts per month ─────────────────
print("\n" + "=" * 70)
print("DIAGNOSTIC: CLARA valid darkening pixel counts")
print(f"{'Month':<12} {'min':>6} {'max':>6} {'mean':>6} "
      f"{'n_below_500':>12} {'n_years':>8}")
print("=" * 70)

for month in MONTHS:
    month_num = MONTHS_NUM[month]
    slope   = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{month}.npy")
    pval    = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{month}.npy")
    sic_e   = np.load(f"{sens_cache_dir}/sic_early_sial_{month}.npy")
    sic_l   = np.load(f"{sens_cache_dir}/sic_late_sial_{month}.npy")
    sic_max = np.fmax(sic_e, sic_l)
    valid          = arctic_mask & np.isfinite(slope) & np.isfinite(pval)
    has_ice        = valid & (sic_max >= SIC_ICE_THRESH)
    extent_mask    = has_ice & (slope <= SIC_SLOPE_THRESH)
    darkening_mask = has_ice & ~extent_mask

    counts = {}
    for year in range(1982, 2025):
        # Try cache first
        p_c = f"{sens_cache_dir}/clara_alb_{month}_{year}.npy"
        if os.path.exists(p_c):
            arr = np.load(p_c)
            if arr.shape == sial_shape:
                counts[year] = int(np.sum(darkening_mask & np.isfinite(arr)))
                continue
        # Fall back to raw
        arr = load_clara_raw(year, month_num)
        if arr is not None:
            counts[year] = int(np.sum(darkening_mask & np.isfinite(arr)))

    if counts:
        vals = list(counts.values())
        print(f"{month:<12} {min(vals):>6d} {max(vals):>6d} "
              f"{int(np.mean(vals)):>6d} "
              f"{sum(1 for v in vals if v < MIN_CLARA_PX):>12d} "
              f"{len(vals):>8d}")
    else:
        print(f"{month:<12} {'no data':>6}")

print("=" * 70)

# ── RECONSTRUCT SIAL-DARKENING TIME SERIES ───────────────────────────────────
sial_dark_ts = {}

for month in MONTHS:
    month_num = MONTHS_NUM[month]
    print(f"\nLoading {month}...", flush=True)

    slope   = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{month}.npy")
    pval    = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{month}.npy")
    sic_e   = np.load(f"{sens_cache_dir}/sic_early_sial_{month}.npy")
    sic_l   = np.load(f"{sens_cache_dir}/sic_late_sial_{month}.npy")
    sic_max = np.fmax(sic_e, sic_l)

    valid          = arctic_mask & np.isfinite(slope) & np.isfinite(pval)
    has_ice        = valid & (sic_max >= SIC_ICE_THRESH)
    extent_mask    = has_ice & (slope <= SIC_SLOPE_THRESH)
    darkening_mask = has_ice & ~extent_mask

    year_vals = {}
    for year in range(1982, 2025):
        clara_map = None
        appx_map  = None

        # ── 1. Try CLARA cache ────────────────────────────────────────────
        p_c = f"{sens_cache_dir}/clara_alb_{month}_{year}.npy"
        if os.path.exists(p_c):
            arr = np.load(p_c)
            if arr.shape == sial_shape:
                clara_map = arr

        # ── 2. Try APP-x cache (regrid to SIAL) ──────────────────────────
        p_a = f"{sens_cache_dir}/appx_alb_{month}_{year}.npy"
        if os.path.exists(p_a):
            arr = np.load(p_a)
            if arr.shape == appx_shape:
                appx_map = regrid_appx_to_sial(arr)
            elif arr.shape == sial_shape:
                appx_map = arr

        # ── 3. Fall back to raw CLARA for uncached middle years ───────────
        if clara_map is None:
            arr = load_clara_raw(year, month_num)
            if arr is not None:
                clara_map = arr

        # ── 4. Check CLARA has sufficient darkening-pixel coverage ────────
        if clara_map is not None:
            clara_valid_px = int(np.sum(darkening_mask & np.isfinite(clara_map)))
            if clara_valid_px < MIN_CLARA_PX:
                print(f"  [!] {month} {year}: CLARA only {clara_valid_px} valid "
                      f"darkening px (< {MIN_CLARA_PX}) — demoting to missing")
                clara_map = None

        # ── 5. Priority: CLARA alone > pixel-merge > APP-x alone ─────────
        if clara_map is not None and appx_map is not None:
            alb = np.where(np.isfinite(clara_map), clara_map, appx_map)
        elif clara_map is not None:
            alb = clara_map
        elif appx_map is not None:
            alb = appx_map
            print(f"  [~] {month} {year}: APP-x only (no CLARA)")
        else:
            print(f"  [x] {month} {year}: no data")
            continue

        m = darkening_mask & np.isfinite(alb)
        if m.sum() > 0:
            year_vals[year] = float(np.nanmean(alb[m]))

    sial_dark_ts[month] = year_vals
    print(f"  → {len(year_vals)} years loaded for {month}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor('white')
ax.grid(True, alpha=0.25, linestyle='--', color='#cccccc', zorder=0)

for month in MONTHS:
    d = sial_dark_ts[month]
    if len(d) < 5:
        print(f"  [!] Skipping {month} — only {len(d)} years")
        continue

    yrs  = np.array(sorted(d.keys()))
    vals = np.array([d[y] for y in yrs])
    col  = COLORS[month]

    sl, ic, r2, p, _ = stats.linregress(yrs, vals)
    fitted  = sl * yrs + ic
    sig     = p < PVAL
    ls      = '-' if sig else '--'
    sig_str = '*' if sig else 'ns'

    ax.scatter(yrs, vals, color=col, s=20, alpha=0.6, zorder=3)
    ax.plot(yrs, fitted, color=col, lw=2.0, linestyle=ls, zorder=4,
            label=f"{month:<10} {sl*10:+.4f} dec⁻¹   R²={r2:.2f}   {sig_str}")

ax.set_xlabel('Year', fontsize=18)
ax.set_ylabel('Mean Surface Albedo (darkening pixels)', fontsize=18)
'''ax.set_title(
    'SIAL Trend on Darkening Pixels — All Months (1982–2024)\n'
    f'CLARA-A3 primary (≥{MIN_CLARA_PX} valid px); APP-x fills missing. '
    'Solid = significant (p < 0.05), Dashed = not significant',
    fontweight='bold', fontsize=18)'''
ax.set_xlim(1981, 2025)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.tick_params(labelsize=18)
ax.legend(loc='upper right', framealpha=0.95, fontsize=18,
          title='Month          slope              fit   sig',
          title_fontsize=18, prop={'family': 'monospace'})
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
fpath = os.path.join(output_dir, 'sial_darkening_all_months_full.png')
plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nSaved: {fpath}")