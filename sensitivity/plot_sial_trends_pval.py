"""
plot_sial_trends_pval.py
------------------------
SIAL & SIC trends by month (AMJJAS), 6-panel figure.
Pixel classification uses per-pixel OLS SIC trend (1982-2024):
  Line 1 (blue)  : SIAL on EXTENT pixels  — significant negative SIC trend (p < PVAL)
  Line 2 (green) : SIAL on DARKENING pixels — stable ice (not significant, SIC_max >= 15%)
  Line 3 (red)   : SIC  on DARKENING pixels — tracks SIC on stable ice domain
Albedo: CLARA-A3 + APP-x averaged (nanmean).
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy import stats
from scipy.interpolate import griddata
from netCDF4 import Dataset as NC4Dataset
import glob, os, re, gc

# =============================================================================
# CONFIG
# =============================================================================

albedo_dir     = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir       = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
sic_dir        = '/glade/derecho/scratch/jrotondo/SIC_data/'
sens_cache_dir = '/glade/work/jrotondo/siaf_extended/sensitivity_cache/'
output_dir     = '/glade/work/jrotondo/siaf_extended/plots/'
os.makedirs(output_dir, exist_ok=True)

PVAL              = 0.05    # p-value threshold for extent classification
SIC_ICE_THRESHOLD = 0.15
PENTAD_DAYS       = 5

years_to_use      = list(range(1982, 2025))
EARLY_YEARS       = list(range(1982, 1992))
months_to_analyze = [4, 5, 6, 7, 8, 9]

month_names_short = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']
month_names_full  = ['January','February','March','April','May','June',
                     'July','August','September','October','November','December']

MONTHS_TO_ANALYZE = {
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
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
print(f"SIAL & SIC TRENDS — p-value classification (p < {PVAL})")
print(f"CLARA-A3 + APP-x averaged  |  1982-2024")
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

# =============================================================================
# STEP 0: LOAD CACHED GRID + OLS TREND MAPS + SIC CLIMATOLOGIES
# =============================================================================

print("\nStep 0: Loading cached grid metadata and OLS trend maps...")

arctic_60N_sial = np.load(f"{sens_cache_dir}/arctic_60N_sial.npy")
arctic_60N_appx = np.load(f"{sens_cache_dir}/arctic_60N_appx.npy")

# Load OLS trend maps and SIC climatologies per month
slope_maps   = {}
pval_maps    = {}
sic_max_maps = {}
sic_early_maps = {}

for month_name, month_num in MONTHS_TO_ANALYZE.items():
    slope_maps[month_name]  = np.load(
        f"{sens_cache_dir}/sic_trend_slope_sial_{month_name}.npy")
    pval_maps[month_name]   = np.load(
        f"{sens_cache_dir}/sic_trend_pval_sial_{month_name}.npy")
    sic_early = np.load(f"{sens_cache_dir}/sic_early_sial_{month_name}.npy")
    sic_late  = np.load(f"{sens_cache_dir}/sic_late_sial_{month_name}.npy")
    sic_max_maps[month_name]   = np.fmax(sic_early, sic_late)
    sic_early_maps[month_name] = sic_early

# Load climo NaN masks
climo_nan_clara = {m: np.load(f"{sens_cache_dir}/climo_nan_clara_{m}.npy")
                   for m in MONTHS_TO_ANALYZE}
climo_nan_appx  = {m: np.load(f"{sens_cache_dir}/climo_nan_appx_{m}.npy")
                   for m in MONTHS_TO_ANALYZE}

print("  Cached trend maps loaded.")

# =============================================================================
# STEP 1: LOAD CLARA-A3 GRID AND BUILD SIC PENTAD DICT (on SIAL grid)
# =============================================================================

print("\nStep 1: Loading SIC pentads onto SIAL grid...")

sample_sial_f = sorted(glob.glob(f"{albedo_dir}/*.nc"))[0]
with xr.open_dataset(sample_sial_f) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

from scipy.spatial import cKDTree
from pyproj import Transformer

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

sample_sic = sorted(glob.glob(f"{sic_dir}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values; sic_y = ds['y'].values
sic_xx, sic_yy = np.meshgrid(sic_x, sic_y)
transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
sic_lon2d_psn, sic_lat2d_psn = transformer.transform(sic_xx, sic_yy)

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

sic_xyz  = latlon_to_xyz(sic_lat2d_psn.ravel(), sic_lon2d_psn.ravel())
sial_xyz = latlon_to_xyz(sial_lat2d.ravel(),    sial_lon2d.ravel())
sic_tree = cKDTree(sic_xyz)
_, idx_sic_to_sial = sic_tree.query(sial_xyz, workers=-1)

def regrid_sic_to_sial(sic_2d):
    return sic_2d.ravel()[idx_sic_to_sial].reshape(sial_shape)

# Month to pentad lookup
month_to_pentads = {}
for month in range(1, 13):
    seen = []
    for day in range(1, 32):
        try:
            p = get_pentad_from_doy(datetime(2020, month, day).timetuple().tm_yday)
            if p not in seen: seen.append(p)
        except ValueError:
            continue
    month_to_pentads[month] = seen

# Load SIC pentads
sic_pentad_dict = {}
for year in years_to_use:
    print(f"  SIC {year}...", end='\r', flush=True)
    for f in sorted(glob.glob(f"{sic_dir}/sic_psn25_{year}*.nc")):
        try:
            with xr.open_dataset(f) as ds:
                t   = pd.Timestamp(ds['time'].values[0])
                raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                if np.nanmax(raw) > 1.5: raw /= 100.0
                raw = np.where(np.isfinite(raw), raw, np.nan)
            pentad = get_pentad_from_doy(t.timetuple().tm_yday)
            key    = (t.year, pentad)
            sic_pentad_dict.setdefault(key, []).append(regrid_sic_to_sial(raw))
        except Exception:
            continue

# Average daily SIC within each pentad
sic_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in sic_pentad_dict.items()}
del sic_pentad_dict; gc.collect()
print(f"\n  Loaded {len(sic_pentad_mean)} SIC pentads on SIAL grid.")

# =============================================================================
# STEP 2: LOAD CLARA-A3 ALBEDO PENTADS
# =============================================================================

print("\nStep 2: Loading CLARA-A3 albedo pentads...")

clara_pentad_dict = {}
for f in sorted(glob.glob(f"{albedo_dir}/*.nc")):
    try:
        date_str = os.path.basename(f)[5:13]
        dt       = datetime.strptime(date_str, "%Y%m%d")
        if dt.year not in years_to_use: continue
        pentad = get_pentad_from_doy(dt.timetuple().tm_yday)
        with xr.open_dataset(f) as ds:
            snow = ds['black_sky_albedo_snow_mean'].squeeze() / 100.0
            reg  = ds['black_sky_albedo_mean'].squeeze()      / 100.0
            alb  = xr.where(~np.isnan(snow), snow, reg).values
        clara_pentad_dict.setdefault((dt.year, pentad), []).append(alb)
    except Exception:
        continue

clara_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in clara_pentad_dict.items()}
del clara_pentad_dict; gc.collect()
print(f"  Loaded {len(clara_pentad_mean)} CLARA-A3 pentads.")

# =============================================================================
# STEP 3: LOAD APP-x ALBEDO PENTADS
# =============================================================================

print("\nStep 3: Loading APP-x albedo pentads...")

sample_appx_f = sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx_f, 'r') as ds:
    alat = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]), np.nan).astype(float)
    alon = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if alat.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(alon, alat)
else:
    appx_lat2d, appx_lon2d = alat, alon
appx_shape = appx_lat2d.shape
appx_xyz   = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
_, idx_appx_to_sial = cKDTree(sial_xyz).query(appx_xyz, workers=-1)

def regrid_appx_to_sial(appx_2d):
    """Scatter-average APP-x onto SIAL grid."""
    flat  = appx_2d.ravel()
    accum = np.zeros(np.prod(sial_shape))
    count = np.zeros(np.prod(sial_shape))
    valid = np.isfinite(flat)
    np.add.at(accum, idx_appx_to_sial[valid], flat[valid])
    np.add.at(count, idx_appx_to_sial[valid], 1)
    out = np.full(np.prod(sial_shape), np.nan)
    m   = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(sial_shape)

# Deduplicate APP-x by obs date
best_by_date = {}
for f in sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True)):
    dt = parse_date_appx(f)
    if dt is None or dt.year not in years_to_use: continue
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
            alb = alb.astype(float)
            alb[alb == appx_fill] = np.nan
            alb[(alb < 0) | (alb > 1)] = np.nan
        pentad = get_pentad_from_doy(dt.timetuple().tm_yday)
        regridded = regrid_appx_to_sial(alb)
        appx_pentad_dict.setdefault((dt.year, pentad), []).append(regridded)
    except Exception:
        continue

appx_pentad_mean = {k: np.nanmean(v, axis=0) for k, v in appx_pentad_dict.items()}
del appx_pentad_dict; gc.collect()
print(f"  Loaded {len(appx_pentad_mean)} APP-x pentads (regridded to SIAL grid).")

# =============================================================================
# STEP 4: COMBINE CLARA + APP-x INTO SINGLE PENTAD DICT
# =============================================================================

print("\nStep 4: Combining CLARA-A3 + APP-x (nanmean)...")

all_keys = set(clara_pentad_mean.keys()) | set(appx_pentad_mean.keys())
sial_pentad_combined = {}
for key in all_keys:
    c = clara_pentad_mean.get(key)
    a = appx_pentad_mean.get(key)
    if c is not None and a is not None:
        sial_pentad_combined[key] = np.where(
            np.isfinite(c) & np.isfinite(a), (c + a) / 2.0,
            np.where(np.isfinite(c), c, a))
    elif c is not None:
        sial_pentad_combined[key] = c
    else:
        sial_pentad_combined[key] = a

print(f"  Combined pentads: {len(sial_pentad_combined)}")

# =============================================================================
# STEP 5: COMPUTE YEARLY TIME SERIES PER MONTH
# =============================================================================

print(f"\nStep 5: Computing yearly time series (p < {PVAL} classification)...")

results = {}

for month_name, month_num in MONTHS_TO_ANALYZE.items():
    print(f"  {month_name}...")
    pentads  = month_to_pentads[month_num]
    slope    = slope_maps[month_name]
    pval     = pval_maps[month_name]
    sic_max  = sic_max_maps[month_name]
    climo_ex = climo_nan_clara[month_name]   # use CLARA NaN mask for valid domain

    valid    = arctic_60N_sial & np.isfinite(slope) & np.isfinite(pval) & ~climo_ex
    has_ice  = valid & (sic_max >= SIC_ICE_THRESHOLD)

    # Pixel masks (static, based on full 1982-2024 OLS trend)
    extent_mask    = has_ice & (slope < 0) & (pval < PVAL)   # significant retreat
    darkening_mask = has_ice & ~extent_mask                   # stable ice

    sial_extent    = {}
    sial_darkening = {}
    sic_darkening  = {}

    for year in years_to_use:
        ext_vals = []; drk_sial_vals = []; drk_sic_vals = []

        for pentad in pentads:
            key = (year, pentad)
            if key not in sial_pentad_combined or key not in sic_pentad_mean:
                continue
            sial = sial_pentad_combined[key]
            sic  = sic_pentad_mean[key]

            # Line 1: SIAL on extent pixels
            m = extent_mask & np.isfinite(sial)
            if m.sum() > 0:
                ext_vals.append(np.nanmean(sial[m]))

            # Line 2: SIAL on darkening pixels
            m = darkening_mask & np.isfinite(sial)
            if m.sum() > 0:
                drk_sial_vals.append(np.nanmean(sial[m]))

            # Line 3: SIC on darkening pixels
            m = darkening_mask & np.isfinite(sic)
            if m.sum() > 0:
                drk_sic_vals.append(np.nanmean(sic[m]))

        if ext_vals:      sial_extent[year]    = np.mean(ext_vals)
        if drk_sial_vals: sial_darkening[year] = np.mean(drk_sial_vals)
        if drk_sic_vals:  sic_darkening[year]  = np.mean(drk_sic_vals)

    results[month_name] = {
        'sial_extent':    sial_extent,
        'sial_darkening': sial_darkening,
        'sic_darkening':  sic_darkening,
        'n_extent':       int(extent_mask.sum()),
        'n_darkening':    int(darkening_mask.sum()),
    }

    print(f"    Extent pixels: {results[month_name]['n_extent']:,}  "
          f"Darkening pixels: {results[month_name]['n_darkening']:,}")

# =============================================================================
# STEP 6: 6-PANEL FIGURE
# =============================================================================

print("\nStep 6: Creating 6-panel figure...")

C_EXT  = '#2166ac'   # blue  — extent
C_DARK = '#4dac26'   # green — darkening SIAL
C_SIC  = '#d62728'   # red   — SIC on darkening pixels

line_styles = {
    'sial_extent':    dict(color=C_EXT,  marker='o', ms=4, lw=1.8,
                           label=f'Surface albedo — extent pixels (p < {PVAL}, slope < 0)'),
    'sial_darkening': dict(color=C_DARK, marker='s', ms=4, lw=1.8,
                           label='Surface albedo — darkening pixels (stable ice)'),
    'sic_darkening':  dict(color=C_SIC,  marker='^', ms=4, lw=1.8,
                           label='Sea ice concentration — darkening pixels'),
}

fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True,
                          facecolor='white')
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
        is_sig  = p < 0.05
        pt_alph = 0.70 if is_sig else 0.20
        ln_alph = 1.00 if is_sig else 0.30
        ls      = '-'  if is_sig else '--'

        ax.scatter(yrs, vals,
                   marker=style['marker'], color=style['color'],
                   s=style['ms']**2, alpha=pt_alph, zorder=3,
                   label=style['label'])
        ax.plot(yrs, fitted,
                color=style['color'], linewidth=style['lw'],
                linestyle=ls, alpha=ln_alph, zorder=4)

        # Slope annotation
        sig_marker = '*' if is_sig else ''
        ann_color  = style['color'] if is_sig else '#aaaaaa'
        line_idx   = list(line_styles.keys()).index(key)
        ax.text(0.97, 0.97 - line_idx * 0.09,
                f"{sig_marker}{sl:+.5f} yr\u207b\u00b9  R\u00b2={r2:.2f}  p={p:.3f}",
                transform=ax.transAxes,
                fontsize=9, color=ann_color, fontweight='bold',
                va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.7,
                          boxstyle='round,pad=0.15', lw=0),
                zorder=5)

    # Pixel count badge
    ax.text(0.03, 0.97,
            f"Extent: {res['n_extent']:,} px  |  Dark: {res['n_darkening']:,} px",
            transform=ax.transAxes,
            fontsize=8, va='top', ha='left', color='#444444',
            bbox=dict(facecolor='white', alpha=0.7,
                      boxstyle='round,pad=0.15', lw=0),
            zorder=5)

    ax.set_title(month_name, fontweight='bold', pad=4)
    ax.set_xlim(years_to_use[0] - 1, years_to_use[-1] + 1)
    ax.set_xticks(range(1985, 2025, 10))

    if idx >= 3:
        ax.set_xlabel('Year', fontweight='bold')
    if idx % 3 == 0:
        ax.set_ylabel('Albedo / Sea Ice Concentration', fontweight='bold')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
seen, h2, l2 = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); h2.append(h); l2.append(l)

# Add legend entries for line styles
h2 += [plt.Line2D([0], [0], color='#555555', lw=1.8, linestyle='-'),
        plt.Line2D([0], [0], color='#555555', lw=1.8, linestyle='--')]
l2 += ['Solid = significant trend (p < 0.05)',
        'Dashed = not significant (p >= 0.05)']

fig.legend(h2, l2,
           loc='lower center', ncol=2,
           fontsize=11, frameon=True, framealpha=0.95,
           edgecolor='#cccccc', fancybox=False,
           bbox_to_anchor=(0.5, -0.01),
           handlelength=1.5, handleheight=0.9,
           columnspacing=1.5)

plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.12,
                    wspace=0.25, hspace=0.20)

fpath = os.path.join(output_dir, 'sial_sic_trends_pval_6panel.png')
fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print(f"\nSaved: {fpath}")

# =============================================================================
# STEP 7: SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 90)
print(f"SUMMARY  |  p < {PVAL} classification  |  CLARA-A3 + APP-x")
print("=" * 90)
hdr = (f"{'Month':<12} {'Series':<35} {'Mean':>7} {'Slope/yr':>11} "
       f"{'Total':>9} {'R2':>6} {'p':>8} {'Sig':>5}")
print(hdr)
print("-" * 90)

for month_name in MONTHS_TO_ANALYZE:
    res = results[month_name]
    for key, label in [
        ('sial_extent',    'Surface albedo (extent pixels)'),
        ('sial_darkening', 'Surface albedo (darkening pixels)'),
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
        print(f"{month_name:<12} {label:<35} "
              f"{np.mean(vals):7.4f} {sl:+11.6f} "
              f"{total:+9.4f} {r2:6.3f} {p:8.4f} {sig:>5}")
    print()

print("=" * 90)
print("DONE")
print("=" * 90)