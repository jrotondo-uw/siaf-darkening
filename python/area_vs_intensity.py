#!/usr/bin/env python
"""
siaf_ext_drk_decomp.py
----------------------
Decomposes observed SIAF change into extent-loss vs darkening components.

    ΔSIAF  =  ΔSIAF_ext  +  ΔSIAF_drk
                  ↑               ↑
           ice retreat       ice darkening
         (area effect)     (intensity effect)

Comparison: CMIP6 window (1982-1991 vs 2005-2014)
        vs  Obs-era window (1982-1991 vs 2015-2024)
Both periods are fully observational; labels indicate time windows only.

Scaling: Huang ERA5 surface-albedo kernel throughout.
Pistone numbers from the previous run are shown at the end for reference.

Outputs:
    siaf_ext_drk_summary.png   — monthly bar chart
    siaf_ext_drk_map_huang.png — Pan-Arctic spatial map
                                 (ext in blue, darkening in orange)

Memory strategy:
  * Scatter indices (SIAL->PSN25, APP-x->SIAL, ERA5->PSN25) are saved to
    OUT_DIR as .npy files and reloaded on subsequent runs — building a
    cKDTree over 26 M SIAL points happens only once ever.
  * All xyz arrays use float32 (3x less RAM than float64).
  * Queries are chunked (200k points at a time).
  * Per-month IS arrays are explicitly deleted before moving on.
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not available -- spatial map uses flat pcolormesh fallback")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DIR_SIAL          = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX          = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC           = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6       = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS         = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
HUANG_KERNEL_FILE = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
OUT_DIR           = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

# Delta-T from previous run (4-source mean)
DELTA_T_CMIP6 = 0.5862
DELTA_T_OBS   = 0.8831

MONTH_NAMES = {4: 'April', 5: 'May', 6: 'June',
               7: 'July',  8: 'August', 9: 'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
SIC_THR     = 0.15
PVAL        = 0.05

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ─────────────────────────────────────────────────────────────────────
# MEMORY-EFFICIENT SCATTER INDEX  (float32 xyz, chunked, disk-cached)
# ─────────────────────────────────────────────────────────────────────
def _xyz32(lat_deg, lon_deg):
    """Unit-sphere xyz in float32 -- 3x less RAM than float64."""
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])


def build_or_load_idx(src_lat, src_lon, tgt_lat, tgt_lon,
                      cache_path, chunk=200_000):
    """
    NN index: src_pixel -> nearest tgt_pixel.
    Saved to disk on first call; loaded instantly on subsequent calls.
    """
    if os.path.exists(cache_path):
        print(f"    loaded  {os.path.basename(cache_path)}", flush=True)
        return np.load(cache_path)

    print(f"    building {os.path.basename(cache_path)} ...", flush=True)
    tgt_xyz = _xyz32(tgt_lat, tgt_lon)
    tree    = cKDTree(tgt_xyz)
    del tgt_xyz; gc.collect()

    n          = int(np.asarray(src_lat).size)
    idx        = np.empty(n, dtype=np.int32)
    src_lat_f  = np.asarray(src_lat, dtype=np.float32).ravel()
    src_lon_f  = np.asarray(src_lon, dtype=np.float32).ravel()

    for i in range(0, n, chunk):
        e = min(i + chunk, n)
        q = _xyz32(src_lat_f[i:e], src_lon_f[i:e])
        _, idx[i:e] = tree.query(q, workers=1)
        del q

    del tree, src_lat_f, src_lon_f; gc.collect()
    np.save(cache_path, idx)
    return idx


# ─────────────────────────────────────────────────────────────────────
# PSN25 GRID + DOMAIN MASK
# ─────────────────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("PSN25 grid...", flush=True)

sample_sic = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy       = np.meshgrid(psn_x, psn_y)
psn_shape             = psn_xx.shape
transformer           = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d  = transformer.transform(psn_xx, psn_yy)
del psn_xx, psn_yy; gc.collect()

phi    = np.deg2rad(psn_lat2d)
phi_ts = np.deg2rad(70.0)
k_ps   = (1.0 + np.sin(phi_ts)) / np.maximum(1.0 + np.sin(phi), 1e-9)
AREA_W = (1.0 / k_ps**2).astype(np.float32)
del phi, k_ps; gc.collect()

try:
    from global_land_mask import globe
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    OCEAN  = globe.is_ocean(psn_lat2d, lon_pm)
    del lon_pm
except ImportError:
    print("  global_land_mask not installed -- land not excluded")
    OCEAN = np.ones(psn_shape, dtype=bool)

DOMAIN  = OCEAN & (psn_lat2d >= 60.0)
A_TOTAL = float(np.sum(AREA_W[DOMAIN]))
del OCEAN; gc.collect()
print(f"  Arctic domain: {int(DOMAIN.sum()):,} PSN25 cells", flush=True)

# ─────────────────────────────────────────────────────────────────────
# SIAL GRID
# ─────────────────────────────────────────────────────────────────────
print("SIAL grid...", flush=True)
sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values.astype(np.float32)
    sial_lon1d = ds['lon'].values.astype(np.float32)
sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
sial_shape = sial_lat2d.shape
print(f"  SIAL: {sial_shape}  ({sial_lat2d.size/1e6:.1f} M pts)", flush=True)

try:
    arctic_sial = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy").astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.0)

# ─────────────────────────────────────────────────────────────────────
# SCATTER INDICES  (built once, saved to disk)
# ─────────────────────────────────────────────────────────────────────
print("Scatter indices...", flush=True)

idx_s2p = build_or_load_idx(
    sial_lat2d, sial_lon2d,
    psn_lat2d,  psn_lon2d,
    os.path.join(OUT_DIR, '_idx_sial_to_psn25.npy'))
gc.collect()

idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        alat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(np.float32)
        alon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(np.float32)
    if alat.ndim == 1:
        alon2d, alat2d = np.meshgrid(alon, alat)
    else:
        alat2d, alon2d = alat, alon
    idx_appx_to_sial = build_or_load_idx(
        alat2d, alon2d,
        sial_lat2d, sial_lon2d,
        os.path.join(OUT_DIR, '_idx_appx_to_sial.npy'))
    del alat2d, alon2d, alat, alon; gc.collect()
except Exception as e:
    print(f"  APP-x index failed ({e}) -- September will use CLARA-A3")

# ERA5 kernel -> PSN25 (small grid, fast)
with xr.open_dataset(HUANG_KERNEL_FILE) as ds:
    lat_n = 'latitude' if 'latitude' in ds else 'lat'
    lon_n = 'longitude' if 'longitude' in ds else 'lon'
    k_lat = ds[lat_n].values.astype(np.float32)
    k_lon = ds[lon_n].values.astype(np.float32)
klon2d, klat2d = np.meshgrid(k_lon, k_lat)
idx_era5_to_psn = build_or_load_idx(
    klat2d, klon2d,
    psn_lat2d, psn_lon2d,
    os.path.join(OUT_DIR, '_idx_era5_to_psn25.npy'))
del klat2d, klon2d, k_lat, k_lon; gc.collect()

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def scatter_avg(arr_flat, idx, shape):
    a = np.zeros(np.prod(shape), dtype=np.float64)
    c = np.zeros(np.prod(shape), dtype=np.float32)
    v = np.isfinite(arr_flat)
    np.add.at(a, idx[v], arr_flat[v].astype(np.float64))
    np.add.at(c, idx[v], 1.0)
    out = np.full(np.prod(shape), np.nan, dtype=np.float32)
    m = c > 0
    out[m] = (a[m] / c[m]).astype(np.float32)
    return out.reshape(shape)


def regrid(arr_sial, zero_fill=True):
    fill = 0. if zero_fill else np.nan
    src  = np.where(arctic_sial,
                    np.where(np.isfinite(arr_sial), arr_sial, fill),
                    np.nan).astype(np.float32)
    return scatter_avg(src.ravel(), idx_s2p, psn_shape)


def amean(arr_psn):
    v  = arr_psn[DOMAIN]
    w  = AREA_W[DOMAIN]
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


# ─────────────────────────────────────────────────────────────────────
# HUANG KERNELS  (all months loaded at once from a single file open)
# ─────────────────────────────────────────────────────────────────────
print("Loading Huang kernels...", flush=True)
huang_maps = {}
with xr.open_dataset(HUANG_KERNEL_FILE) as ds:
    k_all = np.abs(ds['SFC_all'].values.astype(np.float32)) * 100.  # (12, nlat, nlon)
for mn in MONTH_NUMS:
    huang_maps[mn] = scatter_avg(k_all[mn-1].ravel(), idx_era5_to_psn, psn_shape)
del k_all; gc.collect()

# ─────────────────────────────────────────────────────────────────────
# ALBEDO + IS COMPUTATION
# ─────────────────────────────────────────────────────────────────────
def load_alb_sial(cache_dir, mname, years):
    """Climatological mean albedo on SIAL for the given years."""
    use_appx = (mname == 'September') and (idx_appx_to_sial is not None)
    s, c = None, None
    for y in years:
        p = (f"{cache_dir}/appx_alb_{mname}_{y}.npy" if use_appx
             else f"{cache_dir}/clara_alb_{mname}_{y}.npy")
        if not os.path.exists(p):
            continue
        raw = np.load(p).astype(np.float32)
        if use_appx and raw.shape != sial_shape:
            raw = raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
        raw[(raw < 0) | (raw > 1)] = np.nan
        fin = np.isfinite(raw)
        raw[~fin] = 0.
        s = raw.copy()          if s is None else s + raw
        c = fin.astype(np.int16) if c is None else c + fin.astype(np.int16)
    if s is None:
        return None
    v = c > 0
    s[v]  /= c[v].astype(np.float32)
    s[~v]  = np.nan
    return s


def get_IS_sial(cache_dir, mname, early_yrs, late_yrs, delta_t):
    """Returns (IS_ext, IS_drk) in Δalpha/ΔT units on the SIAL grid."""
    ae = load_alb_sial(cache_dir, mname, early_yrs)
    al = load_alb_sial(cache_dir, mname, late_yrs)
    if ae is None or al is None:
        return None, None
    dalpha = (al - ae).astype(np.float32)
    del ae, al

    try:
        slope = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy").astype(np.float32)
        pval  = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy").astype(np.float32)
        sic_e = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy").astype(np.float32)
        sic_l = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy").astype(np.float32)
    except Exception as e:
        print(f"  missing SIC cache for {mname}: {e}")
        return None, None

    valid    = arctic_sial & np.isfinite(dalpha)
    changing = (valid & np.isfinite(slope) & np.isfinite(pval)
                & (slope < 0) & (pval < PVAL))
    stable   = (valid & ~changing & (np.fmax(sic_e, sic_l) >= SIC_THR))

    IS_ext = np.where(changing, -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable,   -dalpha / delta_t, np.nan).astype(np.float32)
    del dalpha, slope, pval, sic_e, sic_l, valid, changing, stable
    return IS_ext, IS_drk


# ─────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Computing SIAF components...", flush=True)

results       = []
delta_ext_acc = np.zeros(psn_shape, dtype=np.float64)
delta_drk_acc = np.zeros(psn_shape, dtype=np.float64)
month_count   = 0

for mn in MONTH_NUMS:
    mname = MONTH_NAMES[mn]
    print(f"  {mname}...", end='', flush=True)

    ext_c, drk_c = get_IS_sial(CACHE_CMIP6, mname, EARLY, LATE_CMIP6, DELTA_T_CMIP6)
    ext_o, drk_o = get_IS_sial(CACHE_OBS,   mname, EARLY, LATE_OBS,   DELTA_T_OBS)
    if ext_c is None or ext_o is None:
        print(" SKIP (missing data)", flush=True)
        continue

    K = huang_maps[mn]

    # Regrid one array at a time, free SIAL arrays immediately
    ext_c_psn = regrid(ext_c) * K;  del ext_c
    drk_c_psn = regrid(drk_c) * K;  del drk_c
    ext_o_psn = regrid(ext_o) * K;  del ext_o
    drk_o_psn = regrid(drk_o) * K;  del drk_o
    gc.collect()

    # Arctic-mean for summary table
    for period, ep, dp in [('CMIP6',   ext_c_psn, drk_c_psn),
                            ('Obs-era', ext_o_psn, drk_o_psn)]:
        tot = np.where(DOMAIN,
                       np.where(np.isfinite(ep), ep, 0.)
                       + np.where(np.isfinite(dp), dp, 0.),
                       np.nan)
        results.append(dict(period=period, month=mname, month_num=mn,
                            SIAF_ext=amean(ep),
                            SIAF_drk=amean(dp),
                            SIAF_tot=amean(tot)))
        del tot

    # Spatial accumulation (DELTA = Obs-era - CMIP6)
    d_ext = np.where(DOMAIN, ext_o_psn - ext_c_psn, np.nan)
    d_drk = np.where(DOMAIN, drk_o_psn - drk_c_psn, np.nan)
    delta_ext_acc += np.where(np.isfinite(d_ext), d_ext, 0.)
    delta_drk_acc += np.where(np.isfinite(d_drk), d_drk, 0.)
    month_count   += 1

    del ext_c_psn, drk_c_psn, ext_o_psn, drk_o_psn, d_ext, d_drk
    gc.collect()
    print(" done", flush=True)

# Seasonal-mean DELTA maps
delta_ext_map = np.where(DOMAIN,
                         delta_ext_acc / max(month_count, 1),
                         np.nan).astype(np.float32)
delta_drk_map = np.where(DOMAIN,
                         delta_drk_acc / max(month_count, 1),
                         np.nan).astype(np.float32)
del delta_ext_acc, delta_drk_acc; gc.collect()

# ─────────────────────────────────────────────────────────────────────
# BUILD DELTA TABLE
# ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
deltas = []
for mn in MONTH_NUMS:
    mname = MONTH_NAMES[mn]
    rc = df[(df.period == 'CMIP6')   & (df.month_num == mn)]
    ro = df[(df.period == 'Obs-era') & (df.month_num == mn)]
    if rc.empty or ro.empty:
        continue
    deltas.append(dict(
        month=mname, month_num=mn,
        delta_ext = float(ro['SIAF_ext'].values[0]) - float(rc['SIAF_ext'].values[0]),
        delta_drk = float(ro['SIAF_drk'].values[0]) - float(rc['SIAF_drk'].values[0]),
    ))
ddf = pd.DataFrame(deltas)
ddf['delta_tot'] = ddf['delta_ext'] + ddf['delta_drk']

seas_ext = float(np.nanmean(ddf['delta_ext']))
seas_drk = float(np.nanmean(ddf['delta_drk']))
seas_tot = seas_ext + seas_drk
pct_ext  = seas_ext / seas_tot * 100 if abs(seas_tot) > 1e-9 else np.nan
pct_drk  = seas_drk / seas_tot * 100 if abs(seas_tot) > 1e-9 else np.nan

# ─────────────────────────────────────────────────────────────────────
# FIG 1 — monthly bar chart
# ─────────────────────────────────────────────────────────────────────
EXT_COL = '#2166ac'
DRK_COL = '#d6604d'

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ddf))
w = 0.35
ax.bar(x - w/2, ddf['delta_ext'], w, color=EXT_COL, alpha=0.85,
       label='DELTA_SIAF_ext  (extent loss)')
ax.bar(x + w/2, ddf['delta_drk'], w, color=DRK_COL, alpha=0.85,
       label='DELTA_SIAF_drk  (ice darkening)')
ax.plot(x, ddf['delta_tot'], 'ko--', ms=5, lw=1.4, label='DELTA_SIAF_total')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_xticks(x)
ax.set_xticklabels([m[:3] for m in ddf['month']])
ax.set_ylabel('DELTA_SIAF  (W/m2/K)', fontsize=11)
ax.set_title('DELTA_SIAF  (Obs-era - CMIP6)  --  Huang ERA5 kernel\n'
             'Extent loss (blue)  vs  Ice darkening (orange-red)',
             fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.legend(fontsize=10)
ax.text(0.98, 0.04,
        f"Season mean (Apr-Sep)\n"
        f"Ext: {seas_ext:+.3f}  ({pct_ext:.0f}%)\n"
        f"Drk: {seas_drk:+.3f}  ({pct_drk:.0f}%)\n"
        f"Tot: {seas_tot:+.3f}",
        transform=ax.transAxes, va='bottom', ha='right',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'siaf_ext_drk_summary.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────
# FIG 2 — Pan-Arctic spatial map
# ─────────────────────────────────────────────────────────────────────
vmax = float(np.nanpercentile(
    np.abs(np.concatenate([delta_ext_map[DOMAIN], delta_drk_map[DOMAIN]])), 97))
vmax = max(round(vmax * 10) / 10, 0.05)

if HAS_CARTOPY:
    proj = ccrs.NorthPolarStereo(central_longitude=-45)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={'projection': proj})
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

fig.suptitle('Huang DELTA_SIAF  (Obs-era - CMIP6)  --  Apr-Sep seasonal mean\n'
             'negative = weaker feedback in Obs-era',
             fontsize=11, fontweight='bold')

for ax, data, cmap, title, col in [
    (axes[0], delta_ext_map, 'RdBu_r',  'DELTA_SIAF_ext  (extent loss)',    EXT_COL),
    (axes[1], delta_drk_map, 'PuOr',    'DELTA_SIAF_drk  (ice darkening)',  DRK_COL),
]:
    plot_data = np.where(DOMAIN, data, np.nan)
    if HAS_CARTOPY:
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor='#d9d9d9', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4,       zorder=3)
        ax.gridlines(linewidth=0.3, color='gray', alpha=0.5,
                     xlocs=range(-180, 181, 30), ylocs=range(60, 91, 10))
        im = ax.pcolormesh(psn_lon2d, psn_lat2d, plot_data,
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           transform=ccrs.PlateCarree(), zorder=1, shading='auto')
    else:
        im = ax.pcolormesh(psn_lon2d, psn_lat2d, plot_data,
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlim(-180, 180); ax.set_ylim(60, 90)

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      pad=0.04, fraction=0.046, extend='both')
    cb.set_label('DELTA_SIAF  (W/m2/K)', fontsize=10)
    dmean = float(np.nanmean(plot_data[DOMAIN]))
    ax.set_title(f'{title}\n(Arctic mean {dmean:+.3f} W/m2/K)',
                 fontsize=10, fontweight='bold', color=col)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'siaf_ext_drk_map_huang.png')
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SEASONAL MEAN  DELTA_SIAF  (Obs-era - CMIP6)  --  Huang")
print("=" * 60)
print(f"  DELTA_SIAF_ext  (extent loss) : {seas_ext:+.3f}  ({pct_ext:.0f}%)")
print(f"  DELTA_SIAF_drk  (darkening)   : {seas_drk:+.3f}  ({pct_drk:.0f}%)")
print(f"  DELTA_SIAF_total              : {seas_tot:+.3f}")
print()
print(f"  {'Month':<10} {'D-ext':>8} {'D-drk':>8} {'D-tot':>8}")
print("  " + "-" * 38)
for _, r in ddf.sort_values('month_num').iterrows():
    print(f"  {r['month']:<10} {r['delta_ext']:>+8.3f} "
          f"{r['delta_drk']:>+8.3f} {r['delta_tot']:>+8.3f}")

print()
print("-- Pistone reference (previous run) --")
print("  Pistone seas.mean:  ext -1.26,  drk -1.52,  tot -2.78")
print("  Huang   seas.mean:  ext -1.00,  drk -1.66,  tot -2.66  (recomputed above)")
print("\nDone.")