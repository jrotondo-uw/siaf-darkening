#!/usr/bin/env python
"""
compute_sial_darkening_ts.py
============================
Builds regrid index, runs the CLARA/APP-x coverage diagnostic,
assembles the darkening-pixel albedo time series, and saves:

    {sens_cache_dir}/sial_darkening_ts.csv
        columns = month names, index = year
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
import glob, os
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
sens_cache_dir = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
albedo_dir     = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir       = '/glade/derecho/scratch/jrotondo/APP-x_observations/'

SIC_SLOPE_THRESH = -0.001
SIC_ICE_THRESH   = 0.15
MIN_CLARA_PX     = 500

MONTHS     = ['April', 'May', 'June', 'July', 'August', 'September']
MONTHS_NUM = {'April': 4, 'May': 5, 'June': 6,
              'July': 7, 'August': 8, 'September': 9}

OUT_CSV = os.path.join(sens_cache_dir, 'sial_darkening_ts.csv')

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
sial_tree  = cKDTree(latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel()))

sample_appx = sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
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
            month_data = (sial_raw.sel(time=sial_raw['time.month'] == month_num)
                          .mean(dim='time').values)
        return month_data if month_data.shape == sial_shape else None
    except Exception as e:
        print(f"  [!] CLARA raw load failed {year}: {e}")
        return None

# ── DIAGNOSTIC: coverage counts ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSTIC: CLARA valid darkening pixel counts per month")
print(f"{'Month':<12} {'min':>6} {'max':>6} {'mean':>6} "
      f"{'n_below_500':>12} {'n_years':>8}")
print("=" * 70)

for month in MONTHS:
    month_num      = MONTHS_NUM[month]
    slope          = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{month}.npy")
    pval           = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{month}.npy")
    sic_e          = np.load(f"{sens_cache_dir}/sic_early_sial_{month}.npy")
    sic_l          = np.load(f"{sens_cache_dir}/sic_late_sial_{month}.npy")
    sic_max        = np.fmax(sic_e, sic_l)
    valid          = arctic_mask & np.isfinite(slope) & np.isfinite(pval)
    has_ice        = valid & (sic_max >= SIC_ICE_THRESH)
    darkening_mask = has_ice & ~(has_ice & (slope <= SIC_SLOPE_THRESH))

    counts = {}
    for year in range(1982, 2025):
        p_c = f"{sens_cache_dir}/clara_alb_{month}_{year}.npy"
        if os.path.exists(p_c):
            arr = np.load(p_c)
            if arr.shape == sial_shape:
                counts[year] = int(np.sum(darkening_mask & np.isfinite(arr)))
                continue
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
        print(f"{month:<12}  no data")

print("=" * 70)

# ── MAIN: assemble time series ────────────────────────────────────────────────
sial_dark_ts = {}

for month in MONTHS:
    month_num = MONTHS_NUM[month]
    print(f"\nLoading {month}...", flush=True)

    slope          = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{month}.npy")
    pval           = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{month}.npy")
    sic_e          = np.load(f"{sens_cache_dir}/sic_early_sial_{month}.npy")
    sic_l          = np.load(f"{sens_cache_dir}/sic_late_sial_{month}.npy")
    sic_max        = np.fmax(sic_e, sic_l)
    valid          = arctic_mask & np.isfinite(slope) & np.isfinite(pval)
    has_ice        = valid & (sic_max >= SIC_ICE_THRESH)
    extent_mask    = has_ice & (slope <= SIC_SLOPE_THRESH)
    darkening_mask = has_ice & ~extent_mask

    year_vals = {}
    for year in range(1982, 2025):
        clara_map = None
        appx_map  = None

        p_c = f"{sens_cache_dir}/clara_alb_{month}_{year}.npy"
        if os.path.exists(p_c):
            arr = np.load(p_c)
            if arr.shape == sial_shape:
                clara_map = arr

        p_a = f"{sens_cache_dir}/appx_alb_{month}_{year}.npy"
        if os.path.exists(p_a):
            arr = np.load(p_a)
            appx_map = (regrid_appx_to_sial(arr) if arr.shape == appx_shape
                        else arr if arr.shape == sial_shape else None)

        if clara_map is None:
            clara_map = load_clara_raw(year, month_num)

        if clara_map is not None:
            n_valid = int(np.sum(darkening_mask & np.isfinite(clara_map)))
            if n_valid < MIN_CLARA_PX:
                print(f"  [!] {month} {year}: CLARA only {n_valid} px — demoting to missing")
                clara_map = None

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

# ── SAVE ──────────────────────────────────────────────────────────────────────
all_years = sorted({y for d in sial_dark_ts.values() for y in d})
df_out    = pd.DataFrame(index=all_years, columns=MONTHS, dtype=float)
df_out.index.name = 'year'
for month in MONTHS:
    for year, val in sial_dark_ts[month].items():
        df_out.loc[year, month] = val

df_out.to_csv(OUT_CSV)
print(f"\nSaved → {OUT_CSV}")
print(df_out.describe().round(4))