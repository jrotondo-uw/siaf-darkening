#!/usr/bin/env python
"""
check_era5_tcwv.py
──────────────────
Quick terminal diagnostic: ERA5 TCWV trends over SIAF extent vs darkening
footprints. Same structure as check_modis_cwp.py. No plots.

Run on Derecho:
  python /glade/work/jrotondo/ncc_paper/python/check_era5_tcwv.py
"""
import glob
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer

# ── PATHS ─────────────────────────────────────────────────────────────────────
DIR_SIC    = '/glade/derecho/scratch/jrotondo/SIC_data/'
TCWV_FILE  = '/glade/work/jrotondo/era5_tcwv_arctic_amjjas.nc'
CACHE_OBS  = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
DIR_SIAL   = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'

MONTH_NUMS = [4, 5, 6, 7, 8, 9]
MN_NAME    = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}
MN_FULL    = {4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
YEARS_ALL  = list(range(1979, 2026))
YEARS_CERES= list(range(2000, 2026))
ARCTIC_LAT = 60.0
SIC_THRESH = 0.15

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ── PSN25 grid ─────────────────────────────────────────────────────────────────
print('Setting up PSN25 grid ...')
_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(_sic) as ds:
    _px = ds['x'].values.astype(np.float64)
    _py = ds['y'].values.astype(np.float64)
_pxx, _pyy = np.meshgrid(_px, _py)
PSN_SHAPE   = _pxx.shape
_tfm        = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = _tfm.transform(_pxx, _pyy)
del _pxx, _pyy
DOMAIN = PSN_LAT >= ARCTIC_LAT

_phi = np.deg2rad(70.)
_k   = (1. + np.sin(_phi)) / np.maximum(1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W = (1. / _k**2).astype(np.float32)

def _xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel())
    lo = np.deg2rad(np.asarray(lon).ravel())
    c  = np.cos(la)
    return np.column_stack([c*np.cos(lo), c*np.sin(lo), np.sin(la)])

def scatter_avg(flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32); cnt = np.zeros(n, np.float32)
    ok  = np.isfinite(flat)
    np.add.at(acc, idx[ok], flat[ok].astype(np.float32))
    np.add.at(cnt, idx[ok], 1.)
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0; out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def wmean(arr, mask):
    v = arr[DOMAIN & mask]; w = AREA_W[DOMAIN & mask]
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.sum() else np.nan

def ols(x, y):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 5: return None
    return stats.linregress(x[ok].astype(float), y[ok])

def fmt(r, unit='kg/m²/yr'):
    if r is None: return '       ---    '
    sig = '*' if r.pvalue < 0.05 else ' '
    return f'{r.slope:+.3f}±{r.stderr:.3f}{sig} {unit}'

# ── Load TCWV → PSN25 ─────────────────────────────────────────────────────────
print('Loading ERA5 TCWV ...')
tcwv_2d = {}
with xr.open_dataset(TCWV_FILE) as ds:
    print(f'  Vars: {list(ds.data_vars)}  shape: {ds["tcwv"].shape}')
    lats = ds['lat'].values; lons = ds['lon'].values
    lo2d, la2d = np.meshgrid(lons, lats)
    _, IDX = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(_xyz(la2d, lo2d), workers=-1)
    IDX = IDX.astype(np.int32); del la2d, lo2d

    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS: continue
        arr = ds['tcwv'].isel(time=i).values.astype(np.float32)
        arr = np.where(np.isfinite(arr) & (arr >= 0.), arr, np.nan)
        tcwv_2d[(t.year, t.month)] = scatter_avg(arr.ravel(), IDX, PSN_SHAPE)

print(f'  {len(tcwv_2d)} month-fields  '
      f'({min(k[0] for k in tcwv_2d)}–{max(k[0] for k in tcwv_2d)})')

# ── Load SIAF masks ───────────────────────────────────────────────────────────
print('Loading SIAF masks ...')
with xr.open_dataset(sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]) as ds:
    la = ds['lat'].values; lo = ds['lon'].values
lo2d, la2d = np.meshgrid(lo, la) if la.ndim == 1 else (lo, la)
_, IDX_S = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(_xyz(la2d, lo2d), workers=-1)
IDX_S = IDX_S.astype(np.int32); del la2d, lo2d

EXT = {}; DRK = {}
for mn in MONTH_NUMS:
    mf = MN_FULL[mn]
    sl = np.load(f'{CACHE_OBS}/sic_trend_slope_sial_{mf}.npy').astype(np.float32)
    pv = np.load(f'{CACHE_OBS}/sic_trend_pval_sial_{mf}.npy').astype(np.float32)
    se = np.load(f'{CACHE_OBS}/sic_early_sial_{mf}.npy').astype(np.float32)
    sl2= np.load(f'{CACHE_OBS}/sic_late_sial_{mf}.npy').astype(np.float32)
    ext_s = np.isfinite(sl) & np.isfinite(pv) & (sl < 0) & (pv < 0.05) & (se >= SIC_THRESH)
    sm    = np.fmax(np.where(np.isfinite(se), se, 0.), np.where(np.isfinite(sl2), sl2, 0.))
    drk_s = np.isfinite(sl) & ~ext_s & (sm >= SIC_THRESH)
    EXT[mn] = scatter_avg(ext_s.astype(np.float32).ravel(), IDX_S, PSN_SHAPE) > 0.3
    DRK[mn] = scatter_avg(drk_s.astype(np.float32).ravel(), IDX_S, PSN_SHAPE) > 0.3

# ── Compute per-month time series ─────────────────────────────────────────────
tcwv_ext = {mn: {} for mn in MONTH_NUMS}
tcwv_drk = {mn: {} for mn in MONTH_NUMS}
tcwv_dom = {mn: {} for mn in MONTH_NUMS}

for (yr, mn), arr in tcwv_2d.items():
    tcwv_ext[mn][yr] = wmean(arr, EXT[mn])
    tcwv_drk[mn][yr] = wmean(arr, DRK[mn])
    tcwv_dom[mn][yr] = wmean(arr, np.ones(PSN_SHAPE, bool))

# ── Print: full record (1979–2025) ────────────────────────────────────────────
yr_all = np.array(YEARS_ALL)

def _ts(d, mn, yrs): return np.array([d[mn].get(y, np.nan) for y in yrs])

print('\n' + '='*80)
print('  ERA5 Total Column Water Vapour  (kg m⁻²)')
print('  FULL RECORD  1979–2025')
print('='*80)
print(f"{'Month':<6}  {'Ext mean':>10}  {'Drk mean':>10}  "
      f"{'Ext trend':>22}  {'Drk trend':>22}  {'Domain trend':>22}")
print('-'*80)
for mn in MONTH_NUMS:
    ts_e = _ts(tcwv_ext, mn, yr_all)
    ts_d = _ts(tcwv_drk, mn, yr_all)
    ts_dom= _ts(tcwv_dom, mn, yr_all)
    print(f'{MN_NAME[mn]:<6}  {np.nanmean(ts_e):>8.1f}  {np.nanmean(ts_d):>8.1f}  '
          f'{fmt(ols(yr_all, ts_e)):>25}  '
          f'{fmt(ols(yr_all, ts_d)):>25}  '
          f'{fmt(ols(yr_all, ts_dom)):>25}')
print('-'*80)
print('  * = p < 0.05')

# ── Print: CERES overlap only (2000–2025) ─────────────────────────────────────
yr_ceres = np.array(YEARS_CERES)
print('\n' + '='*80)
print('  ERA5 Total Column Water Vapour  (kg m⁻²)')
print('  CERES OVERLAP  2000–2025  (matches RS trend period)')
print('='*80)
print(f"{'Month':<6}  {'Ext mean':>10}  {'Drk mean':>10}  "
      f"{'Ext trend':>22}  {'Drk trend':>22}  {'Domain trend':>22}")
print('-'*80)
for mn in MONTH_NUMS:
    ts_e  = _ts(tcwv_ext, mn, yr_ceres)
    ts_d  = _ts(tcwv_drk, mn, yr_ceres)
    ts_dom= _ts(tcwv_dom, mn, yr_ceres)
    print(f'{MN_NAME[mn]:<6}  {np.nanmean(ts_e):>8.1f}  {np.nanmean(ts_d):>8.1f}  '
          f'{fmt(ols(yr_ceres, ts_e)):>25}  '
          f'{fmt(ols(yr_ceres, ts_d)):>25}  '
          f'{fmt(ols(yr_ceres, ts_dom)):>25}')
print('-'*80)

# ── AMJJAS pooled ─────────────────────────────────────────────────────────────
print('\n── AMJJAS pooled (2000–2025) ────────────────────────────────────────')
for label, dd in [('Extent', tcwv_ext), ('Darkening', tcwv_drk), ('Full domain', tcwv_dom)]:
    vals = []; yy = []
    for mn in MONTH_NUMS:
        for yr in YEARS_CERES:
            v = dd[mn].get(yr, np.nan)
            if np.isfinite(v): vals.append(v); yy.append(yr)
    r = ols(np.array(yy), np.array(vals))
    sig = 'SIGNIFICANT' if r and r.pvalue < 0.05 else 'n.s.'
    print(f'  {label:<14}  mean={np.nanmean(vals):.1f} kg/m²  '
          f'trend={r.slope:+.3f} kg/m²/yr  p={r.pvalue:.3f}  [{sig}]')

# ── Extent vs Darkening difference trend ──────────────────────────────────────
print('\n── Extent MINUS Darkening TCWV (moisture contrast) ─────────────────')
print('  (positive = extent wetter; does the contrast change over time?)')
for mn in MONTH_NUMS:
    diffs = {}
    for yr in YEARS_CERES:
        e = tcwv_ext[mn].get(yr, np.nan)
        d = tcwv_drk[mn].get(yr, np.nan)
        if np.isfinite(e) and np.isfinite(d): diffs[yr] = e - d
    if diffs:
        yy = np.array(sorted(diffs)); vv = np.array([diffs[y] for y in yy])
        r  = ols(yy, vv)
        print(f'  {MN_NAME[mn]}  mean_diff={np.nanmean(vv):+.1f} kg/m²  '
              f'{fmt(r, "kg/m²/yr")}')

print('\nDone.')
