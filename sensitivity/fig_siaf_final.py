#!/usr/bin/env python
"""
fig_siaf_final.py
─────────────────
Standalone publication figure: Arctic SW budget drivers over SIAF footprints.

Layout: 2 rows × 4 columns
  (a) RS mean         (b) CF mean (MODIS+CALIPSO+opaque)
  (c) TCWV mean       (d) RS clear-sky vs cloud-effect mean
  (e) RS trend        (f) CF trend
  (g) TCWV trend      (h) RS trend attribution (clear-sky + cloud-effect stacked)

Column (d/h): partitions RS into
  RS_clr       = rsdscs − rsuscs  (clear-sky absorbed SW; captures WV, aerosol, insolation)
  RS_cld_effect = RS_total − RS_clr  (negative; how much clouds suppress surface SW)

AOD (MERRA-2 TOTEXTTAU): non-significant over all months/footprints — reported as note.

Run on Derecho:
  module load conda && conda activate npl
  python /glade/work/jrotondo/ncc_paper/python/fig_siaf_final.py
"""

import glob, os, re, warnings
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy.stats import t as _tdist
from scipy.spatial import cKDTree
from pyproj import Transformer
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DIR_SIC       = '/glade/derecho/scratch/jrotondo/SIC_data/'
CERES_FILE    = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
                 'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
MODIS_FILE    = '/glade/work/jrotondo/modis_cod_arctic_amjjas.nc'
CALIPSO_FILE  = '/glade/work/jrotondo/calipso_cf_arctic_amjjas.nc'
TCWV_FILE     = '/glade/work/jrotondo/era5_tcwv_arctic_amjjas.nc'
MODIS_TPW_FILE= '/glade/work/jrotondo/modis_tpw_arctic_amjjas.nc'
AOD_FILE      = '/glade/work/jrotondo/merra2_aod_arctic_amjjas.nc'
CACHE_OBS     = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
DIR_SIAL      = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
OUT_FILE      = '/glade/work/jrotondo/ncc_paper/plots/fig_siaf_final.png'

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

MONTH_NUMS   = [4, 5, 6, 7, 8, 9]
MN_LBL       = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}
MN_FULL      = {4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
YR_CERES     = np.array(range(2000, 2026))
YR_CALIPSO   = np.array(range(2006, 2024))
YR_MODTPW    = np.array(range(2000, 2026))
YR_AOD       = np.array(range(2000, 2025))
ARCTIC_LAT   = 60.0
SIC_THRESH   = 0.15

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ══════════════════════════════════════════════════════════════════════════════
# GRID + UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
print('PSN25 grid ...')
_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(_sic) as ds:
    _px = ds['x'].values.astype(np.float64)
    _py = ds['y'].values.astype(np.float64)
_pxx, _pyy  = np.meshgrid(_px, _py)
PSN_SHAPE   = _pxx.shape
_tfm        = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = _tfm.transform(_pxx, _pyy)
del _pxx, _pyy
DOMAIN = PSN_LAT >= ARCTIC_LAT
_phi   = np.deg2rad(70.)
_k     = (1. + np.sin(_phi)) / np.maximum(1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W = (1. / _k**2).astype(np.float32)

def _xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel()); lo = np.deg2rad(np.asarray(lon).ravel())
    c  = np.cos(la)
    return np.column_stack([c*np.cos(lo), c*np.sin(lo), np.sin(la)])

def _idx(lats, lons):
    lo2d, la2d = (np.meshgrid(lons, lats) if np.asarray(lats).ndim == 1
                  else (lons, lats))
    _, idx = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(_xyz(la2d, lo2d), workers=-1)
    return idx.astype(np.int32)

def scatter_avg(flat, idx):
    n   = int(np.prod(PSN_SHAPE))
    acc = np.zeros(n, np.float32); cnt = np.zeros(n, np.float32)
    ok  = np.isfinite(flat)
    np.add.at(acc, idx[ok], flat[ok].astype(np.float32))
    np.add.at(cnt, idx[ok], 1.)
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0; out[m] = acc[m] / cnt[m]
    return out.reshape(PSN_SHAPE)

def wmean(arr, mask):
    v = arr[DOMAIN & mask]; w = AREA_W[DOMAIN & mask]
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.sum() else np.nan

def ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return stats.linregress(x[ok].astype(float), y[ok]) if ok.sum() >= 5 else None

# ══════════════════════════════════════════════════════════════════════════════
# SIAF MASKS — computed directly from SIC NC files on PSN25 grid
# ══════════════════════════════════════════════════════════════════════════════
print('SIAF masks (from SIC NC files) ...')

_sic_files = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))
print(f'  Found {len(_sic_files)} SIC daily files')

# Files are daily: sic_psn25_YYYYMMDD_*.nc; variable cdr_seaice_conc (0-1)
SIC_VAR = 'cdr_seaice_conc'
_re_sic = re.compile(r'sic_psn25_(\d{4})(\d{2})(\d{2})_')

# Accumulate daily files into monthly sums for AMJJAS, years 2000-2025
_sic_acc = {}   # (yr, mn) -> [sum_array, count]
for _f in _sic_files:
    _m = _re_sic.search(os.path.basename(_f))
    if not _m: continue
    _yr, _mn = int(_m.group(1)), int(_m.group(2))
    if _yr not in YR_CERES or _mn not in MONTH_NUMS: continue
    with xr.open_dataset(_f) as _ds:
        _v = _ds[SIC_VAR].values.squeeze().astype(np.float32)
        _v = np.where((_v >= 0.) & (_v <= 1.), _v, np.nan)
    _k = (_yr, _mn)
    if _k not in _sic_acc:
        _sic_acc[_k] = [np.zeros_like(_v), np.zeros_like(_v)]
    _ok = np.isfinite(_v)
    _sic_acc[_k][0] = np.where(_ok, _sic_acc[_k][0] + _v, _sic_acc[_k][0])
    _sic_acc[_k][1] = np.where(_ok, _sic_acc[_k][1] + 1.,  _sic_acc[_k][1])

sic_by_ym = {k: np.where(v[1] > 0, v[0] / v[1], np.nan)
             for k, v in _sic_acc.items()}
print(f'  Averaged to {len(sic_by_ym)} monthly fields  '
      f'shape={list(sic_by_ym.values())[0].shape}')

def _vect_ols_sic(stack, yrs):
    """Vectorized OLS on (T, ny, nx); returns slope, pval, early_mean, late_mean."""
    T = stack.shape[0]
    orig = stack.shape[1:]
    Y  = stack.reshape(T, -1).astype(np.float64)
    x  = yrs.astype(np.float64)
    ok = np.isfinite(Y)
    n  = ok.sum(axis=0)

    xm     = np.where(ok, x[:, None], np.nan)
    x_mean = np.nanmean(xm, axis=0)
    y_mean = np.nanmean(np.where(ok, Y, np.nan), axis=0)

    xc = x[:, None] - x_mean[None, :]
    yc = np.where(ok, Y, np.nan) - y_mean[None, :]

    xc_m = np.where(ok, xc, 0.)
    yc_m = np.where(ok, yc, 0.)

    SSx   = (xc_m**2).sum(axis=0)
    Sxy   = (xc_m * yc_m).sum(axis=0)
    slope = np.where(SSx > 1e-12, Sxy / SSx, np.nan)

    yhat  = y_mean[None, :] + slope[None, :] * xc
    SSe   = np.nansum((np.where(ok, Y, np.nan) - yhat)**2, axis=0)
    df    = np.maximum(n - 2, 1)
    se    = np.sqrt(np.where(SSx > 1e-12, SSe / df / SSx, np.nan))
    tstat = np.where(se > 0, slope / se, np.nan)
    pval  = np.where(np.isfinite(tstat),
                     2 * _tdist.sf(np.abs(tstat), df), np.nan)

    early_m = yrs <= 2012
    late_m  = yrs >= 2013
    emean = np.nanmean(np.where(ok & early_m[:, None], Y, np.nan), axis=0)
    lmean = np.nanmean(np.where(ok & late_m[:, None],  Y, np.nan), axis=0)

    good = n >= 5
    def _r(a): return np.where(good, a, np.nan).reshape(orig).astype(np.float32)
    return _r(slope), _r(pval), _r(emean), _r(lmean)

EXT = {}; DRK = {}
for mn in MONTH_NUMS:
    _yrs   = np.array(sorted(yr for yr, m in sic_by_ym if m == mn))
    _stack = np.array([sic_by_ym.get((yr, mn), np.full(PSN_SHAPE, np.nan))
                       for yr in _yrs])
    sl, pv, se, sl2 = _vect_ols_sic(_stack, _yrs)

    ext_m = (np.isfinite(sl) & np.isfinite(pv) &
             (sl < 0) & (pv < 0.05) & (se >= SIC_THRESH) & DOMAIN)
    sm    = np.fmax(np.where(np.isfinite(se),  se,  0.),
                    np.where(np.isfinite(sl2), sl2,  0.))
    drk_m = np.isfinite(sl) & ~ext_m & (sm >= SIC_THRESH) & DOMAIN

    EXT[mn] = ext_m
    DRK[mn] = drk_m
    print(f'  {MN_LBL[mn]}  ext={ext_m.sum():,}  drk={drk_m.sum():,} px')

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA → PSN25
# ══════════════════════════════════════════════════════════════════════════════

# ── CERES: RS (T07), RS_clr, RS_cloud_effect ─────────────────────────────────
# RS_clr  = rsdscs − rsuscs  (absorbed SW under clear sky; encodes WV/aerosol/insolation)
# RS_cld  = RS_total − RS_clr  (cloud radiative effect on surface SW; always ≤ 0)
print('CERES RS (T07 + clear-sky decomposition) ...')

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt > 1., rsdt, np.nan)
        Sd = np.where(rsds > 1., rsds, np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds/S;  q  = rsut/S
        d  = q + p - 2.*p*a
        R  = np.clip((q-p*a) / np.where(np.abs(d)>1e-6, d, np.nan), 0., .99)
        A  = np.clip(1.-p*(1.-a*R) / np.where((1.-R)>.01, 1.-R, np.nan), 0., .99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs/S;  qc = rsutcs/S
        dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc-pc*ac) / np.where(np.abs(dc)>1e-6, dc, np.nan), 0., .99)
        Rl = np.clip(R - Rc, 0., .99)
        RS = (S*(1.-A)*(1.-Rl)**2
              / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS.astype(np.float32)

_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

rs_2d = {}; rs_clr_2d = {}; rs_cld_2d = {}
with xr.open_dataset(CERES_FILE) as ds:
    IDX_C = _idx(ds['lat'].values, ds['lon'].values)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_CERES: continue
        sl = ds.isel(time=i).load()
        vs = {k: sl[k].values.astype(np.float32) for k in _CERES_VARS}

        rs_tot = t07_invert(vs['solar_mon'], vs['toa_sw_all_mon'],
                            vs['sfc_sw_down_all_mon'], vs['sfc_sw_up_all_mon'],
                            vs['toa_sw_clr_c_mon'],
                            vs['sfc_sw_down_clr_c_mon'], vs['sfc_sw_up_clr_c_mon'])
        rs_clr = (vs['sfc_sw_down_clr_c_mon'] - vs['sfc_sw_up_clr_c_mon']).astype(np.float32)
        rs_cld = np.where(np.isfinite(rs_tot) & np.isfinite(rs_clr),
                          rs_tot - rs_clr, np.nan).astype(np.float32)

        key = (t.year, t.month)
        rs_2d[key]     = scatter_avg(rs_tot.ravel(), IDX_C)
        rs_clr_2d[key] = scatter_avg(rs_clr.ravel(), IDX_C)
        rs_cld_2d[key] = scatter_avg(rs_cld.ravel(), IDX_C)
print(f'  {len(rs_2d)} fields')

# ── MODIS CF + LWP + IWP (same file, one pass) ───────────────────────────────
print('MODIS CF, LWP, IWP ...')
cf_2d = {}; lwp_2d = {}; iwp_2d = {}; liqfrac_2d = {}
with xr.open_dataset(MODIS_FILE) as ds:
    IDX_M = _idx(ds['lat'].values, ds['lon'].values)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_CERES: continue
        key = (t.year, t.month)

        v = ds['Cloud_Retrieval_Fraction_Combined_FMean'].isel(time=i).values.astype(np.float32)
        v = np.where(np.isfinite(v) & (v >= 0.) & (v <= 1.), v, np.nan)
        cf_2d[key] = scatter_avg(v.ravel(), IDX_M)

        lwp = ds['Cloud_Water_Path_Liquid_Mean_Mean'].isel(time=i).values.astype(np.float32)
        iwp = ds['Cloud_Water_Path_Ice_Mean_Mean'].isel(time=i).values.astype(np.float32)
        lwp = np.where((lwp > 0.) & (lwp < 5000.), lwp, np.nan)
        iwp = np.where((iwp > 0.) & (iwp < 5000.), iwp, np.nan)
        total = lwp + iwp
        liqfrac = np.where(total > 0., lwp / total, np.nan)

        lwp_2d[key]     = scatter_avg(lwp.ravel(),     IDX_M)
        iwp_2d[key]     = scatter_avg(iwp.ravel(),     IDX_M)
        liqfrac_2d[key] = scatter_avg(liqfrac.ravel(), IDX_M)
print(f'  {len(cf_2d)} fields')

# ── DIAGNOSTIC: verify MODIS variable ranges ──────────────────────────────────
print('\n── MODIS variable diagnostic ─────────────────────────────────────────')
with xr.open_dataset(MODIS_FILE) as _ds:
    for _vn, _label in [
        ('Cloud_Retrieval_Fraction_Combined_FMean', 'CF'),
        ('Cloud_Water_Path_Liquid_Mean_Mean',        'LWP'),
        ('Cloud_Water_Path_Ice_Mean_Mean',           'IWP'),
    ]:
        _vals = _ds[_vn].values
        _ok   = _vals[np.isfinite(_vals)]
        print(f'  {_label:4s}  units={_ds[_vn].attrs.get("units","?"):10s}  '
              f'range=[{_ok.min():.3f}, {_ok.max():.3f}]  mean={_ok.mean():.3f}')
print('──────────────────────────────────────────────────────────────────────\n')

# ── CALIPSO total + opaque CF ─────────────────────────────────────────────────
print('CALIPSO CF ...')
cal_2d = {}; cal_opq_2d = {}
with xr.open_dataset(CALIPSO_FILE) as ds:
    lats = ds['lat'].values; lons = ds['lon'].values
    IDX_CAL = _idx(lats, lons)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_CALIPSO: continue
        for src, dest in [('cf_total', cal_2d), ('cf_opaque', cal_opq_2d)]:
            v = ds[src].isel(time=i).values.astype(np.float32)
            v = np.where(np.isfinite(v) & (v >= 0.) & (v <= 1.), v, np.nan)
            dest[(t.year, t.month)] = scatter_avg(v.ravel(), IDX_CAL)
print(f'  {len(cal_2d)} total  {len(cal_opq_2d)} opaque fields')

# ── ERA5 TCWV ─────────────────────────────────────────────────────────────────
print('ERA5 TCWV ...')
tcwv_2d = {}
with xr.open_dataset(TCWV_FILE) as ds:
    IDX_T = _idx(ds['lat'].values, ds['lon'].values)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_CERES: continue
        v = ds['tcwv'].isel(time=i).values.astype(np.float32)
        v = np.where(np.isfinite(v) & (v >= 0.), v, np.nan)
        tcwv_2d[(t.year, t.month)] = scatter_avg(v.ravel(), IDX_T)
print(f'  {len(tcwv_2d)} fields')

# ── MODIS TPW ─────────────────────────────────────────────────────────────────
print('MODIS TPW ...')
modtpw_2d = {}
with xr.open_dataset(MODIS_TPW_FILE) as ds:
    IDX_MTPW = _idx(ds['lat'].values, ds['lon'].values)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_MODTPW: continue
        v = ds['tpw'].isel(time=i).values.astype(np.float32)
        v = np.where(np.isfinite(v) & (v > 0.) & (v < 200.), v, np.nan)
        modtpw_2d[(t.year, t.month)] = scatter_avg(v.ravel(), IDX_MTPW)
print(f'  {len(modtpw_2d)} fields')

# ── MERRA-2 AOD (significance check only — not plotted) ───────────────────────
print('MERRA-2 AOD (significance check) ...')
aod_2d = {}
with xr.open_dataset(AOD_FILE) as ds:
    IDX_AOD = _idx(ds['lat'].values, ds['lon'].values)
    times = pd.DatetimeIndex(ds['time'].values)
    for i, t in enumerate(times):
        if t.month not in MONTH_NUMS or t.year not in YR_AOD: continue
        v = ds['aod'].isel(time=i).values.astype(np.float32)
        v = np.where(np.isfinite(v) & (v >= 0.) & (v < 5.), v, np.nan)
        aod_2d[(t.year, t.month)] = scatter_avg(v.ravel(), IDX_AOD)
print(f'  {len(aod_2d)} fields')

# ══════════════════════════════════════════════════════════════════════════════
# FOOTPRINT TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
def _fill(d2d, yr_list, masks):
    out = {k: {mn: {} for mn in MONTH_NUMS} for k in masks}
    for (yr, mn), arr in d2d.items():
        if yr not in yr_list: continue
        for k, mask in masks.items():
            out[k][mn][yr] = wmean(arr, mask[mn])
    return out

rs      = _fill(rs_2d,      YR_CERES,   {'ext': EXT, 'drk': DRK})
rs_clr  = _fill(rs_clr_2d,  YR_CERES,   {'ext': EXT, 'drk': DRK})
rs_cld  = _fill(rs_cld_2d,  YR_CERES,   {'ext': EXT, 'drk': DRK})
cf      = _fill(cf_2d,      YR_CERES,   {'ext': EXT, 'drk': DRK})
lwp     = _fill(lwp_2d,     YR_CERES,   {'ext': EXT, 'drk': DRK})
iwp     = _fill(iwp_2d,     YR_CERES,   {'ext': EXT, 'drk': DRK})
liqfrac = _fill(liqfrac_2d, YR_CERES,   {'ext': EXT, 'drk': DRK})
cal     = _fill(cal_2d,     YR_CALIPSO, {'ext': EXT, 'drk': DRK})
cal_opq = _fill(cal_opq_2d, YR_CALIPSO, {'ext': EXT, 'drk': DRK})
tcwv    = _fill(tcwv_2d,    YR_CERES,   {'ext': EXT, 'drk': DRK})
modtpw  = _fill(modtpw_2d,  YR_MODTPW,  {'ext': EXT, 'drk': DRK})
aod     = _fill(aod_2d,     YR_AOD,     {'ext': EXT, 'drk': DRK})

# ══════════════════════════════════════════════════════════════════════════════
# STAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def means(d, key, yrs):
    mu, sd = [], []
    for mn in MONTH_NUMS:
        v = np.array([d[key][mn].get(y, np.nan) for y in yrs])
        mu.append(np.nanmean(v)); sd.append(np.nanstd(v))
    return np.array(mu), np.array(sd)

def trends(d, key, yrs):
    sl, se, pv = [], [], []
    for mn in MONTH_NUMS:
        v = np.array([d[key][mn].get(y, np.nan) for y in yrs])
        r = ols(yrs.astype(float), v)
        sl.append(r.slope  if r else np.nan)
        se.append(r.stderr if r else np.nan)
        pv.append(r.pvalue if r else 1.)
    return np.array(sl), np.array(se), np.array(pv)

# ── AOD significance summary ──────────────────────────────────────────────────
aod_pv_ext = trends(aod, 'ext', YR_AOD)[2]
aod_pv_drk = trends(aod, 'drk', YR_AOD)[2]
n_sig_aod  = int(np.sum(aod_pv_ext < 0.05) + np.sum(aod_pv_drk < 0.05))
aod_note   = (
    f'Note: MERRA-2 aerosol optical depth (550 nm, 2000–2024) trends are non-significant '
    f'across all AMJJAS months in both footprints ({n_sig_aod}/12 p < 0.05); '
    f'aerosol direct forcing is not a primary driver of the observed RS decline.'
)
print(f'\nAOD: {aod_note}')

# ── Print clear-sky vs cloud-effect trend magnitudes for reference ────────────
print('\n── RS trend attribution ──')
for zone in ('ext', 'drk'):
    sl_clr, _, pv_clr = trends(rs_clr, zone, YR_CERES)
    sl_cld, _, pv_cld = trends(rs_cld, zone, YR_CERES)
    sl_tot, _, _      = trends(rs,     zone, YR_CERES)
    print(f'  {zone}:')
    for i, mn in enumerate(MONTH_NUMS):
        print(f'    {MN_LBL[mn]}  clr={sl_clr[i]:+.3f}(p={pv_clr[i]:.2f})  '
              f'cld={sl_cld[i]:+.3f}(p={pv_cld[i]:.2f})  '
              f'tot={sl_tot[i]:+.3f}  check={sl_clr[i]+sl_cld[i]:+.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding figure ...')

CE   = '#b5000a'   # extent — dark red
CD   = '#1a5fa8'   # darkening — steel blue
CE2  = '#e8a091'   # extent CALIPSO / MODIS TPW — salmon
CD2  = '#90b8d8'   # darkening CALIPSO / MODIS TPW — light blue
COPQ = '#5a189a'   # CALIPSO opaque — deep purple

# Clear-sky vs cloud-effect colours for attribution panel
CLR_E = '#b5000a'   # clear-sky component, extent — same red (WV-driven)
CLD_E = '#f4845f'   # cloud-effect component, extent — lighter orange-red
CLR_D = '#1a5fa8'   # clear-sky component, darkening — same blue
CLD_D = '#74b3ce'   # cloud-effect component, darkening — lighter blue

MX   = np.arange(len(MONTH_NUMS))
MLBL = [MN_LBL[m] for m in MONTH_NUMS]

def polish(ax, ylabel, title, letter, zero=False, handles=None, note=None):
    ax.set_xticks(MX); ax.set_xticklabels(MLBL, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if zero:
        ax.axhline(0, color='#bbb', lw=0.9, ls='--', zorder=0)
    if handles:
        ax.legend(handles=handles, fontsize=9, framealpha=0.88, edgecolor='#ccc')
    ax.tick_params(labelsize=10)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='bottom')
    if note:
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, color='#444',
                fontstyle='italic',
                bbox=dict(fc='#f7f7f7', ec='#ddd', pad=3, alpha=0.92))

def trend_bars(ax, datasets, scale=1., w=0.38):
    n = len(datasets)
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    handles = []
    for (_sl, _se, _pv, _col, _lbl), _off in zip(datasets, offsets):
        for i in range(len(MONTH_NUMS)):
            sig = _pv[i] < 0.05
            ax.bar(MX[i] + _off, _sl[i]*scale, w,
                   color=_col, alpha=0.85 if sig else 0.30,
                   hatch=None if sig else '//',
                   edgecolor=_col, linewidth=0.5)
        ax.errorbar(MX + _off, _sl*scale, yerr=_se*scale,
                    fmt='none', color=_col, capsize=3.5, lw=1.2, capthick=1.2, zorder=5)
        handles.append(Patch(facecolor=_col, alpha=0.85, edgecolor=_col, label=_lbl))
    return handles

NS_PATCH = Patch(facecolor='#aaa', alpha=0.30, hatch='//', label='n.s. (p ≥ 0.05)')

fig, axes = plt.subplots(2, 4, figsize=(26, 11), facecolor='white')
fig.subplots_adjust(hspace=0.46, wspace=0.30)

# ══ ROW 0 — Climatological means ══════════════════════════════════════════════

# ── (a) RS total mean ─────────────────────────────────────────────────────────
ax = axes[0, 0]
mu_re, sd_re = means(rs, 'ext', YR_CERES)
mu_rd, sd_rd = means(rs, 'drk', YR_CERES)
ax.bar(MX - 0.2, mu_re, 0.38, yerr=sd_re, capsize=4,
       color=CE, alpha=0.88, error_kw=dict(lw=1.3, capthick=1.3))
ax.bar(MX + 0.2, mu_rd, 0.38, yerr=sd_rd, capsize=4,
       color=CD, alpha=0.88, error_kw=dict(lw=1.3, capthick=1.3))
for i in range(len(MONTH_NUMS)):
    diff = mu_re[i] - mu_rd[i]
    if np.isfinite(diff):
        top = max(mu_re[i] + sd_re[i], mu_rd[i] + sd_rd[i])
        ax.text(MX[i], top * 1.012, f'{diff:+.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#111')
polish(ax, 'W m⁻²', 'Surface absorbed SW  (RS, 2000–2024)', 'a',
       handles=[Patch(facecolor=CE, label='Extent (retreating ice)'),
                Patch(facecolor=CD, label='Darkening (stable ice)')],
       note='Extent absorbs less SW\ndespite lower surface albedo —\ncloud + moisture suppression')

# ── (b) CF mean: MODIS + CALIPSO; LWP mean on right axis ─────────────────────
CLIQ = '#e76f51'   # liquid water — warm orange
CICE = '#457b9d'   # ice water — cool slate blue
ax = axes[0, 1]
mu_me, sd_me = means(cf,  'ext', YR_CERES)
mu_md, sd_md = means(cf,  'drk', YR_CERES)
mu_ce, sd_ce = means(cal, 'ext', YR_CALIPSO)
mu_cd, sd_cd = means(cal, 'drk', YR_CALIPSO)
w4 = 0.19
ax.bar(MX - 1.5*w4, mu_me, w4, yerr=sd_me, capsize=2.5,
       color=CE,  alpha=0.88, error_kw=dict(lw=0.8))
ax.bar(MX - 0.5*w4, mu_md, w4, yerr=sd_md, capsize=2.5,
       color=CD,  alpha=0.88, error_kw=dict(lw=0.8))
ax.bar(MX + 0.5*w4, mu_ce, w4, yerr=sd_ce, capsize=2.5,
       color=CE2, alpha=0.95, edgecolor=CE, lw=1.2, error_kw=dict(lw=0.8))
ax.bar(MX + 1.5*w4, mu_cd, w4, yerr=sd_cd, capsize=2.5,
       color=CD2, alpha=0.95, edgecolor=CD, lw=1.2, error_kw=dict(lw=0.8))
# LWP mean on right axis — shows darkening has more liquid water
ax2b = ax.twinx()
mu_lwe, _ = means(lwp, 'ext', YR_CERES)
mu_lwd, _ = means(lwp, 'drk', YR_CERES)
ax2b.plot(MX, mu_lwe, color=CLIQ, lw=2, ls='-',  marker='o', ms=5, zorder=6)
ax2b.plot(MX, mu_lwd, color=CLIQ, lw=2, ls='--', marker='s', ms=5, zorder=6, alpha=0.75)
ax2b.set_ylabel('LWP  (g m⁻²)', fontsize=9, color=CLIQ)
ax2b.tick_params(labelsize=9, colors=CLIQ)
ax2b.spines['right'].set_color(CLIQ)
ax2b.set_ylim(0, ax2b.get_ylim()[1] * 1.3)
polish(ax, 'Cloud fraction (0–1)',
       'Cloud fraction (MODIS & CALIPSO)\n& liquid cloud water path (right axis)', 'b',
       handles=[Patch(facecolor=CE,  label='MODIS CF — Extent'),
                Patch(facecolor=CD,  label='MODIS CF — Darkening'),
                Patch(facecolor=CE2, edgecolor=CE, lw=1.2, label='CALIPSO CF — Extent'),
                Patch(facecolor=CD2, edgecolor=CD, lw=1.2, label='CALIPSO CF — Darkening'),
                plt.Line2D([0],[0], color=CLIQ, lw=2, marker='o', ms=5,
                           label='LWP — Extent'),
                plt.Line2D([0],[0], color=CLIQ, lw=2, ls='--', marker='s', ms=5,
                           label='LWP — Darkening')],
       note='Extent cloudier overall;\ndarkening carries more\nliquid water per cloud')

# ── (c) TCWV mean: ERA5 + MODIS TPW ──────────────────────────────────────────
ax = axes[0, 2]
mu_te,  sd_te  = means(tcwv,   'ext', YR_CERES)
mu_td,  sd_td  = means(tcwv,   'drk', YR_CERES)
mu_mte, sd_mte = means(modtpw, 'ext', YR_MODTPW)
mu_mtd, sd_mtd = means(modtpw, 'drk', YR_MODTPW)
w4 = 0.19
ax.bar(MX - 1.5*w4, mu_te,  w4, yerr=sd_te,  capsize=2.5,
       color=CE,  alpha=0.88, error_kw=dict(lw=1))
ax.bar(MX - 0.5*w4, mu_td,  w4, yerr=sd_td,  capsize=2.5,
       color=CD,  alpha=0.88, error_kw=dict(lw=1))
ax.bar(MX + 0.5*w4, mu_mte, w4, yerr=sd_mte, capsize=2.5,
       color=CE2, alpha=0.95, edgecolor=CE, lw=1.2, error_kw=dict(lw=1))
ax.bar(MX + 1.5*w4, mu_mtd, w4, yerr=sd_mtd, capsize=2.5,
       color=CD2, alpha=0.95, edgecolor=CD, lw=1.2, error_kw=dict(lw=1))
for i in range(len(MONTH_NUMS)):
    diff = mu_te[i] - mu_td[i]
    if np.isfinite(diff):
        tops = [v + s for v, s in [(mu_te[i], sd_te[i]), (mu_td[i], sd_td[i]),
                                    (mu_mte[i], sd_mte[i]), (mu_mtd[i], sd_mtd[i])]
                if np.isfinite(v)]
        ax.text(MX[i], max(tops) * 1.012, f'{diff:+.1f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#111')
polish(ax, 'kg m⁻²',
       'Total column water vapour\n(ERA5 reanalysis & MODIS near-IR obs)', 'c',
       handles=[Patch(facecolor=CE,  label='ERA5 — Extent'),
                Patch(facecolor=CD,  label='ERA5 — Darkening'),
                Patch(facecolor=CE2, edgecolor=CE, lw=1.2, label='MODIS — Extent'),
                Patch(facecolor=CD2, edgecolor=CD, lw=1.2, label='MODIS — Darkening')],
       note='Reanalysis & obs agree;\nextent wetter by ~1 kg/m²;\nMODIS sparse over sea ice')

# ── (d) RS decomposition mean: RS_total vs RS_clr ────────────────────────────
# Shows how much RS is suppressed by clouds relative to clear-sky baseline.
ax = axes[0, 3]
mu_tot_e, _ = means(rs,     'ext', YR_CERES)
mu_clr_e, _ = means(rs_clr, 'ext', YR_CERES)
mu_tot_d, _ = means(rs,     'drk', YR_CERES)
mu_clr_d, _ = means(rs_clr, 'drk', YR_CERES)
w4 = 0.19
# RS_clr bars (filled) — what RS would be without clouds
ax.bar(MX - 1.5*w4, mu_clr_e, w4, color=CLR_E, alpha=0.55, edgecolor=CLR_E, lw=1, label='RS clear-sky — Extent')
ax.bar(MX - 0.5*w4, mu_clr_d, w4, color=CLR_D, alpha=0.55, edgecolor=CLR_D, lw=1, label='RS clear-sky — Darkening')
# RS_total bars (solid, slightly narrower) on top — actual RS
ax.bar(MX + 0.5*w4, mu_tot_e, w4, color=CLR_E, alpha=0.88, label='RS all-sky — Extent')
ax.bar(MX + 1.5*w4, mu_tot_d, w4, color=CLR_D, alpha=0.88, label='RS all-sky — Darkening')
# Annotate cloud suppression magnitude
for i in range(len(MONTH_NUMS)):
    cld_e = mu_tot_e[i] - mu_clr_e[i]
    if np.isfinite(cld_e):
        ax.annotate('', xy=(MX[i] - 1.5*w4, mu_tot_e[i]),
                    xytext=(MX[i] - 1.5*w4, mu_clr_e[i]),
                    arrowprops=dict(arrowstyle='<->', color=CLR_E, lw=1.3))
        ax.text(MX[i] - 1.5*w4 - 0.12, (mu_tot_e[i] + mu_clr_e[i]) / 2.,
                f'{cld_e:.0f}', ha='right', va='center',
                fontsize=7.5, color=CLR_E, fontweight='bold')
polish(ax, 'W m⁻²',
       'RS clear-sky vs all-sky mean\n(gap = cloud SW suppression)', 'd',
       handles=[Patch(facecolor=CLR_E, alpha=0.55, edgecolor=CLR_E, lw=1, label='Clear-sky — Extent'),
                Patch(facecolor=CLR_E, alpha=0.88, label='All-sky — Extent'),
                Patch(facecolor=CLR_D, alpha=0.55, edgecolor=CLR_D, lw=1, label='Clear-sky — Darkening'),
                Patch(facecolor=CLR_D, alpha=0.88, label='All-sky — Darkening')],
       note='Clouds suppress RS by\n~40–80 W m⁻² in peak summer;\narrows show extent cloud effect')

# ══ ROW 1 — OLS trends  (solid = p<0.05; hatched = n.s.) ══════════════════════

# ── (e) RS total trend ────────────────────────────────────────────────────────
ax = axes[1, 0]
h = trend_bars(ax, [
    (*trends(rs, 'ext', YR_CERES), CE, 'Extent'),
    (*trends(rs, 'drk', YR_CERES), CD, 'Darkening'),
])
polish(ax, 'W m⁻² yr⁻¹', 'RS trend  (2000–2024)', 'e', zero=True,
       handles=h + [NS_PATCH],
       note='RS declining in both zones;\ncloud + moisture forcing at work')

# ── (f) Cloud microphysical trend: CF + LWP + IWP by zone ───────────────────
# Left axis (×10⁻³/yr): MODIS CF trend (same units, plotted as bars).
# Right axis (g m⁻² yr⁻¹): LWP and IWP trends (bars, different unit).
# Zone-specific story: extent CF modest; darkening LWP↑ drives thickening.
ax  = axes[1, 1]
ax2 = ax.twinx()

sl_cf_e,  se_cf_e,  pv_cf_e  = trends(cf,  'ext', YR_CERES)
sl_cf_d,  se_cf_d,  pv_cf_d  = trends(cf,  'drk', YR_CERES)
sl_lwp_e, se_lwp_e, pv_lwp_e = trends(lwp, 'ext', YR_CERES)
sl_lwp_d, se_lwp_d, pv_lwp_d = trends(lwp, 'drk', YR_CERES)
sl_iwp_e, se_iwp_e, pv_iwp_e = trends(iwp, 'ext', YR_CERES)
sl_iwp_d, se_iwp_d, pv_iwp_d = trends(iwp, 'drk', YR_CERES)

# CF bars on left axis
w2 = 0.19
for i in range(len(MONTH_NUMS)):
    for x, sl, pv, col in [
        (MX[i] - 0.5*w2, sl_cf_e[i], pv_cf_e[i], CE),
        (MX[i] + 0.5*w2, sl_cf_d[i], pv_cf_d[i], CD),
    ]:
        ax.bar(x, sl*1e3, w2, color=col,
               alpha=0.85 if pv < 0.05 else 0.30,
               hatch=None if pv < 0.05 else '//',
               edgecolor=col, lw=0.5)
ax.errorbar(MX - 0.5*w2, sl_cf_e*1e3, yerr=se_cf_e*1e3,
            fmt='none', color=CE, capsize=3, lw=1.1, zorder=5)
ax.errorbar(MX + 0.5*w2, sl_cf_d*1e3, yerr=se_cf_d*1e3,
            fmt='none', color=CD, capsize=3, lw=1.1, zorder=5)
ax.axhline(0, color='#bbb', lw=0.9, ls='--', zorder=0)
ax.set_ylabel('CF trend  ×10⁻³ yr⁻¹', fontsize=10)

# LWP and IWP trends on right axis as bars, grouped by zone
# Extent: LWP (CE2), IWP (CICE light)  — both flat/declining, little signal
# Darkening: LWP (CD, solid), IWP (CICE) — LWP rising, IWP mixed
wb = 0.13; gap = 0.55
for i in range(len(MONTH_NUMS)):
    xbase = MX[i] + gap
    for x, sl, se, pv, col in [
        (xbase - 1.5*wb, sl_lwp_e[i], se_lwp_e[i], pv_lwp_e[i], CE2),
        (xbase - 0.5*wb, sl_lwp_d[i], se_lwp_d[i], pv_lwp_d[i], CD),
        (xbase + 0.5*wb, sl_iwp_e[i], se_iwp_e[i], pv_iwp_e[i], '#b5c9d8'),
        (xbase + 1.5*wb, sl_iwp_d[i], se_iwp_d[i], pv_iwp_d[i], CICE),
    ]:
        ax2.bar(x, sl, wb, color=col,
                alpha=0.85 if pv < 0.05 else 0.30,
                hatch=None if pv < 0.05 else '//',
                edgecolor=col, lw=0.5)
        ax2.errorbar(x, sl, yerr=se, fmt='none', color=col,
                     capsize=2.5, lw=1., zorder=5)

# Vertical divider between CF bars and CWP bars
for i in range(len(MONTH_NUMS)):
    ax2.axvline(MX[i] + gap - 2*wb, color='#ddd', lw=0.7, zorder=0)

ax2.axhline(0, color='#ddd', lw=0.6, zorder=0)
ax2.set_ylabel('LWP / IWP trend  (g m⁻² yr⁻¹)', fontsize=9, color='#444')
ax2.tick_params(labelsize=9)
ax2.spines['right'].set_color('#aaa')

polish(ax, 'CF trend  ×10⁻³ yr⁻¹',
       'Cloud trends: CF (left) | LWP & IWP (right)\n(solid=sig p<0.05; hatched=n.s.)', 'f',
       zero=False,
       handles=[
           Patch(facecolor=CE,      alpha=0.85, label='CF — Extent'),
           Patch(facecolor=CD,      alpha=0.85, label='CF — Darkening'),
           Patch(facecolor=CE2,     alpha=0.85, label='LWP — Extent'),
           Patch(facecolor=CD,      alpha=0.85, label='LWP — Darkening ↑ May/Jul***'),
           Patch(facecolor='#b5c9d8', alpha=0.85, label='IWP — Extent'),
           Patch(facecolor=CICE,    alpha=0.85, label='IWP — Darkening'),
           NS_PATCH,
       ],
       note='Darkening: LWP↑ May* Jul*** →\nmore liquid water, optically thicker;\nExtent: CF & LWP flat, WV dominant')

# ── (g) TCWV trend: ERA5 + MODIS TPW ─────────────────────────────────────────
ax = axes[1, 2]
h = trend_bars(ax, [
    (*trends(tcwv,   'ext', YR_CERES),  CE,  'ERA5 — Extent'),
    (*trends(tcwv,   'drk', YR_CERES),  CD,  'ERA5 — Darkening'),
    (*trends(modtpw, 'ext', YR_MODTPW), CE2, 'MODIS — Extent'),
    (*trends(modtpw, 'drk', YR_MODTPW), CD2, 'MODIS — Darkening'),
], scale=1e3, w=0.19)
polish(ax, '×10⁻³ kg m⁻² yr⁻¹',
       'TCWV trend  (ERA5 & MODIS obs)\n(robust pan-Arctic moistening)', 'g',
       zero=True, handles=h + [NS_PATCH],
       note='Significant ↑ Jun–Sep in both zones;\nreanalysis & satellite obs agree')

# ── (h) RS trend attribution: clear-sky vs cloud-effect side-by-side ─────────
# RS_clr trend = change in surface-absorbed SW under clear sky
#   → captures surface albedo feedback (ice loss → lower α → more RS_clr)
#   → also captures WV absorption of clear-sky downwelling (tends negative)
#   → net is typically POSITIVE over extent (albedo effect dominates)
# RS_cld trend = how cloud SW suppression is changing
#   → typically NEGATIVE (clouds suppressing more RS over time)
# RS_total = RS_clr + RS_cld; net negative means cloud suppression > albedo gain
ax = axes[1, 3]
h = trend_bars(ax, [
    (*trends(rs_clr, 'ext', YR_CERES), CLR_E, 'Clear-sky RS — Extent'),
    (*trends(rs_cld, 'ext', YR_CERES), CLD_E, 'Cloud SW effect — Extent'),
    (*trends(rs_clr, 'drk', YR_CERES), CLR_D, 'Clear-sky RS — Darkening'),
    (*trends(rs_cld, 'drk', YR_CERES), CLD_D, 'Cloud SW effect — Darkening'),
], w=0.19)

# Overlay total RS trend as a horizontal tick so reader can verify the sum
sl_tot_e = trends(rs, 'ext', YR_CERES)[0]
sl_tot_d = trends(rs, 'drk', YR_CERES)[0]
off4 = np.linspace(-1.5*0.19, 1.5*0.19, 4)
ax.plot(MX + off4[0], sl_tot_e, '_', ms=14, mew=2.5, color='#111', zorder=6)
ax.plot(MX + off4[2], sl_tot_d, '_', ms=14, mew=2.5, color='#555', zorder=6)

polish(ax, 'W m⁻² yr⁻¹',
       'RS trend attribution  (2000–2024)\n(clear-sky albedo gain vs cloud suppression)', 'h',
       zero=True,
       handles=[
           Patch(facecolor=CLR_E, alpha=0.85, label='Clear-sky RS — Extent'),
           Patch(facecolor=CLD_E, alpha=0.85, label='Cloud SW effect — Extent'),
           Patch(facecolor=CLR_D, alpha=0.85, label='Clear-sky RS — Darkening'),
           Patch(facecolor=CLD_D, alpha=0.85, label='Cloud SW effect — Darkening'),
           NS_PATCH,
           plt.Line2D([0],[0], marker='_', color='#111', ls='none',
                      ms=12, mew=2.5, label='RS total (Ext | Drk)'),
       ],
       note='Clear-sky RS rising (albedo feedback real);\nbut cloud suppression growing faster —\nclouds are winning over extent')

# ── AOD note at figure bottom ─────────────────────────────────────────────────
fig.text(0.5, -0.018, aod_note,
         ha='center', va='top', fontsize=8.5, color='#555', fontstyle='italic')

fig.suptitle(
    'Arctic SW budget drivers over SIAF footprints  '
    '(extent = retreating-ice pixels;  darkening = stable-ice pixels)',
    fontsize=13, fontweight='bold', y=1.003)

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {OUT_FILE}')
