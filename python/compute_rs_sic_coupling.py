#!/usr/bin/env python
"""
compute_rs_sic_coupling_v4.py
─────────────────────────────
Changes vs v3:
  • Checkpointing: each CMIP6 model saved to disk immediately after
    completion; subsequent runs skip already-finished models.
  • Bulk I/O in cmip6_streaming_slope: all target (year, month) slices
    loaded in one .compute() per variable instead of 198 separate isel
    calls.  Typical speedup: 3–5× per model.
  • Year-outer / month-inner loop replaces month-outer / year-inner;
    all six OnlineOLS objects updated together per year.
  • _get_year_slice removed (replaced by _bulk_load cache dict).
"""

import gc, glob, os, sys, traceback, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree
from scipy.stats import linregress
from pyproj import Transformer

warnings.filterwarnings('ignore')

try:
    from global_land_mask import globe as _gm_globe
    _HAS_LAND_MASK = True
except ImportError:
    _HAS_LAND_MASK = False
    print('WARNING: global-land-mask not installed; ocean mask = lat >= 60N.')

# ============================================================
# CONFIG
# ============================================================
DIR_SIC    = '/glade/derecho/scratch/jrotondo/SIC_data/'
CERES_FILE = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
              'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')

CMIP6_ROOT = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

# Personal cache dirs (CLARA-A3 / APP-x albedo npy files)
CACHE_OBS  = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
DIR_SIAL   = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX   = '/glade/derecho/scratch/jrotondo/APP-x_observations/'

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/rs_sic_coupling/'
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(OUT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

YEARS_OBS   = list(range(2000, 2025))
YEARS_CMIP6 = list(range(1982, 2015))
YEARS_ERA5  = list(range(1982, 2025))

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep'}

MONTH_NAMES_FULL = {4: 'April', 5: 'May', 6: 'June',
                    7: 'July',  8: 'August', 9: 'September'}

SIC_ICE_THRESH  = 0.15
SIC_MARG_THRESH = 0.05
ARCTIC_LAT_MIN  = 60.
MIN_OBS_FOR_REG = 8
SIG_THRESH      = 0.05

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

RAD_VARS = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs', 'rsuscs']

MODEL_INFO = {
    'CESM2'           : dict(institution='NCAR',                member='r1i1p1f1'),
    'CESM2-WACCM'     : dict(institution='NCAR',                member='r1i1p1f1'),
    'E3SM-1-0'        : dict(institution='E3SM-Project',         member='r1i1p1f1'),
    'IPSL-CM6A-LR'    : dict(institution='IPSL',                member='r1i1p1f1'),
    'MIROC6'          : dict(institution='MIROC',               member='r1i1p1f1'),
    'CNRM-CM6-1'      : dict(institution='CNRM-CERFACS',        member='r10i1p1f2'),
    'CNRM-ESM2-1'     : dict(institution='CNRM-CERFACS',        member='r1i1p1f2'),
    'BCC-CSM2-MR'     : dict(institution='BCC',                 member='r1i1p1f1'),
    'ACCESS-CM2'      : dict(institution='CSIRO-ARCCSS',        member='r1i1p1f1'),
    'ACCESS-ESM1-5'   : dict(institution='CSIRO',               member='r1i1p1f1'),
    'MPI-ESM1-2-LR'   : dict(institution='MPI-M',               member='r1i1p1f1'),
    'MPI-ESM1-2-HR'   : dict(institution='MPI-M',               member='r1i1p1f1'),
    'GFDL-ESM4'       : dict(institution='NOAA-GFDL',           member='r1i1p1f1'),
    'GFDL-CM4'        : dict(institution='NOAA-GFDL',           member='r1i1p1f1'),
    'CanESM5'         : dict(institution='CCCma',               member='r1i1p1f1'),
    'NorESM2-LM'      : dict(institution='NCC',                 member='r1i1p1f1'),
    'NorESM2-MM'      : dict(institution='NCC',                 member='r1i1p1f1'),
    'UKESM1-0-LL'     : dict(institution='MOHC',                member='r1i1p2f2'),
    'HadGEM3-GC31-LL' : dict(institution='MOHC',                member='r1i1p1f3'),
    'HadGEM3-GC31-MM' : dict(institution='MOHC',                member='r1i1p1f3'),
    'INM-CM4-8'       : dict(institution='INM',                 member='r1i1p1f1'),
    'INM-CM5-0'       : dict(institution='INM',                 member='r1i1p1f1'),
    'FGOALS-g3'       : dict(institution='CAS',                 member='r1i1p1f1'),
    'FGOALS-f3-L'     : dict(institution='CAS',                 member='r1i1p1f1'),
    'AWI-CM-1-1-MR'   : dict(institution='AWI',                 member='r1i1p1f1'),
    'EC-Earth3'       : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'EC-Earth3-Veg'   : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'CMCC-CM2-SR5'    : dict(institution='CMCC',                member='r1i1p1f1'),
    'CMCC-ESM2'       : dict(institution='CMCC',                member='r1i1p1f1'),
    'MRI-ESM2-0'      : dict(institution='MRI',                 member='r1i1p1f1'),
    'TaiESM1'         : dict(institution='AS-RCEC',             member='r1i1p1f1'),
    'SAM0-UNICON'     : dict(institution='SNU',                 member='r1i1p1f1'),
    'MCM-UA-1-0'      : dict(institution='UA',                  member='r1i1p1f2'),
    'NESM3'           : dict(institution='NUIST',               member='r1i1p1f1'),
    'KACE-1-0-G'      : dict(institution='NIMS-KMA',            member='r1i1p1f1'),
    'KIOST-ESM'       : dict(institution='KIOST',               member='r1i1p1f1'),
    'FIO-ESM-2-0'     : dict(institution='FIO-QLNM',            member='r1i1p1f1'),
    'CAMS-CSM1-0'     : dict(institution='CAMS',                member='r1i1p1f1'),
    'BCC-ESM1'        : dict(institution='BCC',                 member='r1i1p1f1'),
    'MIROC-ES2L'      : dict(institution='MIROC',               member='r1i1p1f2'),
    'IITM-ESM'        : dict(institution='CCCR-IITM',           member='r1i1p1f1'),
}

VAR_TABLE_MAP = {
    'rsus'   : ['Amon'],
    'rsds'   : ['Amon'],
    'rsdt'   : ['Amon'],
    'rsut'   : ['Amon'],
    'rsutcs' : ['Amon', 'CFmon'],
    'rsdscs' : ['Amon', 'CFmon'],
    'rsuscs' : ['Amon', 'CFmon'],
    'siconc' : ['SImon', 'OImon', 'Omon'],
}
ALL_NEEDED_YEARS = set(YEARS_CMIP6)

KERNEL_FILE  = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR   = 'SFC_all'
KERNEL_SCALE = 100.0


# ============================================================
# PSN25 GRID
# ============================================================
print('\nSetting up PSN25 grid...', flush=True)
_sic_sample = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(_sic_sample) as _ds:
    _px = _ds['x'].values.astype(np.float64)
    _py = _ds['y'].values.astype(np.float64)
_pxx, _pyy   = np.meshgrid(_px, _py)
PSN_SHAPE    = _pxx.shape
_tfm         = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = _tfm.transform(_pxx, _pyy)
del _pxx, _pyy

_phi_ts = np.deg2rad(70.)
_k_ps   = (1. + np.sin(_phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W  = (1. / _k_ps**2).astype(np.float32)

if _HAS_LAND_MASK:
    _lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    DOMAIN_PSN = (PSN_LAT >= ARCTIC_LAT_MIN) & _gm_globe.is_ocean(PSN_LAT, _lp)
else:
    DOMAIN_PSN = PSN_LAT >= ARCTIC_LAT_MIN

print(f'  PSN25 domain: {DOMAIN_PSN.sum():,} ocean pixels >= {ARCTIC_LAT_MIN}°N',
      flush=True)

print('  Building CERES → PSN25 index...', flush=True)
with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
    _cla = _ds['lat'].values; _clo = _ds['lon'].values
_clon2d, _clat2d = np.meshgrid(_clo, _cla)

def _xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel().astype(np.float64))
    lo = np.deg2rad(np.asarray(lon).ravel().astype(np.float64))
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])

_, IDX_CERES_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
    _xyz(_clat2d, _clon2d), workers=-1)
del _clat2d, _clon2d
print('  Done.', flush=True)


# ============================================================
# SIAL (CLARA-A3) GRID + INDEXES
# ============================================================
print('\nSetting up SIAL (CLARA-A3) grid...', flush=True)
_sial_sample = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(_sial_sample) as _ds:
    _sla = _ds['lat'].values; _slo = _ds['lon'].values
if _sla.ndim == 1:
    SIAL_LON, SIAL_LAT = np.meshgrid(_slo, _sla)
else:
    SIAL_LAT, SIAL_LON = _sla, _slo
SIAL_SHAPE = SIAL_LAT.shape

try:
    ARCTIC_SIAL = np.load(f'{CACHE_OBS}/arctic_60N_sial.npy').astype(bool)
    print('  Loaded arctic_60N_sial mask from cache.', flush=True)
except FileNotFoundError:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)
    print('  Built ARCTIC_SIAL from lat >= 60.', flush=True)

IDX_APPX_TO_SIAL = None
try:
    from netCDF4 import Dataset as _NC4
    _appx_sample = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with _NC4(_appx_sample, 'r') as _ds_nc:
        _al = np.ma.filled(_ds_nc.variables['latitude'][:],  np.nan).astype(float)
        _ao = np.ma.filled(_ds_nc.variables['longitude'][:], np.nan).astype(float)
    if _al.ndim == 1:
        _ao2, _al2 = np.meshgrid(_ao, _al)
    else:
        _al2, _ao2 = _al, _ao
    _, IDX_APPX_TO_SIAL = cKDTree(_xyz(_al2, _ao2)).query(
        _xyz(SIAL_LAT, SIAL_LON), workers=-1)
    IDX_APPX_TO_SIAL = IDX_APPX_TO_SIAL.astype(np.int32)
    print('  APP-x → SIAL index built.', flush=True)
except Exception as _e:
    print(f'  WARNING: APP-x → SIAL index failed ({_e})', flush=True)

print('  Building SIAL → PSN25 index...', flush=True)
_, IDX_SIAL_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
    _xyz(SIAL_LAT, SIAL_LON), workers=-1)
IDX_SIAL_TO_PSN = IDX_SIAL_TO_PSN.astype(np.int32)
print('  Done.', flush=True)


# ============================================================
# GRID UTILITIES
# ============================================================
def scatter_avg(flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    ok  = np.isfinite(flat)
    np.add.at(acc, idx[ok], flat[ok].astype(np.float32))
    np.add.at(cnt, idx[ok], 1.)
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def domain_mean(arr, dom=None, wt=None):
    dom = DOMAIN_PSN if dom is None else dom
    wt  = AREA_W     if wt  is None else wt
    v   = arr[dom]; w = wt[dom]
    ok  = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.sum() else np.nan

def _regrid_sial_to_psn(arr):
    src = np.where(ARCTIC_SIAL,
                   np.where(np.isfinite(arr), arr, np.float32(0.)),
                   np.nan)
    return scatter_avg(src.ravel().astype(np.float32),
                       IDX_SIAL_TO_PSN, PSN_SHAPE)


# ============================================================
# STREAMING / ONLINE OLS  RS ~ SIC
# ============================================================
class OnlineOLS:
    """
    Welford-style running OLS for  y ~ x  at every pixel.
    Memory: 5 running-sum arrays (float64) + shape info.
    """
    def __init__(self, shape):
        P       = int(np.prod(shape))
        self._s = shape
        self.n  = np.zeros(P, np.float64)
        self.sx = np.zeros(P, np.float64)
        self.sy = np.zeros(P, np.float64)
        self.sxx= np.zeros(P, np.float64)
        self.sxy= np.zeros(P, np.float64)
        self.syy= np.zeros(P, np.float64)

    def update(self, x_map, y_map):
        x = x_map.ravel().astype(np.float64)
        y = y_map.ravel().astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        self.n[ok]  += 1
        self.sx[ok] += x[ok]
        self.sy[ok] += y[ok]
        self.sxx[ok]+= x[ok]**2
        self.sxy[ok]+= x[ok]*y[ok]
        self.syy[ok]+= y[ok]**2

    def finalize(self, min_n=MIN_OBS_FOR_REG, sic_std_min=0.01):
        n   = self.n
        denom = n*self.sxx - self.sx**2
        slope = np.full(n.size, np.nan, np.float32)
        r2    = np.full(n.size, np.nan, np.float32)
        good  = (n >= min_n) & (denom > n*(sic_std_min**2)*n)
        slope[good] = ((n[good]*self.sxy[good] - self.sx[good]*self.sy[good])
                       / denom[good]).astype(np.float32)
        cov_xy = n*self.sxy - self.sx*self.sy
        var_y  = n*self.syy - self.sy**2
        r2_raw = np.where(
            good & (denom > 0) & (var_y > 0),
            cov_xy**2 / (denom * var_y), np.nan)
        r2[good] = np.clip(r2_raw[good], 0., 1.).astype(np.float32)
        return slope.reshape(self._s), r2.reshape(self._s)

    def domain_slope_mean(self, dom, wt):
        n = self.n
        denom = n*self.sxx - self.sx**2
        sic_var_thresh = n * (0.01**2) * n
        good = (n >= MIN_OBS_FOR_REG) & (denom > sic_var_thresh)
        s = np.full(n.size, np.nan, np.float64)
        s[good] = (n[good]*self.sxy[good] - self.sx[good]*self.sy[good]) / denom[good]
        s2d = s.reshape(self._s)
        s2d[~dom] = np.nan
        return domain_mean(s2d.astype(np.float32), dom, wt)


# ============================================================
# TREND / SIGNIFICANCE HELPERS
# ============================================================
def trend_stats(yrs, vals):
    x = np.asarray(yrs, float); y = np.asarray(vals, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    sl, ic, r, p, se = linregress(x[ok], y[ok])
    return float(sl), float(p), float(r**2)

def sig_stars(p):
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

def _draw_trend(ax, yrs, vals, color, lw=1.2, label=None, zorder=4):
    sl, pv, r2 = trend_stats(yrs, vals)
    if not np.isfinite(sl):
        return sl, pv
    sig = np.isfinite(pv) and pv < SIG_THRESH
    yt  = sl*yrs + (np.nanmean(vals) - sl*np.nanmean(yrs))
    lbl = label or f'{sl:+.3f} W m⁻²/yr {sig_stars(pv)} (p={pv:.2f})'
    ax.plot(yrs, yt, color=color, linewidth=lw,
            linestyle='-.' if sig else '--',
            alpha=0.90 if sig else 0.22, zorder=zorder, label=lbl)
    return sl, pv


# ============================================================
# T07 INVERSION
# ============================================================
def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd = np.where(rsds  > 1., rsds,  np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds/S;  q  = rsut/S
        d  = q + p - 2.*p*a
        R  = np.clip((q-p*a) / np.where(np.abs(d)>1e-6,d,np.nan), 0., .99)
        A  = np.clip(1.-p*(1.-a*R) / np.where((1.-R)>.01,1.-R,np.nan), 0., .99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs/S;  qc = rsutcs/S
        dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc-pc*ac) / np.where(np.abs(dc)>1e-6,dc,np.nan), 0., .99)
        Rl = np.clip(R - Rc, 0., .99)
        RS = (S*(1.-A)*(1.-Rl)**2
              / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS.astype(np.float32)


# ============================================================
# OBS LOADERS  (CERES + PSN25 SIC)
# ============================================================
def load_ceres_rs(year, month):
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idxs  = [i for i, t in enumerate(times)
                 if t.year == year and t.month == month]
        if not idxs:
            return None
        sums = {k: None for k in _CERES_VARS}; cnt = 0
        for i in idxs:
            sl = ds.isel(time=i).load()
            for k in _CERES_VARS:
                v = sl[k].values.astype(np.float32)
                sums[k] = v if cnt == 0 else sums[k] + v
            cnt += 1; del sl
    if cnt == 0:
        return None
    for k in sums:
        sums[k] /= cnt
    rs = t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                    sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                    sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                    sums['sfc_sw_up_clr_c_mon'])
    return scatter_avg(rs.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)


def load_psn_sic(year, month):
    pat   = f'{DIR_SIC}/sic_psn25_{year}{month:02d}*.nc'
    files = sorted(glob.glob(pat))
    if not files:
        return None
    try:
        with xr.open_dataset(files[0]) as ds:
            vname = next((v for v in ds.data_vars
                          if any(k in v.lower()
                                 for k in ('sic', 'siconc', 'conc'))), None)
            if vname is None:
                return None
            arr = ds[vname].values.squeeze().astype(np.float32)
        if np.nanmax(arr) > 1.5:
            arr /= 100.
        return arr
    except Exception as e:
        print(f'  load_psn_sic({year},{month}): {e}', flush=True)
        return None


# ============================================================
# ERA5 RS  — Huang kernel × CLARA-A3/APP-x albedo
# ============================================================
def _load_obs_albedo(year, month):
    mname = MONTH_NAMES_FULL[month]

    if month == 9:
        p = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
        if not os.path.exists(p):
            return None
        raw = np.load(p).astype(np.float32)
        if IDX_APPX_TO_SIAL is not None and raw.shape != SIAL_SHAPE:
            raw = raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
        return raw

    p_c = f'{CACHE_OBS}/clara_alb_{mname}_{year}.npy'
    p_a = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
    clara = np.load(p_c).astype(np.float32) if os.path.exists(p_c) else None
    if os.path.exists(p_a):
        raw_a = np.load(p_a).astype(np.float32)
        if IDX_APPX_TO_SIAL is not None and raw_a.shape != SIAL_SHAPE:
            raw_a = raw_a.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
        appx = raw_a
    else:
        appx = None

    if clara is not None and appx is not None:
        return np.where(np.isfinite(clara), clara, appx)
    return clara if clara is not None else appx


def load_era5_kernel_map():
    if not os.path.exists(KERNEL_FILE):
        print(f'  ERA5 kernel not found: {KERNEL_FILE}', flush=True)
        return {}
    try:
        with xr.open_dataset(KERNEL_FILE) as ds:
            tdim  = next((d for d, s in ds.sizes.items() if s == 12), None)
            vname = KERNEL_VAR
            if vname not in ds:
                vname = next((v for v in ds.data_vars
                              if v.lower() == KERNEL_VAR.lower()), None)
            lat_n = 'latitude' if 'latitude' in ds else 'lat'
            lon_n = 'longitude' if 'longitude' in ds else 'lon'
            k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
            k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
            _, idx_k = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
                _xyz(k_lat2d, k_lon2d), workers=-1)
            out = {}
            for mn in MONTH_NUMS:
                k_raw = (np.abs(ds[vname].isel({tdim: mn-1}).values
                                .astype(np.float32)) * KERNEL_SCALE)
                out[mn] = scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)
        return out
    except Exception as e:
        print(f'  ERA5 kernel load failed: {e}', flush=True)
        return {}


def load_era5_krs(year, month, kernel_psn):
    alb_sial = _load_obs_albedo(year, month)
    if alb_sial is None:
        return None
    alb_psn = _regrid_sial_to_psn(alb_sial)
    return (kernel_psn * alb_psn).astype(np.float32)


# ============================================================
# CMIP6 UTILITIES
# ============================================================
import re

def _files_cover_years(files, years):
    yr_set = set(years)
    for f in files:
        m = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', os.path.basename(f))
        if m:
            if any(int(m.group(1)[:4]) <= y <= int(m.group(2)[:4])
                   for y in yr_set):
                return True
        else:
            return True
    return False

def find_var_files(inst, model, member, variable):
    tables = VAR_TABLE_MAP.get(variable, ['Amon'])
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in tables:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, variable)
            if not os.path.isdir(d):
                continue
            files = sorted(glob.glob(
                os.path.join(d, '**', f'{variable}_*.nc'), recursive=True))
            files = [f for f in files
                     if _files_cover_years([f], ALL_NEEDED_YEARS)]
            if files:
                return files
    return []

def _open_robust(files, variable):
    for kw in [
        dict(combine='by_coords', use_cftime=True, chunks={'time': 12}),
        dict(combine='nested', concat_dim='time', coords='minimal',
             compat='override', use_cftime=True, chunks={'time': 12}),
    ]:
        try:
            return xr.open_mfdataset(files, **kw)
        except Exception:
            pass
    raise RuntimeError(f'Cannot open {variable}')

def _std_lat(da):
    for name in ['lat', 'latitude', 'nav_lat', 'rlat']:
        if name in da.coords:
            v = da.coords[name].values
            if v.ndim == 1 and v.min() >= -90.1 and v.max() <= 90.1:
                return v
    return None

def _get_latlon_flat(ds, varname):
    da = ds[varname]
    spatial_sh = da.shape[1:] if da.ndim >= 3 else da.shape
    ny = spatial_sh[0] if len(spatial_sh) >= 1 else None
    nx = spatial_sh[1] if len(spatial_sh) >= 2 else None
    lat_names = ['lat','latitude','nav_lat','TLAT','qlat','nlat','y']
    lon_names = ['lon','longitude','nav_lon','TLONG','qlon','nlon','x']
    FILL = 1e10
    def _clean(arr):
        return np.where(np.abs(arr) < FILL, arr.astype(float), np.nan)
    lat1d = lon1d = None
    for n in lat_names:
        if n in ds.coords:
            v = _clean(ds.coords[n].values)
            if v.ndim == 1 and np.nanmin(v) >= -90.1 and np.nanmax(v) <= 90.1:
                lat1d = v; break
    for n in lon_names:
        if n in ds.coords:
            v = _clean(ds.coords[n].values)
            if v.ndim == 1:
                lon1d = v; break
    if lat1d is not None and lon1d is not None:
        lo2d, la2d = np.meshgrid(lon1d, lat1d)
        return la2d.ravel(), lo2d.ravel()
    lat2d = lon2d = None
    for n in lat_names:
        if n in ds.coords:
            v = _clean(ds.coords[n].values)
            if v.ndim == 2 and (ny is None or v.shape[0] == ny):
                if np.nanmin(v) >= -90.1 and np.nanmax(v) <= 90.1:
                    lat2d = v; break
    for n in lon_names:
        if n in ds.coords:
            v = _clean(ds.coords[n].values)
            if v.ndim == 2 and (nx is None or v.shape[1] == nx):
                lon2d = v; break
    if lat2d is not None and lon2d is not None:
        return lat2d.ravel(), lon2d.ravel()
    return None, None


# ============================================================
# STREAMING CMIP6 RS-SIC OLS  (one model at a time)
# ============================================================
def cmip6_streaming_slope(model, member, years):
    """
    Returns {mn: domain_mean_slope}.
    Bulk-loads all target (year, month) slices per variable in one
    .compute() call, then iterates year-outer / month-inner over the
    in-memory cache.  Never holds more than one model's worth of data.
    """
    inst  = MODEL_INFO[model]['institution']
    rf    = {v: find_var_files(inst, model, member, v) for v in RAD_VARS}
    sic_f = find_var_files(inst, model, member, 'siconc')
    avail     = {v for v, f in rf.items() if f}
    has_full  = set(RAD_VARS) <= avail
    has_approx= (set(RAD_VARS) - {'rsuscs'}) <= avail
    if (not has_full and not has_approx) or not sic_f:
        return None

    # ── Build atm grid and SIC → atm index once ───────────────────────────
    try:
        ds_g    = _open_robust(rf['rsds'], 'rsds')
        lat_arr = _std_lat(ds_g['rsds'])
        lon_arr = None
        for nm in ['lon', 'longitude']:
            if nm in ds_g.coords:
                lon_arr = ds_g.coords[nm].values; break
        ds_g.close()
        if lat_arr is None or lon_arr is None:
            return None
    except Exception as e:
        print(f'  [{model}] grid: {e}', flush=True)
        return None

    atm_shape = (len(lat_arr), len(lon_arr))
    lon2d, lat2d = np.meshgrid(lon_arr, lat_arr)
    lon_pm = lon2d - 360.*np.floor((lon2d+180.)/360.)
    if _HAS_LAND_MASK:
        dom_m = (lat2d >= ARCTIC_LAT_MIN) & _gm_globe.is_ocean(lat2d, lon_pm)
    else:
        dom_m = lat2d >= ARCTIC_LAT_MIN
    aw_m = np.cos(np.deg2rad(lat2d)).astype(np.float32)

    idx_sic2atm    = None
    sic_valid_mask = None
    try:
        ds_sic = _open_robust(sic_f, 'siconc')
        sic_lat_flat, sic_lon_flat = _get_latlon_flat(ds_sic, 'siconc')
        ds_sic.close()
        if sic_lat_flat is not None:
            ok_sic = (np.isfinite(sic_lat_flat) & np.isfinite(sic_lon_flat)
                      & (np.abs(sic_lat_flat) <= 90.))
            sic_valid_mask = ok_sic
            _, idx_sic2atm = cKDTree(_xyz(lat2d, lon2d)).query(
                _xyz(sic_lat_flat[ok_sic], sic_lon_flat[ok_sic]),
                workers=-1)
            print(f'    [{model}] SIC grid: {ok_sic.sum():,} valid pts → '
                  f'{atm_shape} atm grid', flush=True)
    except Exception as e:
        print(f'    [{model}] SIC grid error: {e}', flush=True)

    # ── Open datasets (lazy) ──────────────────────────────────────────────
    vlist = RAD_VARS if has_full else [v for v in RAD_VARS if v != 'rsuscs']
    try:
        ds_rad  = {v: _open_robust(rf[v], v) for v in vlist}
        ds_sic2 = _open_robust(sic_f, 'siconc')
    except Exception as e:
        print(f'    [{model}] open error: {e}', flush=True)
        return None

    # ── Bulk-load all target (year, month) slices at once ─────────────────
    # One .compute() per variable instead of 198 separate isel calls.
    target_yrs  = list(years)
    target_mons = MONTH_NUMS

    def _bulk_load(ds, varname):
        da       = ds[varname]
        yr_vals  = da.time.dt.year.values.astype(int)
        mon_vals = da.time.dt.month.values.astype(int)
        mask     = np.isin(yr_vals, target_yrs) & np.isin(mon_vals, target_mons)
        if not mask.any():
            return {}
        sub      = da.isel(time=mask).compute()
        sub_yr   = yr_vals[mask]
        sub_mon  = mon_vals[mask]
        cache    = {}
        for i, (y, m) in enumerate(zip(sub_yr, sub_mon)):
            cache[(int(y), int(m))] = sub.isel(time=i).values.astype(np.float32)
        return cache

    try:
        print(f'    [{model}] bulk-loading SIC...', flush=True)
        sic_cache = _bulk_load(ds_sic2, 'siconc')
        rad_cache = {}
        for v in vlist:
            print(f'    [{model}] bulk-loading {v}...', flush=True)
            rad_cache[v] = _bulk_load(ds_rad[v], v)
    except MemoryError:
        for v in vlist:
            try: ds_rad[v].close()
            except: pass
        try: ds_sic2.close()
        except: pass
        raise
    except Exception as e:
        print(f'    [{model}] bulk load error: {e}', flush=True)
        for v in vlist:
            try: ds_rad[v].close()
            except: pass
        try: ds_sic2.close()
        except: pass
        return None

    # Close lazy datasets — everything we need is now in cache dicts
    for v in vlist:
        try: ds_rad[v].close()
        except: pass
    try: ds_sic2.close()
    except: pass

    if not sic_cache:
        return None

    # ── Streaming OLS — year-outer, month-inner ───────────────────────────
    ols  = {mn: OnlineOLS(atm_shape) for mn in MONTH_NUMS}
    n_ok = {mn: 0                    for mn in MONTH_NUMS}

    for yr in target_yrs:
        for mn in MONTH_NUMS:
            sic_arr = sic_cache.get((yr, mn))
            if sic_arr is None:
                continue

            if np.nanmax(sic_arr) > 1.5:
                sic_arr = sic_arr / 100.

            if idx_sic2atm is not None:
                flat = sic_arr.ravel().astype(np.float32)
                if sic_valid_mask is not None:
                    flat = flat[sic_valid_mask]
                sic_arr = scatter_avg(flat, idx_sic2atm, atm_shape)
            elif sic_arr.shape != atm_shape:
                continue

            rad  = {}
            ok   = True
            for v in vlist:
                a = rad_cache[v].get((yr, mn))
                if a is None:
                    ok = False; break
                rad[v] = a
            if not ok:
                continue

            if not has_full:
                with np.errstate(invalid='ignore', divide='ignore'):
                    alb = np.where(rad['rsds'] > 1.,
                                   rad['rsus'] / rad['rsds'], 0.)
                rad['rsuscs'] = rad['rsdscs'] * alb

            rs = t07_invert(rad['rsdt'], rad['rsut'], rad['rsds'], rad['rsus'],
                            rad['rsutcs'], rad['rsdscs'], rad['rsuscs'])
            if rs.shape != atm_shape:
                continue

            rs_max = float(np.nanmax(rs))
            if rs_max > 2000. or rs_max < 1.:
                continue

            ols[mn].update(sic_arr, rs)
            n_ok[mn] += 1
            del rad, rs, sic_arr

    del sic_cache, rad_cache
    gc.collect()

    # ── Finalize ──────────────────────────────────────────────────────────
    results = {}
    for mn in MONTH_NUMS:
        if n_ok[mn] < MIN_OBS_FOR_REG:
            print(f'    {MONTH_NAMES[mn]}: skip (n={n_ok[mn]} < {MIN_OBS_FOR_REG})',
                  flush=True)
            results[mn] = np.nan
        else:
            raw_mean = ols[mn].domain_slope_mean(dom_m, aw_m)
            if np.isfinite(raw_mean) and np.abs(raw_mean) > 500.:
                print(f'    {MONTH_NAMES[mn]}: SUSPECT slope {raw_mean:+.1f} → NaN',
                      flush=True)
                results[mn] = np.nan
            else:
                print(f'    {MONTH_NAMES[mn]}: dRS/dSIC = {raw_mean:+.3f}  '
                      f'(n={n_ok[mn]} yrs)', flush=True)
                results[mn] = raw_mean
        del ols[mn]
        gc.collect()

    return results


# ============================================================
# STREAMING OBS OLS  (pixel-level)
# ============================================================
def obs_streaming_slope(obs_rs_list, obs_sic_list):
    ols = OnlineOLS(PSN_SHAPE)
    for rs, sic in zip(obs_rs_list, obs_sic_list):
        if rs is None or sic is None:
            continue
        ols.update(sic, rs)
    slope, r2 = ols.finalize()
    slope = np.where(DOMAIN_PSN, slope, np.nan)
    return slope, domain_mean(slope), ols.n.reshape(PSN_SHAPE)


# ============================================================
# MAIN  ── OBS
# ============================================================
print('\n' + '='*60, flush=True)
print('LOADING OBS (CERES + PSN25 SIC)', flush=True)
print('='*60, flush=True)

obs_rs  = [[None]*len(YEARS_OBS) for _ in MONTH_NUMS]
obs_sic = [[None]*len(YEARS_OBS) for _ in MONTH_NUMS]
for mi, mn in enumerate(MONTH_NUMS):
    print(f'  {MONTH_NAMES[mn]}...', end=' ', flush=True)
    for yi, yr in enumerate(YEARS_OBS):
        obs_rs[mi][yi]  = load_ceres_rs(yr, mn)
        obs_sic[mi][yi] = load_psn_sic(yr, mn)
    print('done', flush=True)
gc.collect()

# ── ERA5 Huang kernel × CLARA-A3/APP-x albedo RS ─────────────────────────────
print('\nLoading ERA5 Huang-kernel RS (personal cache, 1982–2024)...', flush=True)
ERA5_KERNEL_MAP   = load_era5_kernel_map()
ERA5_KERNEL_DMEAN = {mn: domain_mean(ERA5_KERNEL_MAP[mn])
                     if mn in ERA5_KERNEL_MAP else np.nan
                     for mn in MONTH_NUMS}

era5_krs_domain = {mn: ([], []) for mn in MONTH_NUMS}

for mi, mn in enumerate(MONTH_NUMS):
    print(f'  ERA5 kernel RS {MONTH_NAMES[mn]}...', end=' ', flush=True)
    if mn not in ERA5_KERNEL_MAP:
        print('no kernel — skip', flush=True)
        continue
    k_psn = ERA5_KERNEL_MAP[mn]
    n_ok = 0
    for yr in YEARS_ERA5:
        krs_map = load_era5_krs(yr, mn, k_psn)
        if krs_map is not None:
            era5_krs_domain[mn][0].append(yr)
            era5_krs_domain[mn][1].append(domain_mean(krs_map))
            n_ok += 1
            del krs_map
    print(f'{n_ok} years loaded', flush=True)

era5_rs_domain = {mn: (np.array([]), np.array([])) for mn in MONTH_NUMS}

for mn in MONTH_NUMS:
    era5_krs_domain[mn] = (np.array(era5_krs_domain[mn][0]),
                            np.array(era5_krs_domain[mn][1]))

# ── Analysis A: streaming OBS pixel OLS RS ~ SIC (2000-2014 CERES subset) ────
print('\nAnalysis A: streaming pixel OLS RS ~ SIC (2000-2014 CERES subset)...',
      flush=True)
_cmip6_obs_overlap = set(YEARS_CMIP6) & set(YEARS_OBS)
cmip6_yidx = [i for i, y in enumerate(YEARS_OBS) if y in _cmip6_obs_overlap]

OBS_SLOPE      = {}
OBS_SLOPE_MEAN = {}
for mi, mn in enumerate(MONTH_NUMS):
    rs_sub  = [obs_rs[mi][yi]  for yi in cmip6_yidx]
    sic_sub = [obs_sic[mi][yi] for yi in cmip6_yidx]
    slope, dmean, _ = obs_streaming_slope(rs_sub, sic_sub)
    OBS_SLOPE[mn]      = slope
    OBS_SLOPE_MEAN[mn] = dmean
    n_valid = sum(r is not None and s is not None
                  for r, s in zip(rs_sub, sic_sub))
    print(f'  {MONTH_NAMES[mn]}: dRS/dSIC = {dmean:+.3f} W m⁻²  '
          f'(n={n_valid} years, CERES 2000-2014 overlap)', flush=True)

# ── Analysis B: RS by ice state ───────────────────────────────────────────────
print('\nAnalysis B: RS by ice state (2000-2024)...', flush=True)

def _ice_composites(rs_list, sic_list, years):
    rec = {'years': [], 'ice': [], 'marginal': [], 'free': []}
    for yr, rs, sic in zip(years, rs_list, sic_list):
        if rs is None or sic is None:
            continue
        def _cmean(mask):
            m2 = mask & np.isfinite(rs)
            v, w = rs[m2], AREA_W[m2]
            ok = np.isfinite(w) & (w > 0)
            return float(np.sum(v[ok]*w[ok])/np.sum(w[ok])) if ok.sum() else np.nan
        rec['years'].append(yr)
        rec['ice'].append(     _cmean(DOMAIN_PSN & (sic >= SIC_ICE_THRESH)))
        rec['marginal'].append(_cmean(DOMAIN_PSN & (sic >= SIC_MARG_THRESH)
                                                 & (sic <  SIC_ICE_THRESH)))
        rec['free'].append(    _cmean(DOMAIN_PSN & (sic <  SIC_MARG_THRESH)))
    return {k: np.array(v) for k, v in rec.items()}

rs_by_state = {}
for mi, mn in enumerate(MONTH_NUMS):
    rs_by_state[mn] = _ice_composites(obs_rs[mi], obs_sic[mi], YEARS_OBS)

print('\nBuilding ERA5 kernel-RS ice-state composites (1982-1999)...', flush=True)
era5_rs_by_state = {}
for mi, mn in enumerate(MONTH_NUMS):
    era5_rs_m, era5_sic_m, era5_yr_m = [], [], []
    k_psn = ERA5_KERNEL_MAP.get(mn)
    if k_psn is not None:
        for yr in YEARS_ERA5:
            if yr >= 2000:
                continue
            krs_map = load_era5_krs(yr, mn, k_psn)
            sic_map = load_psn_sic(yr, mn)
            era5_rs_m.append(krs_map)
            era5_sic_m.append(sic_map)
            era5_yr_m.append(yr)
            del krs_map
    era5_rs_by_state[mn] = _ice_composites(era5_rs_m, era5_sic_m, era5_yr_m)
    del era5_rs_m, era5_sic_m
    gc.collect()

obs_rs_domain = {}
for mi, mn in enumerate(MONTH_NUMS):
    yrs, vals = [], []
    for yi, yr in enumerate(YEARS_OBS):
        if obs_rs[mi][yi] is not None:
            yrs.append(yr)
            vals.append(domain_mean(obs_rs[mi][yi]))
    obs_rs_domain[mn] = (np.array(yrs), np.array(vals))


# ============================================================
# MAIN  ── CMIP6  (streaming, with checkpointing + MemoryError guard)
# ============================================================
print('\n' + '='*60, flush=True)
print('LOADING CMIP6 RS and SIC (streaming OLS, 1982–2014)', flush=True)
print(f'Checkpoints: {CHECKPOINT_DIR}', flush=True)
print('='*60, flush=True)

cmip6_slopes = {}
for model, info in MODEL_INFO.items():
    member = info['member']
    ckpt   = os.path.join(CHECKPOINT_DIR, f'{model}_{member}.npy')

    # Resume from checkpoint if available
    if os.path.exists(ckpt):
        cmip6_slopes[model] = np.load(ckpt, allow_pickle=True).item()
        print(f'  {model}: loaded from checkpoint', flush=True)
        continue

    print(f'\n  {model} / {member}...', flush=True)
    try:
        result = cmip6_streaming_slope(model, member, YEARS_CMIP6)
        if result is None:
            print(f'    → skip (missing files)', flush=True)
        else:
            cmip6_slopes[model] = result
            np.save(ckpt, result)   # save immediately so restarts pick it up
            print(f'    → saved checkpoint', flush=True)
    except MemoryError:
        print(f'    → skip (MemoryError — model too large for available RAM)',
              flush=True)
        gc.collect()
    except Exception as e:
        print(f'    → skip ({type(e).__name__}: {e})', flush=True)
        gc.collect()


# ============================================================
# ─────────────────── FIGURES ─────────────────────────────────
# ============================================================
print('\nGenerating figures...', flush=True)
CMAP_YEAR = cm.plasma


# ============================================================
# FIGURE 1 — Domain-mean RS trend
# ============================================================
fig1, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#FAFAFA',
                           sharey=False, constrained_layout=True)
axes = axes.ravel()

for mi, mn in enumerate(MONTH_NUMS):
    ax = axes[mi]
    ax.set_facecolor('#FAFAFA')
    yrs, vals = obs_rs_domain[mn]
    if len(yrs) < 2:
        ax.set_title(MONTH_NAMES[mn]); continue

    norm_c = plt.Normalize(yrs.min(), yrs.max())
    for yi in range(len(yrs)):
        if yi > 0:
            ax.plot([yrs[yi-1], yrs[yi]], [vals[yi-1], vals[yi]],
                    color=CMAP_YEAR(norm_c(yrs[yi])), lw=1.8, zorder=2)
    ax.scatter(yrs, vals, c=yrs, cmap=CMAP_YEAR, s=40, zorder=3,
               edgecolors='white', linewidths=0.4)
    sl_c, pv_c, _ = trend_stats(yrs, vals)
    _draw_trend(ax, yrs, vals, 'k',
                label=f'CERES T07: {sl_c:+.3f} W m⁻²/yr '
                      f'{sig_stars(pv_c)} (p={pv_c:.2f})')

    ky, kv = era5_krs_domain[mn]
    if len(ky) > 1:
        norm_k = plt.Normalize(ky.min(), ky.max())
        for ki in range(len(ky)):
            if ki > 0:
                ax.plot([ky[ki-1], ky[ki]], [kv[ki-1], kv[ki]],
                        color='#2ca02c', lw=1.5, alpha=0.70, zorder=2)
        ax.scatter(ky, kv, c=ky, cmap=CMAP_YEAR, s=22, marker='^', zorder=3,
                   edgecolors='white', linewidths=0.3,
                   label='ERA5 kernel×α (CLARA-A3/APP-x, 1982–2024)')
        sl_k, pv_k, _ = trend_stats(ky, kv)
        _draw_trend(ax, ky, kv, '#2ca02c',
                    label=f'Kernel RS trend: {sl_k:+.3f} '
                          f'W m⁻²/yr {sig_stars(pv_k)} (p={pv_k:.2f})')
    elif mn in ERA5_KERNEL_MAP:
        kv_static = ERA5_KERNEL_DMEAN[mn]
        if np.isfinite(kv_static):
            ax.axhline(kv_static, color='#2ca02c', lw=0.9, ls='-.',
                       alpha=0.6, label=f'ERA5 kernel RS = {kv_static:.1f} W m⁻²')

    ax.legend(fontsize=6.5, framealpha=0.88, loc='best')
    ax.set_title(MONTH_NAMES[mn], fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Domain-mean RS  [W m⁻²]', fontsize=9)
    ax.grid(True, alpha=0.22)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

fig1.suptitle(
    'Domain-Mean RS Trend  (CERES 2000–2024  +  ERA5 kernel×α 1982–2024)\n'
    'Circles = CERES T07  |  Triangles = ERA5 Huang-kernel × CLARA-A3/APP-x α\n'
    'Solid/dash-dot trend = p < 0.05;  Dashed = not significant (α=0.22)',
    fontsize=11, fontweight='bold')
sm = cm.ScalarMappable(cmap=CMAP_YEAR, norm=plt.Normalize(1982, 2024))
sm.set_array([])
fig1.colorbar(sm, ax=axes, location='right', shrink=0.55,
              label='Year', pad=0.02)
fig1.savefig(os.path.join(OUT_DIR, 'fig1_obs_rs_seasonal.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig1)
print('  -> fig1_obs_rs_seasonal.png', flush=True)


# ============================================================
# FIGURE 2 — dRS/dSIC seasonal cycle  OBS vs CMIP6 MMM
# ============================================================
fig2, ax = plt.subplots(figsize=(11, 6), facecolor='#FAFAFA')
ax.set_facecolor('#FAFAFA')

obs_means = [OBS_SLOPE_MEAN[mn] for mn in MONTH_NUMS]
ax.plot(MONTH_NUMS, obs_means, 'o-', color='#b5000a',
        markersize=9, markeredgecolor='white', linewidth=2.8, zorder=5,
        label='OBS (CERES 2000–2014)')

if cmip6_slopes:
    n_models = len(cmip6_slopes)
    model_cmap = cm.tab20b
    all_curves = []
    for mi_m, (model, slopes) in enumerate(cmip6_slopes.items()):
        vals = [slopes.get(mn, np.nan) for mn in MONTH_NUMS]
        all_curves.append(vals)
        ax.plot(MONTH_NUMS, vals, '-',
                color=model_cmap(mi_m / max(n_models-1, 1)),
                lw=0.85, alpha=0.50, zorder=2, label=model)
    arr = np.array(all_curves)
    mmm = np.nanmean(arr, axis=0)
    std = (np.nanstd(arr, axis=0, ddof=1)
           if arr.shape[0] > 1 else np.zeros(len(MONTH_NUMS)))
    ax.fill_between(MONTH_NUMS, mmm-2*std, mmm+2*std,
                    color='#2166ac', alpha=0.15, zorder=1)
    ax.plot(MONTH_NUMS, mmm, 'D--', color='#2166ac', lw=2.2, markersize=7,
            markeredgecolor='white', zorder=4,
            label=f'CMIP6 MMM ± 2σ  ({n_models} models)')

ax.axhline(0, color='#444', lw=0.8, ls=':', alpha=0.7)
ax.set_xticks(MONTH_NUMS)
ax.set_xticklabels([MONTH_NAMES[m] for m in MONTH_NUMS])
ax.set_ylabel('Domain-mean  dRS/dSIC  [W m⁻² per unit SIC]', fontsize=11)
ax.set_title(f'RS–SIC Coupling Seasonal Cycle — OBS vs CMIP6 (1982–2014)\n'
             f'per-pixel streaming OLS  |  {len(cmip6_slopes)} models loaded',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.22)
handles, labels = ax.get_legend_handles_labels()
prim = [(h, l) for h, l in zip(handles, labels)
        if 'OBS' in l or 'MMM' in l]
rest = [(h, l) for h, l in zip(handles, labels)
        if 'OBS' not in l and 'MMM' not in l]
leg1 = ax.legend([h for h,l in prim], [l for h,l in prim],
                 fontsize=10, framealpha=0.92, loc='upper right')
ax.add_artist(leg1)
ax.legend([h for h,l in rest], [l for h,l in rest],
          fontsize=6.5, framealpha=0.85, ncol=4,
          loc='lower right')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_drs_dsic_seasonal.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig2)
print('  -> fig2_drs_dsic_seasonal.png', flush=True)


# ============================================================
# FIGURE 3 — OBS RS by ice state
# ============================================================
COL_ICE  = '#2166ac'
COL_MARG = '#4dac26'
COL_FREE = '#d01c8b'

fig3, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#FAFAFA',
                           sharey=False, constrained_layout=True)
axes = axes.ravel()

for mi, mn in enumerate(MONTH_NUMS):
    ax  = axes[mi]
    ax.set_facecolor('#FAFAFA')
    rec = rs_by_state[mn]
    yrs = rec['years']

    if len(yrs) >= 3:
        ax.plot(yrs, rec['ice'],      'o-', color=COL_ICE,  lw=2.,
                markersize=6, markeredgecolor='white',
                label='Ice-present (≥15%) CERES')
        ax.plot(yrs, rec['marginal'], 's-', color=COL_MARG, lw=1.8,
                markersize=5, markeredgecolor='white',
                label='Marginal (5–15%) CERES')
        ax.plot(yrs, rec['free'],     '^-', color=COL_FREE, lw=1.8,
                markersize=5, markeredgecolor='white',
                label='Ice-free (<5%) CERES')

        ypos = {'ice': 0.97, 'marginal': 0.88, 'free': 0.79}
        for key, col in [('ice', COL_ICE), ('marginal', COL_MARG),
                          ('free', COL_FREE)]:
            v = rec[key]
            sl, pv, _ = trend_stats(yrs, v)
            if np.isfinite(sl):
                yt  = sl*yrs + (np.nanmean(v) - sl*np.nanmean(yrs))
                sig = np.isfinite(pv) and pv < SIG_THRESH
                ax.plot(yrs, yt, color=col, linewidth=1.0,
                        alpha=0.90 if sig else 0.22,
                        linestyle='-.' if sig else '--')
                ax.text(0.98, ypos[key],
                        f'CERES: {sl:+.2f} W m⁻²/yr {sig_stars(pv)} '
                        f'(p={pv:.2f})',
                        transform=ax.transAxes, ha='right', va='top',
                        color=col, fontsize=7.,
                        alpha=0.90 if sig else 0.35)

    erec = era5_rs_by_state[mn]
    eyrs = erec['years']
    if len(eyrs) >= 3:
        ax.plot(eyrs, erec['ice'],  'o--', color=COL_ICE,  lw=1.4,
                markersize=5, markerfacecolor='white',
                markeredgecolor=COL_ICE, alpha=0.75,
                label='Ice-present ERA5 kernel RS')
        ax.plot(eyrs, erec['free'], '^--', color=COL_FREE, lw=1.4,
                markersize=4, markerfacecolor='white',
                markeredgecolor=COL_FREE, alpha=0.75,
                label='Ice-free ERA5 kernel RS')
        for key, col in [('ice', COL_ICE), ('free', COL_FREE)]:
            v = erec[key]
            sl, pv, _ = trend_stats(eyrs, v)
            if np.isfinite(sl):
                yt  = sl*eyrs + (np.nanmean(v) - sl*np.nanmean(eyrs))
                sig = np.isfinite(pv) and pv < SIG_THRESH
                ax.plot(eyrs, yt, color=col, linewidth=0.9,
                        alpha=0.75 if sig else 0.18,
                        linestyle='-.' if sig else ':')
        ax.axvline(1999.5, color='grey', lw=0.8, ls=':', alpha=0.5)

    ax.set_title(MONTH_NAMES[mn], fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Mean RS  [W m⁻²]', fontsize=9)
    ax.grid(True, alpha=0.22)
    if mi == 0:
        ax.legend(fontsize=7.5, framealpha=0.92, loc='upper left', ncol=2)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

fig3.suptitle(
    'OBS RS by Ice State  (CERES 2000–2024  +  ERA5 kernel×α 1982–1999)\n'
    'Filled = CERES  |  Open = ERA5 Huang-kernel × CLARA-A3/APP-x α  |  '
    'Solid/dash-dot = p < 0.05;  Dashed/dotted = not significant',
    fontsize=12, fontweight='bold')
fig3.savefig(os.path.join(OUT_DIR, 'fig3_obs_rs_by_icestate.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig3)
print('  -> fig3_obs_rs_by_icestate.png', flush=True)


# ============================================================
# FIGURE 4 — Spatial dRS/dSIC  (scatter plot on PSN25 lat/lon)
# ============================================================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False
    print('  NOTE: cartopy not available; fig4 uses imshow fallback.', flush=True)

_VMAX = 60.
fig4, axes4 = plt.subplots(
    2, 3, figsize=(15, 9),
    subplot_kw=dict(projection=ccrs.NorthPolarStereo()) if _HAS_CARTOPY else {},
    facecolor='#FAFAFA', constrained_layout=True)
axes4 = axes4.ravel()
norm4 = TwoSlopeNorm(vmin=-_VMAX, vcenter=0., vmax=_VMAX)

_last_sc = None
for mi, mn in enumerate(MONTH_NUMS):
    ax = axes4[mi]
    slope = OBS_SLOPE[mn].copy()
    slope[~DOMAIN_PSN] = np.nan
    valid = np.isfinite(slope) & DOMAIN_PSN

    if _HAS_CARTOPY:
        ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,  facecolor='#D0D0D0', zorder=2)
        ax.add_feature(cfeature.OCEAN, facecolor='#EEF3F7', zorder=0)
        ax.coastlines(linewidth=0.5, zorder=3)
        ax.gridlines(linewidth=0.3, color='grey', alpha=0.35,
                     zorder=4, draw_labels=False)
        if valid.sum() > 0:
            _last_sc = ax.scatter(
                PSN_LON[valid], PSN_LAT[valid],
                c=slope[valid],
                cmap='RdBu_r', norm=norm4,
                s=2,
                linewidths=0,
                transform=ccrs.PlateCarree(),
                zorder=1, rasterized=True)
    else:
        im = ax.imshow(np.flipud(slope), cmap='RdBu_r', norm=norm4,
                       origin='lower', aspect='auto',
                       extent=[-180, 180, 55, 90])
        _last_sc = im
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    _mv = OBS_SLOPE_MEAN[mn]
    _ms = f'{_mv:+.2f}' if np.isfinite(_mv) else 'NaN'
    ax.set_title(f'{MONTH_NAMES[mn]}  (domain mean = {_ms} W m⁻²)',
                 fontsize=10, fontweight='bold')

fig4.suptitle(
    'OBS dRS/dSIC  [W m⁻² per unit SIC]  —  CERES 2000–2014\n'
    'Red = RS increases with more ice  |  Blue = RS decreases with more ice',
    fontsize=12, fontweight='bold')
if _last_sc is not None:
    try:
        fig4.colorbar(_last_sc, ax=axes4, location='bottom', shrink=0.55,
                      label='dRS/dSIC  [W m⁻² / unit SIC]', pad=0.03,
                      extend='both')
    except Exception:
        pass
fig4.savefig(os.path.join(OUT_DIR, 'fig4_drs_dsic_maps.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig4)
print('  -> fig4_drs_dsic_maps.png', flush=True)


# ============================================================
# SUMMARY TABLE
# ============================================================
print('\n' + '='*60, flush=True)
col_w = 12
header = f'{"Month":<8}{"OBS":>{col_w}}' + \
         ''.join(f'{m:>{col_w+4}}' for m in cmip6_slopes.keys())
print(header, flush=True)
print('-'*min(len(header), 120), flush=True)
def _fmt(v, w=col_w):
    return f'{v:>+{w}.3f}' if np.isfinite(v) else f'{"NaN":>{w}}'
for mn in MONTH_NUMS:
    row = f'{MONTH_NAMES[mn]:<8}' + _fmt(OBS_SLOPE_MEAN[mn])
    for model in cmip6_slopes:
        row += _fmt(cmip6_slopes[model].get(mn, np.nan), col_w+4)
    print(row, flush=True)

print(f'\nCMIP6 models loaded: {len(cmip6_slopes)} / {len(MODEL_INFO)}', flush=True)
print(f'All outputs → {OUT_DIR}', flush=True)
print('DONE.', flush=True)