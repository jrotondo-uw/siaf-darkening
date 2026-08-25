#!/usr/bin/env python
"""
compute_siaf_drk_per_ice_area.py
─────────────────────────────────
Computes SIAF_drk normalised by stable-ice area rather than by the full
Arctic-ocean domain area.

Standard SIAF_drk  =  Σ(IS_drk × RS × w) / Σ(w_all_domain)
Per-ice-area SIAF  =  Σ(IS_drk × RS × w) / Σ(w_stable_ice_pixels)

The numerator is identical; only the denominator mask changes.
This removes the confound of models having different amounts of stable
sea ice, answering: "how much is each model/observation darkening
its extant sea ice per unit area?"

OBS  : loads IS_drk and SIC from existing sensitivity caches + CERES RS.
       No recomputation of albedo time series needed.
CMIP6: minimal re-run of the IS_drk + RS pipeline on native grids.

Output
------
  siaf_drk_per_ice_area.csv  — one row per (source, model/obs, member,
                                rs_method, period, month)
  with columns:
      siaf_drk_global        (standard, full-domain denominator)
      siaf_drk_per_ice       (new, stable-ice-area denominator)
      stable_ice_area_frac   (fraction of domain that is stable ice)

OOM FIX (2026-05-13)
---------------------
- Eliminated the double-load of rsus/rsds. The old code loaded them as
  `ru`/`rd` for TREND_YEARS and then loaded them AGAIN inside `_rad_das`
  for RS_YEARS — two full copies of both large arrays in memory at once.
  Now ALL radiation vars (including rsus/rsds) are loaded ONCE inside
  `_rad_das`, using TREND_YEARS for rsus/rsds so the albedo loop has the
  full 1982-2014 range.  After `rs_by_month` is built, the 5 non-albedo
  DataArrays are freed immediately; `ru`/`rd` are popped out of the dict
  (no copy) and kept alive only for the month loop, then freed.
- `si` DataArray freed immediately after numpy extraction.
- `_da.time.dt.month.values` replaced with a list comprehension cast to
  `dtype=bool` — the vectorised `.dt` accessor fails on cftime arrays in
  some xarray versions.

RESUME LOGIC (2026-05-13)
--------------------------
- If a per-model partial CSV already exists and is non-empty, that model
  is skipped and its rows loaded into cmip6_rows.  Allows re-submission
  after an OOM kill without reprocessing completed models.
- Partial CSV saves contain ONLY the current model's rows (not all
  accumulated rows), preventing duplication on re-submission.
"""

import ctypes, gc, glob, os, warnings
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import t as t_dist
from netCDF4 import Dataset as NC4Dataset

warnings.filterwarnings('ignore')

try:
    from global_land_mask import globe as _gm_globe
    _HAS_LAND_MASK = True
except ImportError:
    _HAS_LAND_MASK = False
    print('WARNING: global-land-mask not installed; ocean mask = lat >= 60 N.')

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
CACHE_CMIP6  = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS    = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
DIR_SIAL     = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX     = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC      = '/glade/derecho/scratch/jrotondo/SIC_data/'
CERES_FILE   = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
                'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
KERNEL_FILE  = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR   = 'SFC_all'
KERNEL_SCALE = 100.0

CMIP6_ROOT   = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT   = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

OUT_CSV = ('/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
           'siaf_drk_metrics_v2.csv')

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
PSN25_PROJ4  = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
                '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

MONTH_NUMS   = [4, 5, 6, 7, 8, 9]
MONTH_NAMES  = {4:'April', 5:'May', 6:'June',
                7:'July',  8:'August', 9:'September'}

SIC_THRESH   = 0.15
PVAL         = 0.05
ARCTIC_LAT   = 60.

# Models to skip in this run (set to non-empty to split a large job).
# Skipped models are simply omitted — their partial CSVs are written by a
# separate job.  Run job A with this set, then job B with CMIP6_MODELS
# filtered to only these two and SKIP_MODELS = set().
SKIP_MODELS  = {'MIROC-ES2L', 'ACCESS-ESM1-5'}

# Set True to skip the entire OBS phase (grid setup, CERES cache, kernel
# preload, OBS computation).  Use this when running a follow-up job that
# only processes a subset of CMIP6 models (e.g. MIROC-ES2L, ACCESS-ESM1-5)
# after the main job has already finished the OBS rows and saved them to
# the output CSV.  Skipping the OBS phase saves ~1.1 GB of baseline RSS.
CMIP6_ONLY   = False

AREA_RATIO   = 0.0347
SEASON_FRAC  = 6. / 12.
SCALE        = AREA_RATIO * SEASON_FRAC

CERES_CMIP6  = list(range(2000, 2015))
CERES_OBS    = list(range(2000, 2025))
TREND_CMIP6  = list(range(1982, 2015))
TREND_OBS    = list(range(1982, 2025))
TREND_YEARS  = TREND_CMIP6          # alias used throughout CMIP6 section
EARLY_YEARS  = list(range(1982, 1992))
LATE_YEARS   = list(range(2005, 2015))

CMIP6_MODELS = {
    'CESM2'          : dict(institution='NCAR',           member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',          member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',          member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',           member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',          member='r1i1p1f1'),
    'CanESM5'        : dict(institution='CCCma',          member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',           member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',           member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS',   member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',            member='r1i1p1f1'),
    'ACCESS-CM2'     : dict(institution='CSIRO-ARCCSS',   member='r1i1p1f1'),
    'CNRM-ESM2-1'    : dict(institution='CNRM-CERFACS',   member='r1i1p1f2'),
    'GISS-E2-1-H'    : dict(institution='NASA-GISS',      member='r1i1p1f1'),
    'INM-CM5-0'      : dict(institution='INM',            member='r1i1p1f1'),
    'MRI-ESM2-0'     : dict(institution='MRI',            member='r1i1p1f1'),
    'MIROC-ES2L'     : dict(institution='MIROC',          member='r1i1p1f2'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',          member='r1i1p1f1'),
}

VAR_TABLE_MAP = {
    'rsus'   : ['Amon'],
    'rsds'   : ['Amon'],
    'rsdt'   : ['Amon'],
    'rsut'   : ['Amon'],
    'rsutcs' : ['Amon', 'CFmon'],
    'rsdscs' : ['Amon', 'CFmon'],
    'rsuscs' : ['Amon', 'CFmon'],
    'siconc' : ['SImon'],
    'tas'    : ['Amon'],
}
RAD_VARS    = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs', 'rsuscs']
FILL_THRESH = 1e10


# ══════════════════════════════════════════════════════════════════════════════
# SHARED GRID UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def latlon_to_xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel())
    lo = np.deg2rad(np.asarray(lon).ravel())
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])

def build_scatter_index(src_lat, src_lon, tgt_lat, tgt_lon):
    _, idx = cKDTree(latlon_to_xyz(tgt_lat, tgt_lon)).query(
        latlon_to_xyz(src_lat, src_lon), workers=-1)
    return idx.astype(np.int32)

def scatter_avg(arr_flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    v   = np.isfinite(arr_flat)
    np.add.at(acc, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(cnt, idx[v], np.float32(1.))
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def _mask_fills(arr):
    arr = np.asarray(arr, dtype=float)
    return np.where(np.abs(arr) < FILL_THRESH, arr, np.nan)

def _rss_mb():
    """Current process RSS in MB (Linux /proc; returns 0 elsewhere)."""
    try:
        with open('/proc/self/status') as _f:
            for _line in _f:
                if _line.startswith('VmRSS:'):
                    return float(_line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def _malloc_trim():
    """
    Ask glibc to release freed heap memory back to the OS.

    Python's allocator, scipy KDTree internals, and netCDF4/HDF5 all use
    C malloc.  When large objects (KDTrees, DataArray buffers, CERES maps)
    are freed in Python the memory goes back to glibc's free-list rather
    than to the OS, inflating RSS permanently.  malloc_trim(0) collapses
    the top of the heap and returns those blocks to the kernel.

    This is safe — it only touches genuinely free memory — and is a no-op
    on non-Linux systems or if libc cannot be loaded.
    """
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# THE KEY NEW METRIC
# ══════════════════════════════════════════════════════════════════════════════

def stable_ice_mean(siaf_drk_map, is_drk_map, domain_mask, area_weights,
                    extent_mask=None):
    """
    Compute three normalizations of the darkening numerator.

    Parameters
    ----------
    siaf_drk_map  : IS_drk × RS pixel map (zero outside stable ice)
    is_drk_map    : IS_drk nan-version (NaN outside stable ice) — defines
                    the stable-ice denominator mask
    domain_mask   : full Arctic-ocean domain boolean array
    area_weights  : cos-lat weights
    extent_mask   : boolean array (same shape) — ALL pixels with any ice
                    (SIC ≥ threshold in either early or late period).
                    If None, per-extent values are returned as NaN.

    Returns
    -------
    drk_per_ice     : numerator / stable-ice area  (per unit darkening ice)
    stable_ice_frac : stable-ice area / domain area
    drk_per_extent  : numerator / total-ice-extent area  (per unit extent)
    total_ice_frac  : total-ice-extent area / domain area
    """
    ice_mask = domain_mask & (is_drk_map > 0) & np.isfinite(is_drk_map)
    v_ice    = siaf_drk_map[ice_mask]
    w_ice    = area_weights[ice_mask]
    ok       = np.isfinite(v_ice) & np.isfinite(w_ice) & (w_ice > 0)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    numerator = float(np.sum(v_ice[ok] * w_ice[ok]))

    # ── stable-ice area (denominator for drk_per_ice) ────────────────────
    ok_ice     = np.isfinite(w_ice) & (w_ice > 0)   # all stable-ice pixels
    w_dom      = area_weights[domain_mask]
    ok_dom     = np.isfinite(w_dom) & (w_dom > 0)
    stable_area = float(np.sum(w_ice[ok_ice])) if ok_ice.any() else 0.
    domain_area = float(np.sum(w_dom[ok_dom])) if ok_dom.any() else 1.

    drk_per_ice     = numerator / stable_area if stable_area > 0 else np.nan
    stable_ice_frac = stable_area / domain_area if domain_area > 0 else np.nan

    # ── total ice extent (denominator for drk_per_extent) ────────────────
    if extent_mask is not None:
        w_ext       = area_weights[domain_mask & extent_mask]
        ok_ext      = np.isfinite(w_ext) & (w_ext > 0)
        extent_area = float(np.sum(w_ext[ok_ext])) if ok_ext.any() else 0.
        drk_per_extent  = numerator / extent_area  if extent_area  > 0 else np.nan
        total_ice_frac  = extent_area / domain_area if domain_area > 0 else np.nan
    else:
        drk_per_extent = np.nan
        total_ice_frac = np.nan

    return drk_per_ice, stable_ice_frac, drk_per_extent, total_ice_frac

def domain_mean(field, domain_mask, area_weights):
    """Weighted mean over domain.

    NaN values in `field` are treated as zero contribution to the numerator,
    but the denominator is ALWAYS the full domain area (not just the finite
    pixels).  This ensures the mean is not inflated when RS has NaN at some
    domain pixels (e.g. CERES data gaps, kernel masking).
    """
    v      = field[domain_mask]
    w      = area_weights[domain_mask]
    ok_num = np.isfinite(v) & np.isfinite(w) & (w > 0)
    ok_den = np.isfinite(w) & (w > 0)          # full domain denominator
    if ok_num.sum() == 0 or ok_den.sum() == 0:
        return np.nan
    return float(np.sum(v[ok_num] * w[ok_num]) / np.sum(w[ok_den]))


# ══════════════════════════════════════════════════════════════════════════════
# ONE-TIME GRID SETUP
# ══════════════════════════════════════════════════════════════════════════════

if not CMIP6_ONLY:
    print('Setting up grids ...', flush=True)

    sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
    with xr.open_dataset(sample_sial) as ds:
        sla = ds['lat'].values; slo = ds['lon'].values
    if sla.ndim == 1:
        SIAL_LON, SIAL_LAT = np.meshgrid(slo, sla)
    else:
        SIAL_LAT, SIAL_LON = sla, slo
    SIAL_SHAPE = SIAL_LAT.shape

    try:
        ARCTIC_SIAL = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
        if ARCTIC_SIAL.shape != SIAL_SHAPE:
            ARCTIC_SIAL = (SIAL_LAT >= ARCTIC_LAT)
    except FileNotFoundError:
        ARCTIC_SIAL = (SIAL_LAT >= ARCTIC_LAT)

    sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
    with xr.open_dataset(sample_sic) as ds:
        psn_x = ds['x'].values.astype(np.float64)
        psn_y = ds['y'].values.astype(np.float64)
    psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
    PSN_SHAPE = psn_xx.shape

    tfm        = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
    PSN_LON, PSN_LAT = tfm.transform(psn_xx, psn_yy)

    phi_ts = np.deg2rad(70.)
    k_ps   = (1. + np.sin(phi_ts)) / np.maximum(
        1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
    AREA_W_PSN = (1. / k_ps**2).astype(np.float32)

    if _HAS_LAND_MASK:
        lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
        DOMAIN_PSN = (PSN_LAT >= ARCTIC_LAT) & _gm_globe.is_ocean(PSN_LAT, lp)
    else:
        DOMAIN_PSN = (PSN_LAT >= ARCTIC_LAT)

    print(f'  PSN25 domain: {DOMAIN_PSN.sum():,} ocean pixels', flush=True)

    IDX_SIAL_TO_PSN = build_scatter_index(
        SIAL_LAT.ravel(), SIAL_LON.ravel(),
        PSN_LAT.ravel(),  PSN_LON.ravel())

    with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
        c_la = _ds['lat'].values; c_lo = _ds['lon'].values
    CERES_LON2, CERES_LAT2 = np.meshgrid(c_lo, c_la)
    IDX_CERES_TO_PSN = build_scatter_index(
        CERES_LAT2.ravel(), CERES_LON2.ravel(),
        PSN_LAT.ravel(),    PSN_LON.ravel())

    # APP-x → SIAL index (optional)
    IDX_APPX_TO_SIAL = None
    try:
        sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
        with NC4Dataset(sample_appx, 'r') as ds:
            al = _mask_fills(ds.variables['latitude'][:])
            ao = _mask_fills(ds.variables['longitude'][:])
        if al.ndim == 1:
            ao2, al2 = np.meshgrid(ao, al)
        else:
            al2, ao2 = al, ao
        _, IDX_APPX_TO_SIAL = cKDTree(latlon_to_xyz(
            al2.ravel(), ao2.ravel())).query(
            latlon_to_xyz(SIAL_LAT.ravel(), SIAL_LON.ravel()), workers=-1)
        IDX_APPX_TO_SIAL = IDX_APPX_TO_SIAL.astype(np.int32)
        print('  APP-x → SIAL index built.', flush=True)
    except Exception as e:
        print(f'  WARNING: APP-x → SIAL index failed: {e}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CERES + KERNEL FUNCTIONS  (defined early; used by pre-load block below)
# ══════════════════════════════════════════════════════════════════════════════

_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt>1.,  rsdt,  np.nan)
        Sd  = np.where(rsds>1.,  rsds,  np.nan)
        a   = np.clip(rsus/Sd, 0., 1.)
        p   = rsds/S;  q  = rsut/S
        den = q + p - 2.*p*a
        R   = np.clip((q-p*a)/np.where(np.abs(den)>1e-6,den,np.nan),0.,.99)
        A   = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan),0.,.99)
        Sc  = np.where(rsdscs>1., rsdscs, np.nan)
        ac  = np.clip(rsuscs/Sc, 0., 1.)
        pc  = rsdscs/S;  qc = rsutcs/S
        dc  = qc+pc-2.*pc*ac
        Rc  = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan),0.,.99)
        Rl  = np.clip(R-Rc,0.,.99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               /np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS.astype(np.float32)

def load_kernel_psn(mn):
    with xr.open_dataset(KERNEL_FILE) as ds:
        tdim  = next((d for d, s in ds.sizes.items() if s == 12), None)
        lat_n = 'latitude' if 'latitude' in ds else 'lat'
        lon_n = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[KERNEL_VAR].isel({tdim: mn-1}).values.astype(np.float32)
    k_raw = np.abs(k_raw) * KERNEL_SCALE
    idx_k = build_scatter_index(k_lat2d.ravel(), k_lon2d.ravel(),
                                PSN_LAT.ravel(), PSN_LON.ravel())
    return scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-LOAD ALL CERES RS MAPS  (single file open; no re-reads in loops)
# ══════════════════════════════════════════════════════════════════════════════

if not CMIP6_ONLY:
    print('\nPre-loading CERES RS cache ...', flush=True)
    _CERES_RS_CACHE = {}   # (year, month_num) → PSN25 RS map

    _all_ceres_years = sorted(set(CERES_CMIP6) | set(CERES_OBS))
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as _cds:
        _ctimes = pd.DatetimeIndex(_cds.time.values)
        for _mn in MONTH_NUMS:
            _idxs = [i for i, t in enumerate(_ctimes)
                     if t.year in set(_all_ceres_years) and t.month == _mn]
            if not _idxs:
                continue
            _block = _cds[_CERES_VARS].isel(time=_idxs).load()
            for _li, _gi in enumerate(_idxs):
                _yr = _ctimes[_gi].year
                _sl = _block.isel(time=_li)
                _s  = {k: _sl[k].values.astype(np.float32) for k in _CERES_VARS}
                _rs = t07_invert(_s['solar_mon'], _s['toa_sw_all_mon'],
                                 _s['sfc_sw_down_all_mon'], _s['sfc_sw_up_all_mon'],
                                 _s['toa_sw_clr_c_mon'], _s['sfc_sw_down_clr_c_mon'],
                                 _s['sfc_sw_up_clr_c_mon'])
                _CERES_RS_CACHE[(_yr, _mn)] = scatter_avg(
                    _rs.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
            print(f'  month {_mn}: {len(_idxs)} maps cached', flush=True)

    print(f'  CERES cache ready: {len(_CERES_RS_CACHE)} maps', flush=True)

    # Pre-load kernels (all months, once)
    print('Pre-loading ERA5 kernels ...', flush=True)
    _KERNEL_CACHE = {}
    for _mn in MONTH_NUMS:
        try:
            _KERNEL_CACHE[_mn] = load_kernel_psn(_mn)
        except Exception as e:
            print(f'  WARNING kernel month {_mn}: {e}', flush=True)
            _KERNEL_CACHE[_mn] = None
    print(f'  Kernels ready: {sum(v is not None for v in _KERNEL_CACHE.values())}/6 months',
          flush=True)

# Pre-scan CMIP6 member directories (all Lustre metadata I/O up front)
print('\nScanning CMIP6 member directories ...', flush=True)

def _find_var_files(inst, model, member, variable):
    tables = VAR_TABLE_MAP.get(variable, ['Amon'])
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in tables:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, variable)
            if os.path.isdir(d):
                files = sorted(glob.glob(
                    os.path.join(d, '**', f'{variable}_*.nc'), recursive=True))
                if files:
                    return files
    return []

def _find_all_members(inst, model):
    members = set()
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        base = os.path.join(root, inst, model, 'historical')
        if not os.path.isdir(base): continue
        try:
            for mem in os.listdir(base):
                if (_find_var_files(inst, model, mem, 'rsus') and
                        _find_var_files(inst, model, mem, 'rsds') and
                        _find_var_files(inst, model, mem, 'siconc')):
                    members.add(mem)
        except Exception:
            pass
    return sorted(members)

_MODEL_MEMBERS = {}
for _m, _info in CMIP6_MODELS.items():
    _mems = _find_all_members(_info['institution'], _m)
    _MODEL_MEMBERS[_m] = _mems
    print(f'  {_m}: {len(_mems)} member(s)', flush=True)
print('Directory scan complete.\n', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# OBS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def regrid_sial_to_psn(arr):
    if arr.shape != SIAL_SHAPE:
        tmp = np.full(SIAL_SHAPE, np.nan, np.float32)
        r = min(arr.shape[0], SIAL_SHAPE[0])
        c = min(arr.shape[1], SIAL_SHAPE[1])
        tmp[:r, :c] = arr[:r, :c]
        arr = tmp
    src = np.where(ARCTIC_SIAL & np.isfinite(arr), arr, np.float32(0.))
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)

def load_obs_is_drk(cache_dir, mname):
    """
    Load IS_drk and IS_ext from cached dα/dT + SIC files.

    Returns
    -------
    IS_drk_nan   : IS_drk with NaN outside stable ice
    IS_drk_zero  : IS_drk with 0  at non-stable domain pixels (NaN off-domain)
    extent_mask  : SIAL-grid boolean — any pixel with SIC ≥ threshold in
                   either early or late period (used for per-extent metric)
    IS_ext_zero  : IS_ext (changing-ice sensitivity) with 0 at non-changing
                   domain pixels (NaN off-domain) — for siaf_ext_global
    All four are None on failure.
    """
    p_dadt  = f'{cache_dir}/merged_alb_dalphadT_{mname}.npy'
    p_slope = f'{cache_dir}/sic_trend_slope_sial_{mname}.npy'
    p_pval  = f'{cache_dir}/sic_trend_pval_sial_{mname}.npy'
    p_sic_e = f'{cache_dir}/sic_early_sial_{mname}.npy'
    p_sic_l = f'{cache_dir}/sic_late_sial_{mname}.npy'
    for p in [p_dadt, p_slope, p_pval, p_sic_e, p_sic_l]:
        if not os.path.exists(p):
            return None, None, None, None

    dalpha_dT = np.load(p_dadt).astype(np.float32)
    sic_slope = np.load(p_slope).astype(np.float32)
    sic_pval  = np.load(p_pval).astype(np.float32)
    sic_e     = np.load(p_sic_e).astype(np.float32)
    sic_l     = np.load(p_sic_l).astype(np.float32)

    IS_raw  = (-dalpha_dT).astype(np.float32)
    valid   = ARCTIC_SIAL & np.isfinite(IS_raw)
    any_ice = ARCTIC_SIAL & (np.fmax(sic_e, sic_l) >= SIC_THRESH)
    chang   = (valid & np.isfinite(sic_slope) & np.isfinite(sic_pval)
               & (sic_slope < 0) & (sic_pval < PVAL))
    stable  = (valid & ~chang & (np.fmax(sic_e, sic_l) >= SIC_THRESH))

    IS_drk_nan  = np.where(stable, IS_raw, np.nan).astype(np.float32)
    # Zero-fill non-stable domain pixels so domain_mean uses the full
    # valid-data domain in its denominator (matching the reference script).
    IS_drk_zero = np.where(valid, np.where(stable, IS_raw, np.float32(0.)),
                            np.nan).astype(np.float32)
    # IS_ext: sensitivity at changing (retreating) ice pixels; 0 elsewhere in
    # domain so domain_mean correctly divides by the full domain area.
    IS_ext_zero = np.where(valid, np.where(chang, IS_raw, np.float32(0.)),
                            np.nan).astype(np.float32)
    return IS_drk_nan, IS_drk_zero, any_ice, IS_ext_zero

def load_ceres_rs_psn(years, mn):
    """Return mean PSN25 RS map for the given years+month from cache."""
    maps = [_CERES_RS_CACHE[(y, mn)] for y in years if (y, mn) in _CERES_RS_CACHE]
    if not maps:
        return None
    return np.nanmean(np.stack(maps, 0), 0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# OBS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

# In CMIP6_ONLY mode these are empty; the final combine still works.
obs_rows      = []
obs_seas_rows = []

if not CMIP6_ONLY:
    print('=' * 60, flush=True)
    print('OBS: computing SIAF_drk per ice area from caches', flush=True)
    print('=' * 60, flush=True)

    period_cfg = [
        ('CMIP6',   CACHE_CMIP6, CERES_CMIP6),
        ('Obs-era', CACHE_OBS,   CERES_OBS),
    ]

    for period, cache_dir, ceres_yrs in period_cfg:
        print(f'\n  Period: {period}', flush=True)
        for mn in MONTH_NUMS:
            mname = MONTH_NAMES[mn]
            print(f'    {mname} ...', end=' ', flush=True)

            IS_drk_nan, IS_drk_zero, ext_sial, IS_ext_zero = load_obs_is_drk(
                cache_dir, mname)
            if IS_drk_nan is None:
                print('no cache — skip', flush=True)
                continue

            IS_drk_psn_nan  = regrid_sial_to_psn(IS_drk_nan)
            IS_drk_psn_zero = regrid_sial_to_psn(IS_drk_zero)
            IS_ext_psn_zero = regrid_sial_to_psn(IS_ext_zero)
            # Regrid extent mask: scatter_avg > 0 means at least one source
            # pixel had ice → convert back to boolean on PSN25 grid
            ext_psn = (scatter_avg(ext_sial.astype(np.float32).ravel(),
                                   IDX_SIAL_TO_PSN, PSN_SHAPE) > 0)

            for rs_name, rs_psn in [
                ('CERES',  load_ceres_rs_psn(ceres_yrs, mn)),
                ('Kernel', _KERNEL_CACHE.get(mn)),
            ]:
                if rs_psn is None:
                    continue
                siaf_drk_map = IS_drk_psn_zero * rs_psn
                siaf_ext_map = IS_ext_psn_zero * rs_psn
                drk_standard = domain_mean(siaf_drk_map, DOMAIN_PSN, AREA_W_PSN)
                ext_standard = domain_mean(siaf_ext_map, DOMAIN_PSN, AREA_W_PSN)
                drk_per_ice, stable_frac, drk_per_ext, ext_frac = stable_ice_mean(
                    siaf_drk_map, IS_drk_psn_nan, DOMAIN_PSN, AREA_W_PSN,
                    extent_mask=ext_psn)
                obs_rows.append(dict(
                    source              = 'OBS',
                    model               = 'OBS',
                    member              = 'obs',
                    rs_method           = rs_name,
                    period              = period,
                    month               = mname,
                    month_num           = mn,
                    siaf_drk_global     = round(float(drk_standard) * SCALE, 7)
                                          if np.isfinite(drk_standard) else np.nan,
                    siaf_ext_global     = round(float(ext_standard) * SCALE, 7)
                                          if np.isfinite(ext_standard) else np.nan,
                    siaf_drk_per_ice    = round(float(drk_per_ice)   * SCALE, 7)
                                          if np.isfinite(drk_per_ice) else np.nan,
                    siaf_drk_per_extent = round(float(drk_per_ext)   * SCALE, 7)
                                          if np.isfinite(drk_per_ext) else np.nan,
                    stable_ice_frac     = round(float(stable_frac), 5)
                                          if np.isfinite(stable_frac) else np.nan,
                    total_ice_frac      = round(float(ext_frac), 5)
                                          if np.isfinite(ext_frac) else np.nan,
                ))
            print('done', flush=True)

    obs_df = pd.DataFrame(obs_rows)

    # Seasonal (AMJJAS) means per (rs_method, period)
    for (rs, per), grp in obs_df.groupby(['rs_method', 'period']):
        obs_seas_rows.append(dict(
            source              = 'OBS',
            model               = 'OBS',
            member              = 'obs',
            rs_method           = rs,
            period              = per,
            month               = 'AMJJAS',
            month_num           = -1,
            siaf_drk_global     = float(grp['siaf_drk_global'].mean()),
            siaf_ext_global     = float(grp['siaf_ext_global'].mean()),
            siaf_drk_per_ice    = float(grp['siaf_drk_per_ice'].mean()),
            siaf_drk_per_extent = float(grp['siaf_drk_per_extent'].mean()),
            stable_ice_frac     = float(grp['stable_ice_frac'].mean()),
            total_ice_frac      = float(grp['total_ice_frac'].mean()),
        ))

    print('\nOBS seasonal results (AMJJAS mean):', flush=True)
    for r in obs_seas_rows:
        print(f"  {r['period']:<10} {r['rs_method']:<8}  "
              f"drk_global={r['siaf_drk_global']:.5f}  "
              f"ext_global={r['siaf_ext_global']:.5f}  "
              f"drk_per_ice={r['siaf_drk_per_ice']:.5f}  "
              f"drk_per_ext={r['siaf_drk_per_extent']:.5f}  "
              f"ice_frac={r['stable_ice_frac']:.3f}  "
              f"ext_frac={r['total_ice_frac']:.3f}", flush=True)

    # ── Free all OBS-only arrays before starting CMIP6 processing ────────────
    del _CERES_RS_CACHE, _KERNEL_CACHE
    del AREA_W_PSN, DOMAIN_PSN, IDX_SIAL_TO_PSN, IDX_CERES_TO_PSN
    del PSN_LAT, PSN_LON, psn_xx, psn_yy
    del CERES_LAT2, CERES_LON2
    del SIAL_LAT, SIAL_LON, ARCTIC_SIAL
    if IDX_APPX_TO_SIAL is not None:
        del IDX_APPX_TO_SIAL
    gc.collect()
    _malloc_trim()
    print(f'  (OBS-phase arrays freed + heap trimmed)  [RSS: {_rss_mb():.0f} MB]\n',
          flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CMIP6 UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_lat(arr):
    if arr is None: return False
    flat = arr.ravel(); flat = flat[np.isfinite(flat)]
    return len(flat) > 0 and flat.min() >= -90.1 and flat.max() <= 90.1

def _std_latlon(da):
    spatial_shape = da.shape[1:] if da.ndim >= 3 else da.shape
    ny = spatial_shape[0] if len(spatial_shape) >= 1 else None
    nx = spatial_shape[1] if len(spatial_shape) >= 2 else None

    def _best(primary, aliases):
        for name in [primary] + aliases:
            if name in da.coords:
                arr = _mask_fills(np.asarray(da.coords[name].values, dtype=float))
                if arr.ndim == 2 and ny and nx and arr.shape == (ny, nx):
                    return arr
                if arr.ndim == 1:
                    return arr
        return None

    lat = _best('lat', ['latitude', 'nav_lat', 'rlat'])
    lon = _best('lon', ['longitude', 'nav_lon', 'rlon'])

    if not _is_valid_lat(lat):
        for name in da.coords:
            try:
                arr = _mask_fills(np.asarray(da.coords[name].values, dtype=float))
                if ny and nx and arr.shape == (ny, nx) and _is_valid_lat(arr):
                    lat = arr; break
            except Exception:
                continue

    is_2d = (lat.ndim == 2) if lat is not None else False
    if is_2d and lon is not None and lon.ndim == 1:
        ny2, nx2 = lat.shape
        lon = (np.tile(lon[np.newaxis, :], (ny2, 1))
               if len(lon) == nx2 else
               np.tile(lon[:, np.newaxis], (1, nx2)))
    return lat, lon, is_2d

def _load_filtered(files, variable, years, months):
    """
    Open a multi-file CMIP6 dataset and return the requested variable filtered
    to (years, months).  Tries three xarray strategies in order:

      1. combine='nested' + coords='minimal' + chunks={'time':12}
         Reads 2D curvilinear coordinates only once (from the first file).
         combine='by_coords' reads lat/lon from EVERY file — for a 360×256
         curvilinear grid that is ~280 MB of coordinate reads before any data.
         chunks={'time':12} is critical for models (e.g. MIROC-ES2L) that
         store siconc in a single file covering 1850-2014 (1980 months).
         Without chunks, xarray reads the full ~730 MB file before filtering;
         with chunks it only reads the 33 year-chunks that match TREND_YEARS.

      2. combine='by_coords' + chunks  — fallback for models whose files lack
         the time coordinate needed for automatic ordering (IPSL, CanESM5,
         BCC, CNRM, ACCESS-ESM1-5).

      3. combine='nested' + decode_times=False + xr.decode_cf() — last resort
         for files with non-standard calendar strings that xarray cannot
         decode automatically (e.g. 'days since 0001-01-01' or unusual
         leap-year conventions).

    All strategies deduplicate repeated time steps (present at some CMIP6
    file boundaries) before filtering.
    """
    ys = set(years); ms = set(months)

    _nested_kw = dict(combine='nested', concat_dim='time',
                      data_vars='minimal', coords='minimal', compat='override')

    strategies = [
        (_nested_kw | dict(use_cftime=True,  chunks={'time': 12}), True),
        (dict(combine='by_coords', use_cftime=True, chunks={'time': 12}),  True),
        (_nested_kw | dict(decode_times=False, chunks={'time': 12}),        False),
    ]

    last_e = None
    for i_s, (kw, auto_decoded) in enumerate(strategies):
        try:
            ds = xr.open_mfdataset(sorted(files), **kw)
            if not auto_decoded:
                ds = xr.decode_cf(ds, use_cftime=True)
            da    = ds[variable]
            da    = da.sortby('time')
            tvals = da.time.values
            # Deduplicate repeated timestamps (some CMIP6 files have
            # overlapping records at file boundaries).
            _, first_occ = np.unique(tvals, return_index=True)
            if len(first_occ) < len(tvals):
                da    = da.isel(time=np.sort(first_occ))
                tvals = da.time.values
            # Integer indexing lets dask read only the matching chunks;
            # boolean masking forces it to inspect every chunk first.
            idx = [i for i, t in enumerate(tvals)
                   if t.year in ys and t.month in ms]
            if not idx:
                ds.close()
                continue
            da = da.isel(time=idx).load()
            ds.close()
            if i_s > 0:
                print(f'        [{variable}] used fallback strategy {i_s+1}',
                      flush=True)
            return da
        except Exception as e:
            last_e = e
            try: ds.close()
            except Exception: pass
            continue
    print(f'      load_filtered({variable}): {last_e}', flush=True)
    return None

def _pixel_ols_trend(years_1d, sic_3d):
    T  = years_1d.shape[0]
    x  = years_1d - years_1d.mean()
    ss = float(np.dot(x, x))
    ym = np.nanmean(sic_3d, axis=0)
    dy = sic_3d - ym[None]
    sl = np.nansum(x[:, None, None] * dy, axis=0) / ss
    resid = sic_3d - (ym + sl * x[:, None, None])
    sse   = np.nansum(resid**2, axis=0)
    df    = T - 2
    with np.errstate(invalid='ignore', divide='ignore'):
        se  = np.sqrt(sse / df) / np.sqrt(ss)
        tst = sl / se
        pv  = 2. * (1. - t_dist.cdf(np.abs(tst), df))
    return sl, pv

def _compute_pixel_ols(stk, x_vec):
    T, H, W = stk.shape
    flat    = stk.reshape(T, -1).astype(np.float64)
    N       = flat.shape[1]
    x       = x_vec.astype(np.float64).reshape(-1, 1)
    valid   = np.isfinite(flat)
    n_valid = valid.sum(axis=0).astype(np.float64)
    flat_z  = np.where(valid, flat, 0.0)
    x_bc    = np.broadcast_to(x, (T, N))
    x_z     = np.where(valid, x_bc, 0.0)
    y_mean  = flat_z.sum(0) / np.maximum(n_valid, 1)
    x_mean  = x_z.sum(0)   / np.maximum(n_valid, 1)
    flat_c  = np.where(valid, flat - y_mean, 0.0)
    x_c     = np.where(valid, x_bc - x_mean, 0.0)
    num     = (x_c * flat_c).sum(0)
    den     = (x_c ** 2).sum(0)
    out     = np.full(N, np.nan, np.float32)
    good    = (n_valid >= 10) & (den > 0)
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out.reshape(H, W)

def _build_native_domain(lat, lon, is_2d):
    if is_2d:
        lat2d = _mask_fills(lat).astype(np.float32)
        lon2d = _mask_fills(lon).astype(np.float32)
    else:
        lo2d, la2d = np.meshgrid(lon, lat)
        lat2d = la2d.astype(np.float32)
        lon2d = lo2d.astype(np.float32)
    lat2d  = np.where(np.isfinite(lat2d), lat2d, 0.)
    lon_pm = lon2d - 360. * np.floor((lon2d + 180.) / 360.)
    area_w = np.cos(np.deg2rad(lat2d)).astype(np.float32)
    if _HAS_LAND_MASK:
        ocean = _gm_globe.is_ocean(lat2d, lon_pm.astype(np.float32))
    else:
        ocean = np.ones(lat2d.shape, dtype=bool)
    domain = ocean & (lat2d >= ARCTIC_LAT)
    return lat2d, lon2d, area_w, domain

def _compute_T_annual(model, member, trend_years):
    inst  = CMIP6_MODELS[model]['institution']
    files = _find_var_files(inst, model, member, 'tas')
    if not files: return None
    da    = _load_filtered(files, 'tas', trend_years, list(range(1, 13)))
    if da is None: return None
    lat, lon, is_2d = _std_latlon(da)
    if lat is None: return None
    yrs  = np.array([int(t.dt.year) for t in da.time])
    vals = da.values.astype(float)
    if is_2d:
        lat2d = np.where(np.isfinite(_mask_fills(lat)), _mask_fills(lat), 0.)
    else:
        lo, la = np.meshgrid(lon, lat)
        lat2d  = la
    w = np.cos(np.deg2rad(lat2d))
    T_dict = {}
    for year in trend_years:
        idx = np.where(yrs == year)[0]
        if not idx.size: continue
        ms = []
        for i in idx:
            ok = np.isfinite(vals[i]) & (w > 0)
            if ok.any():
                ms.append(float(np.sum(vals[i][ok]*w[ok]) / np.sum(w[ok])))
        if ms: T_dict[year] = float(np.mean(ms))
    del da; gc.collect()
    return T_dict if T_dict else None


# ══════════════════════════════════════════════════════════════════════════════
# CMIP6 COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

print('=' * 60, flush=True)
print('CMIP6: computing SIAF_drk per ice area on native grids', flush=True)
print('=' * 60, flush=True)

cmip6_rows = []

for model, info in CMIP6_MODELS.items():
    inst    = info['institution']
    prim    = info['member']
    members = _MODEL_MEMBERS[model]
    if not members:
        print(f'  {model}: no members found — skip', flush=True)
        continue

    if model in SKIP_MODELS:
        print(f'  {model}: in SKIP_MODELS — skip', flush=True)
        continue

    # ── RESUME LOGIC ─────────────────────────────────────────────────────────
    # If a per-model partial CSV exists, skip reprocessing and reload its rows.
    _partial_path = OUT_CSV.replace('.csv', f'_partial_{model}.csv')
    if os.path.exists(_partial_path):
        try:
            _existing = pd.read_csv(_partial_path)
            # Filter to this model's rows only — guards against bloated partial
            # CSVs written by old code that accumulated all prior models' rows.
            _existing = _existing[_existing['model'] == model]
            if len(_existing) > 0:
                print(f'  {model}: partial CSV found ({len(_existing)} rows) — skip',
                      flush=True)
                cmip6_rows.extend(_existing.to_dict('records'))
                continue
        except Exception as _e:
            print(f'  {model}: partial CSV unreadable ({_e}) — recomputing', flush=True)
    # ── END RESUME LOGIC ──────────────────────────────────────────────────────

    if prim in members:
        members = [prim] + [m for m in members if m != prim]

    print(f'\n  {model} ({len(members)} members)  [RSS: {_rss_mb():.0f} MB]',
          flush=True)

    T_prim = _compute_T_annual(model, prim, TREND_YEARS)

    # Collect rows for this model only — saved to the model's partial CSV at
    # the end.  Keeping them separate from cmip6_rows prevents duplication on
    # re-submission (each partial CSV contains only its own model's rows).
    model_rows = []

    for member in members:
        T_ann = (_compute_T_annual(model, member, TREND_YEARS)
                 if member != prim else T_prim)
        if T_ann is None:
            T_ann = T_prim
        if T_ann is None or len(T_ann) < 10:
            print(f'    [{member}] insufficient T data — skip', flush=True)
            continue

        T_center  = float(np.mean(list(T_ann.values())))
        T_vec_all = {y: v - T_center for y, v in T_ann.items()}

        rsus_f = _find_var_files(inst, model, member, 'rsus')
        rsds_f = _find_var_files(inst, model, member, 'rsds')
        sic_f  = _find_var_files(inst, model, member, 'siconc')
        if not all([rsus_f, rsds_f, sic_f]):
            print(f'    [{member}] missing files — skip', flush=True)
            continue

        # ── Load SIC first: lightweight, allows early exit before radiation I/O
        print(f'    [{member}] loading siconc  [RSS: {_rss_mb():.0f} MB]',
              flush=True)
        si = _load_filtered(sic_f, 'siconc', TREND_YEARS, MONTH_NUMS)
        if si is None:
            print(f'    [{member}] siconc load failed — skip', flush=True)
            continue
        print(f'    [{member}] siconc loaded: {si.shape}  [RSS: {_rss_mb():.0f} MB]',
              flush=True)

        # ── Load ALL radiation vars ONCE per member ───────────────────────────
        # KEY FIX: rsus and rsds are loaded here for TREND_YEARS (1982-2014)
        # so the albedo regression has the full time span.  Other vars use
        # 1985-2014 (sufficient for a representative RS mean).  Previously,
        # rsus/rsds were loaded AGAIN separately as `ru`/`rd`, doubling their
        # memory footprint.  Now they live only here; `ru`/`rd` are popped out
        # as references (no copy) after rs_by_month is built.
        rf       = {v: _find_var_files(inst, model, member, v) for v in RAD_VARS}
        rf['rsus'] = rsus_f   # reuse already-found file lists
        rf['rsds'] = rsds_f
        RS_YEARS = list(range(1985, 2015))

        _rad_das   = {}
        all_rad_ok = True
        for v in RAD_VARS:
            if rf[v]:
                yrs_v = TREND_YEARS if v in ('rsus', 'rsds') else RS_YEARS
                _da   = _load_filtered(rf[v], v, yrs_v, MONTH_NUMS)
                if _da is not None:
                    _rad_das[v] = _da
                else:
                    all_rad_ok = False
            else:
                all_rad_ok = False

        print(f'    [{member}] rad vars loaded: {list(_rad_das.keys())}  '
              f'[RSS: {_rss_mb():.0f} MB]', flush=True)
        if 'rsus' not in _rad_das or 'rsds' not in _rad_das:
            print(f'    [{member}] rsus/rsds load failed — skip', flush=True)
            del si, _rad_das; gc.collect()
            continue

        # Extract lat/lon and domain from rsus (used in place of old `ru`)
        lat, lon, is_2d = _std_latlon(_rad_das['rsus'])
        if lat is None:
            print(f'    [{member}] lat/lon extraction failed — skip', flush=True)
            del si, _rad_das; gc.collect()
            continue
        lat2d, lon2d, area_w, domain = _build_native_domain(lat, lon, is_2d)
        atm_shape = lat2d.shape

        # ── Extract SIC numpy arrays; free xarray object immediately ─────────
        sic_lat, sic_lon, sic_is2d = _std_latlon(si)
        sv     = si.values.astype(np.float32)  # float32 halves SIC array size
        if np.nanmax(sv) > 1.5: sv /= 100.
        si_yrs = np.array([int(t.dt.year)  for t in si.time])
        si_mon = np.array([int(t.dt.month) for t in si.time])
        del si; gc.collect()

        if sv.shape[1:] != atm_shape:
            if sic_is2d:
                sla = _mask_fills(sic_lat).ravel()
                slo = _mask_fills(sic_lon).ravel()
            else:
                slo2, sla2 = np.meshgrid(sic_lon, sic_lat)
                sla = sla2.ravel(); slo = slo2.ravel()
            ok_s = np.isfinite(sla) & np.isfinite(slo)
            _, idx_s = cKDTree(latlon_to_xyz(
                sla[ok_s], slo[ok_s])).query(
                latlon_to_xyz(lat2d.ravel(), lon2d.ravel()))
            sv_orig   = sv.reshape(sv.shape[0], -1)
            final_idx = np.where(ok_s)[0][idx_s]  # one combined index, no sv_ok copy
            del sv
            sv = sv_orig[:, final_idx].reshape((sv_orig.shape[0],) + atm_shape)
            del sv_orig, final_idx
            gc.collect()

        # ── Build rs_by_month: 6 mean 2D numpy arrays ────────────────────────
        # Three-tier RS method matching the reference script:
        #   full_T07   — all 7 RAD_VARS available; full T07 inversion
        #   approx_T07 — rsuscs missing; approximated as rsdscs*(rsus/rsds)
        #   rsds_proxy — only downward SW available; used directly
        # Use a list comprehension for month masking — the vectorised
        # .dt.month.values accessor fails on cftime arrays in some xarray
        # versions, while an explicit loop is safe everywhere.
        _APPROX_VARS    = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs']
        _avail          = set(_rad_das.keys())
        _has_full       = set(RAD_VARS)   <= _avail
        _has_approx     = set(_APPROX_VARS) <= _avail
        rs_by_month        = {}
        rs_method_by_month = {}
        for mn in MONTH_NUMS:
            _mm = {}
            for v, _da in _rad_das.items():
                _mask = np.array([int(t.dt.month) == mn for t in _da.time], dtype=bool)
                if _mask.any():
                    _mm[v] = _da.isel(time=_mask).mean('time').values.astype(float)

            if _has_full and all(v in _mm for v in RAD_VARS):
                rs_by_month[mn] = t07_invert(
                    _mm['rsdt'], _mm['rsut'], _mm['rsds'], _mm['rsus'],
                    _mm['rsutcs'], _mm['rsdscs'], _mm['rsuscs']
                ).astype(np.float32)
                rs_method_by_month[mn] = 'full_T07'
            elif _has_approx and all(v in _mm for v in _APPROX_VARS):
                # Approximate rsuscs from clear-sky downwelling scaled by the
                # all-sky upwelling/downwelling ratio (matches reference script)
                _rsuscs_approx = (_mm['rsdscs']
                                  * np.where(_mm['rsds'] > 1.,
                                             _mm['rsus'] / _mm['rsds'], 0.))
                rs_by_month[mn] = t07_invert(
                    _mm['rsdt'], _mm['rsut'], _mm['rsds'], _mm['rsus'],
                    _mm['rsutcs'], _mm['rsdscs'], _rsuscs_approx
                ).astype(np.float32)
                rs_method_by_month[mn] = 'approx_T07'
            elif 'rsds' in _mm:
                rs_by_month[mn] = _mm['rsds'].astype(np.float32)
                rs_method_by_month[mn] = 'rsds_proxy'
            del _mm

        # Determine single method label for this member (matches reference
        # script: combined label if methods differ across months, e.g.
        # 'approx_T07+full_T07').
        _month_methods = list(rs_method_by_month.values())
        if not _month_methods:
            rs_method = 'none'
        elif len(set(_month_methods)) == 1:
            rs_method = _month_methods[0]
        else:
            rs_method = '+'.join(sorted(set(_month_methods)))

        # Pop rsus/rsds for the albedo loop; free the 5 remaining DataArrays.
        ru = _rad_das.pop('rsus')
        rd = _rad_das.pop('rsds')
        del _rad_das; gc.collect()

        ru_yrs = np.array([int(t.dt.year)  for t in ru.time])
        ru_mon = np.array([int(t.dt.month) for t in ru.time])

        # ── Main month loop ───────────────────────────────────────────────────
        for mn in MONTH_NUMS:
            mname = MONTH_NAMES[mn]

            yearly_alb = {}
            for year in TREND_YEARS:
                if year not in T_ann: continue
                ma = (ru_yrs == year) & (ru_mon == mn)
                if not ma.any(): continue
                u = ru.isel(time=ma).mean('time').values
                d = rd.isel(time=ma).mean('time').values
                with np.errstate(invalid='ignore', divide='ignore'):
                    alb = np.where(d > 1., u / d, np.nan)
                yearly_alb[year] = np.clip(alb, 0., 1.)

            valid_years = sorted(yearly_alb.keys())
            if len(valid_years) < 10:
                del yearly_alb
                continue

            stk       = np.stack([yearly_alb[y] for y in valid_years], axis=0)
            del yearly_alb
            T_vec     = np.array([T_vec_all[y] for y in valid_years], dtype=np.float64)
            dalpha_dT = _compute_pixel_ols(stk, T_vec)
            del stk, T_vec
            gc.collect()

            tm = (si_mon == mn) & np.isin(si_yrs, TREND_YEARS)
            em = (si_mon == mn) & np.isin(si_yrs, EARLY_YEARS)
            lm = (si_mon == mn) & np.isin(si_yrs, LATE_YEARS)
            if tm.sum() < 10 or em.sum() == 0 or lm.sum() == 0:
                del dalpha_dT
                gc.collect()
                continue
            sic_e = np.nanmean(sv[em], axis=0)
            sic_l = np.nanmean(sv[lm], axis=0)
            sl_sic, pv_sic = _pixel_ols_trend(si_yrs[tm].astype(float), sv[tm])

            all_ice     = (lat2d >= ARCTIC_LAT) & (sic_e >= SIC_THRESH)
            # extent = any pixel with ice in early OR late period
            extent_mask = (lat2d >= ARCTIC_LAT) & (np.fmax(sic_e, sic_l) >= SIC_THRESH)
            changing    = (all_ice & np.isfinite(sl_sic) & np.isfinite(pv_sic)
                           & (sl_sic < 0) & (pv_sic < PVAL))
            stable      = (all_ice & ~changing
                           & (np.fmax(sic_e, sic_l) >= SIC_THRESH))
            del sic_e, sic_l, sl_sic, pv_sic

            IS_raw     = (-dalpha_dT).astype(np.float32)
            del dalpha_dT
            IS_drk_nan = np.where(stable, IS_raw, np.nan).astype(np.float32)
            # Zero-fill non-stable / non-changing domain pixels so domain_mean
            # divides by the full Arctic-ocean domain area, not just ice pixels.
            # (Bug fix: old code left NaN at non-ice domain → denominator was
            # only Σw[ice], inflating drk_global for low-ice models like MRI.)
            IS_drk_dom = np.where(stable,   IS_raw,
                                  np.where(domain, np.float32(0.), np.nan))
            IS_ext_dom = np.where(changing, IS_raw,
                                  np.where(domain, np.float32(0.), np.nan))
            del IS_raw, all_ice, changing, stable

            rs_arr = rs_by_month.get(mn)
            if rs_arr is None or rs_arr.shape != atm_shape:
                del IS_drk_nan, IS_drk_dom, IS_ext_dom, extent_mask
                gc.collect()
                continue

            siaf_drk_map = IS_drk_dom * rs_arr
            siaf_ext_map = IS_ext_dom * rs_arr
            drk_standard = domain_mean(siaf_drk_map, domain, area_w)
            ext_standard = domain_mean(siaf_ext_map, domain, area_w)
            drk_per_ice, stable_frac, drk_per_ext, ext_frac = stable_ice_mean(
                siaf_drk_map, IS_drk_nan, domain, area_w,
                extent_mask=extent_mask)
            del IS_drk_nan, IS_drk_dom, IS_ext_dom, siaf_drk_map, siaf_ext_map
            del extent_mask
            gc.collect()

            model_rows.append(dict(
                source              = 'CMIP6',
                model               = model,
                member              = member,
                rs_method           = rs_method_by_month.get(mn, rs_method),
                period              = 'CMIP6',
                month               = mname,
                month_num           = mn,
                siaf_drk_global     = round(float(drk_standard) * SCALE, 7)
                                      if np.isfinite(drk_standard) else np.nan,
                siaf_ext_global     = round(float(ext_standard) * SCALE, 7)
                                      if np.isfinite(ext_standard) else np.nan,
                siaf_drk_per_ice    = round(float(drk_per_ice)   * SCALE, 7)
                                      if np.isfinite(drk_per_ice) else np.nan,
                siaf_drk_per_extent = round(float(drk_per_ext)   * SCALE, 7)
                                      if np.isfinite(drk_per_ext) else np.nan,
                stable_ice_frac     = round(float(stable_frac), 5)
                                      if np.isfinite(stable_frac) else np.nan,
                total_ice_frac      = round(float(ext_frac), 5)
                                      if np.isfinite(ext_frac) else np.nan,
            ))

        # si deleted early above; ru/rd freed here along with the numpy arrays
        del ru, rd, sv, rs_by_month, rs_method_by_month
        gc.collect()
        _malloc_trim()
        print(f'    [{member}] done  [RSS: {_rss_mb():.0f} MB]', flush=True)

    # Accumulate globally and save ONLY this model's rows to its partial CSV.
    cmip6_rows.extend(model_rows)
    if model_rows:
        pd.DataFrame(model_rows).to_csv(_partial_path, index=False, float_format='%.7f')
        print(f'  → partial save: {_partial_path}', flush=True)


cmip6_df = pd.DataFrame(cmip6_rows)

# Seasonal means per (model, member, rs_method)
cmip6_seas_rows = []
for (model, member, rs_m), grp in cmip6_df.groupby(['model', 'member', 'rs_method']):
    cmip6_seas_rows.append(dict(
        source              = 'CMIP6',
        model               = model,
        member              = member,
        rs_method           = rs_m,
        period              = 'CMIP6',
        month               = 'AMJJAS',
        month_num           = -1,
        siaf_drk_global     = float(grp['siaf_drk_global'].mean()),
        siaf_ext_global     = float(grp['siaf_ext_global'].mean()),
        siaf_drk_per_ice    = float(grp['siaf_drk_per_ice'].mean()),
        siaf_drk_per_extent = float(grp['siaf_drk_per_extent'].mean()),
        stable_ice_frac     = float(grp['stable_ice_frac'].mean()),
        total_ice_frac      = float(grp['total_ice_frac'].mean()),
    ))


# ══════════════════════════════════════════════════════════════════════════════
# COMBINE AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

# In CMIP6_ONLY mode, fold the previously-saved OBS rows AND any CMIP6 rows
# for models NOT processed in this job back in so the output CSV is always
# complete.  (This job only runs MIROC-ES2L / ACCESS-ESM1-5; the other 15
# models' rows live in the previous CSV and in per-model partial CSVs.)
prior_rows = []
if CMIP6_ONLY and os.path.exists(OUT_CSV):
    _prev        = pd.read_csv(OUT_CSV)
    _current_mdl = set(CMIP6_MODELS.keys())
    _obs_prev    = _prev[_prev['source'] == 'OBS']
    _c6_prev     = _prev[(_prev['source'] == 'CMIP6') &
                          (~_prev['model'].isin(_current_mdl))]
    prior_rows   = _obs_prev.to_dict('records') + _c6_prev.to_dict('records')
    print(f'  (CMIP6_ONLY: reloaded {len(_obs_prev)} OBS + {len(_c6_prev)} '
          f'prior-CMIP6 rows from {OUT_CSV})', flush=True)

all_rows = prior_rows + obs_rows + obs_seas_rows + cmip6_rows + cmip6_seas_rows
out_df   = pd.DataFrame(all_rows)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out_df.to_csv(OUT_CSV, index=False, float_format='%.7f')
print(f'\nSaved → {OUT_CSV}', flush=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '='*72, flush=True)
print('SEASONAL (AMJJAS) SUMMARY', flush=True)
print('='*72, flush=True)
seas = out_df[out_df['month'] == 'AMJJAS'].copy()

print('\nOBS:', flush=True)
for _, r in seas[seas['source'] == 'OBS'].iterrows():
    print(f"  {r['period']:<10} {r['rs_method']:<8}  "
          f"drk_global={r['siaf_drk_global']:.5f}  "
          f"ext_global={r['siaf_ext_global']:.5f}  "
          f"drk_per_ice={r['siaf_drk_per_ice']:.5f}  "
          f"drk_per_ext={r['siaf_drk_per_extent']:.5f}  "
          f"stbl_frac={r['stable_ice_frac']:.3f}  "
          f"ext_frac={r['total_ice_frac']:.3f}", flush=True)

print('\nCMIP6 (per-model ensemble means, full_T07 + approx_T07):', flush=True)
c6 = seas[(seas['source'] == 'CMIP6') &
          (seas['rs_method'].isin(['full_T07', 'approx_T07']))]
for model, grp in c6.groupby('model'):
    dg  = grp['siaf_drk_global'].mean()
    eg  = grp['siaf_ext_global'].mean()
    dp  = grp['siaf_drk_per_ice'].mean()
    de  = grp['siaf_drk_per_extent'].mean()
    ifr = grp['stable_ice_frac'].mean()
    efr = grp['total_ice_frac'].mean()
    print(f"  {model:<24}  drk_global={dg:.5f}  ext_global={eg:.5f}  "
          f"drk_per_ice={dp:.5f}  drk_per_ext={de:.5f}  "
          f"stbl={ifr:.3f}  ext={efr:.3f}  (n={len(grp)})", flush=True)

mmm_g  = c6.groupby('model')['siaf_drk_global'].mean()
mmm_eg = c6.groupby('model')['siaf_ext_global'].mean()
mmm_p  = c6.groupby('model')['siaf_drk_per_ice'].mean()
mmm_e  = c6.groupby('model')['siaf_drk_per_extent'].mean()
print(f'\nCMIP6 MMM drk_global     = {mmm_g.mean():.5f} ± {mmm_g.std():.5f}',
      flush=True)
print(f'CMIP6 MMM ext_global     = {mmm_eg.mean():.5f} ± {mmm_eg.std():.5f}',
      flush=True)
print(f'CMIP6 MMM drk_per_ice    = {mmm_p.mean():.5f} ± {mmm_p.std():.5f}',
      flush=True)
print(f'CMIP6 MMM drk_per_extent = {mmm_e.mean():.5f} ± {mmm_e.std():.5f}',
      flush=True)
print('='*72, flush=True)