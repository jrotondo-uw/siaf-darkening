#!/usr/bin/env python
"""
cmip6_siaf_pistone_only.py
──────────────────────────
Computes SIAF for CMIP6 historical simulations using the Pistone (2014)
Method B2, fully on each model's NATIVE atmospheric grid.

IS formula (updated 2026-05):
  IS[px] = −(dα/dT)[px]  [alpha K⁻¹]
  Estimated by direct OLS regression of the per-pixel monthly albedo time
  series against the model's own global annual mean temperature for each
  year in TREND_YEARS (1982–2014).  Robust to non-linear warming because
  temperature is the explicit predictor rather than a proxy for time.

2026-05-05 updates
  • GISS-E2-1-G removed — siconc not published on ESGF or NASA NCCS.
  • EC-Earth3 T255 filter moved outside summary loop.
  • _scale_adT: dT_arctic threshold raised from <= 0 to < 0.1 K.

2026-05-04 coordinate robustness update
  • FILL_THRESHOLD = 1e10 masks fill values before coordinate arithmetic.
  • _std_latlon: validates lat range, fallback 2D scan, 1D→2D broadcast.
  • build_native_ctx: NaN lat/lon replaced with 0°.
  • compute_member_IS_native: SIC lat/lon fill-masked before KDTree.

2026-04-29  Arctic-dT SIAF columns added.
2026-04-27  NorESM2-LM/MM re-enabled.
2026-04-19  Native-grid wedge mode: no PSN25 regrid on CMIP6 side.
"""

import re
import gc
import os
import glob
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

from scipy.spatial import cKDTree
from scipy.stats import t as t_dist
import traceback

try:
    from global_land_mask import globe as _gm_globe
    _HAS_LAND_MASK = True
except ImportError:
    print('WARNING: global-land-mask not installed; ocean mask = all cells >= 60 N.')
    _HAS_LAND_MASK = False


# ============================================================
# CONFIG
# ============================================================
CMIP6_ROOT = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'
PLOT_DIR   = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
os.makedirs(PLOT_DIR, exist_ok=True)

N_WORKERS = 4

EARLY_YEARS = list(range(1982, 1992))   # SIC climo + arctic-dT scaling
LATE_YEARS  = list(range(2005, 2015))   # SIC climo + arctic-dT scaling
TREND_YEARS = list(range(1982, 2015))   # α~T regression window
RS_YEARS    = list(range(1985, 2015))
TAS_YEARS   = list(range(1982, 2015))   # annual T for regression + epoch ΔT

ALL_NEEDED_YEARS = set(range(1982, 2015))

MONTH_NAMES = {4: 'April', 5: 'May', 6: 'June',
               7: 'July',  8: 'August', 9: 'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

PISTONE_REGIONS = [
    ('EU', 'Europe_Barents',               60, 90,    0,   60),
    ('WA', 'WestAsia_KaraLaptev',          60, 90,   60,  120),
    ('WP', 'WestPacific_EastSiberian',     60, 90,  120,  180),
    ('EP', 'EastPacific_ChukchiBeaufort',  60, 90, -180, -120),
    ('NA', 'NorthAmerica_CdnArchBaffin',   60, 90, -120,  -60),
    ('AO', 'Atlantic_LabradorGreenland',   60, 90,  -60,    0),
]
PISTONE_CODES = [r[0] for r in PISTONE_REGIONS]

AREA_RATIO_OCEAN = 0.0347
SEASON_FRAC      = 6. / 12.
SCALE            = AREA_RATIO_OCEAN * SEASON_FRAC

SIC_ICE_THRESHOLD = 0.15
PVAL              = 0.05
ARCTIC_LAT_MIN    = 60.

FILL_THRESHOLD = 1e10

OBS_CSV_PATH  = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_unified_diagnostics.csv'
OBS_B2_GLOBAL = np.nan

if os.path.exists(OBS_CSV_PATH):
    try:
        obs_df = pd.read_csv(OBS_CSV_PATH)
        match  = obs_df[(obs_df['block'] == 'B2') &
                        (obs_df['period'] == 'CMIP6') &
                        (np.isclose(obs_df['pval'], 0.05))]
        if not match.empty:
            OBS_B2_GLOBAL = float(match['SIAF_pixel_global'].iloc[0])
            print(f"Loaded OBS_B2_GLOBAL = {OBS_B2_GLOBAL:.4f} from {OBS_CSV_PATH}")
        else:
            print(f"WARNING: Could not find matching OBS row in {OBS_CSV_PATH}")
    except Exception as e:
        print(f"WARNING: Failed to load OBS from {OBS_CSV_PATH} ({e})")
else:
    print(f"WARNING: OBS CSV not found at {OBS_CSV_PATH}. OBS_B2_GLOBAL set to NaN.")

RAD_VARS = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs', 'rsuscs']

VAR_TABLE_MAP = {
    'tas'    : ['Amon'],
    'rsus'   : ['Amon'],
    'rsds'   : ['Amon'],
    'rsdt'   : ['Amon'],
    'rsut'   : ['Amon'],
    'rsutcs' : ['Amon', 'CFmon'],
    'rsdscs' : ['Amon', 'CFmon'],
    'rsuscs' : ['Amon', 'CFmon'],
    'siconc' : ['SImon', 'SIday'],
}

MODEL_INFO = {
    'CESM2'          : dict(institution='NCAR',                member='r1i1p1f1'),
    'E3SM-1-0'       : dict(institution='E3SM-Project',        member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',                member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',               member='r1i1p1f1'),
    'MIROC-ES2L'     : dict(institution='MIROC',               member='r1i1p1f2'),
    'CanESM5'        : dict(institution='CCCma',               member='r1i1p1f1'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',               member='r1i1p1f1'),
    'GFDL-ESM4'      : dict(institution='NOAA-GFDL',           member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',                member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',                member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS',        member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',                 member='r1i1p1f1'),
    'ACCESS-CM2'     : dict(institution='CSIRO-ARCCSS',        member='r1i1p1f1'),
    'CNRM-ESM2-1'    : dict(institution='CNRM-CERFACS',        member='r1i1p1f2'),
    'EC-Earth3'      : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'EC-Earth3-Veg'  : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'GFDL-CM4'       : dict(institution='NOAA-GFDL',           member='r1i1p1f2'),
    'GISS-E2-1-H'    : dict(institution='NASA-GISS',           member='r1i1p1f1'),
    'INM-CM4-8'      : dict(institution='INM',                 member='r1i1p1f1'),
    'INM-CM5-0'      : dict(institution='INM',                 member='r1i1p1f1'),
    'MRI-ESM2-0'     : dict(institution='MRI',                 member='r1i1p1f1'),
    'NorESM2-LM'     : dict(institution='NCC',                 member='r1i1p1f1'),
    'NorESM2-MM'     : dict(institution='NCC',                 member='r1i1p1f1'),
}

MODEL_COLORS = {
    'CESM2'          : '#1f77b4',
    'E3SM-1-0'       : '#ff7f0e',
    'MPI-ESM1-2-HR'  : '#2ca02c',
    'MPI-ESM1-2-LR'  : '#98df8a',
    'IPSL-CM6A-LR'   : '#d62728',
    'MIROC6'         : '#9467bd',
    'MIROC-ES2L'     : '#c5b0d5',
    'CanESM5'        : '#8c564b',
    'ACCESS-ESM1-5'  : '#e377c2',
    'GFDL-ESM4'      : '#7f7f7f',
    'UKESM1-0-LL'    : '#17becf',
    'HadGEM3-GC31-LL': '#aec7e8',
    'CNRM-CM6-1'     : '#ffbb78',
    'BCC-CSM2-MR'    : '#ff9896',
    'ACCESS-CM2'     : '#bcbd22',
    'CNRM-ESM2-1'    : '#dbdb8d',
    'EC-Earth3'      : '#e41a1c',
    'EC-Earth3-Veg'  : '#f781bf',
    'GFDL-CM4'       : '#a65628',
    'GISS-E2-1-H'    : '#b8e186',
    'INM-CM4-8'      : '#542788',
    'INM-CM5-0'      : '#998ec3',
    'MRI-ESM2-0'     : '#f1a340',
    'NorESM2-LM'     : '#01665e',
    'NorESM2-MM'     : '#5ab4ac',
}


# ============================================================
# FILE DISCOVERY HELPERS
# ============================================================

def _files_cover_years(files, years):
    yr_set = set(years)
    for f in files:
        bn = os.path.basename(f)
        m  = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', bn)
        if m:
            start_yr = int(m.group(1)[:4])
            end_yr   = int(m.group(2)[:4])
            if any(start_yr <= y <= end_yr for y in yr_set):
                return True
        else:
            return True
    return False


def _prefer_regular_grid(files):
    if not files:
        return files
    bns    = [os.path.basename(f) for f in files]
    is_gr  = lambda bn: ('_gr_' in bn) or ('_gr1_' in bn) or ('_gr2_' in bn)
    is_gn  = lambda bn: '_gn_' in bn
    has_gr = any(is_gr(bn) for bn in bns)
    has_gn = any(is_gn(bn) for bn in bns)
    if has_gr and has_gn:
        return [f for f, bn in zip(files, bns) if is_gr(bn)]
    return files


def _is_hdf_error(exc):
    msg = str(exc).lower()
    return ('hdf error' in msg or
            'errno -101' in msg or
            ('netcdf' in msg and 'error' in msg) or
            'did not find a match' in msg)


def _triage_corrupt_files(files, variable):
    good = []
    for f in files:
        try:
            ds = xr.open_dataset(f, use_cftime=True)
            _  = ds[variable].isel(time=0).values
            ds.close()
            good.append(f)
        except Exception:
            print(f'    [triage] skipping corrupt file: {os.path.basename(f)}',
                  flush=True)
    return good


def find_var_files(inst, model, member, variable):
    tables = VAR_TABLE_MAP.get(variable, ['Amon'])
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in tables:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, variable)
            if os.path.isdir(d):
                files = sorted(glob.glob(
                    os.path.join(d, '**', f'{variable}_*.nc'),
                    recursive=True))
                if not files:
                    continue
                files    = _prefer_regular_grid(files)
                relevant = [f for f in files
                            if _files_cover_years([f], ALL_NEEDED_YEARS)]
                if relevant:
                    return relevant
    return []


def find_all_members(inst, model):
    members = set()
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        base = os.path.join(root, inst, model, 'historical')
        if not os.path.isdir(base):
            continue
        try:
            for mem in sorted(os.listdir(base)):
                if (find_var_files(inst, model, mem, 'rsus') and
                        find_var_files(inst, model, mem, 'rsds') and
                        find_var_files(inst, model, mem, 'siconc')):
                    members.add(mem)
        except Exception:
            pass
    return sorted(members)


# ============================================================
# COORDINATE UTILITIES
# ============================================================

def _mask_fills(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    return np.where(np.abs(arr) < FILL_THRESHOLD, arr, np.nan)


def _is_valid_lat(arr):
    if arr is None:
        return False
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    return len(flat) > 0 and float(flat.min()) >= -90.1 and float(flat.max()) <= 90.1


def _std_latlon(da):
    aliases = {'lat': ['latitude', 'nav_lat', 'rlat', 'y'],
               'lon': ['longitude', 'nav_lon', 'rlon', 'x']}
    for new, als in aliases.items():
        for c in list(da.coords):
            if c.lower() in als and c != new:
                try:
                    da = da.rename({c: new})
                except Exception:
                    pass
                break

    spatial_shape = da.shape[1:] if da.ndim >= 3 else da.shape
    ny = spatial_shape[0] if len(spatial_shape) >= 1 else None
    nx = spatial_shape[1] if len(spatial_shape) >= 2 else None

    def _best_coord(primary_name, alias_names):
        candidates = [primary_name] + alias_names
        best_1d, best_2d = None, None
        for name in candidates:
            for src in [da.coords, da.data_vars if hasattr(da, 'data_vars') else {}]:
                if name in src:
                    arr = _mask_fills(np.asarray(src[name].values, dtype=float))
                    if arr.ndim == 2 and ny and nx and arr.shape == (ny, nx):
                        if best_2d is None:
                            best_2d = arr
                    elif arr.ndim == 1:
                        if best_1d is None:
                            best_1d = arr
        return best_2d if best_2d is not None else best_1d

    lat = _best_coord('lat', ['latitude', 'nav_lat', 'rlat', 'y'])
    lon = _best_coord('lon', ['longitude', 'nav_lon', 'rlon', 'x'])

    if not _is_valid_lat(lat):
        priority = [n for n in da.coords if 'lat' in n.lower()]
        rest     = [n for n in da.coords if n not in priority]
        for name in priority + rest:
            try:
                arr = _mask_fills(np.asarray(da[name].values, dtype=float))
            except Exception:
                continue
            if ny and nx and arr.shape == (ny, nx) and _is_valid_lat(arr):
                lat = arr
                for lon_name in [n for n in da.coords if 'lon' in n.lower()]:
                    try:
                        lon_arr = _mask_fills(
                            np.asarray(da[lon_name].values, dtype=float))
                        if lon_arr.shape == (ny, nx):
                            lon = lon_arr
                            break
                    except Exception:
                        continue
                break

    if lat is not None and lat.ndim == 2 and lon is not None and lon.ndim == 1:
        ny2, nx2 = lat.shape
        if len(lon) == nx2:
            lon = np.tile(lon[np.newaxis, :], (ny2, 1))
        elif len(lon) == ny2:
            lon = np.tile(lon[:, np.newaxis], (1, nx2))

    is_2d = (lat.ndim == 2) if lat is not None else False
    return da, lat, lon, is_2d


# ============================================================
# NATIVE-GRID CONTEXT
# ============================================================

def build_native_ctx(lat, lon, is_2d):
    if is_2d:
        lat2d = np.asarray(lat, dtype=np.float32)
        lon2d = np.asarray(lon, dtype=np.float32)
    else:
        la    = np.asarray(lat, dtype=np.float32)
        lo    = np.asarray(lon, dtype=np.float32)
        lon2d, lat2d = np.meshgrid(lo, la)

    lat2d = np.where(np.isfinite(lat2d), lat2d, 0.).astype(np.float32)
    lon2d = np.where(np.isfinite(lon2d), lon2d, 0.).astype(np.float32)

    shape       = lat2d.shape
    area_weight = np.cos(np.deg2rad(lat2d)).astype(np.float32)

    lon_pm = lon2d.copy()
    lon_pm = lon_pm - 360. * np.floor((lon_pm + 180.) / 360.)
    lon_pm = lon_pm.astype(np.float32)

    if _HAS_LAND_MASK:
        ocean_mask = _gm_globe.is_ocean(lat2d, lon_pm)
    else:
        ocean_mask = np.ones(shape, dtype=bool)

    domain_mask = ocean_mask & (lat2d >= 60.)

    region_masks = {}
    for (code, _, lo_b, hi_b, lmn, lmx) in PISTONE_REGIONS:
        lat_ok = (lat2d >= lo_b) & (lat2d <= hi_b)
        lon_ok = ((lon_pm >= lmn) & (lon_pm <= lmx)
                  if lmn <= lmx else (lon_pm >= lmn) | (lon_pm <= lmx))
        region_masks[code] = lat_ok & lon_ok & domain_mask

    return dict(lat2d=lat2d, lon2d=lon2d, shape=shape,
                area_weight=area_weight, domain_mask=domain_mask,
                region_masks=region_masks, is_2d=is_2d)


def region_mean_native(arr, rmask, area_weight):
    v, w = arr[rmask], area_weight[rmask]
    ok   = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan, 0.
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))


def arctic_ocean_mean_native(arr, native_ctx):
    arr = np.where(native_ctx['domain_mask'] & ~np.isfinite(arr), 0., arr)
    num = den = 0.
    for code in PISTONE_CODES:
        v, w = region_mean_native(arr,
                                   native_ctx['region_masks'][code],
                                   native_ctx['area_weight'])
        if np.isfinite(v) and w > 0:
            num += v * w;  den += w
    return (num / den) if den > 0 else np.nan


def component_mean_native(arr, comp_arr, native_ctx):
    """Area-weighted mean of `arr` restricted to domain pixels where `comp_arr` > 0."""
    mask = (native_ctx['domain_mask']
            & np.isfinite(comp_arr)
            & (comp_arr > 0))
    v  = arr[mask]
    w  = native_ctx['area_weight'][mask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def seasonal_to_global(monthly_dict):
    vals = [v for v in monthly_dict.values() if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.nanmean(vals)) * SCALE


def fill_zero_domain(arr2d, lat2d):
    arctic = lat2d >= 60.
    return np.where(arctic,
                    np.where(np.isfinite(arr2d), arr2d, 0.),
                    np.nan).astype(np.float32)


# ============================================================
# SMALL HELPERS
# ============================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat) * np.cos(lon),
                             np.cos(lat) * np.sin(lon),
                             np.sin(lat)])


# ============================================================
# VECTORIZED PER-PIXEL OLS  (albedo ~ any 1-D predictor)
#
# Works for both the α~T regression (x_vec = annual T anomaly)
# and the SIC~time regression via pixel_ols_trend.
# ============================================================

def compute_pixel_ols_slopes(stk, x_vec):
    """
    Vectorized per-pixel OLS slope of stk ~ x_vec.

    Parameters
    ----------
    stk   : (T, H, W) float array — dependent variable (e.g. albedo)
    x_vec : (T,) float array      — predictor (e.g. annual T anomaly, centered)

    Returns
    -------
    slope_map : (H, W) float32; NaN where fewer than 10 valid pairs
    """
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

    flat_c  = np.where(valid, flat  - y_mean, 0.0)
    x_c     = np.where(valid, x_bc  - x_mean, 0.0)

    num     = (x_c * flat_c).sum(0)
    den     = (x_c ** 2).sum(0)

    out     = np.full(N, np.nan, np.float32)
    good    = (n_valid >= 10) & (den > 0)
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out.reshape(H, W)


# ============================================================
# T07 INVERSION
# ============================================================

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd  = np.where(rsds  > 1., rsds,  np.nan)
        a   = np.clip(rsus / Sd, 0., 1.)
        p   = rsds / S;  q = rsut / S
        den = q + p - 2. * p * a
        R   = np.clip((q - p * a) / np.where(np.abs(den) > 1e-6, den, np.nan),
                      0., .99)
        A   = np.clip(1. - p * (1. - a * R) /
                      np.where((1. - R) > .01, 1. - R, np.nan), 0., .99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S
        dc  = qc + pc - 2. * pc * ac
        Rc  = np.clip((qc - pc * ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan),
                      0., .99)
        Rl  = np.clip(R - Rc, 0., .99)
        RS  = (S * (1. - A) * (1. - Rl) ** 2
               / np.where((1. - a * Rl) ** 2 > 1e-9,
                           (1. - a * Rl) ** 2, np.nan))
    return RS.astype(np.float32)


# ============================================================
# CMIP6 LOADER UTILITIES
# ============================================================

def _open_robust(files, variable):
    strategies = [
        (dict(combine='by_coords'),                                       True),
        (dict(combine='nested', concat_dim='time',
              coords='minimal', compat='override'),                       True),
        (dict(combine='nested', concat_dim='time',
              coords='minimal', compat='override'),                       True),
        (dict(combine='nested', concat_dim='time',
              coords='minimal', compat='override',
              decode_times=False),                                        False),
    ]

    def _try_strategies(file_list):
        last_err = None
        for kw, auto_decoded in strategies:
            try:
                open_kw = dict(kw)
                open_kw['chunks'] = {'time': 12}
                if auto_decoded:
                    open_kw['use_cftime'] = True
                ds = xr.open_mfdataset(file_list, **open_kw)
                if not auto_decoded:
                    ds = xr.decode_cf(ds, use_cftime=True)
                return ds
            except Exception as e:
                last_err = e
        return last_err

    result = _try_strategies(files)
    if not isinstance(result, Exception):
        return result

    if _is_hdf_error(result) and len(files) > 1:
        clean = _triage_corrupt_files(files, variable)
        if clean and len(clean) < len(files):
            result2 = _try_strategies(clean)
            if not isinstance(result2, Exception):
                return result2
            result = result2

    raise RuntimeError(
        f'Cannot open {variable} ({type(result).__name__}: {result})')


def load_filtered(files, variable, years, months):
    try:
        ds  = _open_robust(files, variable)
        da  = ds[variable]

        tvals = da.time.values
        _, first_occ = np.unique(tvals, return_index=True)
        if len(first_occ) < len(tvals):
            da = da.isel(time=np.sort(first_occ))

        ys   = set(years); ms = set(months)
        mask = np.array([int(t.dt.year) in ys and int(t.dt.month) in ms
                          for t in da.time], dtype=bool)
        da   = da.isel(time=mask).load(); ds.close()
        da, _, _, _ = _std_latlon(da)
        return da
    except Exception as e:
        print(f'  load_filtered({variable}): {e}', flush=True)
        return None


# ============================================================
# GLOBAL ANNUAL TEMPERATURE SERIES FROM MODEL TAS
#
# Returns {year: T_K} for each year in trend_years where data exist.
# Used as the X-variable in the per-pixel α~T OLS regression.
# Each model/member pair gets its own T series so intra-ensemble
# T variability is respected.
# ============================================================

def compute_T_annual_global(model, member, trend_years):
    """
    Compute area-weighted global annual mean TAS for every year in
    trend_years from the model's own historical TAS files.

    Returns
    -------
    T_dict : {year (int): T_K (float)} — one value per available year.
             Returns None if TAS files cannot be loaded.
    """
    inst = MODEL_INFO[model]['institution']
    tf   = find_var_files(inst, model, member, 'tas')
    if not tf:
        return None

    da = load_filtered(tf, 'tas', trend_years, list(range(1, 13)))
    if da is None:
        return None

    _, lat, lon, is_2d = _std_latlon(da)
    if lat is None:
        return None

    yrs  = np.array([int(t.dt.year) for t in da.time])
    vals = da.values.astype(float)

    if is_2d:
        lat_2d = np.where(np.isfinite(lat), lat, 0.)
    else:
        if lat.ndim == 1 and lon is not None and lon.ndim == 1:
            lat_2d = np.tile(lat[:, None], (1, len(lon)))
        else:
            lat_2d = lat

    w_glob = np.cos(np.deg2rad(lat_2d)).astype(float)
    w_glob = np.where(w_glob > 0, w_glob, 0.)

    T_dict = {}
    for year in trend_years:
        idx = np.where(yrs == year)[0]
        if idx.size == 0:
            continue
        month_means = []
        for i in idx:
            ok = np.isfinite(vals[i]) & (w_glob > 0)
            if ok.any():
                month_means.append(
                    float(np.sum(vals[i][ok] * w_glob[ok]) / np.sum(w_glob[ok])))
        if month_means:
            T_dict[year] = float(np.mean(month_means))

    del da
    gc.collect()
    return T_dict if T_dict else None


# ============================================================
# EPOCH dT  (kept for arctic-dT scaling columns only)
# ============================================================

def compute_delta_T_both(model, member):
    """
    Epoch-based ΔT (late mean − early mean) for global and arctic domains.
    Used only for the _scale_adT supplementary columns; NOT used in IS.
    """
    inst = MODEL_INFO[model]['institution']
    tf   = find_var_files(inst, model, member, 'tas')
    if not tf:
        return None, None
    da = load_filtered(tf, 'tas', EARLY_YEARS + LATE_YEARS, list(range(1, 13)))
    if da is None:
        return None, None
    _, lat, lon, is_2d = _std_latlon(da)
    if lat is None or is_2d or lat.ndim != 1:
        return None, None

    yrs  = np.array([int(t.dt.year) for t in da.time])
    vals = da.values.astype(float)

    lat_2d = np.tile(lat[:, None], (1, len(lon)))
    w_glob = np.cos(np.deg2rad(lat_2d))
    w_arct = np.where(lat_2d >= ARCTIC_LAT_MIN, w_glob, 0.)

    def _wmean_epoch(yr_list, w2d):
        idx = np.where(np.isin(yrs, yr_list))[0]
        if idx.size == 0:
            return np.nan
        step_means = []
        for i in idx:
            ok = np.isfinite(vals[i]) & (w2d > 0)
            if ok.any():
                step_means.append(
                    np.sum(vals[i][ok] * w2d[ok]) / np.sum(w2d[ok]))
        return float(np.mean(step_means)) if step_means else np.nan

    eT_g = _wmean_epoch(EARLY_YEARS, w_glob)
    lT_g = _wmean_epoch(LATE_YEARS,  w_glob)
    eT_a = _wmean_epoch(EARLY_YEARS, w_arct)
    lT_a = _wmean_epoch(LATE_YEARS,  w_arct)

    del da
    gc.collect()

    def _to_dT(early, late):
        if not (np.isfinite(early) and np.isfinite(late)):
            return None
        dT = float(late - early)
        return dT if dT > 0 else None

    return _to_dT(eT_g, lT_g), _to_dT(eT_a, lT_a)


# ============================================================
# PIXEL-WISE OLS TREND (SIC ~ time, for ice classification)
# ============================================================

def pixel_ols_trend(years_1d, sic_3d):
    T      = years_1d.shape[0]
    x      = years_1d - years_1d.mean()
    ss     = float(np.dot(x, x))
    y_mean = np.nanmean(sic_3d, axis=0)
    dy     = sic_3d - y_mean[None]
    slope  = np.nansum(x[:, None, None] * dy, axis=0) / ss
    resid  = sic_3d - (y_mean + slope * x[:, None, None])
    sse    = np.nansum(resid ** 2, axis=0)
    df     = T - 2
    with np.errstate(invalid='ignore', divide='ignore'):
        se     = np.sqrt(sse / df) / np.sqrt(ss)
        t_stat = slope / se
        pval   = 2. * (1. - t_dist.cdf(np.abs(t_stat), df))
    return slope, pval


# ============================================================
# MODEL IS COMPUTATION — direct α~T OLS regression
#
# IS[px] = −(dα/dT)[px]  [alpha K⁻¹]
#
# For each month and each year in TREND_YEARS, compute the spatial
# mean albedo (rsus/rsds) → build a (n_years, H, W) stack → regress
# against the model's own global annual T (centered on trend-period
# mean for numerical stability) → slope is dα/dT → IS = −slope.
#
# SIC-based extent/darkening classification is unchanged.
# ============================================================

def compute_member_IS_native(model, member, T_annual_dict):
    """
    Parameters
    ----------
    T_annual_dict : {year: T_K} from compute_T_annual_global.
                    Only years present in both the dict and the radiation
                    files are used in the regression.
    """
    inst   = MODEL_INFO[model]['institution']
    rsus_f = find_var_files(inst, model, member, 'rsus')
    rsds_f = find_var_files(inst, model, member, 'rsds')
    sic_f  = find_var_files(inst, model, member, 'siconc')
    if not all([rsus_f, rsds_f, sic_f]):
        return None, None

    all_yrs = list(range(1980, 2016))
    ru = load_filtered(rsus_f, 'rsus',   all_yrs, MONTH_NUMS)
    rd = load_filtered(rsds_f, 'rsds',   all_yrs, MONTH_NUMS)
    si = load_filtered(sic_f,  'siconc', all_yrs, MONTH_NUMS)
    if ru is None or rd is None or si is None:
        return None, None

    _, alb_lat, alb_lon, alb_is2d = _std_latlon(ru)
    _, sic_lat, sic_lon, sic_is2d = _std_latlon(si)
    if alb_lat is None or sic_lat is None:
        return None, None

    def _ym(da):
        return set(zip([int(t.dt.year)  for t in da.time],
                        [int(t.dt.month) for t in da.time]))
    comm = _ym(ru) & _ym(rd) & _ym(si)
    if not comm:
        return None, None

    def _tmask(da):
        return np.array([(int(t.dt.year), int(t.dt.month)) in comm
                          for t in da.time], dtype=bool)
    ru = ru.isel(time=_tmask(ru))
    rd = rd.isel(time=_tmask(rd))
    si = si.isel(time=_tmask(si))

    sv = si.values.astype(float)
    if np.nanmax(sv) > 1.5:
        sv /= 100.

    alb_sh = alb_lat.shape if alb_is2d else (len(alb_lat), len(alb_lon))

    if sv.shape[1:] != alb_sh:
        if sic_is2d:
            sla_raw = _mask_fills(sic_lat).ravel()
            slo_raw = _mask_fills(sic_lon).ravel()
        else:
            slo2d, sla2d = np.meshgrid(sic_lon, sic_lat)
            sla_raw = _mask_fills(sla2d).ravel()
            slo_raw = _mask_fills(slo2d).ravel()

        if alb_is2d:
            ala = _mask_fills(alb_lat).ravel()
            alo = _mask_fills(alb_lon).ravel()
        else:
            alo2d, ala2d = np.meshgrid(alb_lon, alb_lat)
            ala = ala2d.ravel()
            alo = alo2d.ravel()

        sic_valid = np.isfinite(sla_raw) & np.isfinite(slo_raw)
        sla_clean = sla_raw[sic_valid]
        slo_clean = slo_raw[sic_valid]

        sv        = sv.reshape(sv.shape[0], -1)
        sv_clean  = sv[:, sic_valid]

        _, idx_s2a = cKDTree(latlon_to_xyz(sla_clean, slo_clean)).query(
            latlon_to_xyz(ala, alo).astype(np.float32), workers=-1)
        sv = sv_clean[:, idx_s2a].reshape((sv.shape[0],) + alb_sh)

    ru_yrs = np.array([int(t.dt.year)  for t in ru.time])
    ru_mon = np.array([int(t.dt.month) for t in ru.time])
    si_yrs = np.array([int(t.dt.year)  for t in si.time])
    si_mon = np.array([int(t.dt.month) for t in si.time])
    del si
    gc.collect()

    native_ctx = build_native_ctx(alb_lat, alb_lon, alb_is2d)
    lat2d_a    = native_ctx['lat2d']

    # Center T on the trend-period mean for numerical stability.
    # Centering does not affect OLS slopes.
    all_T_vals = np.array([T_annual_dict[y] for y in TREND_YEARS
                            if y in T_annual_dict])
    T_center   = float(all_T_vals.mean()) if len(all_T_vals) > 0 else 0.

    result = {}
    for mn in MONTH_NUMS:
        # ── Build per-year albedo stack for TREND_YEARS ───────────────────
        yearly_alb = {}
        for year in TREND_YEARS:
            if year not in T_annual_dict:
                continue
            mask = (ru_yrs == year) & (ru_mon == mn)
            if mask.sum() == 0:
                continue
            u = ru.isel(time=mask)
            d = rd.isel(time=mask)
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(d.mean('time').values > 1.,
                               u.mean('time').values / d.mean('time').values,
                               np.nan)
            yearly_alb[year] = np.clip(alb, 0., 1.)

        valid_years = sorted(yearly_alb.keys())
        if len(valid_years) < 10:
            result[mn] = None
            continue

        # ── Per-pixel OLS: α ~ T_global ──────────────────────────────────
        stk   = np.stack([yearly_alb[y] for y in valid_years], axis=0)  # (T, H, W)
        T_vec = np.array([T_annual_dict[y] - T_center
                           for y in valid_years], dtype=np.float64)       # (T,)

        dalpha_dT = compute_pixel_ols_slopes(stk, T_vec)   # [alpha K⁻¹], neg = feedback

        # ── SIC classification (unchanged) ────────────────────────────────
        tm = (si_mon == mn) & np.isin(si_yrs, TREND_YEARS)
        em = (si_mon == mn) & np.isin(si_yrs, EARLY_YEARS)
        lm = (si_mon == mn) & np.isin(si_yrs, LATE_YEARS)
        if tm.sum() < 10 or em.sum() == 0 or lm.sum() == 0:
            result[mn] = None
            continue

        sic_e = np.nanmean(sv[em], axis=0)
        sic_l = np.nanmean(sv[lm], axis=0)
        slope_sic, pval_sic = pixel_ols_trend(si_yrs[tm].astype(float), sv[tm])

        all_ice  = (lat2d_a >= 60.) & (sic_e >= SIC_ICE_THRESHOLD)
        changing = (all_ice & np.isfinite(slope_sic) & np.isfinite(pval_sic)
                    & (slope_sic < 0) & (pval_sic < PVAL))
        stable   = (all_ice & ~changing
                    & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

        # IS = −dα/dT  (positive = positive feedback)
        IS_raw = (-dalpha_dT).astype(np.float32)

        IS_ext = np.where(changing, IS_raw, np.nan).astype(np.float32)
        IS_drk = np.where(stable,   IS_raw, np.nan).astype(np.float32)
        IS_tot = np.where(
            np.isfinite(IS_ext) | np.isfinite(IS_drk),
            np.where(np.isfinite(IS_ext), IS_ext, 0.)
            + np.where(np.isfinite(IS_drk), IS_drk, 0.),
            np.nan).astype(np.float32)
        IS_ext = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_ext), IS_ext, 0.),
                          np.nan).astype(np.float32)
        IS_drk = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_drk), IS_drk, 0.),
                          np.nan).astype(np.float32)

        result[mn] = {
            'tot': fill_zero_domain(IS_tot, lat2d_a),
            'ext': fill_zero_domain(IS_ext, lat2d_a),
            'drk': fill_zero_domain(IS_drk, lat2d_a),
        }
        del IS_tot, IS_ext, IS_drk
        gc.collect()

    del ru, rd, sv
    gc.collect()
    return result, native_ctx


# ============================================================
# MODEL RS COMPUTATION  -->  NATIVE GRID
# ============================================================

def _load_var_mean(files, variable, years, month_num):
    if not files:
        return None, None, None, None
    try:
        ds   = _open_robust(files, variable)
        da   = ds[variable]
        mask = np.array([int(t.dt.year) in set(years) and
                          int(t.dt.month) == month_num
                          for t in da.time], dtype=bool)
        if not mask.any():
            ds.close()
            return None, None, None, None
        sub = da.isel(time=mask).mean('time').load()
        sub, lat, lon, is2d = _std_latlon(sub)
        ds.close()
        return sub.values.astype(float), lat, lon, is2d
    except Exception as e:
        print(f'  _load_var_mean({variable}): {e}', flush=True)
        return None, None, None, None


def compute_model_rs_native(model, member, native_ctx):
    inst  = MODEL_INFO[model]['institution']
    rf    = {v: find_var_files(inst, model, member, v) for v in RAD_VARS}
    avail = {v for v, f in rf.items() if f}

    has_full   = set(RAD_VARS) <= avail
    has_approx = (set(RAD_VARS) - {'rsuscs'}) <= avail
    has_rsds   = 'rsds' in avail

    atm_shape = native_ctx['shape']
    result    = {}
    methods   = []

    for mn in MONTH_NUMS:
        rs_arr = None
        used   = None

        if has_full or has_approx:
            req_vars = RAD_VARS if has_full else [v for v in RAD_VARS
                                                   if v != 'rsuscs']
            data = {}
            ok   = True
            for v in req_vars:
                arr, _, _, _ = _load_var_mean(rf[v], v, RS_YEARS, mn)
                if arr is None:
                    ok = False
                    break
                data[v] = arr
            if ok:
                if has_approx and not has_full:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        a_s = np.where(data['rsds'] > 1.,
                                       data['rsus'] / data['rsds'], 0.)
                    data['rsuscs'] = data['rsdscs'] * a_s
                    used = 'approx_T07'
                else:
                    used = 'full_T07'
                rs_arr = t07_invert(data['rsdt'], data['rsut'],
                                    data['rsds'],  data['rsus'],
                                    data['rsutcs'], data['rsdscs'],
                                    data['rsuscs']).astype(np.float32)

        if rs_arr is None and has_rsds:
            arr, _, _, _ = _load_var_mean(rf['rsds'], 'rsds', RS_YEARS, mn)
            if arr is not None:
                rs_arr = arr.astype(np.float32)
                used   = 'rsds_proxy'

        if rs_arr is None:
            result[mn] = None
            continue

        if rs_arr.shape != atm_shape:
            print(f'  RS shape {rs_arr.shape} != atm_shape {atm_shape} '
                  f'for {model}/{member} month {mn}; skip', flush=True)
            result[mn] = None
            continue

        result[mn] = rs_arr
        methods.append(used)
        del rs_arr
        gc.collect()

    if not methods:
        method_label = 'none'
    elif len(set(methods)) == 1:
        method_label = methods[0]
    else:
        method_label = '+'.join(sorted(set(methods)))
    return result, method_label


# ============================================================
# SIAF ASSEMBLY
# ============================================================

def compute_siaf_monthly(IS_by_month, rs_by_month, native_ctx):
    out = {}
    for mn in MONTH_NUMS:
        is_d = IS_by_month.get(mn)
        rs_p = rs_by_month.get(mn)
        if is_d is None or rs_p is None:
            out[mn] = None
            continue
        if rs_p.shape != is_d['tot'].shape:
            print(f'  SIAF shape mismatch IS={is_d["tot"].shape} '
                  f'RS={rs_p.shape}', flush=True)
            out[mn] = None
            continue
        SIAF_tot = (is_d['tot'] * rs_p).astype(np.float32)
        SIAF_ext = (is_d['ext'] * rs_p).astype(np.float32)
        SIAF_drk = (is_d['drk'] * rs_p).astype(np.float32)
        out[mn] = {
            'tot': arctic_ocean_mean_native(SIAF_tot, native_ctx),
            'ext': arctic_ocean_mean_native(SIAF_ext, native_ctx),
            'drk': arctic_ocean_mean_native(SIAF_drk, native_ctx),
        }
        del SIAF_tot, SIAF_ext, SIAF_drk
        gc.collect()
    return out


def siaf_global(monthly_siaf, component='tot'):
    return seasonal_to_global({mn: (d[component] if d else np.nan)
                                for mn, d in monthly_siaf.items()})


def compute_spatial_temporal_mean_is(IS_by_month, comp, native_ctx):
    vals = []
    for mn in MONTH_NUMS:
        d = IS_by_month.get(mn)
        if d is not None:
            vals.append(arctic_ocean_mean_native(d[comp], native_ctx))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_spatial_temporal_mean_rs(rs_by_month, native_ctx):
    vals = []
    for mn in MONTH_NUMS:
        r = rs_by_month.get(mn)
        if r is not None:
            vals.append(arctic_ocean_mean_native(r, native_ctx))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_is_per_area(IS_by_month, native_ctx):
    """IS averaged over each component's own footprint (not the full domain)."""
    ext_v, drk_v = [], []
    for mn in MONTH_NUMS:
        d = IS_by_month.get(mn)
        if d is not None:
            ext_v.append(component_mean_native(d['ext'], d['ext'], native_ctx))
            drk_v.append(component_mean_native(d['drk'], d['drk'], native_ctx))
    return (float(np.nanmean(ext_v)) if ext_v else np.nan,
            float(np.nanmean(drk_v)) if drk_v else np.nan)


def compute_rs_per_area(rs_by_month, IS_by_month, native_ctx):
    """RS averaged over the ext and drk component footprints."""
    ext_v, drk_v = [], []
    for mn in MONTH_NUMS:
        r = rs_by_month.get(mn)
        d = IS_by_month.get(mn)
        if r is not None and d is not None:
            ext_v.append(component_mean_native(r, d['ext'], native_ctx))
            drk_v.append(component_mean_native(r, d['drk'], native_ctx))
    return (float(np.nanmean(ext_v)) if ext_v else np.nan,
            float(np.nanmean(drk_v)) if drk_v else np.nan)


def compute_siaf_per_area(IS_by_month, rs_by_month, native_ctx):
    """SIAF (IS×RS) averaged over each component's own footprint × SCALE."""
    ext_v, drk_v = [], []
    for mn in MONTH_NUMS:
        d = IS_by_month.get(mn)
        r = rs_by_month.get(mn)
        if d is None or r is None:
            continue
        if r.shape != d['ext'].shape:
            continue
        SIAF_ext = (d['ext'] * r).astype(np.float32)
        SIAF_drk = (d['drk'] * r).astype(np.float32)
        ext_v.append(component_mean_native(SIAF_ext, d['ext'], native_ctx))
        drk_v.append(component_mean_native(SIAF_drk, d['drk'], native_ctx))
    ext_pa = float(np.nanmean(ext_v)) * SCALE if ext_v else np.nan
    drk_pa = float(np.nanmean(drk_v)) * SCALE if drk_v else np.nan
    return ext_pa, drk_pa


# ============================================================
# ARCTIC-dT SCALING HELPER
# ============================================================

def _scale_adT(val_global_dT, dT_global, dT_arctic):
    """
    Scale a global-dT SIAF value to arctic-dT units.
    dT_arctic < 0.1 K guard prevents blowup from near-zero arctic warming.
    """
    if (dT_arctic is None or dT_global is None
            or not np.isfinite(dT_arctic) or dT_arctic < 0.1
            or not np.isfinite(dT_global) or dT_global <= 0
            or not np.isfinite(val_global_dT)):
        return np.nan
    return float(val_global_dT) * (dT_global / dT_arctic)


# ============================================================
# PER-MODEL WORKER  (called by ProcessPoolExecutor)
# ============================================================

def process_model(model):
    inst = MODEL_INFO[model]['institution']
    prim = MODEL_INFO[model]['member']
    rows = []
    model_rs_method = 'unknown'

    members = find_all_members(inst, model)
    if not members:
        print(f'  [{model}] No qualifying members -- skip', flush=True)
        return rows, model, 'no_members'

    if prim in members:
        members = [prim] + [m for m in members if m != prim]

    print(f'\n{"─" * 60}', flush=True)
    print(f'Model: {model}  ({len(members)} member(s))', flush=True)

    # Compute epoch ΔT for the primary member (arctic-dT scaling only)
    dT_prim, dT_prim_arctic = compute_delta_T_both(model, prim)
    collected_dTs      = [dT_prim]        if dT_prim        is not None else []
    collected_dTs_arct = [dT_prim_arctic] if dT_prim_arctic is not None else []

    # Compute per-year global T for the primary member (α~T regression)
    T_annual_prim = compute_T_annual_global(model, prim, TREND_YEARS)

    _arct_str = f'{dT_prim_arctic:+.3f} K' if dT_prim_arctic else 'N/A'
    print(f'  Primary dT global = {dT_prim:+.3f} K  |  Arctic = {_arct_str}'
          if dT_prim else '  Primary epoch dT: unavailable', flush=True)
    if T_annual_prim:
        print(f'  Primary T_annual: {len(T_annual_prim)}/{len(TREND_YEARS)} '
              f'trend years covered', flush=True)
    else:
        print('  Primary T_annual: unavailable', flush=True)

    for member in members:
        is_prim = (member == prim)

        # ── Epoch ΔT (for arctic-dT scaling columns) ──────────────────
        if is_prim:
            dT        = dT_prim
            dT_arctic = dT_prim_arctic
        else:
            dT_raw, dT_raw_arctic = compute_delta_T_both(model, member)
            if dT_raw is not None:
                dT = dT_raw;  collected_dTs.append(dT_raw)
            elif collected_dTs:
                dT = float(np.mean(collected_dTs))
            else:
                dT = dT_prim

            if dT_raw_arctic is not None:
                dT_arctic = dT_raw_arctic
                collected_dTs_arct.append(dT_raw_arctic)
            elif collected_dTs_arct:
                dT_arctic = float(np.mean(collected_dTs_arct))
            else:
                dT_arctic = dT_prim_arctic

        # ── Per-year global T (for α~T regression) ────────────────────
        if is_prim:
            T_annual = T_annual_prim
        else:
            T_annual = compute_T_annual_global(model, member, TREND_YEARS)
            if T_annual is None:
                T_annual = T_annual_prim   # fall back to primary member's T

        if T_annual is None or len(T_annual) < 10:
            print(f'  [{member}] Insufficient T_annual data -- skip', flush=True)
            continue

        # ── IS via α~T OLS (no delta_t needed) ────────────────────────
        try:
            IS_by_month, native_ctx = compute_member_IS_native(
                model, member, T_annual)
        except Exception as e:
            print(f'  [{member}] IS failed: {e}', flush=True)
            print(''.join(traceback.format_exc().splitlines(keepends=True)[-6:]),
                  flush=True)
            continue

        if IS_by_month is None:
            print(f'  [{member}] IS: no data -- skip', flush=True)
            continue

        n_valid = sum(1 for v in IS_by_month.values() if v is not None)
        if n_valid < 3:
            print(f'  [{member}] IS: only {n_valid}/6 months -- skip',
                  flush=True)
            continue

        try:
            rs_by_month, rs_meth = compute_model_rs_native(
                model, member, native_ctx)
        except Exception as e:
            print(f'  [{member}] RS failed: {e}', flush=True)
            rs_by_month = {mn: None for mn in MONTH_NUMS}
            rs_meth     = 'failed'
        model_rs_method = rs_meth

        b2_monthly = compute_siaf_monthly(IS_by_month, rs_by_month, native_ctx)
        b2_tot = siaf_global(b2_monthly, 'tot')
        b2_ext = siaf_global(b2_monthly, 'ext')
        b2_drk = siaf_global(b2_monthly, 'drk')

        is_tot_val = compute_spatial_temporal_mean_is(IS_by_month, 'tot', native_ctx)
        is_ext_val = compute_spatial_temporal_mean_is(IS_by_month, 'ext', native_ctx)
        is_drk_val = compute_spatial_temporal_mean_is(IS_by_month, 'drk', native_ctx)
        rs_val     = compute_spatial_temporal_mean_rs(rs_by_month, native_ctx)

        is_ext_pa, is_drk_pa     = compute_is_per_area(IS_by_month, native_ctx)
        rs_ext_pa, rs_drk_pa     = compute_rs_per_area(rs_by_month, IS_by_month, native_ctx)
        siaf_ext_pa, siaf_drk_pa = compute_siaf_per_area(IS_by_month, rs_by_month, native_ctx)

        # arctic-dT scaling (uses epoch ΔT, independent of IS method)
        b2_tot_adT = _scale_adT(b2_tot,     dT, dT_arctic)
        b2_ext_adT = _scale_adT(b2_ext,     dT, dT_arctic)
        b2_drk_adT = _scale_adT(b2_drk,     dT, dT_arctic)
        is_tot_adT = _scale_adT(is_tot_val, dT, dT_arctic)
        is_ext_adT = _scale_adT(is_ext_val, dT, dT_arctic)
        is_drk_adT = _scale_adT(is_drk_val, dT, dT_arctic)

        _arct_str2 = f'{dT_arctic:+.3f} K' if dT_arctic else 'N/A'
        print(f'  [{member}] T_annual={len(T_annual)}yr | '
              f'dT_glob={dT:+.3f} K  dT_arct={_arct_str2} | '
              f'RS [{rs_meth}]: '
              f'tot={b2_tot:+.4f} ({b2_tot_adT:+.4f} adT)  '
              f'ext={b2_ext:+.4f}  drk={b2_drk:+.4f}',
              flush=True)

        rows.append(dict(
            model=model, member=member,
            rs_method=rs_meth,
            delta_t=dT,                         # epoch ΔT (reference only)
            n_T_years=len(T_annual),             # years used in α~T regression
            siaf_global=b2_tot,
            siaf_ext_global=b2_ext,
            siaf_drk_global=b2_drk,
            siaf_ext_per_area=siaf_ext_pa,
            siaf_drk_per_area=siaf_drk_pa,
            is_tot_arctic=is_tot_val,
            is_ext_arctic=is_ext_val,
            is_drk_arctic=is_drk_val,
            rs_arctic=rs_val,
            is_ext_per_area=is_ext_pa,
            is_drk_per_area=is_drk_pa,
            rs_ext_per_area=rs_ext_pa,
            rs_drk_per_area=rs_drk_pa,
            grid_ny=native_ctx['shape'][0],
            grid_nx=native_ctx['shape'][1],
            delta_t_arctic=dT_arctic,
            siaf_global_adT=b2_tot_adT,
            siaf_ext_global_adT=b2_ext_adT,
            siaf_drk_global_adT=b2_drk_adT,
            is_tot_arctic_adT=is_tot_adT,
            is_ext_arctic_adT=is_ext_adT,
            is_drk_arctic_adT=is_drk_adT,
        ))

        del IS_by_month, native_ctx
        gc.collect()

    return rows, model, model_rs_method


# ============================================================
# PLOTTING
# ============================================================

def _model_order(summary_df):
    models = [m for m in MODEL_INFO if m in summary_df['model'].values]
    avgs   = {m: summary_df[summary_df['model'] == m]['siaf_global'].mean()
              for m in models}
    return sorted(models, key=lambda m: avgs.get(m, np.nan))


def plot_siaf_bar(summary_df, obs_val, out_path):
    models = _model_order(summary_df)
    if not models:
        return

    x  = np.arange(len(models))
    w  = 0.6
    m_avg, m_err, bar_colors, n_mems = [], [], [], []
    for m in models:
        sub = summary_df[summary_df['model'] == m]['siaf_global'].dropna()
        m_avg.append(sub.mean() if len(sub) else np.nan)
        m_err.append(sub.std()  if len(sub) > 1 else 0.)
        bar_colors.append(MODEL_COLORS.get(m, '#888'))
        n_mems.append(len(sub))

    m_avg = np.array(m_avg)
    m_err = np.array(m_err)
    mmm   = float(np.nanmean(m_avg))

    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.1), 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    ax.bar(x, m_avg, w, yerr=m_err, color=bar_colors, alpha=0.85,
           capsize=4, edgecolor='white',
           error_kw=dict(linewidth=1.2, capthick=1.2), zorder=3)

    for xi, (avg, n) in enumerate(zip(m_avg, n_mems)):
        if np.isfinite(avg) and n > 1:
            ax.text(xi,
                    avg + (m_err[xi] if np.isfinite(m_err[xi]) else 0) + 0.0003,
                    f'n={n}', ha='center', va='bottom', fontsize=7,
                    color='#444')

    if obs_val is not None and np.isfinite(obs_val):
        ax.axhline(obs_val, color='#b5000a', lw=2.5, ls='--', alpha=0.8,
                   label=f'OBS = {obs_val:.3f}', zorder=4)

    ax.axhline(mmm, color='#333333', lw=1.8, ls=':', alpha=0.7,
               label=f'MMM = {mmm:.3f}', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Global Annual SIAF  (W m⁻² K⁻¹)', fontsize=11)
    ax.set_title(
        'CMIP6 SIAF  —  IS = −dα/dT from direct α~T OLS regression  '
        '(1982–2014)\n'
        'Pistone B2 / native-grid wedges / T07 RS  |  '
        'bars = per-model ensemble mean ± 1σ',
        fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.9)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_path}', flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print('=' * 70, flush=True)
    print('CMIP6 SIAF  —  IS = −dα/dT  (direct α~T OLS regression)',
          flush=True)
    print(f'  Trend window   : {TREND_YEARS[0]}–{TREND_YEARS[-1]}', flush=True)
    print(f'  CMIP6_ROOT     : {CMIP6_ROOT}', flush=True)
    print(f'  STAGE_ROOT     : {STAGE_ROOT}', flush=True)
    print(f'  N_WORKERS      : {N_WORKERS}', flush=True)
    print(f'  ARCTIC_LAT_MIN : {ARCTIC_LAT_MIN} N', flush=True)
    print(f'  FILL_THRESHOLD : {FILL_THRESHOLD:.0e}', flush=True)
    print('=' * 70, flush=True)

    all_rows   = []
    rs_methods = {}

    print('\n' + '=' * 70, flush=True)
    print('PROCESSING CMIP6 MODELS  (parallel)', flush=True)
    print('=' * 70, flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_model, m): m for m in MODEL_INFO}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                rows, model_name, rs_meth = fut.result()
                all_rows.extend(rows)
                rs_methods[model_name] = rs_meth
                print(f'  ✓ {model_name}: {len(rows)} member(s) completed',
                      flush=True)
            except Exception as e:
                print(f'  ✗ {model_name} FAILED: {e}', flush=True)

    summary_df = pd.DataFrame(all_rows)

    # Filter EC-Earth3 T255 members (non-standard high-res grid)
    summary_df_clean = summary_df[
        ~((summary_df['model'] == 'EC-Earth3') &
          (summary_df['grid_ny'] == 256) &
          (summary_df['grid_nx'] == 512))
    ].copy()

    csv_path = os.path.join(PLOT_DIR, 'siaf_cmip6_pistone_summary.csv')
    summary_df_clean.to_csv(csv_path, index=False)
    print(f'\nSummary CSV -> {csv_path}', flush=True)

    suspect = summary_df[summary_df['siaf_global'] > 0.35]
    if not suspect.empty:
        print(f'\n  ⚠️  Suspiciously high SIAF (>0.35) — check grid/data:')
        for _, r in suspect.iterrows():
            print(f'    {r.model}/{r.member}: siaf={r.siaf_global:.4f}  '
                  f'rs={r.rs_method}  grid=({r.grid_ny}×{r.grid_nx})')

    print(f'\n{"ENSEMBLE SUMMARY":-^70}', flush=True)
    print(f'  {"Model":<22} {"N":>4} {"T_yrs":>6} {"RS method":<18} '
          f'{"SIAF (glob dT)":>16}  {"SIAF (arct dT)":>16}',
          flush=True)
    for model in MODEL_INFO:
        sub = summary_df_clean[summary_df_clean['model'] == model]
        if sub.empty:
            continue
        b2     = sub['siaf_global'].dropna()
        b2_adT = sub['siaf_global_adT'].dropna()
        n_T    = int(sub['n_T_years'].mean()) if 'n_T_years' in sub else 0
        b2m    = b2.mean()
        b2s    = b2.std()       if len(b2)     > 1 else 0.
        b2m_a  = b2_adT.mean() if len(b2_adT) else np.nan
        b2s_a  = b2_adT.std()  if len(b2_adT) > 1 else 0.
        rsm    = rs_methods.get(model, '?')
        _adT_str = (f'{b2m_a:+.4f} +/- {b2s_a:.4f}'
                    if np.isfinite(b2m_a) else 'N/A')
        print(f'  {model:<22} {len(b2):>4} {n_T:>6} {rsm:<18} '
              f'{b2m:+.4f} +/- {b2s:.4f}   {_adT_str}',
              flush=True)

    per_model     = []
    per_model_adT = []
    for model in MODEL_INFO:
        sub = summary_df_clean[summary_df_clean['model'] == model]
        g   = sub['siaf_global'].dropna()
        a   = sub['siaf_global_adT'].dropna()
        if len(g):
            per_model.append(g.mean())
        if len(a):
            per_model_adT.append(a.mean())

    if per_model:
        print(f'\n  Multi-Model Mean (global dT) : {np.mean(per_model):+.4f} +/- '
              f'{np.std(per_model):.4f}  (1σ, {len(per_model)} models)',
              flush=True)
    if per_model_adT:
        print(f'  Multi-Model Mean (arctic dT) : {np.mean(per_model_adT):+.4f} +/- '
              f'{np.std(per_model_adT):.4f}  (1σ, {len(per_model_adT)} models)',
              flush=True)
    print(f'  [Each model contributes its ensemble mean regardless of N]',
          flush=True)

    print('\nGenerating plot...', flush=True)
    plot_siaf_bar(
        summary_df_clean, OBS_B2_GLOBAL,
        os.path.join(PLOT_DIR, 'siaf_cmip6_pistone_bar.png'))

    print('\nDONE :3', flush=True)