#!/usr/bin/env python
"""
plot_cmip6_sie_comparison.py
────────────────────────────
Compares Arctic Sea Ice Extent (SIE) trends across CMIP6 models and NOAA CDR
observations, then correlates SIE trend bias with SIAF values from the Pistone
summary CSV.

2026-05-06 coordinate robustness update
  • Ported full coordinate machinery from cmip6_siaf_pistone_only.py:
      - FILL_THRESHOLD = 1e10: fill values (CESM2: 9.97e36, NorESM2: ~1e30)
        are masked to NaN before any grid arithmetic.
      - _mask_fills(), _is_valid_lat(): validate and clean lat/lon arrays.
      - _std_latlon(): robust coordinate standardisation with fallback
        scanning for valid 2D lat (fixes MRI-ESM2-0 where 'latitude' is
        j-indices, not geographic lat) and 1D→2D broadcasting.
      - compute_sie_member now uses fill-masked coordinates for both the
        arctic_mask and the fallback area estimate, preventing garbage SIE
        values from curvilinear grid seam cells and fill-value cells.

Outputs (all to PLOT_DIR):
  sie_timeseries_{month}.png
  sie_trend_bar.png
  sie_earlymean_bar.png
  sie_trendbias_vs_siaf.png
  sie_earlybias_vs_siaf.png
  sie_trendbias_vs_extprop.png
  sie_siaf_summary_panel.png
  sie_summary.csv

Caching:
  Computed SIE results are pickled to PLOT_DIR/cache/ so re-runs skip the
  expensive NetCDF processing. Delete the cache files to force recomputation.

Author: J. Rotondo (updated 2026-05-06)
"""

import re, gc, os, glob, warnings, pickle
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CMIP6_SIAF_CSV = (
    '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
    'siaf_cmip6_pistone_summary.csv'
)
NOAA_CDR_DIR  = '/glade/derecho/scratch/jrotondo/SIC_data/'
NOAA_CDR_GLOB = os.path.join(NOAA_CDR_DIR, 'sic_psn25_*.nc')
NOAA_AREA_FILE = os.path.join(NOAA_CDR_DIR, 'psn25area_v3.nc')

CMIP6_ROOT = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT  = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

PLOT_DIR = (
    '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
    'sie_comparison/'
)
CACHE_DIR = os.path.join(PLOT_DIR, 'cache')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

N_WORKERS = 4

# ── Fill value threshold ────────────────────────────────────────────────────
# Anything with |value| > this is a NetCDF fill value, not a real coordinate.
# CESM2 uses 9.97e36, NorESM2 uses ~1e30.
FILL_THRESHOLD = 1e10

# ── Analysis parameters ─────────────────────────────────────────────────────
EARLY_YEARS   = list(range(1982, 1992))
TREND_YEARS   = list(range(1982, 2015))
MONTH_NUMS    = [4, 5, 6, 7, 8, 9]
MONTH_NAMES   = {4: 'April', 5: 'May', 6: 'June',
                 7: 'July',  8: 'August', 9: 'September'}
SIC_THRESHOLD  = 0.15
MIN_MEMBERS    = 3
FOCUS_RS_METHOD = 'full_T07'

# ── Known OBS SIAF values (from siaf_unified_diagnostics.csv, B2/CMIP6/p=0.05)
OBS_SIAF_TOT  = 0.2062
OBS_SIAF_EXT  = 0.1342
OBS_SIAF_DRK  = 0.0720
OBS_SIAF_ERR  = 0.0138   # 1σ

# ── Models to exclude from ALL analysis ────────────────────────────────────
EXCLUDED_MODELS = {'NorESM2-LM'}

# ── Model registry ───────────────────────────────────────────────────────────
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
    'NorESM2-LM'     : dict(institution='NCC',                 member='r1i1p1f1'),  # excluded
    'NorESM2-MM'     : dict(institution='NCC',                 member='r1i1p1f1'),
}

MODEL_COLORS = {
    'CESM2'          : '#1f77b4', 'E3SM-1-0'       : '#ff7f0e',
    'MPI-ESM1-2-HR'  : '#2ca02c', 'MPI-ESM1-2-LR'  : '#98df8a',
    'IPSL-CM6A-LR'   : '#d62728', 'MIROC6'         : '#9467bd',
    'MIROC-ES2L'     : '#c5b0d5', 'CanESM5'        : '#8c564b',
    'ACCESS-ESM1-5'  : '#e377c2', 'GFDL-ESM4'      : '#7f7f7f',
    'UKESM1-0-LL'    : '#17becf', 'HadGEM3-GC31-LL': '#aec7e8',
    'CNRM-CM6-1'     : '#ffbb78', 'BCC-CSM2-MR'    : '#ff9896',
    'ACCESS-CM2'     : '#bcbd22', 'CNRM-ESM2-1'    : '#dbdb8d',
    'EC-Earth3'      : '#e41a1c', 'EC-Earth3-Veg'  : '#f781bf',
    'GFDL-CM4'       : '#a65628', 'GISS-E2-1-H'    : '#b8e186',
    'INM-CM4-8'      : '#542788', 'INM-CM5-0'      : '#998ec3',
    'MRI-ESM2-0'     : '#f1a340', 'NorESM2-MM'     : '#5ab4ac',
}

COL_MODEL = '#3A6BAF'
COL_OBS   = '#b5000a'
COL_CNRM  = '#D95F1A'


# ============================================================
# COORDINATE UTILITIES  (ported from cmip6_siaf_pistone_only.py)
# ============================================================

def _mask_fills(arr):
    """Return float array with fill values (|val| > FILL_THRESHOLD) → NaN."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    return np.where(np.abs(arr) < FILL_THRESHOLD, arr, np.nan)


def _is_valid_lat(arr):
    """Return True if arr contains real geographic latitudes in [-90, 90]."""
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
                        lon_arr = _mask_fills(np.asarray(da[lon_name].values, dtype=float))
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
# CACHE HELPERS
# ============================================================

def _cache_path(name):
    return os.path.join(CACHE_DIR, f'{name}.pkl')

def save_cache(name, obj):
    with open(_cache_path(name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    print(f'  [cache] saved → {_cache_path(name)}', flush=True)

def load_cache(name):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        print(f'  [cache] loaded ← {p}', flush=True)
        return obj
    return None


# ============================================================
# FILE HELPERS
# ============================================================

def _files_cover_years(files, years):
    yr_set = set(years)
    for f in files:
        bn = os.path.basename(f)
        m  = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', bn)
        if m:
            s, e = int(m.group(1)[:4]), int(m.group(2)[:4])
            if any(s <= y <= e for y in yr_set):
                return True
        else:
            return True
    return False


def _prefer_regular_grid(files):
    if not files:
        return files
    bns   = [os.path.basename(f) for f in files]
    is_gr = lambda bn: ('_gr_' in bn) or ('_gr1_' in bn) or ('_gr2_' in bn)
    is_gn = lambda bn: '_gn_' in bn
    if any(is_gr(bn) for bn in bns) and any(is_gn(bn) for bn in bns):
        return [f for f, bn in zip(files, bns) if is_gr(bn)]
    return files


def find_siconc_files(inst, model, member):
    all_needed = set(TREND_YEARS)
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in ['SImon', 'SIday']:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, 'siconc')
            if not os.path.isdir(d):
                continue
            files = sorted(glob.glob(
                os.path.join(d, '**', 'siconc_*.nc'), recursive=True))
            if not files:
                continue
            files = _prefer_regular_grid(files)
            relevant = [f for f in files if _files_cover_years([f], all_needed)]
            if relevant:
                return relevant
    return []


def find_areacello_files(inst, model, member):
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in ['Ofx', 'fx']:
            for var in ['areacello', 'areacella']:
                d = os.path.join(root, inst, model, 'historical',
                                 member, table, var)
                if not os.path.isdir(d):
                    d = os.path.join(root, inst, model, 'piControl',
                                     member, table, var)
                if not os.path.isdir(d):
                    continue
                files = sorted(glob.glob(
                    os.path.join(d, '**', f'{var}_*.nc'), recursive=True))
                if files:
                    return files, var
    return [], None


# ============================================================
# NOAA CDR LOADER
# ============================================================

def _build_psn25_cell_area_km2():
    if os.path.exists(NOAA_AREA_FILE):
        try:
            with xr.open_dataset(NOAA_AREA_FILE) as ds:
                for v in ['cell_area', 'areacello', 'area']:
                    if v in ds:
                        arr = ds[v].values.astype(np.float64)
                        if arr.mean() > 1e6:
                            arr = arr / 1e6
                        print(f'  [CDR] cell area from file (mean={arr.mean():.1f} km²)',
                              flush=True)
                        return arr
        except Exception as e:
            print(f'  [CDR] area file read failed ({e}); using analytic', flush=True)

    try:
        from pyproj import Transformer
        PSN25_PROJ4 = (
            '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
            '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
        nx, ny = 304, 448
        x0, y0 = -3850e3, -5350e3
        xs = x0 + np.arange(nx) * 25e3
        ys = y0 + np.arange(ny) * 25e3
        xx, yy = np.meshgrid(xs, ys)
        tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
        _, lat2d = tfm.transform(xx, yy)
        phi_ts = np.deg2rad(70.)
        k2 = ((1. + np.sin(phi_ts)) /
              np.maximum(1. + np.sin(np.deg2rad(lat2d)), 1e-9)) ** 2
        area_km2 = (625.0 / k2).astype(np.float64)
        print(f'  [CDR] cell area analytic (mean={area_km2.mean():.1f} km²)', flush=True)
        return area_km2
    except Exception as e:
        print(f'  [CDR] analytic area failed ({e}); using 625 km²', flush=True)
        return np.full((448, 304), 625.0, dtype=np.float64)


def load_noaa_cdr_sie(month_nums=MONTH_NUMS, years=TREND_YEARS):
    """Load NOAA CDR daily SIC → monthly Arctic SIE (10⁶ km²)."""
    files = sorted(glob.glob(NOAA_CDR_GLOB))
    if not files:
        print(f'  [CDR] No files: {NOAA_CDR_GLOB}', flush=True)
        return None
    print(f'  [CDR] {len(files)} daily files found', flush=True)

    cell_area_km2 = _build_psn25_cell_area_km2()

    SIC_VARS = ['cdr_seaice_conc', 'cdr_seaice_conc_monthly',
                'goddard_merged_seaice_conc', 'sea_ice_conc']
    sic_var = None
    for f in files[:5]:
        try:
            with xr.open_dataset(f, mask_and_scale=True) as ds:
                for v in SIC_VARS:
                    if v in ds:
                        sic_var = v
                        break
                if sic_var is None:
                    sic_var = next(
                        (v for v in ds.data_vars
                         if ds[v].dtype in (np.float32, np.float64)), None)
            if sic_var:
                break
        except Exception:
            continue
    if sic_var is None:
        print('  [CDR] Cannot determine SIC variable — abort', flush=True)
        return None
    print(f'  [CDR] SIC variable: {sic_var}', flush=True)

    yr_set, mn_set = set(years), set(month_nums)
    groups = defaultdict(list)

    for f in files:
        try:
            with xr.open_dataset(f, mask_and_scale=False) as ds:
                t_vals = ds['time'].values
                if t_vals.ndim == 0:
                    t_vals = [t_vals.item()]
                for tv in t_vals:
                    try:
                        ts = pd.Timestamp(tv)
                    except Exception:
                        if hasattr(tv, 'year'):
                            ts = pd.Timestamp(year=tv.year, month=tv.month,
                                              day=getattr(tv, 'day', 1))
                        else:
                            continue
                    if ts.year in yr_set and ts.month in mn_set:
                        groups[(ts.year, ts.month)].append(f)
                        break
        except Exception:
            continue

    if not groups:
        print('  [CDR] No files matched requested years/months', flush=True)
        return None
    print(f'  [CDR] {len(groups)} (year, month) groups', flush=True)

    try:
        from pyproj import Transformer
        PSN25_PROJ4 = (
            '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
            '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
        xs = -3850e3 + np.arange(304) * 25e3
        ys = -5350e3 + np.arange(448) * 25e3
        xx, yy = np.meshgrid(xs, ys)
        tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
        _, psn_lat2d = tfm.transform(xx, yy)
        arctic_psn = psn_lat2d >= 60.0
    except Exception:
        arctic_psn = np.ones(cell_area_km2.shape, dtype=bool)

    sie_dict = {mn: {} for mn in month_nums}

    for (yr, mn), flist in sorted(groups.items()):
        daily_maps = []
        for f in flist:
            try:
                with xr.open_dataset(f, mask_and_scale=True) as ds:
                    raw = ds[sic_var].squeeze().values.astype(np.float64)
            except Exception:
                continue
            if raw.ndim > 2:
                raw = raw[0]
            if np.nanmax(raw) > 1.5:
                raw = raw / 100.0
            raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
            daily_maps.append(raw)

        if not daily_maps:
            continue
        sic_monthly = np.nanmean(daily_maps, axis=0)

        if cell_area_km2.shape != sic_monthly.shape:
            cell_area_km2 = np.full(sic_monthly.shape, 625.0, dtype=np.float64)
            arctic_psn    = np.ones(sic_monthly.shape, dtype=bool)

        ice = arctic_psn & np.isfinite(sic_monthly) & (sic_monthly >= SIC_THRESHOLD)
        sie_dict[mn][yr] = float(np.sum(cell_area_km2[ice])) / 1e6

    result = {}
    for mn in month_nums:
        result[mn] = pd.Series(sie_dict[mn]).sort_index() if sie_dict[mn] else None

    n_ok = sum(v is not None for v in result.values())
    print(f'  [CDR] SIE for {n_ok}/{len(month_nums)} months', flush=True)
    return result


# ============================================================
# CMIP6 SIE COMPUTATION  (robust coordinate handling)
# ============================================================

def compute_sie_member(model, member):
    """
    Compute monthly Arctic SIE (10⁶ km²) for one CMIP6 ensemble member.

    Uses the same coordinate machinery as cmip6_siaf_pistone_only.py:
    - Fill values masked before any grid arithmetic
    - Fallback lat scanning for curvilinear grids
    - 1D → 2D lat broadcasting for regular grids
    - areacello preferred; robust fallback for curvilinear grids
    """
    inst = MODEL_INFO[model]['institution']
    sic_files = find_siconc_files(inst, model, member)
    if not sic_files:
        return None

    # ── Open siconc ───────────────────────────────────────────────────────────
    try:
        ds_sic = xr.open_mfdataset(
            sic_files, combine='by_coords', use_cftime=True, chunks={'time': 12})
    except Exception:
        try:
            ds_sic = xr.open_mfdataset(
                sic_files, combine='nested', concat_dim='time',
                coords='minimal', compat='override', use_cftime=True,
                chunks={'time': 12})
        except Exception as e:
            print(f'  [{model}/{member}] open failed: {e}', flush=True)
            return None

    da_sic = ds_sic['siconc']

    # ── Drop duplicate time steps (matches siaf pipeline dedup) ──────────────
    tvals = da_sic.time.values
    _, first_occ = np.unique(tvals, return_index=True)
    if len(first_occ) < len(tvals):
        da_sic = da_sic.isel(time=np.sort(first_occ))

    # ── Robust coordinate extraction ──────────────────────────────────────────
    da_sic, lat, lon, is_2d = _std_latlon(da_sic)

    if lat is None:
        print(f'  [{model}/{member}] no valid lat — skip', flush=True)
        ds_sic.close()
        return None

    # Build 2D lat/lon arrays (fill-masked NaN → 0° so they miss arctic_mask)
    if is_2d:
        lat2d = np.where(np.isfinite(lat), lat, 0.).astype(np.float64)
        lon2d = np.where(np.isfinite(lon), lon, 0.).astype(np.float64) \
                if lon is not None else np.zeros_like(lat2d)
    else:
        lat_clean = np.where(np.isfinite(lat), lat, 0.).astype(np.float64)
        lon_clean = np.where(np.isfinite(lon), lon, 0.).astype(np.float64) \
                    if lon is not None else np.zeros(1, dtype=np.float64)
        lon2d, lat2d = np.meshgrid(lon_clean, lat_clean)

    arctic_mask = lat2d >= 60.0

    # ── Cell area ─────────────────────────────────────────────────────────────
    area_km2 = None
    area_files, area_var = find_areacello_files(inst, model, member)
    if area_files:
        try:
            with xr.open_dataset(area_files[0]) as ds_a:
                area_raw = ds_a[area_var].values.astype(np.float64)
            if area_raw.shape == lat2d.shape:
                # areacello is in m²; convert to km²
                area_km2 = area_raw / 1e6
                # Mask any fill values in the area field
                area_km2 = np.where(np.abs(area_km2) < FILL_THRESHOLD,
                                    area_km2, np.nan)
                print(f'  [{model}/{member}] areacello loaded '
                      f'(mean={np.nanmean(area_km2):.1f} km²)', flush=True)
            else:
                print(f'  [{model}/{member}] areacello shape '
                      f'{area_raw.shape} != grid {lat2d.shape}; using fallback',
                      flush=True)
        except Exception as e:
            print(f'  [{model}/{member}] area load failed ({e}); using fallback',
                  flush=True)

    if area_km2 is None:
        # Fallback: cos-weighted cell area using fill-masked lat2d
        # For regular grids: dlat and dlon from the 1D arrays
        # For curvilinear: estimate local spacing from adjacent rows/cols
        # Fallback: cos-weighted cell area using fill-masked lat2d
        if is_2d:
            lat_clean_2d = np.where(np.isfinite(lat), lat, np.nan)
            # Use median of absolute diffs to prevent cancellation and ignore seam outliers
            dlat = float(np.nanmedian(np.abs(np.diff(lat_clean_2d, axis=0))))
            
            if lon is not None:
                lon_clean_2d = np.where(np.isfinite(lon), lon, np.nan)
                lon_diff = np.abs(np.diff(lon_clean_2d, axis=1))
                # Handle longitude wrap-around (e.g., 359 to 0 is a 1° jump, not 359°)
                lon_diff = np.where(lon_diff > 180, 360.0 - lon_diff, lon_diff)
                dlon = float(np.nanmedian(lon_diff))
            else:
                dlon = 1.0
                
            if not (np.isfinite(dlat) and dlat > 0): dlat = 1.0
            if not (np.isfinite(dlon) and dlon > 0): dlon = 1.0
        else:
            dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if len(lat) > 1 else 1.0
            if lon is not None and len(lon) > 1:
                lon_diff = np.abs(np.diff(lon))
                lon_diff = np.where(lon_diff > 180, 360.0 - lon_diff, lon_diff)
                dlon = float(np.nanmedian(lon_diff))
            else:
                dlon = 1.0

        # cos(lat) * dlat * dlon in km²
        lat2d_cos = np.where(np.isfinite(lat2d), lat2d, 0.)
        area_km2 = np.maximum(
            np.cos(np.deg2rad(lat2d_cos)) * dlat * dlon * (111.32 ** 2), 0.0)
        print(f'  [{model}/{member}] fallback area '
              f'(dlat={dlat:.3f}° dlon={dlon:.3f}° '
              f'mean_arctic={float(np.nanmean(area_km2[arctic_mask])):.0f} km²)',
              flush=True)

    # ── Time filtering ────────────────────────────────────────────────────────
    yr_set, mn_set = set(TREND_YEARS), set(MONTH_NUMS)
    try:
        yr_arr = np.array([int(t.dt.year)  for t in da_sic.time])
        mn_arr = np.array([int(t.dt.month) for t in da_sic.time])
    except Exception:
        yr_arr = np.array([t.year  for t in da_sic.time.values])
        mn_arr = np.array([t.month for t in da_sic.time.values])

    time_mask = np.array([y in yr_set and m in mn_set
                           for y, m in zip(yr_arr, mn_arr)])
    if not time_mask.any():
        ds_sic.close()
        return None

    da_sub   = da_sic.isel(time=time_mask).load()
    ds_sic.close()
    yr_sub   = yr_arr[time_mask]
    mn_sub   = mn_arr[time_mask]
    sic_vals = da_sub.values.astype(np.float64)
    if np.nanmax(sic_vals) > 1.5:
        sic_vals /= 100.0

    # ── SIE accumulation ─────────────────────────────────────────────────────
    sie_dict = {mn: {} for mn in MONTH_NUMS}
    for i in range(sic_vals.shape[0]):
        yr, mn = int(yr_sub[i]), int(mn_sub[i])
        sic_i = sic_vals[i]
        ice   = (arctic_mask
                 & np.isfinite(sic_i)
                 & (sic_i >= SIC_THRESHOLD)
                 & np.isfinite(area_km2)
                 & (area_km2 > 0))
        sie_dict[mn][yr] = float(np.sum(area_km2[ice])) / 1e6

    del da_sub, sic_vals
    gc.collect()

    return {mn: (pd.Series(sie_dict[mn]).sort_index() if sie_dict[mn] else None)
            for mn in MONTH_NUMS}


# ============================================================
# TREND CALCULATION
# ============================================================

def _ols_trend(series, y0, y1):
    s = series.dropna()
    s = s[(s.index >= y0) & (s.index <= y1)]
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    res = stats.linregress(s.index.values.astype(float), s.values)
    return float(res.slope), float(res.slope * (y1 - y0)), float(res.pvalue)


def sie_stats(sie_by_month, early_years=EARLY_YEARS,
              trend_y0=TREND_YEARS[0], trend_y1=TREND_YEARS[-1]):
    out = {}
    for mn in MONTH_NUMS:
        s = sie_by_month.get(mn)
        if s is None or s.empty:
            out.update({f'm{mn}_early_mean': np.nan, f'm{mn}_trend_slope': np.nan,
                        f'm{mn}_trend_change': np.nan, f'm{mn}_trend_pval': np.nan})
            continue
        ev = s[s.index.isin(early_years)]
        out[f'm{mn}_early_mean']    = float(ev.mean()) if len(ev) else np.nan
        sl, ch, pv = _ols_trend(s, trend_y0, trend_y1)
        out[f'm{mn}_trend_slope']  = sl
        out[f'm{mn}_trend_change'] = ch
        out[f'm{mn}_trend_pval']   = pv
    return out


# ============================================================
# WORKER
# ============================================================

def process_model_sie(model):
    if model in EXCLUDED_MODELS:
        return []
    inst    = MODEL_INFO[model]['institution']
    rows    = []
    members = set()
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        base = os.path.join(root, inst, model, 'historical')
        if not os.path.isdir(base):
            continue
        for mem in sorted(os.listdir(base)):
            if find_siconc_files(inst, model, mem):
                members.add(mem)
    members = sorted(members)
    if not members:
        return rows

    print(f'\n{"─"*55}\nModel: {model}  ({len(members)} member(s))', flush=True)

    for mem in members:
        try:
            sie = compute_sie_member(model, mem)
        except Exception as e:
            print(f'  [{model}/{mem}] SIE failed: {e}', flush=True)
            continue
        if sie is None:
            continue
        n_ok = sum(v is not None for v in sie.values())
        if n_ok < 3:
            continue
        st  = sie_stats(sie)
        row = dict(model=model, member=mem, **st)
        rows.append(row)
        print(f'  [{model}/{mem}] ok  '
              f"Apr_early={st.get('m4_early_mean', np.nan):.2f}  "
              f"Sep_trend={st.get('m9_trend_change', np.nan):+.3f}", flush=True)
    return rows


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _season_mean(row, key_prefix):
    vals = [row.get(f'm{mn}_{key_prefix}', np.nan) for mn in MONTH_NUMS]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.nanmean(finite)) if finite else np.nan


def _ols_line(ax, xs, ys, label_sfx='', color='#222', ls='--'):
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 3:
        return None, None
    r, p = stats.pearsonr(xs[ok], ys[ok])
    sl, b = np.polyfit(xs[ok], ys[ok], 1)
    xr = np.linspace(xs[ok].min(), xs[ok].max(), 100)
    ax.plot(xr, sl * xr + b, color=color, linestyle=ls, linewidth=1.8,
            label=f'r = {r:.2f}, p = {p:.3f}{label_sfx}', zorder=3)
    return r, p


def _style_ax(ax):
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.xaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


def plot_sie_trend_bar(model_means, obs_trend, out_path):
    models = sorted(model_means, key=lambda m: model_means[m]['trend_change_mean'])
    n = len(models) + 1
    x = np.arange(n)
    w = 0.7
    labels = [f"{m}\n(n={model_means[m]['n']})" for m in models] + ['OBS\n(NOAA CDR)']
    vals   = [model_means[m]['trend_change_mean'] for m in models] + [obs_trend]
    errs   = [model_means[m]['trend_change_std']  for m in models] + [0.0]
    cols   = [MODEL_COLORS.get(m, '#888') for m in models] + [COL_OBS]

    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 5.5))
    fig.patch.set_facecolor('#FAFAFA');  ax.set_facecolor('#FAFAFA')
    bars = ax.bar(x, vals, w, color=cols, alpha=0.85, edgecolor='white',
                  linewidth=0.8, zorder=3)
    ax.errorbar(x[:-1], vals[:-1], yerr=errs[:-1],
                fmt='none', color='#222', capsize=4, linewidth=1.4,
                capthick=1.4, zorder=4)
    bars[-1].set_edgecolor(COL_OBS); bars[-1].set_linewidth(2.5)
    ax.axhline(0, color='#333', linewidth=0.8, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.get_xticklabels()[-1].set_color(COL_OBS)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_ylabel('SIE change, 1982–2014  (10⁶ km²)', fontsize=11)
    ax.set_title('Arctic SIE Trend: CMIP6 vs NOAA CDR (Apr–Sep mean)\n'
                 'bars = ensemble mean ± 1σ', fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {out_path}', flush=True)


def plot_sie_earlymean_bar(model_means, obs_early, out_path):
    models = sorted(model_means, key=lambda m: model_means[m]['early_mean_mean'])
    n = len(models) + 1
    x = np.arange(n)
    w = 0.7
    labels = [f"{m}\n(n={model_means[m]['n']})" for m in models] + ['OBS\n(NOAA CDR)']
    vals   = [model_means[m]['early_mean_mean'] for m in models] + [obs_early]
    errs   = [model_means[m]['early_mean_std']  for m in models] + [0.0]
    cols   = [MODEL_COLORS.get(m, '#888') for m in models] + [COL_OBS]

    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 5.5))
    fig.patch.set_facecolor('#FAFAFA');  ax.set_facecolor('#FAFAFA')
    bars = ax.bar(x, vals, w, color=cols, alpha=0.85, edgecolor='white',
                  linewidth=0.8, zorder=3)
    ax.errorbar(x[:-1], vals[:-1], yerr=errs[:-1],
                fmt='none', color='#222', capsize=4, linewidth=1.4,
                capthick=1.4, zorder=4)
    bars[-1].set_edgecolor(COL_OBS); bars[-1].set_linewidth(2.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.get_xticklabels()[-1].set_color(COL_OBS)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_ylabel('Arctic SIE  (10⁶ km²)', fontsize=11)
    ax.set_title('Arctic SIE Early Period (1982–1991): CMIP6 vs NOAA CDR (Apr–Sep mean)\n'
                 'bars = ensemble mean ± 1σ', fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {out_path}', flush=True)


def plot_sie_timeseries(model_means, obs_sie_by_month, out_dir):
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#FAFAFA');  ax.set_facecolor('#FAFAFA')
        for m, d in model_means.items():
            t0 = TREND_YEARS[0];  t1 = TREND_YEARS[-1]
            e  = d.get(f'm{mn}_early_mean', np.nan)
            sl = d.get(f'm{mn}_trend_slope', np.nan)
            if np.isfinite(e) and np.isfinite(sl):
                yy  = np.array([t0, t1], dtype=float)
                sie = e + sl * (yy - float(np.mean(list(EARLY_YEARS))))
                ax.plot(yy, sie, color=MODEL_COLORS.get(m, '#888'),
                        alpha=0.6, linewidth=1.0, label=m)
        if obs_sie_by_month and obs_sie_by_month.get(mn) is not None:
            obs_s = obs_sie_by_month[mn]
            ax.plot(obs_s.index, obs_s.values,
                    color=COL_OBS, linewidth=2.5, zorder=5, label='NOAA CDR')
        ax.set_title(f'Arctic SIE — {mname} (1982–2014)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('SIE  (10⁶ km²)', fontsize=10)
        _style_ax(ax)
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, lbls, fontsize=6.5, ncol=3,
                      loc='upper right', framealpha=0.85)
        plt.tight_layout()
        out = os.path.join(out_dir, f'sie_timeseries_{mname.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  → {out}', flush=True)


def _scatter_plot(ax, xs, ys, labels, colors,
                  obs_xy=None, obs_label='OBS',
                  highlight='CNRM-CM6-1'):
    xs = np.array(xs, dtype=float);  ys = np.array(ys, dtype=float)
    hi = np.array([l == highlight for l in labels])
    ax.scatter(xs[~hi], ys[~hi],
               c=[c for c, m in zip(colors, ~hi) if m],
               s=80, edgecolors='white', linewidths=0.8, zorder=4)
    for xi, yi, lbl in zip(xs[~hi], ys[~hi], [l for l, m in zip(labels, ~hi) if m]):
        ax.annotate(lbl, (xi, yi), textcoords='offset points',
                    xytext=(4, 3), fontsize=7, zorder=5)
    if hi.any():
        ax.scatter(xs[hi], ys[hi], c=COL_CNRM, s=100,
                   edgecolors='white', linewidths=1.2, zorder=5, label=highlight)
        for xi, yi, lbl in zip(xs[hi], ys[hi], [l for l, m in zip(labels, hi) if m]):
            ax.annotate(lbl, (xi, yi), textcoords='offset points',
                        xytext=(4, 3), fontsize=7, color=COL_CNRM, zorder=6)
    _ols_line(ax, xs, ys, label_sfx=' (all)', color='gray', ls=':')
    if hi.any():
        _ols_line(ax, xs[~hi], ys[~hi],
                  label_sfx=f' (excl. {highlight})', color='#222', ls='--')
    if obs_xy is not None:
        ox, oy = obs_xy
        if np.isfinite(ox) and np.isfinite(oy):
            ax.scatter([ox], [oy], marker='*', s=280, c=COL_OBS,
                       edgecolors='#800000', linewidths=0.8,
                       zorder=7, label=obs_label)


def plot_scatter_standalone(xs, ys, labels, colors,
                            xlabel, ylabel, title, out_path,
                            obs_xy=None, obs_label='OBS',
                            highlight='CNRM-CM6-1'):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FAFAFA');  ax.set_facecolor('#FAFAFA')
    _scatter_plot(ax, xs, ys, labels, colors,
                  obs_xy=obs_xy, obs_label=obs_label, highlight=highlight)
    ax.set_xlabel(xlabel, fontsize=11);  ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    _style_ax(ax)
    ax.legend(fontsize=9, framealpha=0.9, loc='best')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {out_path}', flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print('=' * 70, flush=True)
    print('SIE COMPARISON: CMIP6 historical vs NOAA CDR', flush=True)
    print(f'Excluded models : {sorted(EXCLUDED_MODELS)}', flush=True)
    print(f'FILL_THRESHOLD  : {FILL_THRESHOLD:.0e}', flush=True)
    print('=' * 70, flush=True)

    # ── 1. Load SIAF summary ──────────────────────────────────────────────────
    siaf_df = None
    if os.path.exists(CMIP6_SIAF_CSV):
        siaf_df = pd.read_csv(CMIP6_SIAF_CSV)
        siaf_df = siaf_df[siaf_df['rs_method'] == FOCUS_RS_METHOD].copy()
        counts  = siaf_df.groupby('model').size()
        siaf_df = siaf_df[siaf_df['model'].isin(
            counts[counts >= MIN_MEMBERS].index)].copy()
        siaf_df = siaf_df[~siaf_df['model'].isin(EXCLUDED_MODELS)].copy()
        print(f'SIAF CSV: {len(siaf_df)} rows, '
              f'{siaf_df["model"].nunique()} models', flush=True)
    else:
        print(f'WARNING: SIAF CSV not found — scatter vs SIAF skipped', flush=True)

    # ── 2. NOAA CDR SIE ───────────────────────────────────────────────────────
    obs_sie_by_month = load_cache('obs_sie_by_month')
    if obs_sie_by_month is None:
        print('\nLoading NOAA CDR SIE (no cache)...', flush=True)
        obs_sie_by_month = load_noaa_cdr_sie(MONTH_NUMS, TREND_YEARS)
        if obs_sie_by_month is not None:
            save_cache('obs_sie_by_month', obs_sie_by_month)
    else:
        print('NOAA CDR SIE: loaded from cache', flush=True)

    obs_sie_stats = {}
    obs_seas_early = obs_seas_trend = np.nan

    if obs_sie_by_month:
        for mn in MONTH_NUMS:
            s = obs_sie_by_month.get(mn)
            if s is not None and len(s) >= 5:
                sl, ch, pv = _ols_trend(s, TREND_YEARS[0], TREND_YEARS[-1])
                e_mean = float(s[s.index.isin(EARLY_YEARS)].mean())
                obs_sie_stats[mn] = dict(trend_slope=sl, trend_change=ch,
                                         trend_pval=pv, early_mean=e_mean)
                print(f'  OBS {MONTH_NAMES[mn]:9s}: '
                      f'early={e_mean:.2f}  trend={ch:+.3f}  p={pv:.3f}',
                      flush=True)
        obs_seas_early = float(np.nanmean(
            [obs_sie_stats[mn]['early_mean']   for mn in MONTH_NUMS
             if mn in obs_sie_stats]))
        obs_seas_trend = float(np.nanmean(
            [obs_sie_stats[mn]['trend_change'] for mn in MONTH_NUMS
             if mn in obs_sie_stats]))
        print(f'\n  OBS seasonal: early={obs_seas_early:.3f}  '
              f'trend={obs_seas_trend:+.3f}', flush=True)
    else:
        print('  WARNING: NOAA CDR not loaded.', flush=True)

    # ── 3. CMIP6 SIE  (delete cache to recompute with fixed coordinates) ─────
    # NOTE: The cache key is 'sie_df'. If a stale cache exists from the old
    # code (without curvilinear fixes), DELETE it before running:
    #   rm <CACHE_DIR>/sie_df.pkl
    sie_df = load_cache('sie_df')
    if sie_df is None:
        print('\n' + '='*70, flush=True)
        print('Computing CMIP6 SIE (parallel, robust coordinates)...', flush=True)
        all_rows = []
        active_models = {m: v for m, v in MODEL_INFO.items()
                         if m not in EXCLUDED_MODELS}
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(process_model_sie, m): m
                       for m in active_models}
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                    print(f'  ✓ {model}: {len(rows)} member(s)', flush=True)
                except Exception as e:
                    print(f'  ✗ {model}: {e}', flush=True)

        sie_df = pd.DataFrame(all_rows)
        if not sie_df.empty:
            save_cache('sie_df', sie_df)
    else:
        sie_df = sie_df[~sie_df['model'].isin(EXCLUDED_MODELS)].copy()
        print(f'CMIP6 SIE: loaded from cache '
              f'({sie_df["model"].nunique()} models, {len(sie_df)} members)',
              flush=True)

    if sie_df is None or sie_df.empty:
        print('ERROR: No CMIP6 SIE data. Check paths or delete cache.',
              flush=True)
        exit(1)

    # Save per-member CSV
    csv_path = os.path.join(PLOT_DIR, 'sie_summary.csv')
    sie_df.to_csv(csv_path, index=False)
    print(f'\nPer-member CSV → {csv_path}', flush=True)

    # ── 4. Per-model ensemble means ───────────────────────────────────────────
    model_means = {}
    for m in sie_df['model'].unique():
        sub = sie_df[sie_df['model'] == m]
        d   = {'n': len(sub)}
        tc  = np.array([_season_mean(r, 'trend_change') for _, r in sub.iterrows()])
        em  = np.array([_season_mean(r, 'early_mean')   for _, r in sub.iterrows()])
        d['trend_change_mean'] = float(np.nanmean(tc))
        d['trend_change_std']  = float(np.nanstd(tc))  if len(tc) > 1 else 0.
        d['early_mean_mean']   = float(np.nanmean(em))
        d['early_mean_std']    = float(np.nanstd(em))  if len(em) > 1 else 0.
        for mn in MONTH_NUMS:
            for stat in ['early_mean', 'trend_slope', 'trend_change']:
                vals = sub[f'm{mn}_{stat}'].values.astype(float)
                d[f'm{mn}_{stat}'] = float(np.nanmean(vals))
        model_means[m] = d

    print(f'\n{"MODEL":<22} {"N":>4} {"Early (Mkm²)":>14} {"Trend (Mkm²)":>14}',
          flush=True)
    for m in sorted(model_means, key=lambda x: model_means[x]['trend_change_mean']):
        d = model_means[m]
        print(f'  {m:<22} {d["n"]:>4} '
              f'{d["early_mean_mean"]:>14.3f} {d["trend_change_mean"]:>+14.3f}',
              flush=True)
    print(f'  {"OBS (CDR)":<22} {"—":>4} '
          f'{obs_seas_early:>14.3f} {obs_seas_trend:>+14.3f}', flush=True)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print('\nGenerating plots...', flush=True)

    plot_sie_trend_bar(model_means, obs_seas_trend,
                       os.path.join(PLOT_DIR, 'sie_trend_bar.png'))

    plot_sie_earlymean_bar(model_means, obs_seas_early,
                           os.path.join(PLOT_DIR, 'sie_earlymean_bar.png'))

    plot_sie_timeseries(model_means, obs_sie_by_month, PLOT_DIR)

    # ── 6. Scatter: SIE bias vs SIAF ─────────────────────────────────────────
    if siaf_df is not None and not sie_df.empty:
        print('\nSIAF ↔ SIE scatter plots...', flush=True)

        siaf_model = {}
        for m in siaf_df['model'].unique():
            sub = siaf_df[siaf_df['model'] == m]
            st  = float(sub['siaf_global'].mean())
            ex  = float(sub['siaf_ext_global'].mean())
            siaf_model[m] = dict(
                siaf_tot = st,
                siaf_ext = ex,
                ext_prop = (ex / st) if st > 0 else np.nan,
            )

        common = sorted(set(model_means) & set(siaf_model))
        print(f'  {len(common)} models in both datasets', flush=True)

        if len(common) >= 4:
            xs_tc   = np.array([model_means[m]['trend_change_mean'] for m in common])
            xs_em   = np.array([model_means[m]['early_mean_mean']   for m in common])
            ys_siaf = np.array([siaf_model[m]['siaf_tot']           for m in common])
            ys_prop = np.array([siaf_model[m]['ext_prop']           for m in common])
            lbls    = common
            cols    = [MODEL_COLORS.get(m, '#888') for m in common]
            obs_ext_prop = OBS_SIAF_EXT / OBS_SIAF_TOT

            plot_scatter_standalone(
                xs_tc, ys_siaf, lbls, cols,
                xlabel='SIE trend, 1982–2014  (10⁶ km²)',
                ylabel='Total SIAF  (W m⁻² K⁻¹)',
                title='SIE Trend vs Total SIAF',
                out_path=os.path.join(PLOT_DIR, 'sie_trendbias_vs_siaf.png'),
                obs_xy=(obs_seas_trend, OBS_SIAF_TOT),
                obs_label=f'OBS  (CDR trend, SIAF={OBS_SIAF_TOT:.4f})',
            )
            plot_scatter_standalone(
                xs_em, ys_siaf, lbls, cols,
                xlabel='Early-period SIE, 1982–1991  (10⁶ km²)',
                ylabel='Total SIAF  (W m⁻² K⁻¹)',
                title='Early-Period SIE vs Total SIAF',
                out_path=os.path.join(PLOT_DIR, 'sie_earlybias_vs_siaf.png'),
                obs_xy=(obs_seas_early, OBS_SIAF_TOT),
                obs_label=f'OBS  (CDR early SIE, SIAF={OBS_SIAF_TOT:.4f})',
            )
            plot_scatter_standalone(
                xs_em, ys_prop, lbls, cols,
                xlabel='Early-period SIE, 1982–1991  (10⁶ km²)',
                ylabel='Extent proportion  (SIAFext / SIAFtot)',
                title='Early-Period SIE vs SIAF Extent Proportion',
                out_path=os.path.join(PLOT_DIR, 'sie_earlybias_vs_extprop.png'),
                obs_xy=(obs_seas_early, obs_ext_prop),
                obs_label=f'OBS  ({obs_ext_prop:.2f})',
            )
            plot_scatter_standalone(
                xs_tc, ys_prop, lbls, cols,
                xlabel='SIE trend, 1982–2014  (10⁶ km²)',
                ylabel='Extent proportion  (SIAFext / SIAFtot)',
                title='SIE Trend vs SIAF Extent Proportion',
                out_path=os.path.join(PLOT_DIR, 'sie_trendbias_vs_extprop.png'),
                obs_xy=(obs_seas_trend, obs_ext_prop),
                obs_label=f'OBS  ({obs_ext_prop:.2f})',
            )

            # ── 2×2 summary panel ─────────────────────────────────────────────
            fig_c, axes = plt.subplots(2, 2, figsize=(13, 10))
            fig_c.patch.set_facecolor('#FAFAFA')
            fig_c.suptitle(
                'CMIP6 Arctic SIE Bias vs SIAF Structure  '
                '(red ★ = OBS)',
                fontsize=13, fontweight='bold', y=1.01)

            panel_specs = [
                (axes[0, 0], xs_tc, ys_siaf,
                 (obs_seas_trend, OBS_SIAF_TOT),
                 'SIE trend  (10⁶ km²)', 'Total SIAF  (W m⁻² K⁻¹)',
                 'SIE Trend vs Total SIAF'),
                (axes[0, 1], xs_em, ys_siaf,
                 (obs_seas_early, OBS_SIAF_TOT),
                 'Early SIE  (10⁶ km²)', 'Total SIAF  (W m⁻² K⁻¹)',
                 'Early SIE vs Total SIAF'),
                (axes[1, 0], xs_tc, ys_prop,
                 (obs_seas_trend, obs_ext_prop),
                 'SIE trend  (10⁶ km²)', 'Extent proportion',
                 'SIE Trend vs Extent Proportion'),
                (axes[1, 1], xs_em, ys_prop,
                 (obs_seas_early, obs_ext_prop),
                 'Early SIE  (10⁶ km²)', 'Extent proportion',
                 'Early SIE vs Extent Proportion'),
            ]

            for ax_p, xs, ys, obs_xy, xl, yl, ttl in panel_specs:
                ax_p.set_facecolor('#FAFAFA')
                _scatter_plot(ax_p, xs, ys, lbls, cols,
                              obs_xy=obs_xy, obs_label='OBS')
                ax_p.set_xlabel(xl, fontsize=9)
                ax_p.set_ylabel(yl, fontsize=9)
                ax_p.set_title(ttl, fontsize=10, fontweight='bold')
                ax_p.legend(fontsize=7.5, framealpha=0.85)
                _style_ax(ax_p)

            plt.tight_layout()
            p_combo = os.path.join(PLOT_DIR, 'sie_siaf_summary_panel.png')
            fig_c.savefig(p_combo, dpi=200, bbox_inches='tight',
                          facecolor=fig_c.get_facecolor())
            plt.close(fig_c)
            print(f'  → {p_combo}', flush=True)
        else:
            print('  Not enough common models for scatter plots.', flush=True)

    print('\nDONE', flush=True)
    print(f'Outputs: {PLOT_DIR}', flush=True)
    print(f'Cache:   {CACHE_DIR}  (delete sie_df.pkl to force recompute)',
          flush=True)