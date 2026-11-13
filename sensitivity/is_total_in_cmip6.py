#!/usr/bin/env python
"""
cmip6_IS_total_comparison.py
-----------------------------
Computes IS_total = -(α_late - α_early) / ΔT  on ALL ice pixels
(sic_max >= 0.15, lat >= 60°) — no SIC-trend significance filter.

This is directly comparable to the obs IS_total = IS_ext + IS_dark
from fig8_donohoe_recreation_pval.py, without requiring a significant
SIC decrease assumption that differs between models and observations.

Changes from cmip6_IS_dark_comparison_fullspatial.py:
  - IS scalar = mean over ALL ice pixels (no ~changing filter)
  - OBS IS_total blends CLARA + APP-x on all valid ice pixels
  - Spatial maps show IS_total (full ice coverage)
  - All plot labels updated to IS_total
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from scipy.stats import t as t_dist
from datetime import datetime as _dt
from netCDF4 import Dataset as NC4Dataset
import glob, os, gc, re, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

CMIP6_ROOT     = '/glade/collections/cmip/CMIP6/CMIP'
sens_cache_dir = '/glade/work/jrotondo/siaf_extended/sensitivity_cache/'
clara_dir      = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir       = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
PLOT_DIR       = '/glade/work/jrotondo/ncc_paper/plots/cmip6_IS_total/'
os.makedirs(PLOT_DIR, exist_ok=True)

MONTH_NAMES  = {4:'April', 5:'May', 6:'June',
                7:'July',  8:'August', 9:'September'}
MONTH_NUMS   = [4, 5, 6, 7, 8, 9]

CMIP6_EARLY_YEARS = list(range(1982, 1992))
CMIP6_LATE_YEARS  = list(range(2005, 2015))

OBS_DELTA_T      = 0.388
OBS_DELTA_T_STD  = 0.059

CMIP6_DELTA_T_FALLBACK = 0.388
SIC_ICE_THRESHOLD      = 0.15

SKIP_MODELS = {'NorESM2-LM'}

MODEL_INFO = {
    'CESM2'          : dict(institution='NCAR',          member='r1i1p1f1'),
    'E3SM-1-0'       : dict(institution='E3SM-Project',  member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',         member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',         member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',          member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',          member='r1i1p1f1'),
    'CanESM5'        : dict(institution='CCCma',          member='r1i1p1f1'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',          member='r1i1p1f1'),
    'GFDL-ESM4'      : dict(institution='NOAA-GFDL',      member='r1i1p1f1'),
    'NorESM2-LM'     : dict(institution='NCC',            member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',           member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',           member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS',   member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',            member='r1i1p1f1'),
}

MODEL_COLORS = {
    'CESM2':           '#1f77b4',
    'E3SM-1-0':        '#ff7f0e',
    'MPI-ESM1-2-HR':   '#2ca02c',
    'MPI-ESM1-2-LR':   '#98df8a',
    'IPSL-CM6A-LR':    '#d62728',
    'MIROC6':          '#9467bd',
    'CanESM5':         '#8c564b',
    'ACCESS-ESM1-5':   '#e377c2',
    'GFDL-ESM4':       '#7f7f7f',
    'NorESM2-LM':      '#bcbd22',
    'UKESM1-0-LL':     '#17becf',
    'HadGEM3-GC31-LL': '#aec7e8',
    'CNRM-CM6-1':      '#ffbb78',
    'BCC-CSM2-MR':     '#ff9896',
}

# =============================================================================
# HELPERS
# =============================================================================

def make_circle_boundary(ax, n=100):
    theta = np.linspace(0, 2*np.pi, n)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

def arctic_wmean_native(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid])
            if valid.any() else np.nan)

def arctic_wmean_2d(arr2d, lat2d):
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid])
            if valid.any() else np.nan)

def arctic_wmean_common(arr2d, common_lat):
    return arctic_wmean_native(arr2d, common_lat)

def nanmean2(a, b):
    out  = np.full_like(a, np.nan)
    both = np.isfinite(a) & np.isfinite(b)
    ao   = np.isfinite(a) & ~np.isfinite(b)
    bo   = np.isfinite(b) & ~np.isfinite(a)
    out[both] = (a[both] + b[both]) / 2.
    out[ao]   = a[ao]
    out[bo]   = b[bo]
    return out

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])

def regrid_to_common_kdtree(src_data, src_lat, src_lon,
                              tgt_lat, tgt_lon, max_dist_km=150.):
    lat_1d = src_lat[:, 0] if src_lat.ndim == 2 else src_lat.copy()
    lon_1d = src_lon[0, :] if src_lon.ndim == 2 else src_lon.copy()
    src_lon2d, src_lat2d = np.meshgrid(lon_1d, lat_1d)
    src_flat_lat = src_lat2d.ravel()
    src_flat_lon = src_lon2d.ravel()
    src_flat_dat = src_data.ravel()
    valid_mask   = np.isfinite(src_flat_dat)
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    _, idx  = cKDTree(latlon_to_xyz(src_flat_lat,
                                    src_flat_lon)).query(tgt_xyz, workers=-1)
    result  = src_flat_dat[idx].reshape(tgt_lat2d.shape)
    if valid_mask.any():
        src_xyz_v = latlon_to_xyz(src_flat_lat[valid_mask],
                                   src_flat_lon[valid_mask])
        max_chord = 2. * np.sin(np.deg2rad(max_dist_km / 111.2 / 2.))
        dists, _  = cKDTree(src_xyz_v).query(tgt_xyz, workers=-1)
        result.ravel()[dists > max_chord] = np.nan
    return result

def regrid_scatter(arr, src_lat2d, src_lon2d, tgt_lat, tgt_lon):
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    _, idx  = cKDTree(tgt_xyz).query(src_xyz, workers=-1)
    flat    = arr.ravel()
    accum   = np.zeros(np.prod(tgt_lat2d.shape))
    count   = np.zeros(np.prod(tgt_lat2d.shape))
    valid   = np.isfinite(flat)
    np.add.at(accum, idx[valid], flat[valid])
    np.add.at(count, idx[valid], 1)
    out    = np.full(np.prod(tgt_lat2d.shape), np.nan)
    m      = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(tgt_lat2d.shape)

def regrid_timeseries_to_grid(src_data_3d, src_lat, src_lon,
                               tgt_lat, tgt_lon):
    if src_lat.ndim == 1:
        src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    else:
        src_lat2d, src_lon2d = src_lat, src_lon
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    _, idx  = cKDTree(src_xyz).query(tgt_xyz, workers=-1)
    n_t     = src_data_3d.shape[0]
    return src_data_3d.reshape(n_t, -1)[:, idx].reshape(
        (n_t,) + tgt_lat2d.shape)

def find_var(inst, model, member, table, variable):
    var_dir = os.path.join(CMIP6_ROOT, inst, model, 'historical',
                           member, table, variable)
    if not os.path.isdir(var_dir):
        return []
    return sorted(glob.glob(
        os.path.join(var_dir, '**', f'{variable}_*.nc'), recursive=True))

def find_all_members(inst, model, table='Amon', variable='rsus'):
    base_dir = os.path.join(CMIP6_ROOT, inst, model, 'historical')
    if not os.path.isdir(base_dir):
        return []
    members = []
    try:
        for d in sorted(os.listdir(base_dir)):
            rsus_dir = os.path.join(base_dir, d, table, variable)
            if not os.path.isdir(rsus_dir):
                continue
            rsus_files = glob.glob(
                os.path.join(rsus_dir, '**', f'{variable}_*.nc'),
                recursive=True)
            if not rsus_files:
                continue
            rsds_dir = os.path.join(base_dir, d, table, 'rsds')
            rsds_files = glob.glob(
                os.path.join(rsds_dir, '**', 'rsds_*.nc'), recursive=True)
            if not rsds_files:
                continue
            sic_files = (
                glob.glob(os.path.join(base_dir, d, 'SImon', 'siconc',
                                       '**', 'siconc_*.nc'), recursive=True) +
                glob.glob(os.path.join(base_dir, d, 'SIday', 'siconc',
                                       '**', 'siconc_*.nc'), recursive=True))
            if not sic_files:
                continue
            members.append(d)
    except Exception:
        pass
    return members


def _open_mfdataset_robust(files, variable, use_cftime=True):
    try:
        ds = xr.open_mfdataset(files, combine='by_coords',
                               use_cftime=use_cftime, chunks={'time': 12})
        return ds
    except Exception:
        pass
    try:
        ds = xr.open_mfdataset(files, combine='nested',
                               concat_dim='time',
                               use_cftime=use_cftime, chunks={'time': 12})
        return ds
    except Exception as e:
        raise RuntimeError(f"Cannot open {variable}: {e}")


def _standardise_latlon(da):
    target_names = {'lat': ['latitude', 'nav_lat', 'rlat', 'y'],
                    'lon': ['longitude', 'nav_lon', 'rlon', 'x']}
    for new, aliases in target_names.items():
        if (new in da.coords and da[new].ndim == 1 and new in da.dims):
            continue
        for c in list(da.coords):
            if c.lower() in aliases and c != new:
                try:
                    if new in da.coords:
                        if new in da.dims:
                            da = da.drop_vars(new)
                        da = da.rename({c: new})
                    else:
                        da = da.rename({c: new})
                except Exception:
                    pass
                break
    lat_arr = da['lat'].values if 'lat' in da.coords else None
    lon_arr = da['lon'].values if 'lon' in da.coords else None
    if lat_arr is None or lon_arr is None:
        return da, None, None, False
    is_2d = (lat_arr.ndim == 2)
    return da, lat_arr, lon_arr, is_2d


def load_monthly(files, variable, years, month_nums):
    try:
        ds = _open_mfdataset_robust(files, variable)
        da = ds[variable]
        yr_set = set(years); mn_set = set(month_nums)
        mask   = np.array(
            [int(t.dt.year) in yr_set and int(t.dt.month) in mn_set
             for t in da.time], dtype=bool)
        da = da.isel(time=mask).load()
        ds.close()
        da, lat_arr, lon_arr, is_2d = _standardise_latlon(da)
        return da
    except Exception as e:
        print(f"    load error ({variable}): {e}", flush=True)
        return None


def ols_trend_pval(years, values):
    n   = values.shape[0]
    x   = np.array(years, dtype=float)
    xm  = x.mean(); sx2 = ((x - xm)**2).sum()
    ym  = np.nanmean(values, axis=0)
    sxy = np.nansum((x-xm)[:, None, None] * (values-ym), axis=0)
    slope  = sxy / sx2
    resid  = values - (slope[None]*(x-xm)[:, None, None] + ym)
    sse    = np.nansum(resid**2, axis=0)
    se     = np.sqrt(sse / (n-2) / sx2)
    t_stat = slope / np.where(se > 0, se, np.nan)
    return slope, 2.*t_dist.sf(np.abs(t_stat), df=n-2)

# =============================================================================
# BUILD FILE INDICES
# =============================================================================

def build_obs_file_index():
    print("Building obs file index...", flush=True)
    clara_index = {}
    for f in sorted(glob.glob(f"{clara_dir}/*.nc")):
        try:
            bname = os.path.basename(f)
            dt    = _dt.strptime(bname[5:13], "%Y%m%d")
            clara_index[(dt.year, dt.month)] = f
        except Exception:
            continue
    print(f"  CLARA: {len(clara_index)} files indexed", flush=True)

    appx_index = {}
    for f in sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True)):
        bname = os.path.basename(f)
        if '_1400_' not in bname:
            continue
        m = re.search(r'd(\d{8})', bname)
        if m:
            try:
                dt = _dt.strptime(m.group(1), "%Y%m%d")
                appx_index[(dt.year, dt.month)] = f
            except Exception:
                continue
    print(f"  APP-x: {len(appx_index)} files indexed", flush=True)

    sample_appx = sorted(appx_index.values())[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat2d = np.ma.filled(
            np.ma.asarray(ds.variables['latitude'][:]),
            np.nan).astype(float).squeeze()
        appx_lon2d = np.ma.filled(
            np.ma.asarray(ds.variables['longitude'][:]),
            np.nan).astype(float).squeeze()

    return clara_index, appx_index, appx_lat2d, appx_lon2d

# =============================================================================
# LOAD ALBEDO
# =============================================================================

def load_clara_alb(yr_list, month_num, clara_index):
    mn = MONTH_NAMES[month_num]
    cached = [np.load(f"{sens_cache_dir}/clara_alb_{mn}_{y}.npy")
              for y in yr_list
              if os.path.exists(f"{sens_cache_dir}/clara_alb_{mn}_{y}.npy")]
    if len(cached) >= max(1, len(yr_list) // 2):
        return cached
    albs = []
    for y in yr_list:
        f = clara_index.get((y, month_num))
        if f is None:
            continue
        try:
            with xr.open_dataset(f) as ds:
                snow = ds['black_sky_albedo_snow_mean'].squeeze().values / 100.
                reg  = ds['black_sky_albedo_mean'].squeeze().values      / 100.
                albs.append(np.where(np.isfinite(snow), snow, reg))
        except Exception as e:
            print(f"    CLARA load error ({y},{month_num}): {e}", flush=True)
    return albs

def load_appx_alb(yr_list, month_num, appx_index):
    albs = []
    for y in yr_list:
        f = appx_index.get((y, month_num))
        if f is None:
            continue
        try:
            with NC4Dataset(f, 'r') as ds:
                arr = np.ma.filled(
                    np.ma.asarray(ds.variables['cdr_surface_albedo'][:]),
                    np.nan).astype(float).squeeze()
                arr = np.clip(arr, 0., 1.)
            albs.append(arr)
        except Exception as e:
            print(f"    APP-x load error ({y},{month_num}): {e}", flush=True)
    return albs

# =============================================================================
# COMPUTE PER-MEMBER DELTA T
# =============================================================================

def compute_model_delta_T(model, member, early_years=CMIP6_EARLY_YEARS,
                           late_years=CMIP6_LATE_YEARS):
    inst      = MODEL_INFO[model]['institution']
    tas_files = find_var(inst, model, member, 'Amon', 'tas')
    if not tas_files:
        return None
    tas_da = load_monthly(tas_files, 'tas',
                          early_years + late_years, list(range(1, 13)))
    if tas_da is None:
        return None
    _, tas_lat, tas_lon, is_2d = _standardise_latlon(tas_da)
    if tas_lat is None or tas_lon is None or is_2d or tas_lat.ndim != 1:
        return None
    tas_yrs  = np.array([int(t.dt.year) for t in tas_da.time])
    tas_vals = tas_da.values.astype(float)
    weights  = np.cos(np.deg2rad(
        np.tile(tas_lat[:, None], (1, len(tas_lon)))))

    def gmst(yr_list):
        mask = np.isin(tas_yrs, yr_list)
        if not mask.any(): return np.nan
        means = []
        for i in np.where(mask)[0]:
            f = tas_vals[i]; v = np.isfinite(f)
            if v.any():
                means.append(np.nansum(f[v]*weights[v]) /
                              np.nansum(weights[v]))
        return np.mean(means) if means else np.nan

    eT = gmst(early_years); lT = gmst(late_years)
    del tas_da; gc.collect()
    if not (np.isfinite(eT) and np.isfinite(lT)): return None
    return lT - eT, eT, lT

# =============================================================================
# LOAD OBS IS_TOTAL
# IS_total = -(α_late - α_early) / ΔT  on all valid ice pixels
# (lat >= 60, sic_max >= 0.15, no climatological NaN) — no trend filter.
# This equals IS_ext + IS_dark from the fig8 script.
# =============================================================================

def load_obs_IS_total(common_lat, common_lon,
                      clara_index, appx_index, appx_lat2d, appx_lon2d,
                      early_years=CMIP6_EARLY_YEARS,
                      late_years=CMIP6_LATE_YEARS,
                      delta_t=OBS_DELTA_T):
    print(f"Loading obs IS_total "
          f"({early_years[0]}-{early_years[-1]} vs "
          f"{late_years[0]}-{late_years[-1]}, ΔT={delta_t:.3f})...",
          flush=True)

    common_shape    = (len(common_lat), len(common_lon))
    arctic_60N_sial = np.load(f"{sens_cache_dir}/arctic_60N_sial.npy")
    arctic_60N_appx = np.load(f"{sens_cache_dir}/arctic_60N_appx.npy")

    sample_clara = sorted(glob.glob(f"{clara_dir}/*.nc"))[0]
    with xr.open_dataset(sample_clara) as ds:
        sial_lat = ds['lat'].values
        sial_lon = ds['lon'].values
    sial_lat1d = sial_lat[:, 0] if sial_lat.ndim == 2 else sial_lat
    sial_lon1d = sial_lon[0, :] if sial_lon.ndim == 2 else sial_lon

    IS_total_obs = {}

    for month_num in MONTH_NUMS:
        mn = MONTH_NAMES[month_num]

        climo_nan_c = np.load(f"{sens_cache_dir}/climo_nan_clara_{mn}.npy")
        sic_early_c = np.load(f"{sens_cache_dir}/sic_early_sial_{mn}.npy")
        sic_late_c  = np.load(f"{sens_cache_dir}/sic_late_sial_{mn}.npy")

        p_ea = f"{sens_cache_dir}/sic_early_appx_{mn}.npy"
        p_la = f"{sens_cache_dir}/sic_late_appx_{mn}.npy"
        p_cn = f"{sens_cache_dir}/climo_nan_appx_{mn}.npy"
        have_appx_sic = all(os.path.exists(p) for p in [p_ea, p_la, p_cn])
        if have_appx_sic:
            climo_nan_a = np.load(p_cn)
            sic_early_a = np.load(p_ea)
            sic_late_a  = np.load(p_la)

        ae_c = load_clara_alb(early_years, month_num, clara_index)
        al_c = load_clara_alb(late_years,  month_num, clara_index)
        ae_a = load_appx_alb(early_years,  month_num, appx_index)
        al_a = load_appx_alb(late_years,   month_num, appx_index)

        if not ae_c or not al_c:
            print(f"  OBS {mn}: no CLARA albedo — skipping", flush=True)
            IS_total_obs[month_num] = None
            continue

        dalpha_c  = np.nanmean(al_c, axis=0) - np.nanmean(ae_c, axis=0)
        sic_max_c = np.fmax(sic_early_c, sic_late_c)

        # IS_total: ALL valid ice pixels, no trend significance filter
        all_ice_c    = arctic_60N_sial & ~climo_nan_c & (sic_max_c >= SIC_ICE_THRESHOLD)
        IS_total_c   = np.where(all_ice_c, -dalpha_c / delta_t, np.nan)
        IS_total_c_common = regrid_to_common_kdtree(
            IS_total_c, sial_lat1d, sial_lon1d, common_lat, common_lon)

        if ae_a and al_a and have_appx_sic:
            dalpha_a  = np.nanmean(al_a, axis=0) - np.nanmean(ae_a, axis=0)
            sic_max_a = np.fmax(sic_early_a, sic_late_a)
            all_ice_a = arctic_60N_appx & ~climo_nan_a & (sic_max_a >= SIC_ICE_THRESHOLD)
            IS_total_a = np.where(all_ice_a, -dalpha_a / delta_t, np.nan)
            IS_total_a_common = regrid_scatter(
                IS_total_a, appx_lat2d, appx_lon2d, common_lat, common_lon)
            n_a = int(all_ice_a.sum())
        else:
            IS_total_a_common = np.full(common_shape, np.nan)
            n_a = 0

        # September: prefer APP-x (better coverage); other months: blend
        IS_blend = (IS_total_a_common if (month_num == 9 and n_a > 0)
                    else nanmean2(IS_total_c_common, IS_total_a_common))

        lat2d = np.tile(common_lat[:, None], (1, len(common_lon)))
        am    = arctic_wmean_common(
            np.where(lat2d >= 60., IS_blend, np.nan), common_lat)
        print(f"  OBS {mn}: n_clara={int(all_ice_c.sum()):,}  "
              f"n_appx={n_a:,}  IS_total={am:+.5f}", flush=True)
        IS_total_obs[month_num] = IS_blend

    return IS_total_obs

# =============================================================================
# COMPUTE IS_TOTAL FOR ONE MODEL MEMBER
# IS_total = -(α_late - α_early) / ΔT  on all ice pixels
# (lat >= 60, sic_max >= 0.15) — no trend significance filter.
# =============================================================================

def compute_IS_total_one_member(model, member, delta_t,
                                 common_lat, common_lon,
                                 return_native=False,
                                 early_years=CMIP6_EARLY_YEARS,
                                 late_years=CMIP6_LATE_YEARS):
    inst = MODEL_INFO[model]['institution']

    rsus_files = find_var(inst, model, member, 'Amon', 'rsus')
    rsds_files = find_var(inst, model, member, 'Amon', 'rsds')
    sic_files  = find_var(inst, model, member, 'SImon', 'siconc')
    if not sic_files:
        sic_files = find_var(inst, model, member, 'SIday', 'siconc')

    if not all([rsus_files, rsds_files, sic_files]):
        return None

    all_years = list(range(1980, 2016))
    sat_years = list(range(1982, 2015))

    rsus_da = load_monthly(rsus_files, 'rsus',   all_years, MONTH_NUMS)
    rsds_da = load_monthly(rsds_files, 'rsds',   all_years, MONTH_NUMS)
    sic_da  = load_monthly(sic_files,  'siconc', all_years, MONTH_NUMS)
    if rsus_da is None or rsds_da is None or sic_da is None:
        return None

    _, alb_lat, alb_lon, alb_is_2d = _standardise_latlon(rsus_da)
    if alb_lat is None or alb_lon is None:
        return None
    if not alb_is_2d and alb_lat.ndim != 1:
        return None

    _, sic_lat, sic_lon, sic_is_2d = _standardise_latlon(sic_da)
    if sic_lat is None or sic_lon is None:
        return None

    def _ym_set(da):
        return set(zip(
            [int(t.dt.year)  for t in da.time],
            [int(t.dt.month) for t in da.time]))

    common_ym = _ym_set(rsus_da) & _ym_set(rsds_da) & _ym_set(sic_da)
    if not common_ym:
        print(f"    [{model}/{member}] no common time steps — skip", flush=True)
        return None

    def _ym_mask(da):
        return np.array(
            [(int(t.dt.year), int(t.dt.month)) in common_ym
             for t in da.time], dtype=bool)

    rsus_da = rsus_da.isel(time=_ym_mask(rsus_da))
    rsds_da = rsds_da.isel(time=_ym_mask(rsds_da))
    sic_da  = sic_da.isel( time=_ym_mask(sic_da))

    sic_vals = sic_da.values.astype(float)
    if np.nanmax(sic_vals) > 1.5:
        sic_vals /= 100.

    if alb_is_2d:
        alb_shape = alb_lat.shape
    else:
        alb_shape = (len(alb_lat), len(alb_lon))

    if sic_vals.shape[1:] != alb_shape:
        sl = sic_lat if sic_is_2d else (
            sic_lat[:, 0] if sic_lat.ndim == 2 else sic_lat)
        sw = sic_lon if sic_is_2d else (
            sic_lon[0, :] if sic_lon.ndim == 2 else sic_lon)
        sic_alb = regrid_timeseries_to_grid(
            sic_vals, sl, sw,
            alb_lat.ravel() if alb_is_2d else alb_lat,
            alb_lon.ravel() if alb_is_2d else alb_lon)
        if alb_is_2d:
            sic_alb = sic_alb.reshape((sic_alb.shape[0],) + alb_shape)
        del sic_vals; gc.collect()
    else:
        sic_alb = sic_vals

    rsus_yrs     = np.array([int(t.dt.year)  for t in rsus_da.time])
    rsus_mon     = np.array([int(t.dt.month) for t in rsus_da.time])
    sic_t_yrs    = np.array([int(t.dt.year)  for t in sic_da.time])
    sic_t_months = np.array([int(t.dt.month) for t in sic_da.time])
    del sic_da; gc.collect()

    if sic_alb.shape[0] != len(sic_t_yrs):
        print(f"    [{model}/{member}] sic shape mismatch — skip", flush=True)
        return None

    if alb_is_2d:
        lat2d_alb = alb_lat
        lon2d_alb = alb_lon
    else:
        lon2d_alb, lat2d_alb = np.meshgrid(alb_lon, alb_lat)

    scalars     = {}
    native_maps = {} if return_native else None

    for month_num in MONTH_NUMS:
        mn = MONTH_NAMES[month_num]

        def epoch_alb(yr_list):
            mask = np.isin(rsus_yrs, list(yr_list)) & (rsus_mon == month_num)
            if mask.sum() == 0:
                return None
            ru = rsus_da.isel(time=mask)
            rd = rsds_da.isel(time=mask)
            if len(ru.time) == 0 or len(rd.time) == 0:
                return None
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(rd.mean('time').values > 1.,
                               ru.mean('time').values /
                               rd.mean('time').values, np.nan)
            return np.clip(alb, 0., 1.)

        alb_early = epoch_alb(early_years)
        alb_late  = epoch_alb(late_years)
        if alb_early is None or alb_late is None:
            print(f"    [{model}/{member}] {mn}: missing epoch albedo "
                  f"(early={'ok' if alb_early is not None else 'MISSING'}, "
                  f"late={'ok' if alb_late is not None else 'MISSING'}) — nan",
                  flush=True)
            scalars[month_num] = np.nan
            if return_native: native_maps[month_num] = None
            continue

        dalpha = alb_late - alb_early

        # Build sic_max from epoch means
        sic_e   = sic_alb[(sic_t_months==month_num) &
                           np.isin(sic_t_yrs, early_years)]
        sic_l   = sic_alb[(sic_t_months==month_num) &
                           np.isin(sic_t_yrs, late_years)]

        # Need at least some SIC data to define ice mask
        sic_yrs_mm = sic_t_yrs[(sic_t_months == month_num) &
                                np.isin(sic_t_yrs, sat_years)]
        if len(sic_yrs_mm) < 10:
            scalars[month_num] = np.nan
            if return_native: native_maps[month_num] = None
            continue

        sic_max = np.fmax(
            np.nanmean(sic_e, axis=0) if len(sic_e) else np.zeros(alb_shape),
            np.nanmean(sic_l, axis=0) if len(sic_l) else np.zeros(alb_shape))

        arctic  = lat2d_alb >= 60.

        # IS_total: ALL ice pixels — no trend significance filter
        all_ice  = arctic & (sic_max >= SIC_ICE_THRESHOLD)
        IS_total = np.where(all_ice, -dalpha / delta_t, np.nan)

        # Scalar
        if alb_is_2d:
            am = arctic_wmean_2d(IS_total, lat2d_alb)
        else:
            am = arctic_wmean_native(
                np.where(lat2d_alb >= 60., IS_total, np.nan), alb_lat)
        scalars[month_num] = am

        if return_native:
            native_maps[month_num] = IS_total.copy()

    del rsus_da, rsds_da, sic_alb; gc.collect()

    amjjas = np.nanmean([scalars[m] for m in MONTH_NUMS])

    return {
        'scalars':   scalars,
        'amjjas':    amjjas,
        'native':    native_maps,
        'lat':       alb_lat,
        'lon':       alb_lon,
        'lat_is_2d': alb_is_2d,
        'delta_t':   delta_t,
    }

# =============================================================================
# PLOTTING — SPATIAL ON NATIVE GRID
# =============================================================================

def plot_spatial_native(native_data, model_names, model_delta_T_bar):
    proj_p   = ccrs.NorthPolarStereo(central_longitude=-35)
    data_crs = ccrs.PlateCarree()
    cmap     = plt.cm.RdBu_r
    vmin, vmax = -0.10, 0.10
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ncols = min(5, len(model_names))
    nrows = int(np.ceil(len(model_names) / ncols))

    for month_num in MONTH_NUMS:
        mname = MONTH_NAMES[month_num]
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols*5, nrows*4.5),
                                 subplot_kw={'projection': proj_p},
                                 facecolor='white')
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f'IS$_{{\\rm total}}$ — {mname}  (albedo per K)\n'
            f'Epoch: {CMIP6_EARLY_YEARS[0]}-{CMIP6_EARLY_YEARS[-1]}'
            f' vs {CMIP6_LATE_YEARS[0]}-{CMIP6_LATE_YEARS[-1]}  '
            f'(native grid, all ice pixels sic$_{{\\rm max}}$ ≥ 15%, per-member ΔT)',
            fontsize=11, fontweight='bold', y=1.01)
        axes = np.array(axes).ravel()

        for idx, label in enumerate(model_names):
            ax  = axes[idx]
            ax.set_facecolor('#cce5f0')
            make_circle_boundary(ax)
            ax.set_extent([-180, 180, 60, 90], crs=data_crs)
            ax.add_feature(cfeature.LAND,      facecolor='#d0cfc4', zorder=3)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#555555',
                           linewidth=0.4, zorder=4)
            gl = ax.gridlines(crs=data_crs, draw_labels=False,
                              linewidth=0.35, color='#888888',
                              alpha=0.5, linestyle='--', zorder=2)
            gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
            gl.ylocator = mticker.FixedLocator([60, 70, 80])

            dT_bar = model_delta_T_bar.get(label, np.nan)
            dT_str = (f"ΔT̄={dT_bar:.3f}K" if np.isfinite(dT_bar)
                      else "ΔT̄=N/A")
            entry  = native_data.get(label)

            if entry is None or entry.get('native') is None:
                ax.set_title(f"{label}\nno data  {dT_str}",
                             fontsize=8, color='grey')
                continue

            IS_tot  = entry['native'].get(month_num)
            lat_arr = entry['lat']
            lon_arr = entry['lon']
            is_2d   = entry.get('lat_is_2d', False)

            if IS_tot is None or lat_arr is None or lon_arr is None:
                ax.set_title(f"{label}\nno data  {dT_str}",
                             fontsize=8, color='grey')
                continue

            if is_2d:
                lat2d = lat_arr
                lon2d = lon_arr
            else:
                lon2d, lat2d = np.meshgrid(lon_arr, lat_arr)

            plot_data = np.where(lat2d >= 58., IS_tot, np.nan)

            im = ax.pcolormesh(lon2d, lat2d, plot_data,
                               transform=data_crs,
                               cmap=cmap, norm=norm,
                               shading='auto', zorder=2)

            if is_2d:
                am = arctic_wmean_2d(IS_tot, lat2d)
            else:
                am = arctic_wmean_native(
                    np.where(lat2d >= 60., IS_tot, np.nan), lat_arr)

            is_obs = (label == 'OBS')
            ax.set_title(f"{'★ ' if is_obs else ''}{label}\n"
                         f"mean={am:+.4f}  {dT_str}",
                         fontsize=8,
                         color='#b5000a' if is_obs else 'black',
                         fontweight='bold' if is_obs else 'normal', pad=3)
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                pad=0.04, fraction=0.046, shrink=0.85)
            cbar.ax.tick_params(labelsize=6)
            cbar.outline.set_linewidth(0.4)

        for j in range(len(model_names), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, f'IS_total_spatial_{mname}.png')
        fig.savefig(fpath, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {fpath}", flush=True)

# =============================================================================
# PLOTTING — ENSEMBLE HBAR
# =============================================================================

def plot_ensemble_hbar(obs_amjjas, obs_2sig, all_members_amjjas,
                        model_delta_T_bar):
    model_means = {}
    model_stds  = {}
    model_ns    = {}
    for model, members_dict in all_members_amjjas.items():
        vals = [v for v in members_dict.values() if np.isfinite(v)]
        if vals:
            model_means[model] = np.mean(vals)
            model_stds[model]  = np.std(vals) if len(vals) > 1 else 0.
            model_ns[model]    = len(vals)

    model_means['OBS'] = obs_amjjas
    model_stds['OBS']  = obs_2sig / 2.
    model_ns['OBS']    = 1

    sorted_labels = sorted(model_means,
                            key=lambda l: model_means[l]
                            if np.isfinite(model_means[l]) else -999)
    fig, ax = plt.subplots(figsize=(11, max(6, len(sorted_labels)*0.52)),
                           facecolor='white')
    fig.patch.set_facecolor('white')

    if np.isfinite(obs_amjjas) and np.isfinite(obs_2sig):
        ax.axvspan(obs_amjjas - obs_2sig, obs_amjjas + obs_2sig,
                   color='#b5000a', alpha=0.12, zorder=0,
                   label='OBS ±2σ')

    y = np.arange(len(sorted_labels))
    for yi, label in enumerate(sorted_labels):
        mean  = model_means.get(label, np.nan)
        std   = model_stds.get(label, 0.)
        color = '#b5000a' if label == 'OBS' else MODEL_COLORS.get(label, '#4393c3')

        if np.isfinite(mean):
            ax.barh(yi, mean, color=color, alpha=0.85, height=0.7)
            if std > 0:
                ax.errorbar(mean, yi, xerr=std,
                            fmt='none', color='black',
                            capsize=3, linewidth=1.2, zorder=5)
            ax.text(mean + 0.0008, yi,
                    f'{mean:+.4f}' + (f' ±{std:.4f}' if std > 0 else ''),
                    va='center', ha='left', fontsize=7.5)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)

    yticklabels = []
    for l in sorted_labels:
        dT_bar = model_delta_T_bar.get(l, np.nan)
        n      = model_ns.get(l, 0)
        if l == 'OBS':
            yticklabels.append(f"★ OBS  (ΔT={OBS_DELTA_T:.3f}K observed)")
        elif np.isfinite(dT_bar):
            yticklabels.append(f"{l}  (ΔT̄={dT_bar:.3f}K, n={n} mbrs)")
        else:
            yticklabels.append(l)
    ax.set_yticklabels(yticklabels, fontsize=9)

    ax.set_xlabel('AMJJAS Arctic mean IS$_{\\rm total}$  (albedo per K)',
                  fontsize=11)
    ax.set_title(
        f'CMIP6 ensemble mean vs OBS — AMJJAS IS$_{{\\rm total}}$\n'
        f'{CMIP6_EARLY_YEARS[0]}-{CMIP6_EARLY_YEARS[-1]} vs '
        f'{CMIP6_LATE_YEARS[0]}-{CMIP6_LATE_YEARS[-1]}  '
        f'bars = ensemble mean ± 1σ across members\n'
        f'IS$_{{\\rm total}}$ = all ice pixels (sic$_{{\\rm max}}$ ≥ 15%, '
        f'no trend filter)\n'
        f'OBS = CLARA-A3 + APP-x blend, ΔT={OBS_DELTA_T:.3f}K  |  '
        f'ΔT̄ = ensemble mean warming',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, 'IS_total_AMJJAS_ensemble_mean_hbar.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fpath}", flush=True)

# =============================================================================
# PLOTTING — ENSEMBLE SCATTER
# =============================================================================

def plot_ensemble_scatter_with_obs(obs_amjjas, obs_2sig, obs_monthly,
                                    all_members_amjjas, all_members_monthly,
                                    model_delta_T_bar):
    models_ordered = [m for m in MODEL_INFO
                      if m in all_members_amjjas and m not in SKIP_MODELS
                      and len(all_members_amjjas[m]) > 0]

    total_members = sum(len(v) for v in all_members_amjjas.values())
    within_band   = sum(
        1 for m in all_members_amjjas
        for v in all_members_amjjas[m].values()
        if np.isfinite(v) and np.isfinite(obs_amjjas) and
        abs(v - obs_amjjas) <= obs_2sig)
    pct = 100. * within_band / total_members if total_members > 0 else 0.

    # ── AMJJAS scatter ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')
    fig.patch.set_facecolor('white')

    if np.isfinite(obs_amjjas) and np.isfinite(obs_2sig):
        ax.axhspan(obs_amjjas - obs_2sig, obs_amjjas + obs_2sig,
                   color='#b5000a', alpha=0.12, zorder=0)
        ax.axhline(obs_amjjas, color='#b5000a', linewidth=2.5,
                   linestyle='--', zorder=5,
                   label=f'OBS = {obs_amjjas:+.4f} ± {obs_2sig:.4f}')

    x_cur = 0
    xtick_pos = []; xtick_labels = []

    for model in models_ordered:
        color   = MODEL_COLORS.get(model, '#333333')
        members = sorted(all_members_amjjas[model].keys())
        vals    = [(m, all_members_amjjas[model][m])
                   for m in members
                   if np.isfinite(all_members_amjjas[model][m])]
        if not vals: continue

        n     = len(vals)
        width = max(1.5, n * 0.35)
        xpos  = np.linspace(x_cur, x_cur + width, n)

        for xi, (member, v) in zip(xpos, vals):
            in_b = (np.isfinite(obs_amjjas) and np.isfinite(obs_2sig) and
                    abs(v - obs_amjjas) <= obs_2sig)
            ax.scatter(xi, v, color=color,
                       s=80 if in_b else 35,
                       marker='*' if in_b else 'o',
                       edgecolors='black' if in_b else 'none',
                       linewidths=0.7,
                       zorder=8 if in_b else 4,
                       alpha=1.0 if in_b else 0.65)

        mean_v = np.mean([v for _, v in vals])
        ax.hlines(mean_v, xpos[0]-0.1, xpos[-1]+0.1,
                  colors=color, linewidths=2.2, alpha=0.85, zorder=6)

        xtick_pos.append(np.mean(xpos))
        n_in   = sum(1 for _, v in vals if abs(v - obs_amjjas) <= obs_2sig)
        dT_bar = model_delta_T_bar.get(model, np.nan)
        short  = (model.replace('ESM1-2-', 'E1-2-')
                       .replace('GC31-', 'G31-')
                       .replace('ESM1-5', 'E1-5')
                       .replace('CM6A-', 'CM6-'))
        dT_label = f"ΔT̄={dT_bar:.3f}" if np.isfinite(dT_bar) else ""
        xtick_labels.append(f"{short}\n(n={n}, {n_in}★)\n{dT_label}")
        x_cur = xpos[-1] + 1.8

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel('AMJJAS Arctic mean IS$_{\\rm total}$  (albedo per K)',
                  fontsize=12)
    ax.set_title(
        f'CMIP6 ensemble members vs OBS IS$_{{\\rm total}}$\n'
        f'{CMIP6_EARLY_YEARS[0]}-{CMIP6_EARLY_YEARS[-1]} vs '
        f'{CMIP6_LATE_YEARS[0]}-{CMIP6_LATE_YEARS[-1]}  |  '
        f'★ = within OBS ±2σ  |  line = model mean  |  '
        f'{within_band}/{total_members} ({pct:.0f}%) in OBS ±2σ\n'
        f'IS$_{{\\rm total}}$ = all ice pixels (sic$_{{\\rm max}}$ ≥ 15%, '
        f'no trend filter)  |  ΔT̄ = ensemble mean',
        fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    from matplotlib.lines import Line2D
    leg_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODEL_COLORS.get(m, '#333333'),
               markersize=9, label=m)
        for m in models_ordered]
    leg1 = ax.legend(handles=leg_elements, fontsize=7.5,
                     loc='upper left', ncol=4, framealpha=0.9,
                     title='Models')
    ax.add_artist(leg1)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.85)
    ax.add_artist(leg1)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, 'IS_total_ensemble_scatter.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fpath}", flush=True)
    print(f"  {within_band}/{total_members} ({pct:.1f}%) within "
          f"OBS ±2σ ({obs_amjjas:+.4f} ± {obs_2sig:.4f})", flush=True)

    # ── Per-month scatter ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'CMIP6 ensemble members vs OBS ±2σ — IS$_{{\\rm total}}$ per month\n'
        f'{CMIP6_EARLY_YEARS[0]}-{CMIP6_EARLY_YEARS[-1]} vs '
        f'{CMIP6_LATE_YEARS[0]}-{CMIP6_LATE_YEARS[-1]}  '
        f'(each member uses its own per-member ΔT)',
        fontsize=13, fontweight='bold')

    for mi, (month_num, ax) in enumerate(zip(MONTH_NUMS, axes.ravel())):
        mn  = MONTH_NAMES[month_num]
        ov  = obs_monthly.get(month_num, np.nan)
        o2s = (2. * OBS_DELTA_T_STD / OBS_DELTA_T * abs(ov)
               if np.isfinite(ov) else np.nan)

        if np.isfinite(ov) and np.isfinite(o2s):
            ax.axhspan(ov - o2s, ov + o2s,
                       color='#b5000a', alpha=0.15, zorder=0)
            ax.axhline(ov, color='#b5000a', linewidth=2.0,
                       linestyle='--', zorder=5,
                       label=f'OBS={ov:+.4f}±{o2s:.4f}')

        x2 = 0; tot = 0; win = 0; xt2 = []; xl2 = []

        for model in models_ordered:
            color = MODEL_COLORS.get(model, '#333333')
            vals  = []
            for member in sorted(all_members_amjjas[model].keys()):
                mv = all_members_monthly.get(
                    (model, member), {}).get(month_num, np.nan)
                if np.isfinite(mv): vals.append(mv)
            if not vals: continue

            n2  = len(vals)
            xp2 = np.linspace(x2, x2 + max(0.4, n2*0.18), n2)
            for xi, mv in zip(xp2, vals):
                in_b = (np.isfinite(ov) and np.isfinite(o2s) and
                        abs(mv - ov) <= o2s)
                ax.scatter(xi, mv, color=color,
                           s=45 if in_b else 18,
                           marker='*' if in_b else 'o',
                           edgecolors='black' if in_b else 'none',
                           linewidths=0.5, zorder=8 if in_b else 3,
                           alpha=1.0 if in_b else 0.6)
                tot += 1
                if in_b: win += 1
            ax.hlines(np.mean(vals), xp2[0]-.08, xp2[-1]+.08,
                      colors=color, linewidths=1.5, alpha=0.8)
            xt2.append(np.mean(xp2))
            xl2.append(model.replace('ESM1-2-', 'E1-2-')
                            .replace('GC31-', 'G31-')
                            .replace('ESM1-5', 'E1-5'))
            x2 = xp2[-1] + 0.6

        pct_m = 100.*win/tot if tot > 0 else 0.
        ax.set_title(f'{mn}  ({win}/{tot} = {pct_m:.0f}% in OBS ±2σ)',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('IS$_{\\rm total}$  (albedo per K)', fontsize=9)
        ax.set_xticks(xt2)
        ax.set_xticklabels(xl2, fontsize=6.5, rotation=35, ha='right')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if mi == 0 and np.isfinite(ov):
            ax.legend(fontsize=8, loc='upper left')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, 'IS_total_ensemble_scatter_monthly.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fpath}", flush=True)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    common_lat = np.arange(-90., 91.,  1.)
    common_lon = np.arange(  0., 360., 1.)

    clara_index, appx_index, appx_lat2d, appx_lon2d = build_obs_file_index()

    # ── OBS IS_total ──────────────────────────────────────────────────────────
    obs_IS_total = load_obs_IS_total(
        common_lat, common_lon,
        clara_index, appx_index, appx_lat2d, appx_lon2d,
        early_years=CMIP6_EARLY_YEARS,
        late_years=CMIP6_LATE_YEARS,
        delta_t=OBS_DELTA_T)

    obs_scalars_monthly = {}
    lat2d_c = np.tile(common_lat[:, None], (1, len(common_lon)))
    for mn in MONTH_NUMS:
        raw = obs_IS_total.get(mn)
        if raw is not None:
            obs_scalars_monthly[mn] = arctic_wmean_common(
                np.where(lat2d_c >= 60., raw, np.nan), common_lat)
        else:
            obs_scalars_monthly[mn] = np.nan
    obs_amjjas = np.nanmean([obs_scalars_monthly[m] for m in MONTH_NUMS])
    obs_2sig   = (2. * OBS_DELTA_T_STD / OBS_DELTA_T * abs(obs_amjjas)
                  if np.isfinite(obs_amjjas) else np.nan)
    print(f"\nOBS IS_total AMJJAS = {obs_amjjas:+.4f} ± {obs_2sig:.4f}",
          flush=True)

    # ── Storage ───────────────────────────────────────────────────────────────
    all_members_amjjas  = {}
    all_members_monthly = {}
    member_delta_T      = {}
    model_delta_T_bar   = {}
    native_data         = {}

    native_data['OBS'] = {
        'native': {mn: np.where(lat2d_c >= 60.,
                                obs_IS_total.get(mn)
                                if obs_IS_total.get(mn) is not None
                                else np.full(lat2d_c.shape, np.nan),
                                np.nan)
                   for mn in MONTH_NUMS},
        'lat':       common_lat,
        'lon':       common_lon,
        'lat_is_2d': False,
    }
    model_delta_T_bar['OBS']       = OBS_DELTA_T
    member_delta_T[('OBS', 'obs')] = OBS_DELTA_T

    # ── Loop over models ──────────────────────────────────────────────────────
    for model in MODEL_INFO:
        if model in SKIP_MODELS:
            print(f"\n[{model}] skipped", flush=True)
            continue

        inst           = MODEL_INFO[model]['institution']
        primary_member = MODEL_INFO[model]['member']

        print(f"\n{'='*60}", flush=True)
        print(f"Finding all members for {model}...", flush=True)

        all_members = find_all_members(inst, model, 'Amon', 'rsus')
        if not all_members:
            print(f"  No members found — skip", flush=True)
            continue

        if primary_member in all_members:
            all_members = ([primary_member] +
                           [m for m in all_members if m != primary_member])

        print(f"  Found {len(all_members)} members: "
              f"{all_members[:5]}{'...' if len(all_members)>5 else ''}",
              flush=True)

        all_members_amjjas[model] = {}
        collected_dT = []

        for member in all_members:
            is_primary    = (member == primary_member)
            return_native = is_primary

            dT_result = compute_model_delta_T(model, member)
            if dT_result is not None:
                delta_t_m = dT_result[0]
                if delta_t_m <= 0:
                    delta_t_m = CMIP6_DELTA_T_FALLBACK
                if is_primary:
                    print(f"  [{model}] primary ΔT ({member}): "
                          f"{dT_result[1]:.2f}→{dT_result[2]:.2f}K  "
                          f"ΔT={delta_t_m:+.3f}K", flush=True)
                else:
                    print(f"    [{model}/{member}] per-member "
                          f"ΔT={delta_t_m:+.3f}K", flush=True)
            else:
                delta_t_m = (np.mean(collected_dT) if collected_dT
                             else CMIP6_DELTA_T_FALLBACK)
                print(f"    [{model}/{member}] ΔT fallback: "
                      f"{delta_t_m:.3f}K", flush=True)

            member_delta_T[(model, member)] = delta_t_m
            collected_dT.append(delta_t_m)

            print(f"  Processing {model}/{member} "
                  f"(ΔT={delta_t_m:+.3f}K)...", flush=True)
            try:
                result = compute_IS_total_one_member(
                    model, member, delta_t_m,
                    common_lat, common_lon,
                    return_native=return_native)

                if result is None:
                    print(f"    [{model}/{member}] no result", flush=True)
                    continue

                amjjas_val = result['amjjas']
                all_members_amjjas[model][member]    = amjjas_val
                all_members_monthly[(model, member)] = result['scalars']

                if is_primary and result['native'] is not None:
                    native_data[model] = {
                        'native':    result['native'],
                        'lat':       result['lat'],
                        'lon':       result['lon'],
                        'lat_is_2d': result['lat_is_2d'],
                    }

                print(f"    [{model}/{member}] AMJJAS={amjjas_val:+.4f}  "
                      f"ΔT={delta_t_m:+.3f}K", flush=True)
            except Exception as e:
                print(f"    [{model}/{member}] FAILED: {e}", flush=True)
            finally:
                gc.collect()

        valid_dTs = [
            member_delta_T[(model, m)]
            for m in all_members_amjjas[model]
            if np.isfinite(all_members_amjjas[model][m])
            and (model, m) in member_delta_T
        ]
        dT_bar  = np.mean(valid_dTs) if valid_dTs else np.nan
        n_valid = len(all_members_amjjas[model])
        model_delta_T_bar[model] = dT_bar
        print(f"  [{model}] ΔT̄ = {dT_bar:.3f}K over {len(valid_dTs)} "
              f"valid members  |  AMJJAS processed = {n_valid}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(len(v) for v in all_members_amjjas.values())
    print(f"\nTotal ensemble members processed: {total}", flush=True)

    spatial_names = (['OBS'] +
                     [m for m in MODEL_INFO
                      if m in native_data and m != 'OBS'
                      and m not in SKIP_MODELS])

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting spatial maps (IS_total, native grid, primary member)...",
          flush=True)
    plot_spatial_native(native_data, spatial_names, model_delta_T_bar)

    print("\nPlotting ensemble mean hbar...", flush=True)
    plot_ensemble_hbar(obs_amjjas, obs_2sig, all_members_amjjas,
                       model_delta_T_bar)

    print("\nPlotting ensemble scatter...", flush=True)
    plot_ensemble_scatter_with_obs(obs_amjjas, obs_2sig, obs_scalars_monthly,
                                   all_members_amjjas, all_members_monthly,
                                   model_delta_T_bar)

    print("\nDONE", flush=True)