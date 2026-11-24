"""
siaf_obs_core.py — GENERATED, DO NOT EDIT. Rebuild with make_core.py.

Verbatim first 588 lines of python/siaf_obs_analysis.py: every definition and
grid the published SIAF numbers were produced with, cut immediately before the
ERA5-kernel load and the main loop so that it can be imported. Figures rebuilt
against this module therefore use the published masks, area weights and
conventions rather than a reimplementation of them.

sha256 of the copied body: 62073d052171104d8e639106e535084349f6fa09b583261b733ac0c9763f7020
"""
"""
siaf_analysis.py
----------------
Pixel-wise SIAF calculation using direct α~T OLS regression.

IS[px] = −(dα/dT)[px]   [alpha K⁻¹]

The per-pixel dα/dT is estimated by regressing the merged monthly albedo
time series against the global annual mean temperature anomaly for the same
years.  This is robust to non-linear warming because temperature is the
explicit predictor rather than a proxy for time.

Bootstrap: pairs bootstrap — resample (T_year, albedo_map) together, refit
the OLS slope for each draw.  No separate ΔT samples needed.

Sensor assignment (must match build_albedo_cache.py):
  September → APP-x only
  Apr–Aug   → CLARA-A3 primary; APP-x fills pixels where CLARA-A3 is NaN

2026-05 update: siaf_pixelwise now returns 13 values (indices 0-8 unchanged,
  9-12 = is_tot_arctic, is_ext_arctic, is_drk_arctic, rs_arctic).
  CSV now includes IS_tot/ext/drk_arctic, RS_arctic and their bootstrap 2σ.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, gc
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

# ============================================================
# CONFIG
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
KERNEL_FILE = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR  = 'SFC_all'
KERNEL_SCALE = 100.0
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
OUT_CSV     = os.path.join(OUT_DIR, 'siaf_pixelwise_summary.csv')
PLOT_OUT    = os.path.join(OUT_DIR, 'siaf_pixelwise_p{pval}.png')
os.makedirs(OUT_DIR, exist_ok=True)

GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

TREND_CMIP6  = (1982, 2014)
TREND_OBS    = (1982, 2024)
CERES_CMIP6  = list(range(2000, 2015))
CERES_OBS    = list(range(2000, 2025))

SIC_THRESH      = 0.15
AREA_RATIO      = 0.0347
SEASON_FRAC     = 6. / 12.
SCALE           = AREA_RATIO * SEASON_FRAC
PVAL_THRESHOLDS = [0.01, 0.05, 0.10]
N_BOOT          = 1000
BOOT_SEED       = 42

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

# siaf_pixelwise return-value index map (13 total)
# 0  tot_siaf_arctic   domain_mean(IS_t * RS)
# 1  ext_siaf_arctic   domain_mean(IS_x * RS)
# 2  drk_siaf_arctic   domain_mean(IS_d * RS)
# 3  is_ext_pa         component_mean(IS_x, IS_x footprint)
# 4  is_drk_pa         component_mean(IS_d, IS_d footprint)
# 5  rs_ext_pa         component_mean(RS,   IS_x footprint)
# 6  rs_drk_pa         component_mean(RS,   IS_d footprint)
# 7  siaf_ext_pa       component_mean(IS_x*RS, IS_x footprint)
# 8  siaf_drk_pa       component_mean(IS_d*RS, IS_d footprint)
# 9  is_tot_arctic     domain_mean(IS_t)
# 10 is_ext_arctic     domain_mean(IS_x)
# 11 is_drk_arctic     domain_mean(IS_d)
# 12 rs_arctic         domain_mean(RS)
N_RETURN = 13

# ============================================================
# TEMPERATURE ENSEMBLE
# ============================================================

def build_T_annual_ensemble(trend_years):
    raw_sources = {}

    try:
        df = pd.read_csv(GISTEMP_PATH, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        raw_sources['GISTEMP'] = {
            int(r['Year']): float(r['J-D'])
            for _, r in df.iterrows() if np.isfinite(r['J-D'])
        }
    except Exception:
        pass

    try:
        df = pd.read_csv(COWTANWAY_PATH, sep=r'\s+', header=None,
                         names=['year', 'anomaly', 'u1', 'u2', 'u3'])
        raw_sources['CowtanWay'] = {
            int(r['year']): float(r['anomaly'])
            for _, r in df.iterrows() if np.isfinite(r['anomaly'])
        }
    except Exception:
        pass

    try:
        ds = xr.open_dataset(NCEP_PATH)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat', 'lon']).resample(time='1YE').mean()
        raw_sources['NCEP'] = {
            int(yr): float(v)
            for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
            if np.isfinite(v)
        }
    except Exception:
        pass

    try:
        files = sorted(glob.glob(ERA5_GLOB))
        if files:
            ds = xr.open_mfdataset(files, combine='by_coords')
            w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
            ta = ds['VAR_2T'].weighted(w).mean(
                dim=['latitude', 'longitude']).resample(time='1YE').mean()
            raw_sources['ERA5'] = {
                int(yr): float(v)
                for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
                if np.isfinite(v)
            }
    except Exception:
        pass

    centered = {}
    for name, yr_dict in raw_sources.items():
        common = [y for y in trend_years if y in yr_dict]
        if len(common) < 10:
            continue
        mean_val = float(np.mean([yr_dict[y] for y in common]))
        centered[name] = {y: v - mean_val for y, v in yr_dict.items()}

    if not centered:
        raise RuntimeError('No T source has enough trend-period coverage.')

    ensemble = {}
    for y in trend_years:
        vals = [v[y] for v in centered.values() if y in v]
        if vals:
            ensemble[y] = float(np.mean(vals))

    print(f'  T ensemble: {list(centered.keys())}, '
          f'{len(ensemble)}/{len(trend_years)} years covered')
    return ensemble

# ============================================================
# GRID UTILITIES
# ============================================================

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

def build_nn_index(src_lat, src_lon, tgt_lat, tgt_lon):
    _, idx = cKDTree(latlon_to_xyz(src_lat, src_lon)).query(
        latlon_to_xyz(tgt_lat, tgt_lon), workers=-1)
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

def regrid_sial_to_psn(arr, fill_zero=True):
    src = (np.where(ARCTIC_SIAL,
                    np.where(np.isfinite(arr), arr, np.float32(0.)), np.nan)
           if fill_zero else arr)
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)

def domain_mean(field_psn):
    """Area-weighted mean over DOMAIN_PSN, NaN pixels excluded."""
    v  = field_psn[DOMAIN_PSN]
    w  = AREA_W[DOMAIN_PSN]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def component_mean(arr, comp_arr):
    """Area-weighted mean of `arr` restricted to pixels where `comp_arr` > 0."""
    mask = DOMAIN_PSN & np.isfinite(comp_arr) & (comp_arr > 0)
    v  = arr[mask]
    w  = AREA_W[mask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))

# ============================================================
# VECTORIZED OLS
# ============================================================

def compute_pixel_ols_slopes(stk, x_vec):
    """
    Per-pixel OLS slope of stk ~ x_vec.

    Parameters
    ----------
    stk   : (T, H, W) float array
    x_vec : (T,) float array

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
# DATA LOADERS
# ============================================================

_CERES_VARS = ['solar_mon','toa_sw_all_mon','sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon','toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon','sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt>1.,  rsdt,  np.nan)
        Sd  = np.where(rsds>1.,  rsds,  np.nan)
        a   = np.clip(rsus/Sd, 0., 1.)
        p   = rsds/S;  q  = rsut/S
        den = q + p - 2.*p*a
        R   = np.clip((q-p*a)/np.where(np.abs(den)>1e-6, den, np.nan), 0., .99)
        A   = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan), 0., .99)
        Sc  = np.where(rsdscs>1., rsdscs, np.nan)
        ac  = np.clip(rsuscs/Sc, 0., 1.)
        pc  = rsdscs/S;  qc = rsutcs/S
        dc  = qc+pc-2.*pc*ac
        Rc  = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan),0.,.99)
        Rl  = np.clip(R-Rc,0.,.99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               /np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS.astype(np.float32)

def _ceres_sums_to_RS(sums):
    return t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                      sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                      sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                      sums['sfc_sw_up_clr_c_mon'])

def load_IS_map(cache_dir, mname):
    p = f'{cache_dir}/merged_alb_dalphadT_{mname}.npy'
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)

def load_alb_yearly_with_T(cache_dir, mname, T_dict):
    maps, T_vals, years_out = [], [], []

    for year in sorted(T_dict.keys()):
        if mname == 'September':
            p = f'{cache_dir}/appx_alb_{mname}_{year}.npy'
            if not os.path.exists(p):
                continue
            raw = np.load(p).astype(np.float32)
            if IDX_APPX_TO_SIAL is not None and raw.shape != SIAL_SHAPE:
                raw = raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
            alb = raw
        else:
            p_c = f'{cache_dir}/clara_alb_{mname}_{year}.npy'
            p_a = f'{cache_dir}/appx_alb_{mname}_{year}.npy'
            clara, appx = None, None
            if os.path.exists(p_c):
                clara = np.load(p_c).astype(np.float32)
            if os.path.exists(p_a):
                raw_a = np.load(p_a).astype(np.float32)
                if IDX_APPX_TO_SIAL is not None and raw_a.shape != SIAL_SHAPE:
                    raw_a = raw_a.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
                appx = raw_a

            if clara is not None and appx is not None:
                alb = np.where(np.isfinite(clara), clara, appx)
            elif clara is not None:
                alb = clara
            elif appx is not None:
                alb = appx
            else:
                continue

        maps.append(alb)
        T_vals.append(T_dict[year])
        years_out.append(year)

    if not maps:
        return None, None, []

    stk   = np.stack(maps, axis=0).astype(np.float32)
    T_vec = np.array(T_vals, dtype=np.float64)
    return stk, T_vec, years_out

def load_RS_ceres_yearly(years, mn):
    result = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in years:
            idxs = [i for i,t in enumerate(times)
                    if t.year == y and t.month == mn]
            if not idxs: continue
            sums = {k: None for k in _CERES_VARS}; cnt = 0
            for i in idxs:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1; del sl
            if cnt == 0: continue
            for k in sums: sums[k] /= cnt
            result.append(_ceres_sums_to_RS(sums))
    gc.collect(); return result

def load_RS_ceres_mean(years, mn):
    yearly = load_RS_ceres_yearly(years, mn)
    if not yearly: return None
    return np.nanmean(np.stack(yearly, 0), axis=0).astype(np.float32)

def load_sic_data(cache_dir, mname):
    try:
        return dict(
            slope = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval  = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except:
        return None

def load_kernel_psn(mn):
    with xr.open_dataset(KERNEL_FILE) as ds:
        tdim  = next((d for d,s in ds.sizes.items() if s == 12), None)
        if tdim is None: raise KeyError('No size-12 dim in kernel file')
        vname = KERNEL_VAR
        if vname not in ds:
            vname = next((v for v in ds.data_vars
                          if v.lower() == KERNEL_VAR.lower()), None)
        lat_n = 'latitude' if 'latitude' in ds else 'lat'
        lon_n = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[vname].isel({tdim: mn-1}).values.astype(np.float32)
    k_raw = np.abs(k_raw) * KERNEL_SCALE
    idx_k = build_scatter_index(k_lat2d.ravel(), k_lon2d.ravel(),
                                 PSN_LAT.ravel(), PSN_LON.ravel())
    return scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)

# ============================================================
# IS COMPUTATION
# ============================================================

def compute_IS(dalpha_dT, sic, pval_thresh):
    IS_raw = (-dalpha_dT).astype(np.float32)

    valid  = ARCTIC_SIAL & np.isfinite(IS_raw)
    chang  = (valid & np.isfinite(sic['slope']) & np.isfinite(sic['pval'])
              & (sic['slope'] < 0) & (sic['pval'] < pval_thresh))
    stable = (valid & ~chang
              & (np.fmax(sic['sic_e'], sic['sic_l']) >= SIC_THRESH))

    IS_ext = np.where(chang,  IS_raw, np.nan).astype(np.float32)
    IS_drk = np.where(stable, IS_raw, np.nan).astype(np.float32)
    IS_tot = np.where(
        np.isfinite(IS_ext) | np.isfinite(IS_drk),
        np.where(np.isfinite(IS_ext), IS_ext, 0.)
        + np.where(np.isfinite(IS_drk), IS_drk, 0.),
        np.nan).astype(np.float32)
    IS_ext = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
    IS_drk = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)
    return IS_tot, IS_ext, IS_drk

# ============================================================
# PIXEL-WISE SIAF  (returns N_RETURN = 13 values)
#
# Indices 0-8: SIAF composites (unchanged from original)
# Indices 9-12: IS and RS domain/component means SEPARATELY
#   9  is_tot_arctic  domain_mean(IS_t)
#  10  is_ext_arctic  domain_mean(IS_x)
#  11  is_drk_arctic  domain_mean(IS_d)
#  12  rs_arctic      domain_mean(RS)
# ============================================================

def siaf_pixelwise(IS_t_psn, IS_x_psn, IS_d_psn, RS_psn):
    # ── Original SIAF products (indices 0-8) ──────────────────────────────
    tot = domain_mean(IS_t_psn * RS_psn)
    ext = domain_mean(IS_x_psn * RS_psn)
    drk = domain_mean(IS_d_psn * RS_psn)

    is_ext_pa   = component_mean(IS_x_psn, IS_x_psn)
    is_drk_pa   = component_mean(IS_d_psn, IS_d_psn)
    rs_ext_pa   = component_mean(RS_psn,   IS_x_psn)
    rs_drk_pa   = component_mean(RS_psn,   IS_d_psn)
    siaf_ext_pa = component_mean(IS_x_psn * RS_psn, IS_x_psn)
    siaf_drk_pa = component_mean(IS_d_psn * RS_psn, IS_d_psn)

    # ── NEW: IS and RS domain means SEPARATELY (indices 9-12) ─────────────
    is_tot_arctic = domain_mean(IS_t_psn)
    is_ext_arctic = domain_mean(IS_x_psn)
    is_drk_arctic = domain_mean(IS_d_psn)
    rs_arctic     = domain_mean(RS_psn)

    return [tot, ext, drk,
            is_ext_pa, is_drk_pa,
            rs_ext_pa, rs_drk_pa,
            siaf_ext_pa, siaf_drk_pa,
            is_tot_arctic, is_ext_arctic, is_drk_arctic, rs_arctic]

# ============================================================
# BOOTSTRAP
# ============================================================

def _resamp(stk, rng):
    n = stk.shape[0]
    return np.nanmean(stk[rng.integers(0, n, size=n)], axis=0).astype(np.float32)

def _stack(arr_list):
    valid = [a for a in arr_list if a is not None]
    return np.stack(valid, 0).astype(np.float32) if valid else None

def bootstrap_month(alb_stk, T_vec, rs_stk, sic, rng, pval):
    """
    Pairs bootstrap over (albedo map, T_year) samples.

    Returns
    -------
    boot : (N_BOOT, N_RETURN=13) float64 array
    """
    n_years = alb_stk.shape[0]
    boot    = np.full((N_BOOT, N_RETURN), np.nan, np.float64)

    for b in range(N_BOOT):
        idx = rng.integers(0, n_years, size=n_years)

        dalpha_dT_b = compute_pixel_ols_slopes(alb_stk[idx], T_vec[idx])
        RS          = _resamp(rs_stk, rng)

        IS_t, IS_x, IS_d = compute_IS(dalpha_dT_b, sic, pval)
        IS_t_p = regrid_sial_to_psn(IS_t)
        IS_x_p = regrid_sial_to_psn(IS_x)
        IS_d_p = regrid_sial_to_psn(IS_d)

        r = siaf_pixelwise(IS_t_p, IS_x_p, IS_d_p, RS)
        if all(np.isfinite(v) for v in r):
            boot[b] = r
    return boot

# ============================================================
# ONE-TIME SETUP
# ============================================================
print('='*70)
print('SETUP')
print('='*70)

# SIAL grid
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
except:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)

# APP-x -> SIAL nearest-neighbour index
IDX_APPX_TO_SIAL = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        al = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        ao = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if al.ndim == 1:
        ao2, al2 = np.meshgrid(ao, al)
    else:
        al2, ao2 = al, ao
    IDX_APPX_TO_SIAL = build_nn_index(
        al2.ravel(), ao2.ravel(), SIAL_LAT.ravel(), SIAL_LON.ravel())
    print('  APP-x -> SIAL index built.')
except Exception as e:
    print(f'  WARNING: APP-x -> SIAL failed ({e})')

# CERES grid
with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
    c_la = _ds['lat'].values; c_lo = _ds['lon'].values
CERES_LON, CERES_LAT = np.meshgrid(c_lo, c_la)

# PSN25 grid
from pyproj import Transformer
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
PSN_SHAPE = psn_xx.shape
tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = tfm.transform(psn_xx, psn_yy)

phi_ts = np.deg2rad(70.)
k_ps   = (1. + np.sin(phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W = (1. / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe as _globe
    lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    DOMAIN_PSN = (PSN_LAT >= 60.) & _globe.is_ocean(PSN_LAT, lp)
except ImportError:
    DOMAIN_PSN = (PSN_LAT >= 60.)
    print('  WARNING: global-land-mask not available; land not excluded.')

print(f'  PSN25 domain: {DOMAIN_PSN.sum():,} ocean pixels >= 60N')

print('  Building SIAL -> PSN25 index ...')
IDX_SIAL_TO_PSN  = build_scatter_index(
    SIAL_LAT.ravel(), SIAL_LON.ravel(),
    PSN_LAT.ravel(),  PSN_LON.ravel())
print('  Building CERES -> PSN25 index ...')
IDX_CERES_TO_PSN = build_scatter_index(
    CERES_LAT.ravel(), CERES_LON.ravel(),
    PSN_LAT.ravel(),   PSN_LON.ravel())
