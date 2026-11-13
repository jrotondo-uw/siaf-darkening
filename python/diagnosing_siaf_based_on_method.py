#!/usr/bin/env python
"""
diagnose_siaf_unified.py  (2026-04-19  full revision)
──────────────────────────────────────────────────────
Two diagnostic blocks with 2σ bootstrap uncertainty.
  - CERES EBAF Regional Approach (Pistone et al. 2014)  [Block 2]
  - Radiative Kernel Approach    (Huang and Huang, 2023) [Block 5]

Evaluates three significance thresholds: p < 0.01, 0.05, 0.10

Bootstrap propagates uncertainty from ALL three sources for BOTH blocks:

  Block 2 (CERES):
    1. Interannual variability in surface albedo  (SIAL yearly stacks)
    2. Interannual variability in RS              (CERES yearly stacks)
    3. Structural uncertainty in ΔT              (resampled across 4 datasets)

  Block 5 (ERA5 kernel):
    1. Interannual variability in surface albedo  (SIAL yearly stacks)
    2. RS kernel uncertainty                      (multiplicative perturbation
                                                   drawn from N(1, σ_kernel²)
                                                   where σ_kernel = 0.025,
                                                   giving 2σ = ±5% consistent
                                                   with Huang & Huang 2023)
    3. Structural uncertainty in ΔT              (resampled across 4 datasets)

ΔT computed via OLS trend (slope × N years) for each source:
    1. GISTEMP   (NASA GISS Land-Ocean)
    2. Cowtan & Way
    3. NCEP reanalysis  (sigma-995 air temperature, area-weighted)
    4. ERA5 reanalysis  (2 m temperature, area-weighted)

Also saves IS_arctic_mean / RS_arctic_mean / IS_arctic_2sigma / RS_arctic_2sigma
per row so the IS–RS scatter plot can read them directly without hardcoding.

Outputs:
  siaf_unified_diagnostics_monthly.csv
  siaf_unified_diagnostics.csv
  siaf_breakdown_barplot_p{pval}.png
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, gc, resource
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
OUT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
OUT_CSV  = os.path.join(OUT_DIR, 'siaf_unified_diagnostics.csv')
PLOT_OUT = os.path.join(OUT_DIR, 'siaf_breakdown_barplot.png')
os.makedirs(OUT_DIR, exist_ok=True)

KERNEL_CONFIGS = {
    'Huang_ERA5': dict(
        file     = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc',
        var      = 'SFC_all',
        scale    = 100.0,
        time_dim = 'month',
        desc     = 'ERA5 offline (Huang & Huang 2023)',
    ),
}

# ── Temperature dataset paths ────────────────────────────────────────────────
GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

# Kernel fractional uncertainty (1σ, multiplicative).
# Huang & Huang (2023) report ~±5% (2σ) over Arctic melt season from
# RRTMG linearisation error + ERA5 atmospheric state variability.
KERNEL_RS_FRAC_SIGMA = 0.025   # 1σ = 2.5%  →  2σ band = ±5%

EARLY       = list(range(1982, 1992))
LATE_CMIP6  = list(range(2005, 2015))
LATE_OBS    = list(range(2015, 2025))
CERES_CMIP6 = list(range(2000, 2015))
CERES_OBS   = list(range(2000, 2025))

TREND_START_CMIP6, TREND_END_CMIP6 = 1982, 2014
TREND_START_OBS,   TREND_END_OBS   = 1982, 2024

AREA_RATIO_OCEAN  = 0.0347
SEASON_FRAC       = 6. / 12.
SCALE             = AREA_RATIO_OCEAN * SEASON_FRAC   # 0.01735
PVAL_THRESHOLDS   = [0.01, 0.05, 0.10]
SIC_ICE_THRESHOLD = 0.15
N_BOOTSTRAP       = 1000
BOOTSTRAP_SEED    = 42

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

PISTONE_REGIONS = [
    ('EU', 'Europe_Barents',              60, 90,    0,   60),
    ('WA', 'WestAsia_KaraLaptev',         60, 90,   60,  120),
    ('WP', 'WestPacific_EastSiberian',    60, 90,  120,  180),
    ('EP', 'EastPacific_ChukchiBeaufort', 60, 90, -180, -120),
    ('NA', 'NorthAmerica_CdnArchBaffin',  60, 90, -120,  -60),
    ('AO', 'Atlantic_LabradorGreenland',  60, 90,  -60,    0),
]

def log_mem(tag=''):
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.
    print(f'[MEM] {tag:<55} {gb:.2f} GB', flush=True)


# ============================================================
# TEMPERATURE DATASET LOADING
# ============================================================

def _load_gistemp(path):
    if not os.path.exists(path):
        print(f'  T-source GISTEMP not found: {path}', flush=True)
        return None, None
    try:
        df = pd.read_csv(path, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        return df['Year'].values.astype(float), df['J-D'].values.astype(float)
    except Exception as e:
        print(f'  T-source GISTEMP load error: {e}', flush=True)
        return None, None


def _load_cowtan_way(path):
    if not os.path.exists(path):
        print(f'  T-source CowtanWay not found: {path}', flush=True)
        return None, None
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None,
                         names=['year', 'anomaly', 'u1', 'u2', 'u3'])
        return df['year'].values.astype(float), df['anomaly'].values.astype(float)
    except Exception as e:
        print(f'  T-source CowtanWay load error: {e}', flush=True)
        return None, None


def _load_ncep(path):
    if not os.path.exists(path):
        print(f'  T-source NCEP not found: {path}', flush=True)
        return None, None
    try:
        ds = xr.open_dataset(path)
        t  = ds['air']
        w  = np.cos(np.deg2rad(t.lat))
        tg = t.weighted(w).mean(dim=['lat', 'lon'])
        ta = tg.resample(time='1YE').mean()
        yy = ta.time.dt.year.values.astype(float)
        vv = ta.values.squeeze().astype(float)
        ds.close()
        return yy, vv
    except Exception as e:
        print(f'  T-source NCEP load error: {e}', flush=True)
        return None, None


def _load_era5_t2m(glob_pattern):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        print(f'  T-source ERA5 no files matching: {glob_pattern}', flush=True)
        return None, None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        t  = ds['VAR_2T']
        w  = np.cos(np.deg2rad(t.latitude))
        tg = t.weighted(w).mean(dim=['latitude', 'longitude'])
        ta = tg.resample(time='1YE').mean()
        yy = ta.time.dt.year.values.astype(float)
        vv = ta.values.squeeze().astype(float)
        ds.close()
        return yy, vv
    except Exception as e:
        print(f'  T-source ERA5 load error: {e}', flush=True)
        return None, None


def _build_T_sources():
    """
    Load all four temperature datasets once at setup time.
    Returns list of (name, years_array, vals_array) for successfully
    loaded sources.  Raises RuntimeError if fewer than 2 succeed.
    """
    candidates = [
        ('GISTEMP',   _load_gistemp(GISTEMP_PATH)),
        ('CowtanWay', _load_cowtan_way(COWTANWAY_PATH)),
        ('NCEP',      _load_ncep(NCEP_PATH)),
        ('ERA5',      _load_era5_t2m(ERA5_GLOB)),
    ]
    sources = []
    for name, (yy, vv) in candidates:
        if yy is not None and vv is not None and np.isfinite(vv).sum() > 10:
            sources.append((name, yy, vv))
            print(f'  T-source {name:<12}: {len(yy)} years  '
                  f'range {int(yy[np.isfinite(yy)].min())}–'
                  f'{int(yy[np.isfinite(yy)].max())}', flush=True)
        else:
            print(f'  T-source {name:<12}: FAILED — excluded from bootstrap',
                  flush=True)
    if len(sources) < 2:
        raise RuntimeError(
            f'Only {len(sources)} T-source(s) loaded. Need at least 2.')
    return sources


# ============================================================
# SHARED HELPERS
# ============================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])


def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tree    = cKDTree(tgt_xyz)
    idx     = np.empty(src_xyz.shape[0], dtype=np.int32)
    for i in range(0, src_xyz.shape[0], chunk):
        e = min(i + chunk, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx, tgt_lat2d.shape


def build_nn_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d):
    tree = cKDTree(latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel()))
    _, idx = tree.query(latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel()),
                        workers=-1)
    del tree; gc.collect()
    return idx


def scatter_avg(arr_flat, idx, out_shape):
    n     = int(np.prod(out_shape))
    accum = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    v     = np.isfinite(arr_flat)
    np.add.at(accum, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(count, idx[v], np.float32(1.))
    out   = np.full(n, np.nan, dtype=np.float32)
    m     = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)


def regrid_sial_to_psn(arr, fill_zero=True):
    if fill_zero:
        src = np.where(arctic_sial,
                       np.where(np.isfinite(arr), arr, np.float32(0.)),
                       np.nan).astype(np.float32)
    else:
        src = arr.astype(np.float32)
    return scatter_avg(src.ravel(), idx_sial_to_psn, psn_shape)


def region_mean_psn(arr_psn, rmask):
    v, w = arr_psn[rmask], AREA_WEIGHT_PSN[rmask]
    ok   = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan, 0.
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))


def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd  = np.where(rsds  > 1., rsds,  np.nan)
        a   = np.clip(rsus / Sd, 0., 1.)
        p   = rsds / S;  q = rsut / S
        den = q + p - 2.*p*a
        R   = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan), 0., .99)
        A   = np.clip(1. - p*(1.-a*R) / np.where((1.-R) > .01, 1.-R, np.nan), 0., .99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S
        dc  = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan), 0., .99)
        Rl  = np.clip(R - Rc, 0., .99)
        RS  = (S * (1.-A) * (1.-Rl)**2
               / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS


def trend_dT(years, vals, y0, y1):
    """OLS slope × (y1 − y0).  Identical to companion delta-T script."""
    m = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if m.sum() < 5:
        return np.nan
    return stats.linregress(years[m], vals[m]).slope * (y1 - y0)


def compute_delta_T(y0, y1):
    """
    Returns (mean_dT, std_dT, per_source_array).

    Calls trend_dT for each loaded T-source and collects results.
    per_source_array is used in the bootstrap to resample structural
    ΔT uncertainty across datasets.
    """
    vals = []
    for name, yy, vv in T_SOURCES_LOADED:
        dT = trend_dT(yy, vv, y0, y1)
        if np.isfinite(dT):
            vals.append(dT)
        else:
            print(f'  compute_delta_T: {name} NaN for {y0}–{y1} '
                  f'(insufficient coverage)', flush=True)

    if not vals:
        default = 0.64 if y1 <= 2014 else 0.84
        print(f'  WARNING: no valid ΔT for {y0}–{y1}, '
              f'using default {default}', flush=True)
        return default, 0.10, np.array([default], dtype=np.float32)

    arr     = np.array(vals, dtype=np.float32)
    mean_dT = float(np.mean(arr))
    std_dT  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.

    print(f'  ΔT {y0}–{y1}: mean={mean_dT:+.4f} K  '
          f'std={std_dT:.4f} K  '
          f'n_sources={len(arr)}  '
          f'samples={arr}', flush=True)

    return mean_dT, std_dT, arr


# ============================================================
# DATA LOADERS
# ============================================================

def load_alb_sial(cache_dir, mname, years):
    use_appx = (mname == 'September')
    s, c = None, None
    for y in years:
        p = (f'{cache_dir}/appx_alb_{mname}_{y}.npy' if use_appx
             else f'{cache_dir}/clara_alb_{mname}_{y}.npy')
        if not os.path.exists(p):
            continue
        raw = np.load(p).astype(np.float32)
        if use_appx and idx_appx_to_sial is not None:
            a = (raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
                 if raw.shape != sial_shape else raw)
        else:
            a = raw
        fin = np.isfinite(a);  a[~fin] = 0.
        s   = a.copy()               if s is None else s + a
        c   = fin.astype(np.int16)   if c is None else c + fin.astype(np.int16)
        del raw, a, fin
    if s is None:
        return None
    v      = c > 0
    out    = np.full(sial_shape, np.nan, dtype=np.float32)
    out[v] = s[v] / c[v].astype(np.float32)
    return out


def load_alb_sial_yearly(cache_dir, mname, years):
    use_appx = (mname == 'September')
    result   = []
    for y in years:
        p = (f'{cache_dir}/appx_alb_{mname}_{y}.npy' if use_appx
             else f'{cache_dir}/clara_alb_{mname}_{y}.npy')
        if not os.path.exists(p):
            continue
        raw = np.load(p).astype(np.float32)
        if use_appx and idx_appx_to_sial is not None:
            a = (raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
                 if raw.shape != sial_shape else raw.copy())
        else:
            a = raw.copy()
        result.append(a);  del raw
    return result


_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']


def _t07_from_sums(sums):
    return t07_invert(sums['solar_mon'],
                      sums['toa_sw_all_mon'],
                      sums['sfc_sw_down_all_mon'],
                      sums['sfc_sw_up_all_mon'],
                      sums['toa_sw_clr_c_mon'],
                      sums['sfc_sw_down_clr_c_mon'],
                      sums['sfc_sw_up_clr_c_mon']).astype(np.float32)


def load_RS_ceres(years, mn):
    sums = {k: None for k in _CERES_VARS}
    cnt  = 0
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idxs  = [i for i, t in enumerate(times)
                 if t.year in years and t.month == mn]
        if not idxs:
            return None
        for i in idxs:
            sl = ds.isel(time=i).load()
            for k in _CERES_VARS:
                a = sl[k].values.astype(np.float32)
                sums[k] = a if cnt == 0 else sums[k] + a
            cnt += 1;  del sl
    if cnt == 0:
        return None
    for k in sums:
        sums[k] /= np.float32(cnt)
    RS = _t07_from_sums(sums)
    del sums;  gc.collect()
    return RS


def load_RS_ceres_yearly(years, mn):
    result = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in years:
            idxs = [i for i, t in enumerate(times)
                    if t.year == y and t.month == mn]
            if not idxs:
                continue
            sums = {k: None for k in _CERES_VARS}
            cnt  = 0
            for i in idxs:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1;  del sl
            if cnt == 0:
                continue
            for k in sums:
                sums[k] /= np.float32(cnt)
            result.append(_t07_from_sums(sums))
            del sums
    gc.collect()
    return result


def load_sic_data(cache_dir, mname):
    try:
        return dict(
            slope = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval  = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except Exception:
        return None


def compute_IS_from_alb(ae, al, sic_data, delta_t, pval_thresh):
    if sic_data is None:
        return None, None, None
    dalpha   = al.astype(np.float32) - ae.astype(np.float32)
    slope    = sic_data['slope']
    pval_arr = sic_data['pval']
    sic_e    = sic_data['sic_e']
    sic_l    = sic_data['sic_l']

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval_arr)
                  & (slope < 0) & (pval_arr < pval_thresh))
    stable_ice = (valid & ~changing
                  & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

    IS_ext = np.where(changing,    -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable_ice,  -dalpha / delta_t, np.nan).astype(np.float32)

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
    return IS_tot, IS_ext, IS_drk


# ============================================================
# BOOTSTRAP HELPERS
# ============================================================

def _stack(arr_list):
    valid = [a for a in arr_list if a is not None]
    if not valid:
        return None
    return np.stack(valid, axis=0).astype(np.float32)


def _resample_mean(stack, rng):
    n = stack.shape[0]
    return np.nanmean(stack[rng.integers(0, n, size=n)], axis=0).astype(np.float32)


def _region_weighted_mean_5col(IS_tot_p, IS_ext_p, IS_drk_p, RS_psn):
    """
    Area-weighted Pistone region means for SIAF_tot, SIAF_ext, SIAF_drk,
    IS_mean, RS_mean in a single pass.  Returns 5-element list or None.
    """
    reg_siaf = []
    reg_is   = []
    reg_rs   = []
    reg_wts  = []

    RS_dom = np.where(DOMAIN_PSN, RS_psn, np.nan)

    for code, *_ in PISTONE_REGIONS:
        rm = REGION_MASKS[code]
        if rm.sum() == 0:
            continue
        ST, w = region_mean_psn(IS_tot_p * RS_psn, rm)
        if not (np.isfinite(ST) and w > 0):
            continue
        SX, _   = region_mean_psn(IS_ext_p * RS_psn, rm)
        SD, _   = region_mean_psn(IS_drk_p * RS_psn, rm)
        IS_r, _ = region_mean_psn(IS_tot_p,           rm)
        RS_r, _ = region_mean_psn(RS_dom,             rm)
        reg_siaf.append([ST, SX, SD])
        reg_is.append(IS_r)
        reg_rs.append(RS_r)
        reg_wts.append(w)

    if not reg_wts:
        return None

    wts      = np.array(reg_wts,  dtype=np.float64)
    arr_siaf = np.array(reg_siaf, dtype=np.float64)
    arr_is   = np.array(reg_is,   dtype=np.float64)
    arr_rs   = np.array(reg_rs,   dtype=np.float64)

    siaf_wmean = (np.sum(arr_siaf * wts[:, None], axis=0) / np.sum(wts)).tolist()
    is_wmean   = float(np.nansum(arr_is * wts) / np.nansum(wts[np.isfinite(arr_is)]))
    rs_wmean   = float(np.nansum(arr_rs * wts) / np.nansum(wts[np.isfinite(arr_rs)]))

    return siaf_wmean + [is_wmean, rs_wmean]


def bootstrap_b2_month(early_stack, late_stack, rs_psn_stack,
                        sic_data, delta_t_samples, rng, pvals):
    """
    Bootstrap Block-2 SIAF for one month.

    Uncertainty sources propagated each iteration:
      1. Albedo early/late fields  — resample over years (interannual)
      2. CERES RS                  — resample over years (interannual)
      3. ΔT                        — draw one T-source at random (structural)
    """
    boot_res = {p: np.full((N_BOOTSTRAP, 5), np.nan, dtype=np.float64)
                for p in pvals}
    n_t = len(delta_t_samples)

    for b in range(N_BOOTSTRAP):
        ae     = _resample_mean(early_stack,  rng)
        al     = _resample_mean(late_stack,   rng)
        RS_psn = _resample_mean(rs_psn_stack, rng)
        dT     = float(delta_t_samples[rng.integers(0, n_t)])

        for pval in pvals:
            IS_tot, IS_ext, IS_drk = compute_IS_from_alb(
                ae, al, sic_data, dT, pval)
            if IS_tot is None:
                continue
            IS_t_p = regrid_sial_to_psn(IS_tot, fill_zero=True)
            IS_x_p = regrid_sial_to_psn(IS_ext, fill_zero=True)
            IS_d_p = regrid_sial_to_psn(IS_drk, fill_zero=True)
            result = _region_weighted_mean_5col(IS_t_p, IS_x_p, IS_d_p, RS_psn)
            if result is not None:
                boot_res[pval][b] = result

    return boot_res


def bootstrap_b5_month(early_stack, late_stack, k_psn,
                        sic_data, delta_t_samples, rng, pvals):
    """
    Bootstrap Block-5 SIAF for one month.

    Uncertainty sources propagated each iteration:
      1. Albedo early/late fields  — resample over years (interannual)
      2. ERA5 kernel RS            — multiplicative scalar ~ N(1, σ²)
                                     clamped to [0.80, 1.20]
                                     σ = KERNEL_RS_FRAC_SIGMA = 0.025
                                     (2σ = ±5%, Huang & Huang 2023)
      3. ΔT                        — draw one T-source at random (structural)

    The kernel perturbation is spatially uniform because Huang & Huang
    do not provide spatially-resolved uncertainty fields.  This is
    conservative: spatial correlation in kernel errors would partially
    cancel at the domain-mean level.
    """
    boot_res = {p: np.full((N_BOOTSTRAP, 5), np.nan, dtype=np.float64)
                for p in pvals}
    n_t = len(delta_t_samples)

    for b in range(N_BOOTSTRAP):
        ae = _resample_mean(early_stack, rng)
        al = _resample_mean(late_stack,  rng)
        dT = float(delta_t_samples[rng.integers(0, n_t)])

        # Kernel RS uncertainty: uniform multiplicative perturbation
        kernel_scale    = float(np.clip(
            rng.normal(1.0, KERNEL_RS_FRAC_SIGMA), 0.80, 1.20))
        k_psn_perturbed = (k_psn * kernel_scale).astype(np.float32)

        for pval in pvals:
            IS_tot, IS_ext, IS_drk = compute_IS_from_alb(
                ae, al, sic_data, dT, pval)
            if IS_tot is None:
                continue
            IS_t_p = regrid_sial_to_psn(IS_tot, fill_zero=True)
            IS_x_p = regrid_sial_to_psn(IS_ext, fill_zero=True)
            IS_d_p = regrid_sial_to_psn(IS_drk, fill_zero=True)
            result = _region_weighted_mean_5col(
                IS_t_p, IS_x_p, IS_d_p, k_psn_perturbed)
            if result is not None:
                boot_res[pval][b] = result

    return boot_res


def seasonal_boot_stats(monthly_boot_list):
    """
    Aggregate per-month bootstrap arrays into seasonal mean and 2σ.
    Handles 5-column arrays [SIAF_tot, SIAF_ext, SIAF_drk, IS_mean, RS_mean].
    """
    valid = [b for b in monthly_boot_list
             if b is not None and not np.all(np.isnan(b))]
    if not valid:
        return dict(pix=np.nan, ext=np.nan, drk=np.nan,
                    pix_2s=np.nan, ext_2s=np.nan, drk_2s=np.nan,
                    is_mean=np.nan, rs_mean=np.nan,
                    is_2s=np.nan,   rs_2s=np.nan)

    seas = np.nanmean(np.stack(valid, axis=1), axis=1)  # (N_BOOTSTRAP, ncol)

    out = dict(
        pix    = float(np.nanmean(seas[:, 0])),
        ext    = float(np.nanmean(seas[:, 1])),
        drk    = float(np.nanmean(seas[:, 2])),
        pix_2s = float(2 * np.nanstd(seas[:, 0])),
        ext_2s = float(2 * np.nanstd(seas[:, 1])),
        drk_2s = float(2 * np.nanstd(seas[:, 2])),
    )

    if seas.shape[1] >= 5:
        out.update(
            is_mean = float(np.nanmean(seas[:, 3])),
            rs_mean = float(np.nanmean(seas[:, 4])),
            is_2s   = float(2 * np.nanstd(seas[:, 3])),
            rs_2s   = float(2 * np.nanstd(seas[:, 4])),
        )
    else:
        out.update(is_mean=np.nan, rs_mean=np.nan,
                   is_2s=np.nan,   rs_2s=np.nan)
    return out


# ============================================================
# ONE-TIME SETUP
# ============================================================
print('='*70, flush=True)
print('SETUP', flush=True)
print('='*70, flush=True)

# --- Load temperature datasets first (expensive for ERA5) ---
print('Loading temperature datasets...', flush=True)
T_SOURCES_LOADED = _build_T_sources()

# --- SIAL grid ---
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

try:
    arctic_sial = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.)

# --- CERES grid ---
with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
    ceres_lat1d = _ds['lat'].values
    ceres_lon1d = _ds['lon'].values
ceres_lon2d, ceres_lat2d = np.meshgrid(ceres_lon1d, ceres_lat1d)
ceres_shape = ceres_lat2d.shape

# --- Land/ocean mask ---
_globe_mod = None
try:
    from global_land_mask import globe as _globe_mod
except ImportError:
    print('  WARNING: global-land-mask not installed; land not excluded.', flush=True)

def make_arctic_ocean_mask(lat2d, lon2d):
    if _globe_mod is None:
        return lat2d >= 60.
    lp = np.where(lon2d > 180, lon2d - 360, lon2d)
    return (lat2d >= 60.) & _globe_mod.is_ocean(lat2d, lp)

# --- APP-x -> SIAL index ---
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon
    idx_appx_to_sial = build_nn_index(appx_lat2d, appx_lon2d,
                                       sial_lat2d, sial_lon2d)
    del appx_lat2d, appx_lon2d, appx_lat, appx_lon;  gc.collect()
    print('  APP-x -> SIAL index built.', flush=True)
except Exception as e:
    print(f'  WARNING: APP-x -> SIAL failed ({e}).', flush=True)

# --- PSN25 grid ---
PSN25_AVAILABLE = False
psn_lat2d = psn_lon2d = psn_shape = None
AREA_WEIGHT_PSN = DOMAIN_PSN = None
REGION_MASKS = {}
idx_sial_to_psn = idx_ceres_to_psn = None

try:
    from pyproj import Transformer
    sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
    with xr.open_dataset(sample_sic) as ds:
        psn_x = ds['x'].values.astype(np.float64)
        psn_y = ds['y'].values.astype(np.float64)
    psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
    psn_shape  = psn_xx.shape
    tfm        = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
    psn_lon2d, psn_lat2d = tfm.transform(psn_xx, psn_yy)
    phi_ts          = np.deg2rad(70.)
    k_ps            = (1. + np.sin(phi_ts)) / np.maximum(
                          1. + np.sin(np.deg2rad(psn_lat2d)), 1e-9)
    AREA_WEIGHT_PSN = (1. / k_ps**2).astype(np.float32)
    OCEAN_PSN       = make_arctic_ocean_mask(psn_lat2d, psn_lon2d)
    DOMAIN_PSN      = OCEAN_PSN & (psn_lat2d >= 60.)
    for code, rname, lo, hi, lmn, lmx in PISTONE_REGIONS:
        lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
        lp     = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
        lon_ok = ((lp >= lmn) & (lp <= lmx)
                  if lmn <= lmx else (lp >= lmn) | (lp <= lmx))
        REGION_MASKS[code] = lat_ok & lon_ok & DOMAIN_PSN
    print(f'  PSN25 grid: {DOMAIN_PSN.sum():,} ocean cells >= 60N', flush=True)
    print('  SIAL  -> PSN25 scatter index...', flush=True)
    idx_sial_to_psn,  _ = build_scatter_index(
        sial_lat2d,  sial_lon2d, psn_lat2d, psn_lon2d)
    print('  CERES -> PSN25 scatter index...', flush=True)
    idx_ceres_to_psn, _ = build_scatter_index(
        ceres_lat2d, ceres_lon2d, psn_lat2d, psn_lon2d)
    PSN25_AVAILABLE = True
except Exception as e:
    print(f'  WARNING: PSN25 setup failed ({e}). Blocks 2 and 5 skipped.',
          flush=True)

# --- Delta T (4 sources, trend method) ---
DELTA_T_CMIP6, DELTA_T_CMIP6_STD, DELTA_T_CMIP6_SAMPLES = compute_delta_T(
    TREND_START_CMIP6, TREND_END_CMIP6)
DELTA_T_OBS, DELTA_T_OBS_STD, DELTA_T_OBS_SAMPLES = compute_delta_T(
    TREND_START_OBS, TREND_END_OBS)

PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6,
     DELTA_T_CMIP6, CERES_CMIP6, DELTA_T_CMIP6_SAMPLES),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,
     DELTA_T_OBS,   CERES_OBS,   DELTA_T_OBS_SAMPLES),
]

RNG_B2 = np.random.default_rng(BOOTSTRAP_SEED)
RNG_B5 = np.random.default_rng(BOOTSTRAP_SEED + 2)


# ============================================================
# BLOCK 2 — CERES EBAF Regional Approach (Pistone et al. 2014)
# ============================================================
print('\n' + '='*70, flush=True)
print('BLOCK 2: CERES EBAF Regional (PSN25, 6 wedges) + full bootstrap',
      flush=True)
print('  Bootstrap: albedo interannual + CERES-RS interannual + ΔT structural',
      flush=True)
print('='*70, flush=True)

block2_rows  = []
boot_b2_dict = {}

if PSN25_AVAILABLE:
    for (period, cache_dir, early_yrs, late_yrs,
         delta_t, ceres_yrs, delta_t_samples) in PERIOD_CONFIGS:

        print(f'\n  Period: {period}  DT={delta_t:.4f} K  '
              f'(bootstrap draws from {len(delta_t_samples)} T sources)',
              flush=True)

        for p in PVAL_THRESHOLDS:
            boot_b2_dict[(period, p)] = []

        for mn in MONTH_NUMS:
            mname    = MONTH_NAMES[mn]
            ae       = load_alb_sial(cache_dir, mname, early_yrs)
            al       = load_alb_sial(cache_dir, mname, late_yrs)
            sic_data = load_sic_data(cache_dir, mname)

            if ae is None or al is None or sic_data is None:
                for p in PVAL_THRESHOLDS:
                    boot_b2_dict[(period, p)].append(None)
                continue

            RS_c = load_RS_ceres(ceres_yrs, mn)
            if RS_c is None:
                for p in PVAL_THRESHOLDS:
                    boot_b2_dict[(period, p)].append(None)
                del ae, al, sic_data;  gc.collect()
                continue

            RS_psn = scatter_avg(RS_c.ravel(), idx_ceres_to_psn, psn_shape)
            del RS_c

            # Point estimates
            for pval_thresh in PVAL_THRESHOLDS:
                IS_tot, IS_ext, IS_drk = compute_IS_from_alb(
                    ae, al, sic_data, delta_t, pval_thresh)
                if IS_tot is None:
                    continue
                IS_tot_p = regrid_sial_to_psn(IS_tot, fill_zero=True)
                IS_ext_p = regrid_sial_to_psn(IS_ext, fill_zero=True)
                IS_drk_p = regrid_sial_to_psn(IS_drk, fill_zero=True)
                SIAF_t   = (IS_tot_p * RS_psn).astype(np.float32)
                SIAF_x   = (IS_ext_p * RS_psn).astype(np.float32)
                SIAF_d   = (IS_drk_p * RS_psn).astype(np.float32)
                for code, rname, *_ in PISTONE_REGIONS:
                    rm = REGION_MASKS[code]
                    if rm.sum() == 0:
                        continue
                    ST_r, w_r = region_mean_psn(SIAF_t, rm)
                    SX_r, _   = region_mean_psn(SIAF_x, rm)
                    SD_r, _   = region_mean_psn(SIAF_d, rm)
                    if not np.isfinite(ST_r):
                        continue
                    block2_rows.append(dict(
                        block='B2', config='PSN25',
                        period=period, pval=pval_thresh,
                        month=mname, region=code,
                        SIAF_pixel=ST_r, SIAF_ext=SX_r, SIAF_drk=SD_r,
                        area_weight=w_r))

            # Bootstrap
            early_alb_stk = _stack(load_alb_sial_yearly(cache_dir, mname, early_yrs))
            late_alb_stk  = _stack(load_alb_sial_yearly(cache_dir, mname, late_yrs))
            rs_c_list     = load_RS_ceres_yearly(ceres_yrs, mn)

            if rs_c_list:
                rs_psn_stk = _stack([
                    scatter_avg(r.ravel(), idx_ceres_to_psn, psn_shape)
                    for r in rs_c_list])
            else:
                rs_psn_stk = None

            if (early_alb_stk is not None
                    and late_alb_stk is not None
                    and rs_psn_stk is not None):
                boot_res = bootstrap_b2_month(
                    early_alb_stk, late_alb_stk, rs_psn_stk,
                    sic_data, delta_t_samples, RNG_B2, PVAL_THRESHOLDS)
                for p in PVAL_THRESHOLDS:
                    boot_b2_dict[(period, p)].append(boot_res[p])
            else:
                for p in PVAL_THRESHOLDS:
                    boot_b2_dict[(period, p)].append(None)

            del ae, al, sic_data, RS_psn
            del early_alb_stk, late_alb_stk, rs_psn_stk, rs_c_list
            gc.collect()
            print(f'    {mname} done', flush=True)

else:
    print('  PSN25 not available — Block 2 skipped.', flush=True)

log_mem('Block 2 complete')


# ============================================================
# BLOCK 5 — RADIATIVE KERNEL (Huang/ERA5, PSN25)
# ============================================================
print('\n' + '='*70, flush=True)
print('BLOCK 5: Radiative Kernel (Huang/ERA5, PSN25) + full bootstrap',
      flush=True)
print('  Bootstrap: albedo interannual + kernel RS uncertainty + ΔT structural',
      flush=True)
print(f'  Kernel uncertainty: 1σ = {KERNEL_RS_FRAC_SIGMA*100:.1f}%  '
      f'2σ = ±{KERNEL_RS_FRAC_SIGMA*200:.1f}%  (Huang & Huang 2023)',
      flush=True)
print('='*70, flush=True)

_kernel_psn_cache = {}

def load_kernel_psn(kname, mn):
    if kname in _kernel_psn_cache and mn in _kernel_psn_cache[kname]:
        return _kernel_psn_cache[kname][mn]
    cfg = KERNEL_CONFIGS[kname]
    if not os.path.exists(cfg['file']):
        print(f'  [B5] Kernel file missing: {cfg["file"]}', flush=True)
        return None
    try:
        with xr.open_dataset(cfg['file']) as ds:
            tdim = cfg['time_dim']
            if tdim not in ds.dims:
                tdim = next((d for d, s in ds.dims.items() if s == 12), None)
                if tdim is None:
                    raise KeyError(f'No size-12 dim in {cfg["file"]}')
            vname = cfg['var']
            if vname not in ds:
                vname = next((v for v in ds.data_vars
                               if v.lower() == cfg['var'].lower()), None)
                if vname is None:
                    raise KeyError(f'Var {cfg["var"]} not found.')
            lat_name = 'latitude'  if 'latitude'  in ds else 'lat'
            lon_name = 'longitude' if 'longitude' in ds else 'lon'
            k_lat    = ds[lat_name].values
            k_lon    = ds[lon_name].values
            k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
            k_raw    = ds[vname].isel({tdim: mn - 1}).values.astype(np.float32)
        k_raw = np.abs(k_raw) * np.float32(cfg['scale'])
        if kname not in _kernel_psn_cache:
            _kernel_psn_cache[kname] = {}
            print(f'  [B5] Building scatter index for {kname} -> PSN25',
                  flush=True)
            idx_k, _ = build_scatter_index(
                k_lat2d, k_lon2d, psn_lat2d, psn_lon2d)
            _kernel_psn_cache[kname]['_idx'] = idx_k
            del k_lat2d, k_lon2d;  gc.collect()
        idx_k = _kernel_psn_cache[kname]['_idx']
        k_psn = scatter_avg(k_raw.ravel(), idx_k, psn_shape).astype(np.float32)
        _kernel_psn_cache[kname][mn] = k_psn
        del k_raw;  gc.collect()
        return k_psn
    except Exception as e:
        print(f'  [B5] Failed loading {kname} month {mn}: {e}', flush=True)
        return None


block5_rows  = []
boot_b5_dict = {}

if PSN25_AVAILABLE:
    for (period, cache_dir, early_yrs, late_yrs,
         delta_t, ceres_yrs, delta_t_samples) in PERIOD_CONFIGS:

        print(f'\n  Period: {period}  DT={delta_t:.4f} K', flush=True)

        for p in PVAL_THRESHOLDS:
            boot_b5_dict[(period, p)] = []

        for mn in MONTH_NUMS:
            mname    = MONTH_NAMES[mn]
            ae       = load_alb_sial(cache_dir, mname, early_yrs)
            al       = load_alb_sial(cache_dir, mname, late_yrs)
            sic_data = load_sic_data(cache_dir, mname)
            kname    = 'Huang_ERA5'
            k_psn    = load_kernel_psn(kname, mn)

            if ae is None or al is None or sic_data is None or k_psn is None:
                for p in PVAL_THRESHOLDS:
                    boot_b5_dict[(period, p)].append(None)
                continue

            # Point estimates
            for pval_thresh in PVAL_THRESHOLDS:
                IS_tot, IS_ext, IS_drk = compute_IS_from_alb(
                    ae, al, sic_data, delta_t, pval_thresh)
                if IS_tot is None:
                    continue
                IS_tot_p = regrid_sial_to_psn(IS_tot, fill_zero=True)
                IS_ext_p = regrid_sial_to_psn(IS_ext, fill_zero=True)
                IS_drk_p = regrid_sial_to_psn(IS_drk, fill_zero=True)
                SIAF_t   = (IS_tot_p * k_psn).astype(np.float32)
                SIAF_x   = (IS_ext_p * k_psn).astype(np.float32)
                SIAF_d   = (IS_drk_p * k_psn).astype(np.float32)
                for code, rname, *_ in PISTONE_REGIONS:
                    rm = REGION_MASKS[code]
                    if rm.sum() == 0:
                        continue
                    ST_r, w_r = region_mean_psn(SIAF_t, rm)
                    SX_r, _   = region_mean_psn(SIAF_x, rm)
                    SD_r, _   = region_mean_psn(SIAF_d, rm)
                    if not np.isfinite(ST_r):
                        continue
                    block5_rows.append(dict(
                        block='B5', config=f'Kernel_{kname}',
                        period=period, pval=pval_thresh,
                        month=mname, region=code,
                        SIAF_pixel=ST_r, SIAF_ext=SX_r, SIAF_drk=SD_r,
                        area_weight=w_r))

            # Bootstrap
            early_alb_stk = _stack(load_alb_sial_yearly(cache_dir, mname, early_yrs))
            late_alb_stk  = _stack(load_alb_sial_yearly(cache_dir, mname, late_yrs))

            if early_alb_stk is not None and late_alb_stk is not None:
                boot_res = bootstrap_b5_month(
                    early_alb_stk, late_alb_stk, k_psn,
                    sic_data, delta_t_samples, RNG_B5, PVAL_THRESHOLDS)
                for p in PVAL_THRESHOLDS:
                    boot_b5_dict[(period, p)].append(boot_res[p])
            else:
                for p in PVAL_THRESHOLDS:
                    boot_b5_dict[(period, p)].append(None)

            del ae, al, sic_data, early_alb_stk, late_alb_stk
            gc.collect()
            print(f'    {mname} done', flush=True)

else:
    print('  PSN25 not available — Block 5 skipped.', flush=True)

log_mem('Block 5 complete')


# ============================================================
# MASTER SUMMARY
# ============================================================
print('\n' + '='*70, flush=True)
print('MASTER SUMMARY — global annual SIAF (W m⁻² K⁻¹)', flush=True)
print(f'  Scale = {AREA_RATIO_OCEAN} x {SEASON_FRAC:.3f} = {SCALE:.5f}',
      flush=True)
print('='*70, flush=True)

MONTH_LIST = ['April', 'May', 'June', 'July', 'August', 'September']

b2_agg = []
if block2_rows:
    b2_df = pd.DataFrame(block2_rows)
    for (period, pval, month), g in b2_df.groupby(['period', 'pval', 'month']):
        W = g['area_weight'].values
        def _wm(col):
            v  = g[col].values;  ok = np.isfinite(v) & (W > 0)
            return float(np.sum(v[ok]*W[ok]) / np.sum(W[ok])) if ok.sum() > 0 else np.nan
        
        # Extract monthly bootstrap uncertainty (index matches MONTH_LIST)
        m_idx = MONTH_LIST.index(month)
        month_boot = boot_b2_dict.get((period, pval), [])[m_idx]
        
        pix_2s = float(2 * np.nanstd(month_boot[:, 0])) if month_boot is not None else np.nan
        ext_2s = float(2 * np.nanstd(month_boot[:, 1])) if month_boot is not None else np.nan
        drk_2s = float(2 * np.nanstd(month_boot[:, 2])) if month_boot is not None else np.nan

        b2_agg.append(dict(block='B2', config='PSN25',
                           period=period, pval=pval, month=month,
                           SIAF_pixel=_wm('SIAF_pixel'),
                           SIAF_ext  =_wm('SIAF_ext'),
                           SIAF_drk  =_wm('SIAF_drk'),
                           sigma2_pix=pix_2s, sigma2_ext=ext_2s, sigma2_drk=drk_2s))

b5_agg = []
if block5_rows:
    b5_df = pd.DataFrame(block5_rows)
    for (period, config, pval, month), g in b5_df.groupby(['period', 'config', 'pval', 'month']):
        W = g['area_weight'].values
        def _wm5(col):
            v  = g[col].values;  ok = np.isfinite(v) & (W > 0)
            return float(np.sum(v[ok]*W[ok]) / np.sum(W[ok])) if ok.sum() > 0 else np.nan
        
        m_idx = MONTH_LIST.index(month)
        month_boot = boot_b5_dict.get((period, pval), [])[m_idx]
        
        pix_2s = float(2 * np.nanstd(month_boot[:, 0])) if month_boot is not None else np.nan
        ext_2s = float(2 * np.nanstd(month_boot[:, 1])) if month_boot is not None else np.nan
        drk_2s = float(2 * np.nanstd(month_boot[:, 2])) if month_boot is not None else np.nan

        b5_agg.append(dict(block='B5', config=config,
                           period=period, pval=pval, month=month,
                           SIAF_pixel=_wm5('SIAF_pixel'),
                           SIAF_ext  =_wm5('SIAF_ext'),
                           SIAF_drk  =_wm5('SIAF_drk'),
                           sigma2_pix=pix_2s, sigma2_ext=ext_2s, sigma2_drk=drk_2s))

all_monthly = pd.DataFrame(b2_agg + b5_agg)
all_monthly.to_csv(OUT_CSV.replace('.csv', '_monthly.csv'), index=False)

summary_rows = []
for (block, config, period, pval), g in all_monthly.groupby(
        ['block', 'config', 'period', 'pval']):

    s_pix = float(np.nanmean(g['SIAF_pixel']))
    s_ext = float(np.nanmean(g['SIAF_ext']))
    s_drk = float(np.nanmean(g['SIAF_drk']))

    boot_lookup = {'B2': boot_b2_dict, 'B5': boot_b5_dict}
    mb  = boot_lookup.get(block, {}).get((period, pval), [])
    bst = seasonal_boot_stats(mb)

    summary_rows.append(dict(
        block  = block,
        config = config,
        period = period,
        pval   = pval,
        n_months = len(g),
        SIAF_pixel_arctic  = round(s_pix, 5),
        SIAF_ext_arctic    = round(s_ext, 5),
        SIAF_drk_arctic    = round(s_drk, 5),
        SIAF_pixel_global  = round(s_pix * SCALE, 6),
        SIAF_ext_global    = round(s_ext * SCALE, 6),
        SIAF_drk_global    = round(s_drk * SCALE, 6),
        sigma2_pix_global  = round(bst['pix_2s'] * SCALE, 6),
        sigma2_ext_global  = round(bst['ext_2s'] * SCALE, 6),
        sigma2_drk_global  = round(bst['drk_2s'] * SCALE, 6),
        IS_arctic_mean     = round(bst['is_mean'], 6),
        RS_arctic_mean     = round(bst['rs_mean'], 4),
        IS_arctic_2sigma   = round(bst['is_2s'],   6),
        RS_arctic_2sigma   = round(bst['rs_2s'],   4),
        ext_pct    = round(s_ext / s_pix * 100 if s_pix != 0 else np.nan, 1),
        drk_pct    = round(s_drk / s_pix * 100 if s_pix != 0 else np.nan, 1),
        scale_used = round(SCALE, 6),
    ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_CSV, index=False)

hdr = (f"{'Blk':<4} {'Config':<22} {'Period':<8} "
       f"{'pix_glo ± 2σ':^22} {'ext_glo ± 2σ':^22} {'drk_glo ± 2σ':^22} "
       f"{'IS ± 2σ':^20} {'RS ± 2σ':^18}")

for pval in PVAL_THRESHOLDS:
    print(f'\n--- p < {pval} ---', flush=True)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    sub = summary[summary['pval'] == pval].sort_values(
        ['block', 'config', 'period'])
    for _, r in sub.iterrows():
        pix_s = f"{r['SIAF_pixel_global']:+.4f} ± {r['sigma2_pix_global']:.4f}"
        ext_s = f"{r['SIAF_ext_global']:+.4f} ± {r['sigma2_ext_global']:.4f}"
        drk_s = f"{r['SIAF_drk_global']:+.4f} ± {r['sigma2_drk_global']:.4f}"
        is_s  = (f"{r['IS_arctic_mean']:.5f} ± {r['IS_arctic_2sigma']:.5f}"
                 if np.isfinite(r['IS_arctic_mean']) else 'n/a')
        rs_s  = (f"{r['RS_arctic_mean']:.2f} ± {r['RS_arctic_2sigma']:.2f}"
                 if np.isfinite(r['RS_arctic_mean']) else 'n/a')
        print(f"{r['block']:<4} {r['config']:<22} {r['period']:<8} "
              f"{pix_s:^22} {ext_s:^22} {drk_s:^22} "
              f"{is_s:^20} {rs_s:^18}", flush=True)

print('\n--- Obs-era minus CMIP6 ---', flush=True)
for pval in PVAL_THRESHOLDS:
    print(f'  p < {pval}:', flush=True)
    for block in summary['block'].unique():
        for config in summary.loc[summary['block']==block, 'config'].unique():
            c = summary.loc[(summary['block']  == block) &
                            (summary['config'] == config) &
                            (summary['period'] == 'CMIP6') &
                            (summary['pval']   == pval)]
            o = summary.loc[(summary['block']  == block) &
                            (summary['config'] == config) &
                            (summary['period'] == 'Obs-era') &
                            (summary['pval']   == pval)]
            if c.empty or o.empty:
                continue
            dg = float(o['SIAF_pixel_global'].iloc[0]
                       - c['SIAF_pixel_global'].iloc[0])
            print(f'    {block:<4} {config:<22}  '
                  f'delta_global = {dg:+.6f}', flush=True)

print(f'\nMonthly -> {OUT_CSV.replace(".csv","_monthly.csv")}', flush=True)
print(f'Summary -> {OUT_CSV}', flush=True)


# ============================================================
# BAR CHART
# ============================================================
print('\nGenerating bar charts...', flush=True)

def make_barplot(summary_df, out_path, pval):
    methods = [
        ('B2', 'PSN25',             'CERES EBAF Regional Approach\n(Pistone et al. 2014)'),
        ('B5', 'Kernel_Huang_ERA5', 'Radiative Kernel Approach\n(Huang and Huang, 2023)'),
    ]
    periods  = ['CMIP6', 'Obs-era']
    COL_EXT  = '#3A6BAF'
    COL_DRK  = '#D95F1A'
    ALPHA    = {'CMIP6': 0.50, 'Obs-era': 1.00}
    HATCHES  = {'CMIP6': '////', 'Obs-era': ''}
    n        = len(methods)
    x        = np.arange(n)
    w        = 0.30
    offsets  = {'CMIP6': -(w/2 + 0.03), 'Obs-era': (w/2 + 0.03)}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    legend_handles = []

    for period in periods:
        a   = ALPHA[period]
        hat = HATCHES[period]
        for i, (blk, cfg, _) in enumerate(methods):
            row = summary_df[(summary_df['block']  == blk) &
                             (summary_df['config'] == cfg) &
                             (summary_df['period'] == period)]
            if row.empty:
                continue
            ext_val = float(row['SIAF_ext_global'].iloc[0])
            drk_val = float(row['SIAF_drk_global'].iloc[0])
            tot_val = float(row['SIAF_pixel_global'].iloc[0])
            err     = float(row['sigma2_pix_global'].iloc[0])
            xp      = x[i] + offsets[period]

            ax.bar(xp, ext_val, w, color=COL_EXT, alpha=a, hatch=hat,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.bar(xp, drk_val, w, bottom=ext_val,
                   color=COL_DRK, alpha=a, hatch=hat,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.errorbar(xp, tot_val, yerr=err, fmt='none',
                        color='#222222', capsize=5,
                        linewidth=1.5, capthick=1.5, zorder=5)

            ext_pct = ext_val / tot_val * 100 if tot_val > 0 else 0
            drk_pct = drk_val / tot_val * 100 if tot_val > 0 else 0
            if ext_val > 0.004:
                ax.text(xp, ext_val / 2, f'{ext_pct:.0f}%',
                        ha='center', va='center',
                        color='white', fontsize=8.5, fontweight='bold', zorder=6)
            if drk_val > 0.004:
                ax.text(xp, ext_val + drk_val / 2, f'{drk_pct:.0f}%',
                        ha='center', va='center',
                        color='white', fontsize=8.5, fontweight='bold', zorder=6)

        legend_handles.append(mpatches.Patch(
            facecolor=COL_EXT, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Extent ({period})'))
        legend_handles.append(mpatches.Patch(
            facecolor=COL_DRK, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Darkening ({period})'))

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in methods], fontsize=11)

    for i in range(n):
        for period in periods:
            row = summary_df[(summary_df['block']  == methods[i][0]) &
                             (summary_df['config'] == methods[i][1]) &
                             (summary_df['period'] == period)]
            if row.empty:
                continue
            tot = float(row['SIAF_pixel_global'].iloc[0])
            err = float(row['sigma2_pix_global'].iloc[0])
            xp  = x[i] + offsets[period]
            ax.text(xp, tot + err + 0.002,
                    'C6' if period == 'CMIP6' else 'Obs',
                    ha='center', va='bottom',
                    color='#555555', fontsize=7.5, style='italic')

    ax.set_ylabel('Global Annual SIAF  (W m⁻² K⁻¹)', fontsize=11)
    ax.set_title(f'Sea Ice–Albedo Feedback: Extent vs Darkening  (p < {pval})',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', colors='#444444')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, ncol=2, columnspacing=0.8,
              handlelength=1.4, handleheight=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Bar chart -> {out_path}', flush=True)


for pval in PVAL_THRESHOLDS:
    sub_df = summary[summary['pval'] == pval]
    if not sub_df.empty:
        make_barplot(sub_df,
                     PLOT_OUT.replace('.png', f'_p{pval}.png'),
                     pval)

# ============================================================
# SPATIAL PLOTS OF RS (CERES vs HUANG KERNEL) — UPDATED
# ============================================================

print('\n' + '='*70, flush=True)
print('SPATIAL PLOTS: CERES RS vs Huang ERA5 Kernel RS', flush=True)
print('='*70, flush=True)

if PSN25_AVAILABLE:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.path import Path
        CARTOPY_OK = True
    except ImportError:
        print('  cartopy not available — install with: conda install -c conda-forge cartopy',
              flush=True)
        CARTOPY_OK = False

    from scipy.interpolate import NearestNDInterpolator

    # ------------------------------------------------------------------
    # 1.  Build monthly mean SIC on PSN25 (Apr–Sep) for RS gating
    # ------------------------------------------------------------------
    print('  Building monthly mean SIC for RS gating...', flush=True)
    sic_monthly_mean = {}   # {month_num: 2D array on psn_shape}
    for mn in MONTH_NUMS:
        _sum   = np.zeros(psn_shape, dtype=np.float64)
        _count = np.zeros(psn_shape, dtype=np.int32)
        for _f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc')):
            try:
                with xr.open_dataset(_f) as _ds:
                    if pd.Timestamp(_ds['time'].values[0]).month != mn:
                        continue
                    _raw = _ds['cdr_seaice_conc'].squeeze().values.astype(float)
                    if _raw.ndim > 2: _raw = _raw[0]
                    if np.nanmax(_raw) > 1.5: _raw /= 100.
                    _raw = np.where((_raw >= 0.) & (_raw <= 1.), _raw, np.nan)
                    _ok  = np.isfinite(_raw)
                    _sum[_ok]   += _raw[_ok]
                    _count[_ok] += 1
            except Exception:
                continue
        _mean = np.full(psn_shape, np.nan)
        _v = _count > 0
        _mean[_v] = _sum[_v] / _count[_v]
        sic_monthly_mean[mn] = _mean
        print(f'    Month {mn}: {np.sum(_mean > SIC_ICE_THRESHOLD):,} cells > 15% SIC',
              flush=True)

    # ------------------------------------------------------------------
    # 2.  Accumulate RS means — only where monthly mean SIC > 15%
    # ------------------------------------------------------------------
    print('  Accumulating Apr–Sep RS means (SIC > 15% gate)...', flush=True)
    rs_ceres_sum   = np.zeros(psn_shape, dtype=np.float64)
    rs_ceres_count = np.zeros(psn_shape, dtype=np.int32)
    rs_kernel_sum  = np.zeros(psn_shape, dtype=np.float64)
    rs_kernel_cnt  = np.zeros(psn_shape, dtype=np.int32)

    for mn in MONTH_NUMS:
        RS_c  = load_RS_ceres(CERES_OBS, mn)
        k_psn = load_kernel_psn('Huang_ERA5', mn)

        # Ice gate: only accumulate where this month's mean SIC > 15%
        ice_gate = sic_monthly_mean[mn] > SIC_ICE_THRESHOLD

        if RS_c is not None:
            RS_psn = scatter_avg(RS_c.ravel(), idx_ceres_to_psn, psn_shape)
            ok = np.isfinite(RS_psn) & ice_gate
            rs_ceres_sum[ok]   += RS_psn[ok]
            rs_ceres_count[ok] += 1

        if k_psn is not None:
            ok = np.isfinite(k_psn) & ice_gate
            rs_kernel_sum[ok] += k_psn[ok]
            rs_kernel_cnt[ok] += 1

        del RS_c, k_psn;  gc.collect()

    rs_ceres_mean  = np.full(psn_shape, np.nan, dtype=np.float32)
    rs_kernel_mean = np.full(psn_shape, np.nan, dtype=np.float32)
    ok = rs_ceres_count > 0
    rs_ceres_mean[ok]  = (rs_ceres_sum[ok]  / rs_ceres_count[ok]).astype(np.float32)
    ok = rs_kernel_cnt > 0
    rs_kernel_mean[ok] = (rs_kernel_sum[ok] / rs_kernel_cnt[ok]).astype(np.float32)

    # ------------------------------------------------------------------
    # 2.  Mask PSN25 pole hole (lat > 87.2°)
    # ------------------------------------------------------------------
    POLE_HOLE = psn_lat2d > 87.2
    rs_ceres_mean[POLE_HOLE]  = np.nan
    rs_kernel_mean[POLE_HOLE] = np.nan

    # ------------------------------------------------------------------
    # 3.  Fill sparse PSN25 grid cells within DOMAIN_PSN
    # ------------------------------------------------------------------
    def fill_domain(arr, xx, yy, mask):
        src = np.isfinite(arr) & mask
        if not src.any():
            return arr
        interp = NearestNDInterpolator(
            np.column_stack([xx[src], yy[src]]), arr[src])
        out = interp(np.column_stack([xx.ravel(), yy.ravel()])).reshape(arr.shape)
        out = out.astype(np.float32)
        out[~mask] = np.nan
        return out

    rs_ceres_plot  = fill_domain(rs_ceres_mean,  psn_xx, psn_yy, DOMAIN_PSN)
    rs_kernel_plot = fill_domain(rs_kernel_mean, psn_xx, psn_yy, DOMAIN_PSN)
    rs_diff_plot   = rs_ceres_plot - rs_kernel_plot

    # ------------------------------------------------------------------
    # 4.  April SIC mask — only retain cells with nonzero mean April SIC
    # ------------------------------------------------------------------
    print('  Building April SIC mask...', flush=True)
    april_sic_sum   = np.zeros(psn_shape, dtype=np.float64)
    april_sic_count = np.zeros(psn_shape, dtype=np.int32)

    for _f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc')):
        try:
            with xr.open_dataset(_f) as _ds:
                if pd.Timestamp(_ds['time'].values[0]).month != 4:
                    continue
                _raw = _ds['cdr_seaice_conc'].squeeze().values.astype(float)
                if _raw.ndim > 2:
                    _raw = _raw[0]
                if np.nanmax(_raw) > 1.5:
                    _raw /= 100.
                _raw = np.where((_raw >= 0.) & (_raw <= 1.), _raw, np.nan)
                _ok  = np.isfinite(_raw)
                april_sic_sum[_ok]   += _raw[_ok]
                april_sic_count[_ok] += 1
        except Exception:
            continue

    april_sic_mean = np.full(psn_shape, np.nan)
    _v = april_sic_count > 0
    april_sic_mean[_v] = april_sic_sum[_v] / april_sic_count[_v]

    # Cells must be in DOMAIN_PSN AND have nonzero mean April SIC
    APRIL_ICE_MASK = DOMAIN_PSN & (april_sic_mean > 0.)
    print(f'  April ice mask: {APRIL_ICE_MASK.sum():,} cells '
          f'(of {DOMAIN_PSN.sum():,} domain cells)', flush=True)

    # Apply mask — NaN out non-ice cells in all three fields
    rs_ceres_plot  = np.where(APRIL_ICE_MASK, rs_ceres_plot,  np.nan).astype(np.float32)
    rs_kernel_plot = np.where(APRIL_ICE_MASK, rs_kernel_plot, np.nan).astype(np.float32)
    rs_diff_plot   = np.where(APRIL_ICE_MASK, rs_diff_plot,   np.nan).astype(np.float32)

    # ------------------------------------------------------------------
    # 5.  Agreement statistics (masked to April ice cells only)
    # ------------------------------------------------------------------
    dom = APRIL_ICE_MASK & np.isfinite(rs_ceres_plot) & np.isfinite(rs_kernel_plot)
    c_v = rs_ceres_plot[dom].ravel()
    k_v = rs_kernel_plot[dom].ravel()
    r_val = float(np.corrcoef(c_v, k_v)[0, 1])
    bias  = float(np.mean(c_v - k_v))
    rmse  = float(np.sqrt(np.mean((c_v - k_v) ** 2)))
    print(f'  RS agreement (April ice cells)  r={r_val:.3f}  '
          f'bias={bias:+.2f} W m⁻²  RMSE={rmse:.2f} W m⁻²', flush=True)

    # ------------------------------------------------------------------
    # 6.  Normalise PSN25 longitudes to −180 … 180 for cartopy
    # ------------------------------------------------------------------
    lon_plot = np.where(psn_lon2d > 180, psn_lon2d - 360., psn_lon2d)

    # ------------------------------------------------------------------
    # 7.  Colour scale
    # ------------------------------------------------------------------
    vmax_rs = float(np.nanpercentile(np.concatenate([c_v, k_v]), 98))
    vmax_rs = max(50., vmax_rs)
    diff_lim = 50.

    if CARTOPY_OK:
        proj     = ccrs.NorthPolarStereo(central_longitude=0.)
        data_crs = ccrs.PlateCarree()
        LAND_COLOR = '#C8C8C8'

        def add_map_features(ax):
            ax.add_feature(
                cfeature.LAND.with_scale('50m'),
                facecolor=LAND_COLOR, edgecolor='none', zorder=2)
            ax.add_feature(
                cfeature.COASTLINE.with_scale('50m'),
                linewidth=0.9, edgecolor='#333333', zorder=4)
            ax.add_feature(
                cfeature.BORDERS.with_scale('50m'),
                linewidth=0.45, edgecolor='#666666', linestyle=':', zorder=4)
            lons60 = np.linspace(-180., 180., 721)
            ax.plot(lons60, np.full_like(lons60, 60.),
                    transform=data_crs,
                    color='#333333', linewidth=0.8, linestyle='--', zorder=5)

        def set_circular_boundary(ax):
            theta  = np.linspace(0, 2 * np.pi, 200)
            center, radius = [0.5, 0.5], 0.5
            verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(22, 6.2), facecolor='white')
        gs  = GridSpec(
            1, 4, figure=fig,
            width_ratios=[1, 1, 1, 1.15],
            wspace=0.08,
            left=0.01, right=0.99,
            top=0.87, bottom=0.10,
        )

        map_specs = [
            (rs_ceres_plot,
             r'$R_S^{\mathrm{iso}}$',
             'YlGnBu_r', 0., vmax_rs,
             r'$R_S$  [W m$^{-2}$]'),
            (rs_kernel_plot,
             r'$R_S^{\mathrm{ERA5}}$',
             'YlGnBu_r', 0., vmax_rs,
             r'$R_S$  [W m$^{-2}$]'),
            (rs_diff_plot,
             r'$R_S^{\mathrm{iso}} - R_S^{\mathrm{ERA5}}$',
             'RdBu_r', -diff_lim, diff_lim,
             r'$\Delta R_S$  [W m$^{-2}$]'),
        ]

        for col, (data, title, cmap, vmin, vmax, cblabel) in enumerate(map_specs):
            ax = fig.add_subplot(gs[0, col], projection=proj)
            ax.set_extent([-180, 180, 55., 90.], crs=data_crs)
            set_circular_boundary(ax)
            ax.set_facecolor('#D8EEFF')

            im = ax.pcolormesh(
                lon_plot, psn_lat2d, data,
                transform=data_crs,
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading='nearest', zorder=1, rasterized=True)

            add_map_features(ax)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

            cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                              pad=0.04, fraction=0.046, shrink=0.88)
            cb.set_label(cblabel, fontsize=9.5)
            cb.ax.tick_params(labelsize=8.5)

            if col == 2:
                ax.text(0.5, -0.20,
                        f'Bias = {bias:+.1f} W m⁻²\n'
                        f'RMSE = {rmse:.1f} W m⁻²',
                        ha='center', va='top', fontsize=9.5,
                        transform=ax.transAxes, color='#333333')

        # ------ Scatter panel ------
        ax_sc = fig.add_subplot(gs[0, 3])

        step = max(1, len(c_v) // 10_000)
        cs, ks = c_v[::step], k_v[::step]

        hb = ax_sc.hexbin(cs, ks, gridsize=55, cmap='YlOrRd',
                          mincnt=1, linewidths=0.15)
        cb3 = plt.colorbar(hb, ax=ax_sc, pad=0.02)
        cb3.set_label('Grid cells', fontsize=9)
        cb3.ax.tick_params(labelsize=8)

        ax_sc.plot([0, vmax_rs], [0, vmax_rs], 'k--',
                   linewidth=1.2, label='1 : 1', zorder=5)

        m_reg, b_reg, *_ = stats.linregress(cs, ks)
        x_fit = np.array([0., vmax_rs])
        ax_sc.plot(x_fit, m_reg * x_fit + b_reg,
                   color='#1A5CA8', linewidth=1.2,
                   label=f'OLS  (slope={m_reg:.2f})', zorder=5)

        ax_sc.set_xlim(0, vmax_rs * 1.04)
        ax_sc.set_ylim(0, vmax_rs * 1.04)
        ax_sc.set_aspect('equal')
        ax_sc.set_xlabel(r'$R_S^{\mathrm{iso}}$  [W m$^{-2}$]', fontsize=10.5)
        ax_sc.set_ylabel(r'$R_S^{\mathrm{ERA5}}$  [W m$^{-2}$]', fontsize=10.5)
        ax_sc.set_title(
            r'$R_S^{\mathrm{iso}}$ vs $R_S^{\mathrm{ERA5}}$'
            + '\n(April ice-covered cells, ≥ 60°N)',
            fontsize=11, fontweight='bold')
        ax_sc.legend(fontsize=9, loc='upper left')
        ax_sc.grid(True, alpha=0.30, linewidth=0.6)

        stats_txt = (f'$r$ = {r_val:.3f}\n'
                     f'Bias = {bias:+.1f} W m⁻²\n'
                     f'RMSE = {rmse:.1f} W m⁻²')
        ax_sc.text(0.97, 0.06, stats_txt,
                   ha='right', va='bottom', fontsize=9.5,
                   transform=ax_sc.transAxes, color='#222222',
                   bbox=dict(boxstyle='round,pad=0.35',
                             facecolor='white', alpha=0.85,
                             edgecolor='#CCCCCC'))

        fig.suptitle(
            r'Shortwave Radiative Effect: $R_S^{\mathrm{iso}}$ (CERES EBAF) '
            r'vs $R_S^{\mathrm{ERA5}}$ (Radiative Kernel)',
            fontsize=13.5, fontweight='bold', y=0.97)

        rs_plot_path = os.path.join(OUT_DIR, 'rs_spatial_comparison.png')
        fig.savefig(rs_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Spatial RS comparison -> {rs_plot_path}', flush=True)

    else:
        print('  Cartopy unavailable; generating fallback plain-grid plot.',
              flush=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')
        for ax, data, title, cmap, vmin, vmax in [
            (axes[0], rs_ceres_plot,
             r'$R_S^{\mathrm{iso}}$',       'YlGnBu_r', 0, vmax_rs),
            (axes[1], rs_kernel_plot,
             r'$R_S^{\mathrm{ERA5}}$',      'YlGnBu_r', 0, vmax_rs),
            (axes[2], rs_diff_plot,
             r'$R_S^{\mathrm{iso}} - R_S^{\mathrm{ERA5}}$',
             'RdBu_r', -diff_lim, diff_lim),
        ]:
            im = ax.pcolormesh(psn_xx, psn_yy, data,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='nearest')
            cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                              pad=0.05, fraction=0.046)
            cb.set_label(r'$R_S$  [W m$^{-2}$]', fontsize=9)
            ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
        fig.tight_layout()
        rs_plot_path = os.path.join(OUT_DIR, 'rs_spatial_comparison.png')
        fig.savefig(rs_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Fallback spatial plot -> {rs_plot_path}', flush=True)

else:
    print('  PSN25 not available — spatial plots skipped.', flush=True)

log_mem('DONE')