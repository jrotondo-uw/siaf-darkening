#!/usr/bin/env python
"""
cesm2_gdex_siaf.py
───────────────────
Compute scalar SIAF_global for the CESM2 tuned sea-ice-albedo GDEX dataset
(Kay et al. 2022 IMPACTS, d651070) and compare to standard CESM2 CMIP6 members.

Two experiment groups on GDEX (/gdex/data/d651070/):
  BHIST   012-015  Standard historical CESM2 (extra members not in CMIP6 archive)
                   Period: 1982-2014  (files span 1850-2014)
  BHISTsmbb 118-122  Tuned sea-ice broadband melt-pond albedo
                   Period: 1990-2014  (files only start 1990)

Output
──────
  PLOT_DIR/cesm2_gdex_siaf_comparison.png   — bar chart + table
  PLOT_DIR/cesm2_gdex_siaf_results.csv      — scalar SIAF_global per member
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse math utilities from the main CMIP6 script
from cmip6_siaf_calculation import (
    MONTH_NUMS, EARLY_YEARS, LATE_YEARS,
    SIC_ICE_THRESHOLD, PVAL,
    MODEL_INFO, find_var_files,
    latlon_to_xyz,
    build_native_ctx,
    compute_pixel_ols_slopes,
    pixel_ols_trend,
    fill_zero_domain,
    arctic_ocean_mean_native,
    seasonal_to_global,
    t07_invert,
    _open_robust,
)

# ============================================================
# CONFIG
# ============================================================
GDEX_ROOT = '/gdex/data/d651070'
PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
CSV_CMIP6 = os.path.join(PLOT_DIR, 'siaf_cmip6_pistone_summary.csv')
OUT_PNG   = os.path.join(PLOT_DIR, 'cesm2_gdex_siaf_comparison.png')
OUT_CSV   = os.path.join(PLOT_DIR, 'cesm2_gdex_siaf_results.csv')

BHIST_CASES = [
    'b.e21.BHIST.f09_g17.CMIP6-historical.012',
    'b.e21.BHIST.f09_g17.CMIP6-historical.013',
    'b.e21.BHIST.f09_g17.CMIP6-historical.014',
    'b.e21.BHIST.f09_g17.CMIP6-historical.015',
]

SMBB_CASES = [
    'b.e21.BHISTsmbb.f09_g17.CMIP6-historical.118',
    'b.e21.BHISTsmbb.f09_g17.CMIP6-historical.119',
    'b.e21.BHISTsmbb.f09_g17.CMIP6-historical.120',
    'b.e21.BHISTsmbb.f09_g17.CMIP6-historical.121',
    'b.e21.BHISTsmbb.f09_g17.CMIP6-historical.122',
]

# BHIST has full 1850-2014, BHISTsmbb only 1990-2014
BHIST_TREND_YEARS = list(range(1982, 2015))   # 33 years
SMBB_TREND_YEARS  = list(range(1990, 2015))   # 25 years

# OLS uses [1980,1981] as pre-period for stability (BHIST only)
BHIST_ALL_YEARS = list(range(1980, 2015))
SMBB_ALL_YEARS  = list(range(1990, 2015))

N_WORKERS = 4


# ============================================================
# FILE FINDERS
# ============================================================

def _is_smbb(case):
    return 'BHISTsmbb' in case


def gdex_cam_files(case, var):
    """Sorted monthly CAM h0 files for a variable."""
    if _is_smbb(case):
        base = os.path.join(GDEX_ROOT, case, 'atm', 'proc', 'tseries', 'month_1')
    else:
        base = GDEX_ROOT
    return sorted(glob.glob(os.path.join(base, f'{case}.cam.h0.{var}.*.nc')))


def gdex_cice_files(case, var='aice'):
    """Sorted monthly CICE h files for a variable."""
    if _is_smbb(case):
        base = os.path.join(GDEX_ROOT, case, 'ice', 'proc', 'tseries', 'month_1')
    else:
        base = GDEX_ROOT
    return sorted(glob.glob(os.path.join(base, f'{case}.cice.h.{var}.*.nc')))


# ============================================================
# LOADERS
# ============================================================

_OPEN_KW = dict(use_cftime=True, decode_times=True)


def _open_mfd(files, var):
    """Open a multi-file monthly dataset, returning DataArray."""
    ds = xr.open_mfdataset(
        files, combine='by_coords', data_vars='minimal',
        coords='minimal', compat='override', **_OPEN_KW)
    return ds[var]


def load_cam_years_months(case, var, years, months):
    """
    Load a CAM h0 variable filtered to given years and months.
    Returns xr.DataArray on f09 (192 lat × 288 lon) grid.
    """
    files = gdex_cam_files(case, var)
    if not files:
        return None
    da = _open_mfd(files, var)
    yr = np.array([int(t.dt.year)  for t in da.time])
    mn = np.array([int(t.dt.month) for t in da.time])
    mask = np.isin(yr, years) & np.isin(mn, months)
    if not mask.any():
        return None
    return da.isel(time=mask).load()


def load_cice_years_months(case, var, years, months):
    """
    Load a CICE h variable filtered to given years and months.
    Returns xr.DataArray on gx1v7 (384 nj × 320 ni) grid.
    TLAT/TLON are 2D coords.
    """
    files = gdex_cice_files(case, var)
    if not files:
        return None
    da = _open_mfd(files, var)
    yr = np.array([int(t.dt.year)  for t in da.time])
    mn = np.array([int(t.dt.month) for t in da.time])
    mask = np.isin(yr, years) & np.isin(mn, months)
    if not mask.any():
        return None
    return da.isel(time=mask).load()


# ============================================================
# T ANNUAL
# ============================================================

def compute_gdex_T_annual(case, all_years):
    """
    Area-weighted global mean annual TREFHT.
    Returns dict {year: T_K}.
    """
    da = load_cam_years_months(case, 'TREFHT', all_years, list(range(1, 13)))
    if da is None:
        return {}
    lat = da.lat.values
    lon = da.lon.values
    weights = np.cos(np.deg2rad(lat))
    da_w = da.weighted(xr.DataArray(weights, coords=[da.lat], dims=['lat']))
    yr = np.array([int(t.dt.year) for t in da.time])
    result = {}
    for y in sorted(set(yr)):
        mask = yr == y
        if mask.sum() == 12:
            result[y] = float(da_w.mean(['lat', 'lon']).isel(time=mask).mean().values)
    return result


# ============================================================
# IS COMPUTATION  (mirrors compute_member_IS_native)
# ============================================================

def compute_gdex_IS(case, T_annual, trend_years, early_years, late_years):
    """
    Compute pixel-level IS (ice shading) from GDEX CESM2 native files.
    Uses FSDS/FSNS for surface albedo, aice for SIC classification.

    Returns
    -------
    IS_result : dict {month: {'tot', 'ext', 'drk'} on f09 grid}
    native_ctx: standard native context (lat2d, lon2d, shape, …)
    """
    all_years = sorted(set(trend_years) | set(early_years) | set(late_years))

    # ── Load atmosphere SW (f09 grid) ─────────────────────────────────────
    fsds_da = load_cam_years_months(case, 'FSDS', all_years, MONTH_NUMS)
    fsns_da = load_cam_years_months(case, 'FSNS', all_years, MONTH_NUMS)
    if fsds_da is None or fsns_da is None:
        return None, None

    lat  = fsds_da.lat.values   # (192,)
    lon  = fsds_da.lon.values   # (288,)
    atm_shape = (len(lat), len(lon))

    # Build native context from f09 regular grid
    native_ctx = build_native_ctx(lat, lon, False)

    alb_yr  = np.array([int(t.dt.year)  for t in fsds_da.time])
    alb_mon = np.array([int(t.dt.month) for t in fsds_da.time])

    # ── Load CICE aice (gx1v7, 384×320) ──────────────────────────────────
    aice_da = load_cice_years_months(case, 'aice', all_years, MONTH_NUMS)
    if aice_da is None:
        return None, None

    # Extract TLAT/TLON
    ds_tmp = xr.open_dataset(gdex_cice_files(case)[0], **_OPEN_KW)
    cice_lat = ds_tmp['TLAT'].values.astype(np.float32)   # (384,320)
    cice_lon = ds_tmp['TLON'].values.astype(np.float32)
    ds_tmp.close()

    sv = aice_da.values.astype(np.float64)   # already 0-1 fraction
    si_yr  = np.array([int(t.dt.year)  for t in aice_da.time])
    si_mon = np.array([int(t.dt.month) for t in aice_da.time])

    # Regrid aice gx1v7 → f09 via KDTree (same approach as main script)
    cice_valid = np.isfinite(cice_lat) & np.isfinite(cice_lon)
    cla = cice_lat[cice_valid].ravel()
    clo = cice_lon[cice_valid].ravel()

    lon2d_atm, lat2d_atm = np.meshgrid(lon, lat)
    kd = cKDTree(latlon_to_xyz(cla, clo).astype(np.float32))
    _, idx_c2a = kd.query(
        latlon_to_xyz(lat2d_atm.ravel(), lon2d_atm.ravel()).astype(np.float32),
        workers=-1)

    sv_flat = sv.reshape(sv.shape[0], -1)[:, cice_valid.ravel()]
    sv_atm  = sv_flat[:, idx_c2a].reshape((sv.shape[0],) + atm_shape)

    del sv_flat, sv, aice_da
    gc.collect()

    lat2d_a = native_ctx['lat2d']

    T_center = float(np.mean([T_annual[y] for y in trend_years
                               if y in T_annual])) if T_annual else 0.

    result = {}
    for mn in MONTH_NUMS:
        # Per-year albedo: α = rsus/rsds = (FSDS - FSNS)/FSDS
        yearly_alb = {}
        for year in trend_years:
            if year not in T_annual:
                continue
            m = (alb_yr == year) & (alb_mon == mn)
            if m.sum() == 0:
                continue
            fsds = fsds_da.isel(time=m).mean('time').values
            fsns = fsns_da.isel(time=m).mean('time').values
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(fsds > 1., (fsds - fsns) / fsds, np.nan)
            yearly_alb[year] = np.clip(alb, 0., 1.)

        valid_yrs = sorted(yearly_alb)
        if len(valid_yrs) < 10:
            result[mn] = None
            continue

        stk   = np.stack([yearly_alb[y] for y in valid_yrs], axis=0)
        T_vec = np.array([T_annual[y] - T_center for y in valid_yrs],
                         dtype=np.float64)

        dalpha_dT = compute_pixel_ols_slopes(stk, T_vec)
        IS_raw    = (-dalpha_dT).astype(np.float32)

        # SIC classification using regridded aice on f09
        tm = (si_mon == mn) & np.isin(si_yr, trend_years)
        em = (si_mon == mn) & np.isin(si_yr, early_years)
        lm = (si_mon == mn) & np.isin(si_yr, late_years)
        if tm.sum() < 10 or em.sum() == 0 or lm.sum() == 0:
            result[mn] = None
            continue

        sic_e = np.nanmean(sv_atm[em], axis=0)
        sic_l = np.nanmean(sv_atm[lm], axis=0)
        slope_sic, pval_sic = pixel_ols_trend(
            si_yr[tm].astype(float), sv_atm[tm])

        all_ice  = (lat2d_a >= 60.) & (sic_e >= SIC_ICE_THRESHOLD)
        changing = (all_ice & np.isfinite(slope_sic) & np.isfinite(pval_sic)
                    & (slope_sic < 0) & (pval_sic < PVAL))
        stable   = (all_ice & ~changing
                    & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

        IS_ext = np.where(changing, IS_raw, np.nan).astype(np.float32)
        IS_drk = np.where(stable,   IS_raw, np.nan).astype(np.float32)
        IS_tot = np.where(
            np.isfinite(IS_ext) | np.isfinite(IS_drk),
            np.where(np.isfinite(IS_ext), IS_ext, 0.)
            + np.where(np.isfinite(IS_drk), IS_drk, 0.),
            np.nan).astype(np.float32)
        IS_ext = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
        IS_drk = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)

        result[mn] = {
            'tot': fill_zero_domain(IS_tot, lat2d_a),
            'ext': fill_zero_domain(IS_ext, lat2d_a),
            'drk': fill_zero_domain(IS_drk, lat2d_a),
        }

    del fsds_da, fsns_da, sv_atm
    gc.collect()
    return result, native_ctx


# ============================================================
# RS COMPUTATION  (full T07 Taylor inversion — matches siaf_obs_analysis.py
# and compute_model_rs_native's 'full_T07' path used for the untuned CMIP6
# CESM2 ensemble we compare against)
# ============================================================

_RSDT_SOURCE_MODEL  = 'CESM2'
_RSDT_SOURCE_MEMBER = 'r1i1p1f1'


_RSDT_CACHE = None   # lazily-loaded (yr, mon, values) arrays, shared across calls


def _load_rsdt_borrowed(years, mn):
    """
    rsdt (TOA incident SW) is purely orbital/astronomical — it depends only
    on solar constant + Earth-Sun geometry for the given calendar month,
    not on a model's albedo tuning or aerosol forcing. So it is valid to
    reuse CESM2's published CMIP6 rsdt (any member; same orbital forcing
    for all members) for the GDEX runs, rather than approximating it.

    Uses _open_robust (same loader the main CMIP6 pipeline uses) because
    find_var_files returns 3 ESGF DRS symlink paths (files/, latest/,
    vYYYYMMDD/) to the SAME physical file — xr.open_mfdataset with
    combine='by_coords' can't order 3 datasets with identical, fully
    overlapping time ranges and raises; _open_robust's fallback strategy
    (nested concat) tolerates the harmless duplication.

    Returns the seasonal monthly-mean rsdt array on the f09 grid (192,288),
    or None if the CMIP6 file isn't found.
    """
    global _RSDT_CACHE
    if _RSDT_CACHE is None:
        inst  = MODEL_INFO[_RSDT_SOURCE_MODEL]['institution']
        files = find_var_files(inst, _RSDT_SOURCE_MODEL, _RSDT_SOURCE_MEMBER, 'rsdt')
        if not files:
            _RSDT_CACHE = False
        else:
            ds  = _open_robust(files, 'rsdt')
            da  = ds['rsdt'].load()
            yr  = np.array([int(t.dt.year)  for t in da.time])
            mon = np.array([int(t.dt.month) for t in da.time])
            _RSDT_CACHE = (yr, mon, da.values.astype(np.float32))
            ds.close()

    if _RSDT_CACHE is False:
        return None
    yr, mon, vals = _RSDT_CACHE
    m = (mon == mn) & np.isin(yr, years)
    if m.sum() == 0:
        return None
    return vals[m].mean(axis=0).astype(np.float32)


def compute_gdex_RS(case, native_ctx, trend_years):
    """
    Full T07 Taylor-inversion RS, matching siaf_obs_analysis.py's CERES
    path and compute_model_rs_native's 'full_T07' path exactly:
      rsds   = FSDS            rsus   = FSDS  - FSNS
      rsdscs = FSDSC            rsuscs = FSDSC - FSNSC
      rsut   = rsdt - FSNTOA    rsutcs = rsdt  - FSNTOAC
    rsdt is borrowed from the published CMIP6 CESM2 archive (see
    _load_rsdt_borrowed) since it is model-tuning-independent.
    Falls back to the rsds_proxy (FSDS only) if any input is unavailable.
    Returns dict {month: rs_array} on f09 grid.
    """
    from cmip6_siaf_calculation import RS_YEARS
    all_yrs   = [y for y in RS_YEARS if y >= min(trend_years)]
    atm_shape = native_ctx['shape']

    vars_needed = ['FSDS', 'FSNS', 'FSDSC', 'FSNSC', 'FSNTOA', 'FSNTOAC']
    das = {v: load_cam_years_months(case, v, all_yrs, MONTH_NUMS)
           for v in vars_needed}
    have_full = all(d is not None for d in das.values())

    result = {}
    method = 'full_T07' if have_full else 'rsds_proxy'
    for mn in MONTH_NUMS:
        if have_full:
            yr  = np.array([int(t.dt.year)  for t in das['FSDS'].time])
            mon = np.array([int(t.dt.month) for t in das['FSDS'].time])
            m   = (mon == mn) & np.isin(yr, all_yrs)
            if m.sum() == 0:
                result[mn] = None
                continue
            fsds   = das['FSDS'].isel(time=m).mean('time').values.astype(np.float64)
            fsns   = das['FSNS'].isel(time=m).mean('time').values.astype(np.float64)
            fsdsc  = das['FSDSC'].isel(time=m).mean('time').values.astype(np.float64)
            fsnsc  = das['FSNSC'].isel(time=m).mean('time').values.astype(np.float64)
            fsntoa = das['FSNTOA'].isel(time=m).mean('time').values.astype(np.float64)
            fsntoac= das['FSNTOAC'].isel(time=m).mean('time').values.astype(np.float64)

            rsdt = _load_rsdt_borrowed(all_yrs, mn)
            if rsdt is None or rsdt.shape != atm_shape:
                result[mn] = None
                continue
            rsdt = rsdt.astype(np.float64)

            rsds, rsus     = fsds,  fsds  - fsns
            rsdscs, rsuscs = fsdsc, fsdsc - fsnsc
            rsut           = rsdt - fsntoa
            rsutcs         = rsdt - fsntoac

            arr = t07_invert(rsdt, rsut, rsds, rsus,
                             rsutcs, rsdscs, rsuscs).astype(np.float32)
            if arr.shape != atm_shape:
                result[mn] = None
                continue
            result[mn] = arr
        else:
            if das['FSDS'] is None:
                result[mn] = None
                continue
            yr  = np.array([int(t.dt.year)  for t in das['FSDS'].time])
            mon = np.array([int(t.dt.month) for t in das['FSDS'].time])
            m   = (mon == mn) & np.isin(yr, all_yrs)
            if m.sum() == 0:
                result[mn] = None
                continue
            arr = das['FSDS'].isel(time=m).mean('time').values.astype(np.float32)
            if arr.shape != atm_shape:
                result[mn] = None
                continue
            result[mn] = arr

    print(f'    [{case[-3:]}] RS method: {method}', flush=True)
    return result


# ============================================================
# SCALAR SIAF_GLOBAL
# ============================================================

def compute_siaf_global(IS_result, RS_result, native_ctx):
    """
    Mirrors compute_siaf_monthly + siaf_global from the main CMIP6 script
    exactly: per-month Pistone-region ocean-only area-weighted mean via
    arctic_ocean_mean_native, then seasonal_to_global applies
    SCALE = AREA_RATIO_OCEAN * SEASON_FRAC to express the Apr-Sep Arctic
    Ocean signal as a global-mean-equivalent forcing (W m⁻² K⁻¹).
    Returns (siaf_tot, siaf_ext, siaf_drk).
    """
    monthly = {'tot': {}, 'ext': {}, 'drk': {}}
    for mn in MONTH_NUMS:
        IS = IS_result.get(mn)
        RS = RS_result.get(mn)
        if IS is None or RS is None:
            continue
        for comp in ('tot', 'ext', 'drk'):
            field = (IS[comp].astype(np.float64) * RS.astype(np.float64)).astype(np.float32)
            monthly[comp][mn] = arctic_ocean_mean_native(field, native_ctx)

    tot = seasonal_to_global(monthly['tot'])
    ext = seasonal_to_global(monthly['ext'])
    drk = seasonal_to_global(monthly['drk'])
    return tot, ext, drk


# ============================================================
# PER-CASE WORKER
# ============================================================

def process_case(case):
    """Run full SIAF computation for one GDEX case. Returns (case, tot, ext, drk)."""
    smbb = _is_smbb(case)
    trend_years = SMBB_TREND_YEARS  if smbb else BHIST_TREND_YEARS
    all_years   = SMBB_ALL_YEARS    if smbb else BHIST_ALL_YEARS

    # EARLY/LATE based on available period
    if smbb:
        n = len(trend_years)
        n3 = n // 3
        early = trend_years[:n3]
        late  = trend_years[-n3:]
    else:
        early = EARLY_YEARS
        late  = LATE_YEARS

    print(f'  [{case[-3:]}] computing T_annual...', flush=True)
    T_annual = compute_gdex_T_annual(case, all_years)
    if not T_annual:
        print(f'  [{case[-3:]}] T_annual failed', flush=True)
        return case, np.nan, np.nan, np.nan

    print(f'  [{case[-3:]}] computing IS...', flush=True)
    IS_result, native_ctx = compute_gdex_IS(
        case, T_annual, trend_years, early, late)
    if IS_result is None:
        print(f'  [{case[-3:]}] IS failed', flush=True)
        return case, np.nan, np.nan, np.nan

    print(f'  [{case[-3:]}] computing RS...', flush=True)
    RS_result = compute_gdex_RS(case, native_ctx, trend_years)

    tot, ext, drk = compute_siaf_global(IS_result, RS_result, native_ctx)
    print(f'  [{case[-3:]}] siaf_tot={tot:.4f}  ext={ext:.4f}  drk={drk:.4f}',
          flush=True)
    return case, tot, ext, drk


# ============================================================
# PLOT
# ============================================================

def plot_comparison(gdex_rows, csv_cmip6, out_png):
    """
    Bar chart comparing untuned CMIP6 CESM2 vs the GDEX tuned-sea-ice-albedo
    ensemble (r_snw=1.5, dt_mlt=1.0 per Kay et al. 2022 / IMPACTS), against
    the observed SIAF reference.

    IMPORTANT: within the GDEX d651070 archive, BOTH the BHIST (012-015) and
    BHISTsmbb (118-122) cases have the tuned sea-ice albedo applied — the
    "smbb" suffix only marks a different, unrelated biomass-burning aerosol
    forcing treatment (smoothed vs standard CMIP6 forcing), not an albedo
    tuning on/off switch. So both groups are "tuned"; there is no untuned
    control inside this archive. The untuned baseline is the separate,
    standard CMIP6 CESM2 historical ensemble (r1-r11).
    """
    # Current authoritative obs value, from siaf_obs_analysis.py's own output
    # (siaf_pixelwise_summary.csv) — NOT the stale siaf_unified_diagnostics.csv
    # cached value (OBS_B2_GLOBAL=0.2062, a week older and superseded).
    # CMIP6-period (1982-2014, matches the model trend window) at p=0.05;
    # CERES and Kernel RS methods agree to within 0.0005 (~0.225).
    PIXELWISE_CSV = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
    obs_df  = pd.read_csv(PIXELWISE_CSV)
    obs_row = obs_df[(obs_df['period'] == 'CMIP6') & (obs_df['pval'] == 0.05)]
    obs_ceres  = float(obs_row[obs_row['rs_method'] == 'CERES']['SIAF_tot_global'].iloc[0])
    obs_kernel = float(obs_row[obs_row['rs_method'] == 'Kernel']['SIAF_tot_global'].iloc[0])
    OBS_B2_GLOBAL = float(np.mean([obs_ceres, obs_kernel]))

    df = pd.read_csv(csv_cmip6)
    cesm2_df    = df[df['model'] == 'CESM2'].sort_values('siaf_global', ascending=False)
    cesm2_mean  = cesm2_df['siaf_global'].mean()
    cesm2_ext_m = cesm2_df['siaf_ext_global'].mean()
    cesm2_drk_m = cesm2_df['siaf_drk_global'].mean()

    # Organise GDEX results — both groups are tuned, differ only in aerosol forcing
    cmip6aer_rows = [(c, t, e, d) for c, t, e, d in gdex_rows if 'BHISTsmbb' not in c]
    smbbaer_rows  = [(c, t, e, d) for c, t, e, d in gdex_rows if 'BHISTsmbb'     in c]
    tuned_rows    = [(c, t, e, d) for c, t, e, d in (cmip6aer_rows + smbbaer_rows)
                      if np.isfinite(t)]
    tuned_mean    = np.nanmean([t for _, t, *_ in tuned_rows])
    tuned_ext_m   = np.nanmean([e for _, _, e, _ in tuned_rows])
    tuned_drk_m   = np.nanmean([d for _, _, _, d in tuned_rows])

    COL_EXT, COL_DRK = '#3A6BAF', '#D95F1A'   # consistent with siaf_obs_analysis.py

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('white')

    y_pos   = 0
    yticks  = []
    ylabels = []

    def _stacked_bar(label, ext, drk, hatch=''):
        nonlocal y_pos
        if np.isfinite(ext) and np.isfinite(drk):
            ax.barh(y_pos, ext, color=COL_EXT, alpha=0.85, height=0.7,
                    hatch=hatch, edgecolor='white', linewidth=0.4)
            ax.barh(y_pos, drk, left=ext, color=COL_DRK, alpha=0.85, height=0.7,
                    hatch=hatch, edgecolor='white', linewidth=0.4)
            tot = ext + drk
            ax.text(tot + 0.003, y_pos, f'{tot:.3f}', va='center', fontsize=8)
        yticks.append(y_pos)
        ylabels.append(label)
        y_pos += 1

    for _, r in cesm2_df.iterrows():
        _stacked_bar(f"untuned CMIP6  {r['member']}",
                    r['siaf_ext_global'], r['siaf_drk_global'], hatch='////')

    y_pos += 0.5
    for c, tot, ext, drk in cmip6aer_rows:
        _stacked_bar(f'tuned albedo, cmip6-aer  {c[-3:]}', ext, drk)

    y_pos += 0.5
    for c, tot, ext, drk in smbbaer_rows:
        _stacked_bar(f'tuned albedo, smbb-aer   {c[-3:]}', ext, drk)

    ax.axvline(cesm2_mean, color='#555555', lw=1.5, ls='--',
               label=f'untuned CMIP6 CESM2 mean = {cesm2_mean:.3f}')
    ax.axvline(tuned_mean, color='#d62728', lw=2.0, ls='-',
               label=f'tuned (all, n={len(tuned_rows)}) mean = {tuned_mean:.3f}')
    if np.isfinite(OBS_B2_GLOBAL):
        ax.axvline(OBS_B2_GLOBAL, color='black', lw=2.0, ls=':',
                   label=(f'observed SIAF (CERES/Kernel mean, CMIP6-period, '
                          f'p=0.05) = {OBS_B2_GLOBAL:.3f}'))

    import matplotlib.patches as mpatches
    handles, labels_ = ax.get_legend_handles_labels()
    handles += [mpatches.Patch(facecolor=COL_EXT, alpha=0.85, label='SIAF$_{ext}$'),
               mpatches.Patch(facecolor=COL_DRK, alpha=0.85, label='SIAF$_{drk}$')]
    labels_ += [r'SIAF$_{\rm ext}$  (extent)', r'SIAF$_{\rm drk}$  (darkening)']

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r'SIAF$_{\rm tot}$  (W m$^{-2}$ K$^{-1}$)', fontsize=11)

    d_tot  = tuned_mean  - cesm2_mean
    d_ext  = tuned_ext_m - cesm2_ext_m
    d_drk  = tuned_drk_m - cesm2_drk_m
    ax.set_title(
        'CESM2 SIAF: untuned CMIP6 (r1-r11)  vs  GDEX tuned sea-ice-albedo '
        'ensemble (r_snw=1.5, dt_mlt=1.0)\n'
        'Both tuned groups (cmip6-aer 012-015, smbb-aer 118-122) differ only '
        'in biomass-burning aerosol forcing, not albedo tuning  ·  '
        'smbb-aer trend period 1990-2014 (25 yr); others 1982-2014 (33 yr)\n'
        f'Mean shift: tot {cesm2_mean:.3f}'
        f'→{tuned_mean:.3f} (+{d_tot:.3f})   |   ext {cesm2_ext_m:.3f}'
        f'→{tuned_ext_m:.3f} (+{d_ext:.3f}, {100*d_ext/d_tot:.0f}% of Δtot)'
        f'   |   drk {cesm2_drk_m:.3f}→{tuned_drk_m:.3f} '
        f'(+{d_drk:.3f}, {100*d_drk/d_tot:.0f}% of Δtot)',
        fontsize=8.5)
    ax.legend(handles=handles, labels=labels_, fontsize=8, loc='lower right', ncol=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_png}')
    print(f'\n  untuned CMIP6 CESM2 mean : {cesm2_mean:.4f}')
    print(f'  tuned (all GDEX) mean    : {tuned_mean:.4f}')
    print(f'  observed SIAF reference  : {OBS_B2_GLOBAL:.4f}')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print('=' * 70)
    print('CESM2 GDEX SIAF — tuned albedo comparison')
    print(f'  GDEX root : {GDEX_ROOT}')
    print(f'  BHIST     : {len(BHIST_CASES)} members  (1982-2014)')
    print(f'  BHISTsmbb : {len(SMBB_CASES)}  members  (1990-2014)')
    print('=' * 70)

    all_cases = BHIST_CASES + SMBB_CASES
    gdex_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_case, c): c for c in all_cases}
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                row = fut.result()
                gdex_rows.append(row)
                print(f'  ✓ {row[0][-18:]}  siaf={row[1]:.4f}', flush=True)
            except Exception as exc:
                print(f'  ✗ {case[-18:]}  {exc}', flush=True)
                gdex_rows.append((case, np.nan, np.nan, np.nan))

    gdex_rows.sort(key=lambda r: r[0])

    # Save CSV
    rows_df = pd.DataFrame(
        [(c, 'BHIST' if 'BHISTsmbb' not in c else 'BHISTsmbb', t, e, d)
         for c, t, e, d in gdex_rows],
        columns=['case', 'group', 'siaf_tot', 'siaf_ext', 'siaf_drk'])
    rows_df.to_csv(OUT_CSV, index=False)
    print(f'\nResults saved to {OUT_CSV}')
    print(rows_df.to_string(index=False))

    print('\nPlotting comparison...')
    plot_comparison(gdex_rows, CSV_CMIP6, OUT_PNG)

    print(f'\nDone  ->  {OUT_PNG}')
