'''#!/usr/bin/env python
"""
diagnose_regridding.py
-------------------------------------------------------
Sensitivity sweep: C0–C6 regridding configurations for IS/RS/SIAF.
Memory-safe streaming version (no Dask).

C0 : IS × RS on native SIAL grid → sial_arctic_wmean  [preferred, C0]
C1 : IS → 1°,   RS → 1°,   multiply → arctic_wmean    [standard remap]
C3 : IS → 2°,   RS → 2°,   multiply → arctic_wmean
C4 : IS → 5°,   RS → 5°,   multiply → arctic_wmean
C5 : IS × RS on native SIAL, regrid product → 1°  → arctic_wmean
C6 : IS → 1°,   RS scalar mean,  multiply → arctic_wmean

C2 (0.5°) was REMOVED.  The 0.5° target grid is finer than the ~25 km
SIAL source spacing, so scatter-regrid leaves most 0.5° cells with no
votes.  fill_zero=True then converts those empty cells to 0, massively
diluting the Arctic mean (~3.5× lower than C0).  The result is an
artifact of grid resolution mismatch, not a physically meaningful
sensitivity estimate.

Albedo loading (streaming_alb_mean):
  April - August : Strictly uses CLARA-A3.
  September      : Strictly uses APP-x (regridded to SIAL via NN).
  No blending or gap-filling between datasets.

arctic_wmean convention:
  fill_zero=True  → SIAF: non-ice pixels treated as 0, all ≥60°N in denom.
                    Correct companion to AREA_RATIO_60N = 0.065.
  fill_zero=False → RS, IS: ice-pixel-only mean.
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc, warnings, resource, sys
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# MEMORY AUDITOR
# ============================================================
def log_mem(tag=""):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem_kb / (1024.0 * 1024.0)
    print(f"[MEM] {tag:<45} | Peak RAM: {mem_gb:.3f} GB", flush=True)
    sys.stdout.flush()

log_mem("STARTING SCRIPT")

# ============================================================
# CONFIG
# ============================================================

DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')

OUT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
OUT_CSV  = os.path.join(OUT_DIR, 'siaf_sensitivity_results.csv')
OUT_SUM  = os.path.join(OUT_DIR, 'siaf_sensitivity_summary.csv')
PLOT_DIR = os.path.join(OUT_DIR, 'siaf_sensitivity_plots/')
os.makedirs(PLOT_DIR, exist_ok=True)

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

TREND_START_CMIP6 = 1982;  TREND_END_CMIP6 = 2014
TREND_START_OBS   = 1982;  TREND_END_OBS   = 2024

CERES_YRS_CMIP6 = list(range(2000, 2015))
CERES_YRS_OBS   = list(range(2000, 2025))

PVAL              = 0.05
SIC_ICE_THRESHOLD = 0.15

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

# C2 (0.5°) removed — see module docstring for explanation.
GRIDS = {
    '1deg': (np.arange(-90., 91.,  1.0), np.arange(0., 360., 1.0)),
    '2deg': (np.arange(-90., 91.,  2.0), np.arange(0., 360., 2.0)),
    '5deg': (np.arange(-90., 91.,  5.0), np.arange(0., 360., 5.0)),
}

# ============================================================
# HELPERS
# ============================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg, dtype=np.float32)
    lon = np.deg2rad(lon_deg, dtype=np.float32)
    xyz = np.empty((lat.size, 3), dtype=np.float32)
    xyz[:, 0] = np.cos(lat) * np.cos(lon)
    xyz[:, 1] = np.cos(lat) * np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    return xyz


def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d,
                        chunk_size=100_000):
    """Each src pixel votes into its nearest tgt cell (scatter average)."""
    log_mem("  -> build_scatter_index")
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    idx     = np.empty(src_xyz.shape[0], dtype=np.int32)
    tree    = cKDTree(tgt_xyz)
    for i in range(0, src_xyz.shape[0], chunk_size):
        e = min(i + chunk_size, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    log_mem("  -> build_scatter_index done")
    return idx, tgt_lat2d.shape


def scatter_avg(arr_flat, idx, out_shape):
    accum = np.zeros(np.prod(out_shape), dtype=np.float32)
    count = np.zeros(np.prod(out_shape), dtype=np.float32)
    valid = np.isfinite(arr_flat)
    np.add.at(accum, idx[valid], arr_flat[valid])
    np.add.at(count, idx[valid], np.float32(1.0))
    out    = np.full(np.prod(out_shape), np.nan, dtype=np.float32)
    m      = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)


def arctic_wmean(arr2d, lat1d, fill_zero=True):
    """
    Area-weighted mean over ≥60°N.

    fill_zero=True  (SIAF): NaN → 0, all ≥60°N in denominator.
                    Pairs correctly with AREA_RATIO_60N = 0.065.
    fill_zero=False (RS, IS): ice-pixel-only mean (NaN excluded from both
                    numerator and denominator).
    """
    lat2d  = np.tile(lat1d[:, None], (1, arr2d.shape[1]))
    w      = np.cos(np.deg2rad(lat2d))
    domain = lat2d >= 60.
    if not domain.any():
        return np.nan
    if fill_zero:
        arr_z = np.where(np.isnan(arr2d), 0.0, arr2d)
        return float(np.sum(arr_z[domain] * w[domain]) / np.sum(w[domain]))
    else:
        valid = domain & np.isfinite(arr2d)
        if not valid.any():
            return np.nan
        return float(np.sum(arr2d[valid] * w[valid]) / np.sum(w[valid]))


def regrid_2d(arr, idx, out_shape):
    return scatter_avg(arr.ravel(), idx, out_shape)

# ============================================================
# T07 INVERSION
# ============================================================

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd = np.where(rsds  > 1., rsds,  np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds / S;  q = rsut / S;  den = q + p - 2.*p*a
        R  = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan), 0., 0.99)
        A  = np.clip(1. - p*(1.-a*R) / np.where((1.-R) > 0.01, 1.-R, np.nan), 0., 0.99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs / S;  qc = rsutcs / S;  dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan), 0., 0.99)
        Rl = np.clip(R - Rc, 0., 0.99)
        RS = (S * (1.-A) * (1.-Rl)**2
              / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS

# ============================================================
# STEP 1: LOAD SIAL GRID + BUILD APP-x → SIAL MAPPING
# ============================================================

print("=" * 70, flush=True)
print("STEP 1: Loading SIAL grid and building APP-x mapping ...", flush=True)
print("=" * 70, flush=True)

sial_shape  = tuple(np.load(f"{CACHE_CMIP6}/sial_shape.npy"))
arctic_sial = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy")

sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values

if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d

assert sial_lat2d.shape == sial_shape, (
    f"SIAL shape mismatch: {sial_lat2d.shape} vs {sial_shape}")

log_mem("Loaded SIAL grid")

# Build APP-x → SIAL nearest-neighbour mapping.
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon

    appx_xyz = latlon_to_xyz(appx_lat2d.ravel().astype(np.float32),
                              appx_lon2d.ravel().astype(np.float32))
    sial_xyz  = latlon_to_xyz(sial_lat2d.ravel().astype(np.float32),
                               sial_lon2d.ravel().astype(np.float32))
    _, idx_appx_to_sial = cKDTree(appx_xyz).query(sial_xyz, workers=-1)
    del appx_xyz, sial_xyz, appx_lat2d, appx_lon2d, appx_lat, appx_lon
    gc.collect()
    print(f"  APP-x → SIAL mapping built (shape {sial_shape})", flush=True)
    log_mem("APP-x mapping done")
except Exception as e:
    print(f"  WARNING: APP-x mapping failed ({e}); September data will be missing.",
          flush=True)

# ============================================================
# STEP 2: CERES RS LOADER (manual time-step streaming)
# ============================================================

def load_ceres_RS_single_month(years, mn):
    log_mem(f"Loading CERES month {mn}")
    sums  = {k: None for k in ['solar','toa_all','sfc_d_all','sfc_u_all',
                                 'toa_clr','sfc_d_clr','sfc_u_clr']}
    count = 0
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times  = pd.DatetimeIndex(ds.time.values)
        clon2d, clat2d = np.meshgrid(ds.lon.values, ds.lat.values)
        tidxs  = np.where([t.year in years and t.month == mn
                            for t in times])[0]
        if len(tidxs) == 0:
            return None
        log_mem(f"  -> Streaming {len(tidxs)} CERES time steps")
        for tidx in tidxs:
            sl = ds.isel(time=tidx).load()
            def _v(name):
                return sl[name].values.astype(np.float32)
            arrs = [_v('solar_mon'), _v('toa_sw_all_mon'),
                    _v('sfc_sw_down_all_mon'), _v('sfc_sw_up_all_mon'),
                    _v('toa_sw_clr_c_mon'),   _v('sfc_sw_down_clr_c_mon'),
                    _v('sfc_sw_up_clr_c_mon')]
            for k, a in zip(sums, arrs):
                sums[k] = a if count == 0 else sums[k] + a
            count += 1
            del sl

    for k in sums:
        sums[k] /= np.float32(count)

    log_mem("  -> T07 inversion")
    RS = t07_invert(sums['solar'],    sums['toa_all'],   sums['sfc_d_all'],
                    sums['sfc_u_all'], sums['toa_clr'],   sums['sfc_d_clr'],
                    sums['sfc_u_clr']).astype(np.float32)
    del sums; gc.collect()

    rs_month = {}

    # Regrid to regular grids (scatter average: CERES → target)
    for gname, (glat, glon) in GRIDS.items():
        log_mem(f"  -> Regrid CERES → {gname}")
        glon2d, glat2d = np.meshgrid(glon, glat)
        idx, shp = build_scatter_index(clat2d, clon2d, glat2d, glon2d)
        rs_month[gname] = scatter_avg(RS.ravel(), idx, shp).astype(np.float32)
        del idx, glon2d, glat2d

    # Native SIAL NN gather: for each SIAL pixel → nearest CERES pixel
    log_mem("  -> Regrid CERES → native SIAL (NN)")
    sial_xyz_q = latlon_to_xyz(sial_lat2d.ravel().astype(np.float32),
                                sial_lon2d.ravel().astype(np.float32))
    _, idx_cs = cKDTree(latlon_to_xyz(clat2d.ravel().astype(np.float32),
                                       clon2d.ravel().astype(np.float32))
                        ).query(sial_xyz_q, workers=-1)
    rs_month['sial_nn'] = RS.ravel()[idx_cs].reshape(sial_shape).astype(np.float32)
    del idx_cs, sial_xyz_q, RS, clat2d, clon2d; gc.collect()

    log_mem(f"Finished CERES month {mn}")
    return rs_month

# ============================================================
# STEP 3: DELTA-T
# ============================================================

def trend_dT(years, vals, y0, y1):
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5:
        return np.nan
    return stats.linregress(years[mask], vals[mask]).slope * (y1 - y0)

print("\n--- Computing ΔT ---", flush=True)
dT_cmip6_vals = []; dT_obs_vals = []

for label, path, reader in [
    ('Cowtan&Way', '/glade/work/jrotondo/cowtan_way.txt',
     lambda p: (pd.read_csv(p, sep=r'\s+', header=None,
                             names=['year','anomaly','unc1','unc2','unc3'])
                  .rename(columns={'year':'Year','anomaly':'val'}))),
    ('GISTEMP',   '/glade/work/jrotondo/gistemp_global.csv',
     lambda p: (pd.read_csv(p, skiprows=1, na_values='***')
                  .assign(val=lambda d: pd.to_numeric(d['J-D'], errors='coerce'))
                  .rename(columns={'Year':'Year'}))),
]:
    try:
        df_tmp = reader(path)
        y = df_tmp['Year'].values.astype(float)
        v = df_tmp['val'].values.astype(float)
        dT_cmip6_vals.append(trend_dT(y, v, TREND_START_CMIP6, TREND_END_CMIP6))
        dT_obs_vals.append(  trend_dT(y, v, TREND_START_OBS,   TREND_END_OBS))
        del df_tmp, y, v
    except Exception as e:
        print(f"  ΔT source {label} failed: {e}", flush=True)

DELTA_T_CMIP6 = float(np.nanmean(dT_cmip6_vals)) if dT_cmip6_vals else 0.64
DELTA_T_OBS   = float(np.nanmean(dT_obs_vals))   if dT_obs_vals   else 0.84
print(f"  DELTA_T_CMIP6 = {DELTA_T_CMIP6:.4f} K", flush=True)
print(f"  DELTA_T_OBS   = {DELTA_T_OBS:.4f} K",   flush=True)

# ============================================================
# STEP 4: ALBEDO LOADER
# ============================================================

def streaming_alb_mean(cache_dir, mname, years):
    """
    Per-pixel streaming mean of surface albedo over `years`.

    April - August : Strictly uses CLARA-A3.
    September      : Strictly uses APP-x (regridded to SIAL via NN).
    No blending or gap-filling between datasets.
    """
    alb_sum = None
    alb_cnt = None

    use_appx = (mname == 'September')

    for y in years:
        if use_appx:
            # --- STRICTLY APP-x FOR SEPTEMBER ---
            if idx_appx_to_sial is None:
                continue # Skip if mapping failed in Step 1
            appx_path = f"{cache_dir}/appx_alb_{mname}_{y}.npy"
            if not os.path.exists(appx_path):
                continue
                
            appx_raw = np.load(appx_path).astype(np.float32)
            # Remap APP-x pixels → SIAL grid via NN mapping
            a = appx_raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
            del appx_raw
            
        else:
            # --- STRICTLY CLARA-A3 FOR ALL OTHER MONTHS ---
            clara_path = f"{cache_dir}/clara_alb_{mname}_{y}.npy"
            if not os.path.exists(clara_path):
                continue
            a = np.load(clara_path).astype(np.float32)

        # Accumulate the valid pixels safely
        finite_mask = np.isfinite(a)
        a[~finite_mask] = 0.0

        if alb_sum is None:
            alb_sum = a
            alb_cnt = finite_mask.astype(np.int16)
        else:
            alb_sum += a
            alb_cnt += finite_mask.astype(np.int16)
        del a, finite_mask

    if alb_sum is None:
        return None

    # Compute final mean
    valid = alb_cnt > 0
    alb_sum[valid] /= alb_cnt[valid].astype(np.float32)
    alb_sum[~valid] = np.nan
    del alb_cnt
    
    return alb_sum

# ============================================================
# STEP 5: COMPUTE ALL CONFIGS FOR ONE MONTH × PERIOD
# ============================================================

def run_configs(cache_dir, early_years, late_years, delta_t, rs_maps_dict, mn):
    mname = MONTH_NAMES[mn]
    log_mem(f"run_configs {mname}")

    mean_e = streaming_alb_mean(cache_dir, mname, early_years)
    mean_l = streaming_alb_mean(cache_dir, mname, late_years)
    if mean_e is None or mean_l is None:
        print(f"  {mname}: missing albedo — skip", flush=True)
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dalpha = (mean_l - mean_e).astype(np.float32)
    del mean_e, mean_l

    slope = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy").astype(np.float32)
    pval  = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy").astype(np.float32)
    sic_e = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy").astype(np.float32)
    sic_l = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy").astype(np.float32)

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval)
                  & (slope < 0) & (pval < PVAL))
    stable_ice = (valid & ~changing
                  & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

    IS_ext_nat = np.where(changing,   -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk_nat = np.where(stable_ice, -dalpha / delta_t, np.nan).astype(np.float32)
    IS_tot_nat = np.where(
        np.isfinite(IS_ext_nat) | np.isfinite(IS_drk_nat),
        np.where(np.isfinite(IS_ext_nat), IS_ext_nat, 0.)
        + np.where(np.isfinite(IS_drk_nat), IS_drk_nat, 0.),
        np.nan).astype(np.float32)
    # Zero-fill so IS_ext + IS_drk == IS_tot at every ice pixel
    IS_ext_nat = np.where(np.isfinite(IS_tot_nat),
                          np.where(np.isfinite(IS_ext_nat), IS_ext_nat, 0.),
                          np.nan).astype(np.float32)
    IS_drk_nat = np.where(np.isfinite(IS_tot_nat),
                          np.where(np.isfinite(IS_drk_nat), IS_drk_nat, 0.),
                          np.nan).astype(np.float32)

    del dalpha, slope, pval, sic_e, sic_l, valid, changing, stable_ice
    gc.collect()
    log_mem(f"  IS computed")

    # Zero-fill IS within the Arctic domain BEFORE regridding to coarser grids.
    IS_tot_for_rg = np.where(arctic_sial,
                             np.where(np.isfinite(IS_tot_nat), IS_tot_nat, 0.),
                             np.nan).astype(np.float32)
    IS_ext_for_rg = np.where(arctic_sial,
                             np.where(np.isfinite(IS_ext_nat), IS_ext_nat, 0.),
                             np.nan).astype(np.float32)
    IS_drk_for_rg = np.where(arctic_sial,
                             np.where(np.isfinite(IS_drk_nat), IS_drk_nat, 0.),
                             np.nan).astype(np.float32)

    rs  = rs_maps_dict[mn]
    results = {}

    # ------------------------------------------------------------------
    # C0: IS × RS on native SIAL grid
    # SIAF mean: all-Arctic (fill_zero=True, pairs with AREA_RATIO_60N=0.065)
    # RS mean:   ice-pixel-only (fill_zero=False)
    # ------------------------------------------------------------------
    log_mem("  -> C0")
    if rs is not None and 'sial_nn' in rs:
        RS_sial = rs['sial_nn']

        SIAF_tot_nat = (IS_tot_nat * RS_sial).astype(np.float32)
        SIAF_ext_nat = (IS_ext_nat * RS_sial).astype(np.float32)
        SIAF_drk_nat = (IS_drk_nat * RS_sial).astype(np.float32)
        mask_tot = np.isnan(SIAF_tot_nat)
        SIAF_ext_nat[np.isnan(SIAF_ext_nat)] = 0.0; SIAF_ext_nat[mask_tot] = np.nan
        SIAF_drk_nat[np.isnan(SIAF_drk_nat)] = 0.0; SIAF_drk_nat[mask_tot] = np.nan

        lat2d_s = (np.tile(sial_lat1d[:, None], (1, sial_lon2d.shape[1]))
                   if sial_lat1d.ndim == 1 else sial_lat2d)
        w_s     = np.cos(np.deg2rad(lat2d_s))

        # --- THE MATH PROOF: STANDARD 60N vs ICE-ONLY ---
        domain_60n = lat2d_s >= 60.
        ice_mask   = np.isfinite(IS_tot_nat) & domain_60n
        
        area_60n_sum = np.sum(w_s[domain_60n])
        area_ice_sum = np.sum(w_s[ice_mask])
        
        # 1. The Standard 60N Approach
        # Mean of 60N (with zeros fed in) * 0.065 global constant
        arr_z = np.where(np.isnan(SIAF_tot_nat), 0., SIAF_tot_nat)
        mean_60n = float(np.sum(arr_z[domain_60n] * w_s[domain_60n]) / area_60n_sum)
        global_forcing_60n = mean_60n * 0.065
        
        # 2. The "Ice-Only" Approach
        # Mean strictly over the ice mask * Exact Global Area Ratio of that ice mask
        ice_area_ratio_global = (area_ice_sum / area_60n_sum) * 0.065
        mean_ice_only = float(np.sum(SIAF_tot_nat[ice_mask] * w_s[ice_mask]) / area_ice_sum)
        global_forcing_ice = mean_ice_only * ice_area_ratio_global
        
        # Print the proof to the console
        print(f"\n      [MATH CHECK - {mname} C0]")
        print(f"      Standard 60N Mean : {mean_60n:.4f} W/m2  * 0.06500 Area = {global_forcing_60n:.5f} W/m2 Global Forcing")
        print(f"      Ice-Only Mean     : {mean_ice_only:.4f} W/m2  * {ice_area_ratio_global:.5f} Area = {global_forcing_ice:.5f} W/m2 Global Forcing")
        # ------------------------------------------------

        def _sial_mean(arr, fill_zero):
            if fill_zero:
                a = np.where(np.isnan(arr), 0., arr)
                return float(np.sum(a[domain_60n] * w_s[domain_60n]) / area_60n_sum)
            else:
                valid = domain_60n & np.isfinite(arr)
                if not valid.any(): return np.nan
                return float(np.sum(arr[valid] * w_s[valid]) / np.sum(w_s[valid]))

        results['C0'] = dict(
            SIAF_tot=_sial_mean(SIAF_tot_nat, fill_zero=True),
            SIAF_ext=_sial_mean(SIAF_ext_nat, fill_zero=True),
            SIAF_drk=_sial_mean(SIAF_drk_nat, fill_zero=True),
            IS_mean =_sial_mean(IS_tot_nat,    fill_zero=False),   # ice-pixel-only
            RS_mean =_sial_mean(np.where(np.isfinite(IS_tot_nat), RS_sial, np.nan),
                                fill_zero=False),
        )
        del SIAF_tot_nat, SIAF_ext_nat, SIAF_drk_nat
        gc.collect()
    else:
        results['C0'] = None

    # ------------------------------------------------------------------
    # C1, C3, C4: IS and RS both regridded to the same regular grid,
    #             product computed there.
    # ------------------------------------------------------------------
    grid_map = {'C1': '1deg', 'C3': '2deg', 'C4': '5deg'}
    for cname, gname in grid_map.items():
        log_mem(f"  -> {cname}")
        if rs is None or gname not in rs:
            results[cname] = None; continue

        glat, glon = GRIDS[gname]
        glon2d, glat2d = np.meshgrid(glon, glat)
        RS_g = rs[gname]

        idx, shp = build_scatter_index(sial_lat2d, sial_lon2d, glat2d, glon2d)
        IS_tot_g = regrid_2d(IS_tot_for_rg, idx, shp)
        IS_ext_g = regrid_2d(IS_ext_for_rg, idx, shp)
        IS_drk_g = regrid_2d(IS_drk_for_rg, idx, shp)
        IS_tot_g_diag = regrid_2d(IS_tot_nat, idx, shp)
        del idx, glon2d, glat2d

        SIAF_tot = (IS_tot_g * RS_g).astype(np.float32)
        SIAF_ext = (IS_ext_g * RS_g).astype(np.float32)
        SIAF_drk = (IS_drk_g * RS_g).astype(np.float32)
        mask_tot = np.isnan(SIAF_tot)
        SIAF_ext[np.isnan(SIAF_ext)] = 0.; SIAF_ext[mask_tot] = np.nan
        SIAF_drk[np.isnan(SIAF_drk)] = 0.; SIAF_drk[mask_tot] = np.nan

        results[cname] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot,   glat, fill_zero=True),
            SIAF_ext=arctic_wmean(SIAF_ext,   glat, fill_zero=True),
            SIAF_drk=arctic_wmean(SIAF_drk,   glat, fill_zero=True),
            IS_mean =arctic_wmean(IS_tot_g_diag, glat, fill_zero=False),
            RS_mean =arctic_wmean(
                np.where(np.isfinite(IS_tot_g_diag), RS_g, np.nan),
                glat, fill_zero=False),
        )
        del IS_tot_g, IS_ext_g, IS_drk_g, IS_tot_g_diag, SIAF_tot, SIAF_ext, SIAF_drk
        gc.collect()

    # ------------------------------------------------------------------
    # C5: IS × RS on native SIAL, then regrid PRODUCT to 1°
    # ------------------------------------------------------------------
    log_mem("  -> C5")
    if rs is not None and 'sial_nn' in rs:
        RS_sial = rs['sial_nn']

        SIAF_tot_p = (IS_tot_nat * RS_sial).astype(np.float32)
        SIAF_ext_p = (IS_ext_nat * RS_sial).astype(np.float32)
        SIAF_drk_p = (IS_drk_nat * RS_sial).astype(np.float32)
        mask_p = np.isnan(SIAF_tot_p)
        SIAF_ext_p[np.isnan(SIAF_ext_p)] = 0.; SIAF_ext_p[mask_p] = np.nan
        SIAF_drk_p[np.isnan(SIAF_drk_p)] = 0.; SIAF_drk_p[mask_p] = np.nan

        SIAF_tot_p_rg = np.where(arctic_sial,
                                  np.where(np.isfinite(SIAF_tot_p), SIAF_tot_p, 0.),
                                  np.nan).astype(np.float32)
        SIAF_ext_p_rg = np.where(arctic_sial,
                                  np.where(np.isfinite(SIAF_ext_p), SIAF_ext_p, 0.),
                                  np.nan).astype(np.float32)
        SIAF_drk_p_rg = np.where(arctic_sial,
                                  np.where(np.isfinite(SIAF_drk_p), SIAF_drk_p, 0.),
                                  np.nan).astype(np.float32)
        del SIAF_tot_p, SIAF_ext_p, SIAF_drk_p

        glat1, glon1 = GRIDS['1deg']
        glon1d2d, glat1d2d = np.meshgrid(glon1, glat1)
        idx1, shp1 = build_scatter_index(sial_lat2d, sial_lon2d, glat1d2d, glon1d2d)

        SIAF_tot_1d = regrid_2d(SIAF_tot_p_rg, idx1, shp1)
        SIAF_ext_1d = regrid_2d(SIAF_ext_p_rg, idx1, shp1)
        SIAF_drk_1d = regrid_2d(SIAF_drk_p_rg, idx1, shp1)
        IS_1d_diag  = regrid_2d(IS_tot_nat, idx1, shp1)
        RS_1d       = regrid_2d(np.where(np.isfinite(IS_tot_nat), RS_sial, np.nan),
                                idx1, shp1)
        del idx1, glon1d2d, glat1d2d, SIAF_tot_p_rg, SIAF_ext_p_rg, SIAF_drk_p_rg

        results['C5'] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot_1d, glat1, fill_zero=True),
            SIAF_ext=arctic_wmean(SIAF_ext_1d, glat1, fill_zero=True),
            SIAF_drk=arctic_wmean(SIAF_drk_1d, glat1, fill_zero=True),
            IS_mean =arctic_wmean(IS_1d_diag,  glat1, fill_zero=False),
            RS_mean =arctic_wmean(RS_1d,        glat1, fill_zero=False),
        )
        del SIAF_tot_1d, SIAF_ext_1d, SIAF_drk_1d, IS_1d_diag, RS_1d
        gc.collect()
    else:
        results['C5'] = None

    # ------------------------------------------------------------------
    # C6: IS → 1°,  scalar RS mean × IS_map
    # ------------------------------------------------------------------
    log_mem("  -> C6")
    if rs is not None and '1deg' in rs:
        glat1, glon1  = GRIDS['1deg']
        RS_1d_full    = rs['1deg']
        glon1d2d, glat1d2d = np.meshgrid(glon1, glat1)
        idx1, shp1    = build_scatter_index(sial_lat2d, sial_lon2d, glat1d2d, glon1d2d)
        IS_tot_1d      = regrid_2d(IS_tot_for_rg, idx1, shp1)
        IS_ext_1d      = regrid_2d(IS_ext_for_rg, idx1, shp1)
        IS_drk_1d      = regrid_2d(IS_drk_for_rg, idx1, shp1)
        IS_tot_1d_diag = regrid_2d(IS_tot_nat,    idx1, shp1)
        del idx1, glon1d2d, glat1d2d

        RS_scalar = arctic_wmean(
            np.where(np.isfinite(IS_tot_1d_diag), RS_1d_full, np.nan),
            glat1, fill_zero=False)

        SIAF_tot_s = (IS_tot_1d * RS_scalar).astype(np.float32)
        SIAF_ext_s = (IS_ext_1d * RS_scalar).astype(np.float32)
        SIAF_drk_s = (IS_drk_1d * RS_scalar).astype(np.float32)
        mask_s = np.isnan(SIAF_tot_s)
        SIAF_ext_s[np.isnan(SIAF_ext_s)] = 0.; SIAF_ext_s[mask_s] = np.nan
        SIAF_drk_s[np.isnan(SIAF_drk_s)] = 0.; SIAF_drk_s[mask_s] = np.nan

        results['C6'] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot_s, glat1, fill_zero=True),
            SIAF_ext=arctic_wmean(SIAF_ext_s, glat1, fill_zero=True),
            SIAF_drk=arctic_wmean(SIAF_drk_s, glat1, fill_zero=True),
            IS_mean =arctic_wmean(IS_tot_1d_diag, glat1, fill_zero=False),
            RS_mean =RS_scalar,
        )
        del IS_tot_1d, IS_ext_1d, IS_drk_1d, IS_tot_1d_diag, SIAF_tot_s, SIAF_ext_s, SIAF_drk_s
    else:
        results['C6'] = None

    del IS_tot_nat, IS_ext_nat, IS_drk_nat
    del IS_tot_for_rg, IS_ext_for_rg, IS_drk_for_rg
    gc.collect()
    log_mem(f"run_configs {mname} done")
    return results

# ============================================================
# STEP 6: RUN BOTH PERIODS
# ============================================================

all_rows = []

PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, CERES_YRS_CMIP6),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   DELTA_T_OBS,   CERES_YRS_OBS),
]

for (period_label, cache_dir, early_years, late_years,
     delta_t, ceres_years) in PERIOD_CONFIGS:
    print(f"\n{'='*70}", flush=True)
    print(f"Period: {period_label}  (ΔT={delta_t:.4f} K)", flush=True)
    print(f"{'='*70}", flush=True)

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f"\n  {mname}", flush=True)

        rs_month = load_ceres_RS_single_month(ceres_years, mn)
        if rs_month is None:
            print("    [SKIPPED — missing CERES data]", flush=True); continue

        cfg_results = run_configs(cache_dir, early_years, late_years,
                                   delta_t, {mn: rs_month}, mn)
        del rs_month; gc.collect()
        log_mem(f"Cleared memory after {mname}")

        if cfg_results is None:
            continue
        for cname, vals in cfg_results.items():
            if vals is None: continue
            all_rows.append(dict(period=period_label, month=mname,
                                  month_num=mn, config=cname, **vals))

# ============================================================
# STEP 7: DATAFRAMES, CSV, SUMMARY
# ============================================================
log_mem("Starting summary and plots")

df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nRaw results → {OUT_CSV}", flush=True)

summary_rows = []
for (period, cname), grp in df.groupby(['period', 'config']):
    summary_rows.append(dict(
        period=period, config=cname,
        SIAF_tot=grp['SIAF_tot'].mean(),
        SIAF_ext=grp['SIAF_ext'].mean(),
        SIAF_drk=grp['SIAF_drk'].mean(),
        IS_mean =grp['IS_mean'].mean(),
        RS_mean =grp['RS_mean'].mean(),
    ))

summary = pd.DataFrame(summary_rows)

for period in summary['period'].unique():
    c0_row = summary[(summary['period'] == period) & (summary['config'] == 'C0')]
    if c0_row.empty: continue
    c0_tot = c0_row['SIAF_tot'].values[0]
    mask   = summary['period'] == period
    summary.loc[mask, 'delta_vs_C0_tot'] = summary.loc[mask, 'SIAF_tot'] - c0_tot
    summary.loc[mask, 'pct_diff_vs_C0']  = (
        (summary.loc[mask, 'SIAF_tot'] - c0_tot) / np.abs(c0_tot) * 100.)

summary.to_csv(OUT_SUM, index=False)
print(f"Summary      → {OUT_SUM}", flush=True)
print(summary.to_string(index=False), flush=True)

log_mem("DONE")'''

#!/usr/bin/env python
"""
diagnose_pistone_2014.py   (revised — PSN25 + 6 wedges)
-------------------------------------------------------
Applies Pistone et al. (2014, PNAS) methodology faithfully:

  * "The Arctic is defined as the area north of 60°N, and only nonland
    area is considered."
  * Six Pistone Fig S2 regions — 60° longitudinal wedges between 60°N
    and 90°N, counterclockwise from 0°:
        EU (Europe, 0-60°E, Barents Sea)
        WA (West Asia, 60-120°E, Kara + Laptev Seas)
        WP (West Pacific, 120-180°E, East Siberian Sea)
        EP (East Pacific, -180 - -120°, Chukchi + Beaufort)
        NA (North America, -120 - -60°, Canadian Archipelago + Baffin)
        AO (Atlantic, -60 - 0°, Labrador + Greenland waters)
  * Native polar-stereographic PSN25 grid (25 × 25 km).  PSN25 is
    approximately equal-area across the Arctic, so "weighted by grid
    cell area" (Pistone) reduces to *equal per-cell weights*.  A small
    polar-stereographic scale-factor correction is applied as a
    refinement so the weights are true surface-area weights, not
    projected-area weights.
  * ΔT from GISS annual-mean global-mean surface temperature,
    "estimated the total change during [interval] as the product of the
    linear trend and the time interval."

Inputs -- user's framework:
  * Cached monthly albedo on SIAL grid (Apr-Aug: CLARA-A3; Sep: APP-x),
    regridded here to PSN25 via scatter average.
  * CERES EBAF Ed4.2.1 T07-inverted RS, regridded to PSN25.
  * GISTEMP annual global-mean anomaly for ΔT.

Outputs:
  siaf_pistone_results.csv  -- one row per (period, month, region)
  siaf_pistone_summary.csv  -- Arctic-wide + global, per (period, month)
"""

#!/usr/bin/env python
"""
diagnose_pistone_2014.py   (revised — PSN25 + 6 wedges)
-------------------------------------------------------
Applies Pistone et al. (2014, PNAS) methodology faithfully:

  * "The Arctic is defined as the area north of 60°N, and only nonland
    area is considered."
  * Six Pistone Fig S2 regions — 60° longitudinal wedges between 60°N
    and 90°N, counterclockwise from 0°:
        EU (Europe, 0-60°E, Barents Sea)
        WA (West Asia, 60-120°E, Kara + Laptev Seas)
        WP (West Pacific, 120-180°E, East Siberian Sea)
        EP (East Pacific, -180 - -120°, Chukchi + Beaufort)
        NA (North America, -120 - -60°, Canadian Archipelago + Baffin)
        AO (Atlantic, -60 - 0°, Labrador + Greenland waters)
  * Native polar-stereographic PSN25 grid (25 × 25 km).  PSN25 is
    approximately equal-area across the Arctic, so "weighted by grid
    cell area" (Pistone) reduces to *equal per-cell weights*.  A small
    polar-stereographic scale-factor correction is applied as a
    refinement so the weights are true surface-area weights, not
    projected-area weights.
  * ΔT from GISS annual-mean global-mean surface temperature,
    "estimated the total change during [interval] as the product of the
    linear trend and the time interval."

Inputs -- user's framework:
  * Cached monthly albedo on SIAL grid (Apr-Aug: CLARA-A3; Sep: APP-x),
    regridded here to PSN25 via scatter average.
  * CERES EBAF Ed4.2.1 T07-inverted RS, regridded to PSN25.
  * GISTEMP annual global-mean anomaly for ΔT.

Outputs:
  siaf_pistone_results.csv  -- one row per (period, month, region)
  siaf_pistone_summary.csv  -- Arctic-wide + global, per (period, month)
"""

#!/usr/bin/env python
"""
diagnose_pistone_2014.py   (revised — PSN25 + 6 wedges)
-------------------------------------------------------
Applies Pistone et al. (2014, PNAS) methodology faithfully:

  * "The Arctic is defined as the area north of 60°N, and only nonland
    area is considered."
  * Six Pistone Fig S2 regions — 60° longitudinal wedges between 60°N
    and 90°N, counterclockwise from 0°:
        EU (Europe, 0-60°E, Barents Sea)
        WA (West Asia, 60-120°E, Kara + Laptev Seas)
        WP (West Pacific, 120-180°E, East Siberian Sea)
        EP (East Pacific, -180 - -120°, Chukchi + Beaufort)
        NA (North America, -120 - -60°, Canadian Archipelago + Baffin)
        AO (Atlantic, -60 - 0°, Labrador + Greenland waters)
  * Native polar-stereographic PSN25 grid (25 × 25 km).  PSN25 is
    approximately equal-area across the Arctic, so "weighted by grid
    cell area" (Pistone) reduces to *equal per-cell weights*.  A small
    polar-stereographic scale-factor correction is applied as a
    refinement so the weights are true surface-area weights, not
    projected-area weights.
  * ΔT from GISS annual-mean global-mean surface temperature,
    "estimated the total change during [interval] as the product of the
    linear trend and the time interval."

Inputs -- user's framework:
  * Cached monthly albedo on SIAL grid (Apr-Aug: CLARA-A3; Sep: APP-x),
    regridded here to PSN25 via scatter average.
  * CERES EBAF Ed4.2.1 T07-inverted RS, regridded to PSN25.
  * GISTEMP annual global-mean anomaly for ΔT.

Outputs:
  siaf_pistone_results.csv  -- one row per (period, month, region)
  siaf_pistone_summary.csv  -- Arctic-wide + global, per (period, month)
"""

#!/usr/bin/env python
"""
diagnose_kernel_method.py
-------------------------------------------------------
Calculates Sea Ice Albedo Feedback (SIAF) using the Radiative Kernel approach 
(e.g., Flanner et al. 2011).

Instead of relying on CERES TOA fluxes to calculate regional sensitivity, 
this uses a pre-computed Radiative Kernel (e.g., CAM5 or ERA-Interim). 
The kernel is scatter-regridded to the native PSN25 projection, allowing 
for pixel-by-pixel energy flux calculations to entirely eliminate 
neighboring-pixel contamination before spatial averaging.

Inputs:
  * Cached monthly albedo on SIAL grid (Apr-Aug: CLARA-A3; Sep: APP-x)
  * Surface Albedo Radiative Kernel NetCDF (month, lat, lon)
  * GISTEMP annual global-mean anomaly for ΔT

Outputs:
  siaf_kernel_results.csv
  siaf_kernel_summary.csv
"""

'''import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc, resource, sys
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')

# ============================================================
# MEMORY AUDITOR
# ============================================================
def log_mem(tag=""):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem_kb / (1024.0 * 1024.0)
    print(f"[MEM] {tag:<45} | Peak RAM: {mem_gb:.3f} GB", flush=True)
    sys.stdout.flush()

log_mem("STARTING Radiative Kernel Method (PSN25)")

# ============================================================
# CONFIG
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'

# --- KERNEL CONFIGURATION ---
KERNEL_FILE  = '/glade/derecho/scratch/jrotondo/alb.kernel.nc'
KERNEL_VAR   = 'FSNT' # All-sky Net Shortwave Flux at TOA
KERNEL_SCALE = 100.0  # The CAM5 kernel is in W/m2 per 1% change. Our albedo is 0-1, so we multiply by 100.

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'siaf_kernel_results.csv')
OUT_SUM = os.path.join(OUT_DIR, 'siaf_kernel_summary.csv')

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

TREND_START_CMIP6, TREND_END_CMIP6 = 1979, 2014
TREND_START_OBS,   TREND_END_OBS   = 1979, 2024

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

AREA_RATIO_60N  = 0.0347       # True Arctic Ocean (>60°N) area ratio
SEASON_FRAC     = 6.0 / 12.0   
PVAL              = 0.05       
SIC_ICE_THRESHOLD = 0.15       

T_SOURCES = [
    ('GISTEMP',   '/glade/work/jrotondo/gistemp_global.csv',   'gistemp'),
    ('CowtanWay', '/glade/work/jrotondo/cowtan_way.txt',       'cowtan_way'),
    ('HadCRUT5',  '/glade/work/jrotondo/hadcrut5_global.csv',  'hadcrut5'),
    ('NOAA',      '/glade/work/jrotondo/noaa_global.csv',      'noaa'),
]

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

# ============================================================
# STEP 1: BUILD PSN25 GRID + MASKS + REGIONS
# ============================================================
print("=" * 70, flush=True)
print("STEP 1: PSN25 grid, non-land mask, 6 Regions", flush=True)
print("=" * 70, flush=True)

sample_sic = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
psn_shape = psn_xx.shape

transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d = transformer.transform(psn_xx, psn_yy)

phi    = np.deg2rad(psn_lat2d)
phi_ts = np.deg2rad(70.0)
k_ps   = (1.0 + np.sin(phi_ts)) / np.maximum(1.0 + np.sin(phi), 1e-9)
AREA_WEIGHT_PSN = (1.0 / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    OCEAN_PSN = globe.is_ocean(psn_lat2d, lon_pm)
except ImportError:
    print("  WARNING: global-land-mask not installed; land NOT excluded.")
    OCEAN_PSN = np.ones(psn_shape, dtype=bool)

ARCTIC_60N_PSN = (psn_lat2d >= 60.0)
DOMAIN_PSN     = OCEAN_PSN & ARCTIC_60N_PSN

REGION_MASKS = {}
for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
    lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    if lmn <= lmx:
        lon_ok = (lon_pm >= lmn) & (lon_pm <= lmx)
    else:
        lon_ok = (lon_pm >= lmn) | (lon_pm <= lmx)
    REGION_MASKS[code] = lat_ok & lon_ok & DOMAIN_PSN

# ============================================================
# STEP 2: BUILD REGRID INDICES → PSN25
# ============================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg, dtype=np.float32), np.deg2rad(lon_deg, dtype=np.float32)
    xyz = np.empty((lat.size, 3), dtype=np.float32)
    xyz[:, 0] = np.cos(lat) * np.cos(lon)
    xyz[:, 1] = np.cos(lat) * np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    return xyz

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk_size=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    idx  = np.empty(src_xyz.shape[0], dtype=np.int32)
    tree = cKDTree(tgt_xyz)
    for i in range(0, src_xyz.shape[0], chunk_size):
        e = min(i + chunk_size, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx, tgt_lat2d.shape

def scatter_avg(arr_flat, idx, out_shape):
    accum = np.zeros(np.prod(out_shape), dtype=np.float32)
    count = np.zeros(np.prod(out_shape), dtype=np.float32)
    valid = np.isfinite(arr_flat)
    np.add.at(accum, idx[valid], arr_flat[valid])
    np.add.at(count, idx[valid], np.float32(1.0))
    out = np.full(np.prod(out_shape), np.nan, dtype=np.float32)
    m = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

print("  SIAL → PSN25 scatter index ...", flush=True)
sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d

sial_shape = sial_lat2d.shape
try:
    arctic_sial = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy").astype(bool)
except:
    arctic_sial = (sial_lat2d >= 60.0)

idx_sial_to_psn, _ = build_scatter_index(sial_lat2d, sial_lon2d, psn_lat2d, psn_lon2d)

# Build APP-x → SIAL mapping for September data
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon

    appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
    sial_xyz_target = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
    _, idx_appx_to_sial = cKDTree(appx_xyz).query(sial_xyz_target, workers=-1)
    del appx_xyz, sial_xyz_target, appx_lat2d, appx_lon2d, appx_lat, appx_lon; gc.collect()
except Exception as e:
    print(f"  WARNING: APP-x → SIAL mapping failed ({e})")

# ============================================================
# STEP 3: KERNEL LOADER
# ============================================================
print("\nSTEP 3: Loading Radiative Kernel", flush=True)
def load_kernel_psn_month(mn):
    if not os.path.exists(KERNEL_FILE):
        print(f"  [ERROR] Kernel file not found: {KERNEL_FILE}")
        return None
        
    try:
        with xr.open_dataset(KERNEL_FILE) as ds:
            # CAM5 kernels use 'time' (size 12) to index the months
            month_idx = mn - 1 
            
            k_lat = ds.lat.values
            k_lon = ds.lon.values
            k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
            
            # Extract the 2D kernel slice for this month using 'time' instead of 'month'
            k_raw = ds[KERNEL_VAR].isel(time=month_idx).values.astype(np.float32)
            
            # We want the absolute value of the kernel for mathematically matching your IS equation
            # (An increase in albedo yields a negative SW flux change)
            k_raw = np.abs(k_raw) * KERNEL_SCALE
            
        # Scatter the regular lat/lon kernel onto our native PSN25 grid
        idx, shp = build_scatter_index(k_lat2d, k_lon2d, psn_lat2d, psn_lon2d)
        k_psn = scatter_avg(k_raw.ravel(), idx, shp).astype(np.float32)
        
        del idx, k_raw, k_lat2d, k_lon2d; gc.collect()
        return k_psn
    except Exception as e:
        print(f"  [ERROR] Failed to load kernel: {e}")
        return None

# ============================================================
# STEP 4: DELTA-T 
# ============================================================
def trend_dT(years, vals, y0, y1):
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5: return np.nan
    return stats.linregress(years[mask], vals[mask]).slope * (y1 - y0)

def _load_t_dataset(kind, path):
    if not os.path.exists(path): return None, None
    try:
        if kind == 'gistemp':
            df = (pd.read_csv(path, skiprows=1, na_values='***').assign(val=lambda d: pd.to_numeric(d['J-D'], errors='coerce')))
            return df['Year'].values.astype(float), df['val'].values.astype(float)
        if kind == 'cowtan_way':
            df = pd.read_csv(path, sep=r'\s+', header=None, names=['year','anomaly','unc1','unc2','unc3'])
            return df['year'].values.astype(float), df['anomaly'].values.astype(float)
        if kind == 'hadcrut5':
            df = pd.read_csv(path)
            vcol = [c for c in df.columns if 'anom' in c.lower() or 'mean' in c.lower()]
            return df.iloc[:, 0].values.astype(float), df[vcol[0]].values.astype(float) if vcol else df.iloc[:, 1].values.astype(float)
        if kind == 'noaa':
            df = pd.read_csv(path, comment='#')
            return df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)
    except: return None, None

def delta_T_4source(y0, y1):
    vals = []; labels = []
    for (lab, path, kind) in T_SOURCES:
        yy, vv = _load_t_dataset(kind, path)
        if yy is not None:
            dT = trend_dT(yy, vv, y0, y1)
            if np.isfinite(dT): vals.append(dT)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0

DELTA_T_CMIP6, DELTA_T_CMIP6_STD = delta_T_4source(TREND_START_CMIP6, TREND_END_CMIP6)
DELTA_T_OBS,   DELTA_T_OBS_STD   = delta_T_4source(TREND_START_OBS,   TREND_END_OBS)

# ============================================================
# STEP 5: ALBEDO STREAMING & SPLITTING
# ============================================================
def streaming_alb_sial(cache_dir, mname, years):
    use_appx = (mname == 'September')
    alb_sum, alb_cnt = None, None
    for y in years:
        p = f"{cache_dir}/{'appx' if use_appx else 'clara'}_alb_{mname}_{y}.npy"
        if not os.path.exists(p): continue
        
        raw_data = np.load(p).astype(np.float32)
        
        # If September, remap the 360x360 APP-x to the 361x361 SIAL shape
        if use_appx and idx_appx_to_sial is not None:
            if raw_data.shape != sial_shape:
                a_sial = raw_data.ravel()[idx_appx_to_sial].reshape(sial_shape)
            else:
                a_sial = raw_data
        else:
            a_sial = raw_data
            
        fin = np.isfinite(a_sial)
        a_sial[~fin] = 0.0
        if alb_sum is None:
            alb_sum, alb_cnt = a_sial, fin.astype(np.int16)
        else:
            alb_sum += a_sial; alb_cnt += fin.astype(np.int16)
            
    if alb_sum is None: return None
    v = alb_cnt > 0
    alb_sum[v] /= alb_cnt[v].astype(np.float32); alb_sum[~v] = np.nan
    return alb_sum

def compute_IS_split_sial(cache_dir, mname, early_years, late_years, delta_t):
    alb_e = streaming_alb_sial(cache_dir, mname, early_years)
    alb_l = streaming_alb_sial(cache_dir, mname, late_years)
    if alb_e is None or alb_l is None: return None, None, None

    dalpha = (alb_l - alb_e).astype(np.float32)
    try:
        slope = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy").astype(np.float32)
        pval  = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy").astype(np.float32)
        sic_e = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy").astype(np.float32)
        sic_l = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy").astype(np.float32)
    except: return None, None, None

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = valid & np.isfinite(slope) & np.isfinite(pval) & (slope < 0) & (pval < PVAL)
    stable_ice = valid & ~changing & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD)

    # IS = -dalpha / dT 
    IS_ext = np.where(changing,   -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable_ice, -dalpha / delta_t, np.nan).astype(np.float32)
    
    IS_tot = np.where(np.isfinite(IS_ext) | np.isfinite(IS_drk),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.) + np.where(np.isfinite(IS_drk), IS_drk, 0.),
                      np.nan).astype(np.float32)
    
    IS_ext = np.where(np.isfinite(IS_tot), np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan).astype(np.float32)
    IS_drk = np.where(np.isfinite(IS_tot), np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan).astype(np.float32)
    return IS_tot, IS_ext, IS_drk

def regrid_sial_to_psn(arr_sial, fill_zero_in_arctic=True):
    src = np.where(arctic_sial, np.where(np.isfinite(arr_sial), arr_sial, 0.), np.nan).astype(np.float32) if fill_zero_in_arctic else arr_sial.astype(np.float32)
    return scatter_avg(src.ravel(), idx_sial_to_psn, psn_shape)

def region_mean_psn(arr_psn, rmask):
    v, w = arr_psn[rmask], AREA_WEIGHT_PSN[rmask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0: return np.nan, 0.0
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))

# ============================================================
# STEP 6: KERNEL PIPELINE
# ============================================================
def run_kernel_method(period_label, cache_dir, early_years, late_years, delta_t, delta_t_std):
    rows = []
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f"\n  {period_label} / {mname}", flush=True)

        IS_tot_sial, IS_ext_sial, IS_drk_sial = compute_IS_split_sial(cache_dir, mname, early_years, late_years, delta_t)
        if IS_tot_sial is None: continue

        IS_tot_psn = regrid_sial_to_psn(IS_tot_sial, fill_zero_in_arctic=True)
        IS_ext_psn = regrid_sial_to_psn(IS_ext_sial, fill_zero_in_arctic=True)
        IS_drk_psn = regrid_sial_to_psn(IS_drk_sial, fill_zero_in_arctic=True)
        IS_tot_psn_icepx = regrid_sial_to_psn(IS_tot_sial, fill_zero_in_arctic=False)

        # Retrieve the pixel-wise Kernel
        kernel_psn = load_kernel_psn_month(mn)
        if kernel_psn is None: 
            print("    Kernel missing — skipping calculation.", flush=True)
            continue

        # Pixel-wise Energy Flux Formulations
        # SIAF = IS * |K| 
        SIAF_tot_psn = (IS_tot_psn * kernel_psn).astype(np.float32)
        SIAF_ext_psn = (IS_ext_psn * kernel_psn).astype(np.float32)
        SIAF_drk_psn = (IS_drk_psn * kernel_psn).astype(np.float32)

        def _sc(sigma_sign):
            if delta_t_std <= 0 or not np.isfinite(delta_t_std): return 1.0
            denom = delta_t + sigma_sign * delta_t_std
            return delta_t / denom if np.abs(denom) > 1e-6 else np.nan

        scale_hi, scale_lo = _sc(-1.0), _sc(+1.0)

        for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
            rmask = REGION_MASKS[code]
            if rmask.sum() == 0: continue

            k_r,         _      = region_mean_psn(kernel_psn,      rmask)
            SIAF_r,      wsum_r = region_mean_psn(SIAF_tot_psn,    rmask)
            SIAF_ext_r,  _      = region_mean_psn(SIAF_ext_psn,    rmask)
            SIAF_drk_r,  _      = region_mean_psn(SIAF_drk_psn,    rmask)
            IS_r,        _      = region_mean_psn(IS_tot_psn,      rmask)
            IS_ext_r,    _      = region_mean_psn(IS_ext_psn,      rmask)
            IS_drk_r,    _      = region_mean_psn(IS_drk_psn,      rmask)
            IS_icepx_r,  _      = region_mean_psn(IS_tot_psn_icepx, rmask)

            if not (np.isfinite(SIAF_r) and np.isfinite(k_r)): continue

            rows.append(dict(
                period=period_label, month=mname, month_num=mn,
                region_code=code, region=rname,
                Kernel_Mean=k_r, # Replaces 'RS'
                IS=IS_r, IS_ext=IS_ext_r, IS_drk=IS_drk_r, IS_icepx=IS_icepx_r,
                SIAF=SIAF_r, SIAF_ext=SIAF_ext_r, SIAF_drk=SIAF_drk_r,
                SIAF_lo=SIAF_r*scale_lo, SIAF_hi=SIAF_r*scale_hi,
                SIAF_ext_lo=SIAF_ext_r*scale_lo, SIAF_ext_hi=SIAF_ext_r*scale_hi,
                SIAF_drk_lo=SIAF_drk_r*scale_lo, SIAF_drk_hi=SIAF_drk_r*scale_hi,
                area_weight=wsum_r, delta_T=delta_t, delta_T_std=delta_t_std,
            ))
    return rows

# ============================================================
# STEP 7: RUN EXECUTION 
# ============================================================
all_rows = []
PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, DELTA_T_CMIP6_STD),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   DELTA_T_OBS,   DELTA_T_OBS_STD),
]

for (pl, cd, ey, ly, dt, dt_std) in PERIOD_CONFIGS:
    print(f"\n{'='*70}\nPeriod {pl} (ΔT={dt:+.4f} K)\n{'='*70}", flush=True)
    all_rows.extend(run_kernel_method(pl, cd, ey, ly, dt, dt_std))

# ============================================================
# STEP 8: OUTPUT SUMMARIES
# ============================================================
if not all_rows:
    print("\nNo rows generated. Ensure the KERNEL_FILE and KERNEL_VAR are set correctly!")
    sys.exit()

df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nRegional results → {OUT_CSV}", flush=True)

summary_rows = []
for (period, month), g in df.groupby(['period', 'month']):
    W = g['area_weight'].values
    if W.sum() == 0: continue
    def _wavg(col):
        v = g[col].values
        ok = np.isfinite(v) & (W > 0)
        return float(np.sum(v[ok] * W[ok]) / np.sum(W[ok])) if ok.sum() > 0 else np.nan

    siaf_arctic, siaf_arctic_ext, siaf_arctic_drk = _wavg('SIAF'), _wavg('SIAF_ext'), _wavg('SIAF_drk')
    
    S_season, S_annual = AREA_RATIO_60N, AREA_RATIO_60N * SEASON_FRAC

    summary_rows.append(dict(
        period=period, month=month,
        delta_T=g['delta_T'].iloc[0],
        Kernel_arctic=_wavg('Kernel_Mean'),
        SIAF_arctic=siaf_arctic, SIAF_arctic_ext=siaf_arctic_ext, SIAF_arctic_drk=siaf_arctic_drk,
        SIAF_global_season=siaf_arctic * S_season,
        SIAF_global_annual=siaf_arctic * S_annual,
        SIAF_global_annual_ext=siaf_arctic_ext * S_annual,
        SIAF_global_annual_drk=siaf_arctic_drk * S_annual,
    ))

_tmp = pd.DataFrame(summary_rows)
season_cols = [c for c in _tmp.columns if c not in ('period', 'month')]
for period, g in _tmp.groupby('period'):
    row = dict(period=period, month='MEAN_Apr-Sep')
    for c in season_cols: row[c] = float(np.nanmean(g[c].values))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUM, index=False)

print("\n--- Kernel-Method Summary (6 wedges on PSN25) ---", flush=True)
cols_show = ['period', 'month', 'delta_T', 'Kernel_arctic', 'SIAF_arctic', 
             'SIAF_global_annual', 'SIAF_global_annual_ext', 'SIAF_global_annual_drk']
with pd.option_context('display.float_format', '{:.4f}'.format):
    print(summary[cols_show].to_string(index=False), flush=True)

log_mem("DONE")'''

#!/usr/bin/env python
"""
diagnose_pistone_2014.py
-------------------------------------------------------
Applies Pistone et al. (2014, PNAS) methodology faithfully:

  * "The Arctic is defined as the area north of 60°N, and only nonland
    area is considered."
  * Six Pistone Fig S2 regions — 60° longitudinal wedges between 60°N
    and 90°N, counterclockwise from 0°.
  * Native polar-stereographic PSN25 grid (25 × 25 km) area-weighting.
  * ΔT from annual global-mean surface temperature.
  * Extent vs Darkening split preserved on the native SIAL grid before
    mapping to PSN25 for regional integration.

Outputs:
  siaf_pistone_results.csv  -- one row per (period, month, region)
  siaf_pistone_summary.csv  -- Arctic-wide + global, per (period, month)
"""

'''import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc, resource, sys
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# MEMORY AUDITOR
# ============================================================
def log_mem(tag=""):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem_kb / (1024.0 * 1024.0)
    print(f"[MEM] {tag:<45} | Peak RAM: {mem_gb:.3f} GB", flush=True)
    sys.stdout.flush()

log_mem("STARTING Pistone-2014 (PSN25, 6 wedges)")

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

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'siaf_pistone_results.csv')
OUT_SUM = os.path.join(OUT_DIR, 'siaf_pistone_summary.csv')

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

CERES_YRS_CMIP6 = list(range(2000, 2015))
CERES_YRS_OBS   = list(range(2000, 2025))

TREND_START_CMIP6, TREND_END_CMIP6 = 1979, 2014
TREND_START_OBS,   TREND_END_OBS   = 1979, 2024

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

AREA_RATIO_60N  = 0.0347       # Arctic OCEAN (>60°N) area / global surface area
SEASON_FRAC     = 6.0 / 12.0   # Apr-Sep insolation season / full year
PVAL              = 0.05       
SIC_ICE_THRESHOLD = 0.15       

T_SOURCES = [
    ('GISTEMP',   '/glade/work/jrotondo/gistemp_global.csv',   'gistemp'),
    ('CowtanWay', '/glade/work/jrotondo/cowtan_way.txt',       'cowtan_way'),
    ('HadCRUT5',  '/glade/work/jrotondo/hadcrut5_global.csv',  'hadcrut5'),
    ('NOAA',      '/glade/work/jrotondo/noaa_global.csv',      'noaa'),
]

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

# ============================================================
# STEP 1: BUILD PSN25 GRID + MASKS + REGIONS
# ============================================================
print("=" * 70, flush=True)
print("STEP 1: PSN25 grid, non-land mask, 6 Pistone regions", flush=True)
print("=" * 70, flush=True)

sample_sic = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
psn_shape = psn_xx.shape

transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d = transformer.transform(psn_xx, psn_yy)

phi    = np.deg2rad(psn_lat2d)
phi_ts = np.deg2rad(70.0)
k_ps   = (1.0 + np.sin(phi_ts)) / np.maximum(1.0 + np.sin(phi), 1e-9)
AREA_WEIGHT_PSN = (1.0 / k_ps**2).astype(np.float32)

print("  Non-land mask ...", flush=True)
try:
    from global_land_mask import globe
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    OCEAN_PSN = globe.is_ocean(psn_lat2d, lon_pm)
except ImportError:
    print("  WARNING: global-land-mask not installed; land NOT excluded.", flush=True)
    OCEAN_PSN = np.ones(psn_shape, dtype=bool)

ARCTIC_60N_PSN = (psn_lat2d >= 60.0)
DOMAIN_PSN     = OCEAN_PSN & ARCTIC_60N_PSN
print(f"  Ocean ∧ ≥60°N cells on PSN25: {int(DOMAIN_PSN.sum()):,}", flush=True)

print("  Regional masks:", flush=True)
REGION_MASKS = {}
for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
    lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    lon_ok = (lon_pm >= lmn) & (lon_pm <= lmx) if lmn <= lmx else (lon_pm >= lmn) | (lon_pm <= lmx)
    m = lat_ok & lon_ok & DOMAIN_PSN
    REGION_MASKS[code] = m
    print(f"    {code}  {rname:<38s}  {int(m.sum()):>6d} cells", flush=True)

# ============================================================
# STEP 2: BUILD REGRID INDICES → PSN25 & SIAL
# ============================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg, dtype=np.float32), np.deg2rad(lon_deg, dtype=np.float32)
    xyz = np.empty((lat.size, 3), dtype=np.float32)
    xyz[:, 0] = np.cos(lat) * np.cos(lon)
    xyz[:, 1] = np.cos(lat) * np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    return xyz

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk_size=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    idx  = np.empty(src_xyz.shape[0], dtype=np.int32)
    tree = cKDTree(tgt_xyz)
    for i in range(0, src_xyz.shape[0], chunk_size):
        e = min(i + chunk_size, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx, tgt_lat2d.shape

def scatter_avg(arr_flat, idx, out_shape):
    accum = np.zeros(np.prod(out_shape), dtype=np.float32)
    count = np.zeros(np.prod(out_shape), dtype=np.float32)
    valid = np.isfinite(arr_flat)
    np.add.at(accum, idx[valid], arr_flat[valid])
    np.add.at(count, idx[valid], np.float32(1.0))
    out = np.full(np.prod(out_shape), np.nan, dtype=np.float32)
    m = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

print("  SIAL → PSN25 scatter index ...", flush=True)
sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d) if sial_lat1d.ndim == 1 else (sial_lon1d, sial_lat1d)
sial_shape = sial_lat2d.shape

try:
    arctic_sial = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy").astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.0)

idx_sial_to_psn, _ = build_scatter_index(sial_lat2d, sial_lon2d, psn_lat2d, psn_lon2d)

# APP-x mapping for September
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat) if appx_lat.ndim == 1 else (appx_lon, appx_lat)
    
    appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
    sial_xyz_target = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
    _, idx_appx_to_sial = cKDTree(appx_xyz).query(sial_xyz_target, workers=-1)
    del appx_xyz, sial_xyz_target, appx_lat2d, appx_lon2d, appx_lat, appx_lon; gc.collect()
    print("  APP-x → SIAL scatter index built (for September)", flush=True)
except Exception as e:
    print(f"  WARNING: APP-x → SIAL mapping failed ({e})")

# ============================================================
# STEP 3: CERES + T07 → RS on PSN25
# ============================================================
def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd = np.where(rsds  > 1., rsds,  np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds / S;  q = rsut / S;  den = q + p - 2.*p*a
        R  = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan), 0., 0.99)
        A  = np.clip(1. - p*(1.-a*R) / np.where((1.-R) > 0.01, 1.-R, np.nan), 0., 0.99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs / S;  qc = rsutcs / S;  dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan), 0., 0.99)
        Rl = np.clip(R - Rc, 0., 0.99)
        RS = (S * (1.-A) * (1.-Rl)**2 / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS

def load_ceres_RS_psn_month(years, mn):
    sums = {k: None for k in ['solar','toa_all','sfc_d_all','sfc_u_all','toa_clr','sfc_d_clr','sfc_u_clr']}
    count = 0
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        clon2d, clat2d = np.meshgrid(ds.lon.values, ds.lat.values)
        tidxs = np.where([t.year in years and t.month == mn for t in times])[0]
        if len(tidxs) == 0: return None
        for tidx in tidxs:
            sl = ds.isel(time=tidx).load()
            arrs = [sl['solar_mon'].values.astype(np.float32),
                    sl['toa_sw_all_mon'].values.astype(np.float32),
                    sl['sfc_sw_down_all_mon'].values.astype(np.float32),
                    sl['sfc_sw_up_all_mon'].values.astype(np.float32),
                    sl['toa_sw_clr_c_mon'].values.astype(np.float32),
                    sl['sfc_sw_down_clr_c_mon'].values.astype(np.float32),
                    sl['sfc_sw_up_clr_c_mon'].values.astype(np.float32)]
            for k, a in zip(sums, arrs): sums[k] = a if count == 0 else sums[k] + a
            count += 1
            del sl
    for k in sums: sums[k] /= np.float32(count)

    RS = t07_invert(sums['solar'], sums['toa_all'], sums['sfc_d_all'],
                    sums['sfc_u_all'], sums['toa_clr'], sums['sfc_d_clr'],
                    sums['sfc_u_clr']).astype(np.float32)
    del sums; gc.collect()

    idx, shp = build_scatter_index(clat2d, clon2d, psn_lat2d, psn_lon2d)
    RS_psn = scatter_avg(RS.ravel(), idx, shp).astype(np.float32)
    del idx, RS, clat2d, clon2d; gc.collect()
    return RS_psn

# ============================================================
# STEP 4: ΔT from GISTEMP
# ============================================================
def trend_dT(years, vals, y0, y1):
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5: return np.nan
    return stats.linregress(years[mask], vals[mask]).slope * (y1 - y0)

def _load_t_dataset(kind, path):
    if not os.path.exists(path): return None, None
    try:
        if kind == 'gistemp':
            df = (pd.read_csv(path, skiprows=1, na_values='***').assign(val=lambda d: pd.to_numeric(d['J-D'], errors='coerce')))
            return df['Year'].values.astype(float), df['val'].values.astype(float)
        if kind == 'cowtan_way':
            df = pd.read_csv(path, sep=r'\s+', header=None, names=['year','anomaly','unc1','unc2','unc3'])
            return df['year'].values.astype(float), df['anomaly'].values.astype(float)
        if kind == 'hadcrut5':
            df = pd.read_csv(path)
            vcol = [c for c in df.columns if 'anom' in c.lower() or 'mean' in c.lower()]
            return df.iloc[:, 0].values.astype(float), df[vcol[0]].values.astype(float) if vcol else df.iloc[:, 1].values.astype(float)
        if kind == 'noaa':
            df = pd.read_csv(path, comment='#')
            return df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)
    except Exception: return None, None
    return None, None

def delta_T_4source(y0, y1):
    vals, labels = [], []
    for (lab, path, kind) in T_SOURCES:
        yy, vv = _load_t_dataset(kind, path)
        if yy is not None:
            dT = trend_dT(yy, vv, y0, y1)
            if np.isfinite(dT):
                vals.append(dT); labels.append(lab)
    if not vals: return (0.64 if y1 <= 2014 else 0.84), 0.10, []
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, labels

DELTA_T_CMIP6, DELTA_T_CMIP6_STD, _ = delta_T_4source(TREND_START_CMIP6, TREND_END_CMIP6)
DELTA_T_OBS,   DELTA_T_OBS_STD,   _ = delta_T_4source(TREND_START_OBS,   TREND_END_OBS)

# ============================================================
# STEP 5: ALBEDO ON NATIVE SIAL (for decomposition)
# ============================================================
def streaming_alb_sial(cache_dir, mname, years):
    use_appx = (mname == 'September')
    alb_sum, alb_cnt = None, None
    for y in years:
        p = f"{cache_dir}/{'appx' if use_appx else 'clara'}_alb_{mname}_{y}.npy"
        if not os.path.exists(p): continue
        
        raw_data = np.load(p).astype(np.float32)
        
        if use_appx and idx_appx_to_sial is not None:
            a_sial = raw_data.ravel()[idx_appx_to_sial].reshape(sial_shape) if raw_data.shape != sial_shape else raw_data
        else:
            a_sial = raw_data
            
        fin = np.isfinite(a_sial)
        a_sial[~fin] = 0.0
        if alb_sum is None:
            alb_sum, alb_cnt = a_sial, fin.astype(np.int16)
        else:
            alb_sum += a_sial; alb_cnt += fin.astype(np.int16)
            
    if alb_sum is None: return None
    v = alb_cnt > 0
    alb_sum[v] /= alb_cnt[v].astype(np.float32); alb_sum[~v] = np.nan
    return alb_sum

def compute_IS_split_sial(cache_dir, mname, early_years, late_years, delta_t):
    alb_e = streaming_alb_sial(cache_dir, mname, early_years)
    alb_l = streaming_alb_sial(cache_dir, mname, late_years)
    if alb_e is None or alb_l is None: return None, None, None

    dalpha = (alb_l - alb_e).astype(np.float32)
    try:
        slope = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy").astype(np.float32)
        pval  = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy").astype(np.float32)
        sic_e = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy").astype(np.float32)
        sic_l = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy").astype(np.float32)
    except Exception: return None, None, None

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval) & (slope < 0) & (pval < PVAL))
    stable_ice = (valid & ~changing & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

    IS_ext = np.where(changing,   -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable_ice, -dalpha / delta_t, np.nan).astype(np.float32)
    
    IS_tot = np.where(np.isfinite(IS_ext) | np.isfinite(IS_drk),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.) + np.where(np.isfinite(IS_drk), IS_drk, 0.),
                      np.nan).astype(np.float32)
                      
    IS_ext = np.where(np.isfinite(IS_tot), np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan).astype(np.float32)
    IS_drk = np.where(np.isfinite(IS_tot), np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan).astype(np.float32)

    return IS_tot, IS_ext, IS_drk

def regrid_sial_to_psn(arr_sial, fill_zero_in_arctic=True):
    src = np.where(arctic_sial, np.where(np.isfinite(arr_sial), arr_sial, 0.), np.nan).astype(np.float32) if fill_zero_in_arctic else arr_sial.astype(np.float32)
    return scatter_avg(src.ravel(), idx_sial_to_psn, psn_shape)

def region_mean_psn(arr_psn, rmask):
    v, w = arr_psn[rmask], AREA_WEIGHT_PSN[rmask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0: return np.nan, 0.0
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))

# ============================================================
# STEP 6: PISTONE PIPELINE 
# ============================================================
def run_pistone(period_label, cache_dir, early_years, late_years, delta_t, delta_t_std, ceres_years):
    rows = []
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f"\n  {period_label} / {mname}", flush=True)

        IS_tot_sial, IS_ext_sial, IS_drk_sial = compute_IS_split_sial(cache_dir, mname, early_years, late_years, delta_t)
        if IS_tot_sial is None:
            print("    missing albedo or SIC cache — skip", flush=True)
            continue

        IS_tot_psn = regrid_sial_to_psn(IS_tot_sial, fill_zero_in_arctic=True)
        IS_ext_psn = regrid_sial_to_psn(IS_ext_sial, fill_zero_in_arctic=True)
        IS_drk_psn = regrid_sial_to_psn(IS_drk_sial, fill_zero_in_arctic=True)
        IS_tot_psn_icepx = regrid_sial_to_psn(IS_tot_sial, fill_zero_in_arctic=False)

        rs_psn = load_ceres_RS_psn_month(ceres_years, mn)
        if rs_psn is None:
            print("    missing CERES — skip", flush=True); continue

        SIAF_tot_psn = (IS_tot_psn * rs_psn).astype(np.float32)
        SIAF_ext_psn = (IS_ext_psn * rs_psn).astype(np.float32)
        SIAF_drk_psn = (IS_drk_psn * rs_psn).astype(np.float32)

        def _sc(sigma_sign):
            if delta_t_std <= 0 or not np.isfinite(delta_t_std): return 1.0
            denom = delta_t + sigma_sign * delta_t_std
            return delta_t / denom if np.abs(denom) > 1e-6 else np.nan

        scale_hi, scale_lo = _sc(-1.0), _sc(+1.0)

        for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
            rmask = REGION_MASKS[code]
            if rmask.sum() == 0: continue

            rs_r,        _      = region_mean_psn(rs_psn,          rmask)
            SIAF_r,      wsum_r = region_mean_psn(SIAF_tot_psn,    rmask)
            SIAF_ext_r,  _      = region_mean_psn(SIAF_ext_psn,    rmask)
            SIAF_drk_r,  _      = region_mean_psn(SIAF_drk_psn,    rmask)
            IS_r,        _      = region_mean_psn(IS_tot_psn,      rmask)
            IS_ext_r,    _      = region_mean_psn(IS_ext_psn,      rmask)
            IS_drk_r,    _      = region_mean_psn(IS_drk_psn,      rmask)
            IS_icepx_r,  _      = region_mean_psn(IS_tot_psn_icepx, rmask)

            if not (np.isfinite(SIAF_r) and np.isfinite(rs_r)): continue

            rows.append(dict(
                period=period_label, month=mname, month_num=mn,
                region_code=code, region=rname,
                RS=rs_r, IS=IS_r, IS_ext=IS_ext_r, IS_drk=IS_drk_r, IS_icepx=IS_icepx_r,
                SIAF=SIAF_r, SIAF_ext=SIAF_ext_r, SIAF_drk=SIAF_drk_r,
                SIAF_lo=SIAF_r*scale_lo, SIAF_hi=SIAF_r*scale_hi,
                SIAF_ext_lo=SIAF_ext_r*scale_lo, SIAF_ext_hi=SIAF_ext_r*scale_hi,
                SIAF_drk_lo=SIAF_drk_r*scale_lo, SIAF_drk_hi=SIAF_drk_r*scale_hi,
                area_weight=wsum_r, delta_T=delta_t, delta_T_std=delta_t_std,
                n_cells=int(rmask.sum()),
            ))
    return rows

# ============================================================
# STEP 7: RUN BOTH PERIODS
# ============================================================
all_rows = []
PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, DELTA_T_CMIP6_STD, CERES_YRS_CMIP6),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   DELTA_T_OBS,   DELTA_T_OBS_STD,   CERES_YRS_OBS),
]
for (pl, cd, ey, ly, dt, dt_std, cy) in PERIOD_CONFIGS:
    print(f"\n{'='*70}\nPeriod {pl} (ΔT={dt:+.4f} K)\n{'='*70}", flush=True)
    all_rows.extend(run_pistone(pl, cd, ey, ly, dt, dt_std, cy))

# ============================================================
# STEP 8: OUTPUT CSVs
# ============================================================
df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nRegional results → {OUT_CSV}", flush=True)

summary_rows = []
for (period, month), g in df.groupby(['period', 'month']):
    W = g['area_weight'].values
    if W.sum() == 0: continue
    def _wavg(col):
        v = g[col].values
        ok = np.isfinite(v) & (W > 0)
        return float(np.sum(v[ok] * W[ok]) / np.sum(W[ok])) if ok.sum() > 0 else np.nan

    siaf_arctic, siaf_arctic_ext, siaf_arctic_drk = _wavg('SIAF'), _wavg('SIAF_ext'), _wavg('SIAF_drk')
    siaf_arctic_lo, siaf_arctic_hi = _wavg('SIAF_lo'), _wavg('SIAF_hi')
    siaf_arctic_ext_lo, siaf_arctic_ext_hi = _wavg('SIAF_ext_lo'), _wavg('SIAF_ext_hi')
    siaf_arctic_drk_lo, siaf_arctic_drk_hi = _wavg('SIAF_drk_lo'), _wavg('SIAF_drk_hi')

    S_season = AREA_RATIO_60N
    S_annual = AREA_RATIO_60N * SEASON_FRAC

    summary_rows.append(dict(
        period=period, month=month, delta_T=g['delta_T'].iloc[0], delta_T_std=g['delta_T_std'].iloc[0],
        IS_arctic=_wavg('IS'), IS_arctic_ext=_wavg('IS_ext'), IS_arctic_drk=_wavg('IS_drk'), IS_arctic_icepx=_wavg('IS_icepx'),
        RS_arctic=_wavg('RS'),
        SIAF_arctic=siaf_arctic, SIAF_arctic_lo=siaf_arctic_lo, SIAF_arctic_hi=siaf_arctic_hi,
        SIAF_arctic_ext=siaf_arctic_ext, SIAF_arctic_ext_lo=siaf_arctic_ext_lo, SIAF_arctic_ext_hi=siaf_arctic_ext_hi,
        SIAF_arctic_drk=siaf_arctic_drk, SIAF_arctic_drk_lo=siaf_arctic_drk_lo, SIAF_arctic_drk_hi=siaf_arctic_drk_hi,
        SIAF_global_season=siaf_arctic * S_season, SIAF_global_season_ext=siaf_arctic_ext * S_season, SIAF_global_season_drk=siaf_arctic_drk * S_season,
        SIAF_global_annual=siaf_arctic * S_annual, SIAF_global_annual_lo=siaf_arctic_lo * S_annual, SIAF_global_annual_hi=siaf_arctic_hi * S_annual,
        SIAF_global_annual_ext=siaf_arctic_ext * S_annual, SIAF_global_annual_ext_lo=siaf_arctic_ext_lo * S_annual, SIAF_global_annual_ext_hi=siaf_arctic_ext_hi * S_annual,
        SIAF_global_annual_drk=siaf_arctic_drk * S_annual, SIAF_global_annual_drk_lo=siaf_arctic_drk_lo * S_annual, SIAF_global_annual_drk_hi=siaf_arctic_drk_hi * S_annual,
        n_regions=int(len(g)),
    ))

_tmp = pd.DataFrame(summary_rows)
season_cols = [c for c in _tmp.columns if c not in ('period', 'month', 'n_regions')]
for period, g in _tmp.groupby('period'):
    row = dict(period=period, month='MEAN_Apr-Sep')
    for c in season_cols: row[c] = float(np.nanmean(g[c].values))
    row['n_regions'] = int(np.nanmean(g['n_regions'].values))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUM, index=False)
print(f"Summary        → {OUT_SUM}", flush=True)

cols_show = ['period', 'month', 'delta_T', 'SIAF_arctic', 'SIAF_arctic_ext', 'SIAF_arctic_drk',
             'SIAF_global_annual', 'SIAF_global_annual_ext', 'SIAF_global_annual_drk']
print("\n--- Pistone-2014 summary (6 wedges on PSN25) ---", flush=True)
with pd.option_context('display.float_format', '{:.4f}'.format):
    print(summary[cols_show].to_string(index=False), flush=True)

log_mem("DONE")'''

'''#!/usr/bin/env python
"""
diagnose_kernel_method_dual.py
-------------------------------------------------------
Calculates Sea Ice Albedo Feedback (SIAF) using the Radiative Kernel approach,
running TWO kernel datasets side-by-side for direct comparison:

  1. CAM5 kernel  — Pendergrass et al. (2018, NCAR)
       GCM-derived. Variable: FSNT (W/m² per 1% Δα). Scale: ×100.
       Source: https://doi.org/10.5065/D6F47MT6
       On GLADE: /glade/collections/cdg/data/radiative-kernels/
                 (file: alb.kernel.nc)

  2. ERA5 kernel — Huang & Huang (2023, Earth Syst. Sci. Data)
       Reanalysis-based (RRTMG offline on ERA5 profiles).
       Variable: alb (W/m² per Δα, 0–1 scale). Scale: ×1.
       DOI: 10.17632/vmg3s67568.4
       Data: https://data.mendeley.com/datasets/vmg3s67568/4
             (file: ERA5_kernel_alb_SFC.nc)
       Download helper: run  python diagnose_kernel_method_dual.py --download
       to fetch the Huang ERA5 kernel automatically.

Key differences:
  * CAM5  uses a GCM Arctic cloud climatology (low-cloud bias → overestimates
    clear-sky transmittance → tends to overestimate SIAF).
  * ERA5 uses highly constrained reanalysis cloud profiles → better MIZ representation.
  * Both are all-sky surface-albedo kernels, so cloud masking of the surface
    albedo signal is already folded in.

The spread between the two kernel sets is reported as an "atmospheric
state uncertainty" in the final summary.

Outputs:
  siaf_kernel_dual_results.csv   — one row per (period, month, region, kernel)
  siaf_kernel_dual_summary.csv   — Arctic-wide + global, per (period, month),
                                   columns for each kernel + spread
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc, resource, sys, urllib.request, shutil
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# DOWNLOAD HELPER  (run once: python script.py --download)
# ============================================================

ERA5_REMOTE_URL  = 'https://data.mendeley.com/public-files/datasets/vmg3s67568/files/14846067-2790-449e-990a-0985558e806d/file_download'
ERA5_LOCAL_PATH  = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'

def download_era5_kernel():
    """Fetch Huang ERA5 surface albedo kernel if not already present."""
    os.makedirs(os.path.dirname(ERA5_LOCAL_PATH), exist_ok=True)
    if os.path.exists(ERA5_LOCAL_PATH):
        print(f"  ERA5 kernel already present at {ERA5_LOCAL_PATH}")
        return True
    print(f"  Downloading ERA5 kernel from Mendeley Data ...")
    try:
        # Mendeley requires a User-Agent to prevent 403 Forbidden errors
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        with urllib.request.urlopen(ERA5_REMOTE_URL, timeout=120) as r, \
             open(ERA5_LOCAL_PATH, 'wb') as f:
            shutil.copyfileobj(r, f)
        print(f"  Saved to {ERA5_LOCAL_PATH}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Download manually from https://data.mendeley.com/datasets/vmg3s67568/4")
        print(f"  (Download the file named ERA5_kernel_alb_SFC.nc)")
        print(f"  and place at           {ERA5_LOCAL_PATH}")
        return False

# ============================================================
# KERNEL REGISTRY
# Each entry: file path, variable name, unit scale, description.
#
# Scale convention:
#   CAM5  : W/m² per 1% albedo change  → multiply by 100 to get W/m² per Δα(0–1)
#   Huang : W/m² per Δα(0–1)           → scale = 1.0
#
# Sign convention:
#   We take abs() because a decrease in albedo (negative Δα) produces a
#   positive forcing, and IS = −Δα/ΔT is already defined as positive.
#   SIAF = IS × |K|  (both positive for warming/darkening feedback)
#
# 'time_dim': CAM5 uses 'time' (size 12); Huang uses 'month' (size 12).
# ============================================================

KERNEL_CONFIGS = {
    'CAM5': dict(
        file        = '/glade/derecho/scratch/jrotondo/alb.kernel.nc',
        var         = 'FSNT',
        scale       = 100.0,    # kernel is W/m² per 1%, our Δα is 0–1
        time_dim    = 'time',
        description = 'CAM5 GCM kernel (Pendergrass et al. 2018)',
    ),
    'Huang_ERA5': dict(
        file        = ERA5_LOCAL_PATH,
        var         = 'SFC_all',      
        scale       = 100.0,          # <--- Now converts 1% to 0-1 scale
        time_dim    = 'month',
        description = 'ERA5 offline kernel (Huang & Huang 2023)',
    ),
}

# ============================================================
# MEMORY AUDITOR
# ============================================================
def log_mem(tag=""):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem_kb / (1024.0 * 1024.0)
    print(f"[MEM] {tag:<50} | Peak RAM: {mem_gb:.3f} GB", flush=True)

log_mem("STARTING dual-kernel SIAF")

# ============================================================
# CONFIG
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'siaf_kernel_dual_results.csv')
OUT_SUM = os.path.join(OUT_DIR, 'siaf_kernel_dual_summary.csv')

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

TREND_START_CMIP6, TREND_END_CMIP6 = 1979, 2014
TREND_START_OBS,   TREND_END_OBS   = 1979, 2024

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

AREA_RATIO_60N  = 0.0347       # Arctic ocean (>60°N) / global surface area
SEASON_FRAC     = 6.0 / 12.0   # Apr–Sep / full year
PVAL            = 0.05
SIC_ICE_THRESHOLD = 0.15

T_SOURCES = [
    ('GISTEMP',   '/glade/work/jrotondo/gistemp_global.csv',   'gistemp'),
    ('CowtanWay', '/glade/work/jrotondo/cowtan_way.txt',       'cowtan_way'),
    ('HadCRUT5',  '/glade/work/jrotondo/hadcrut5_global.csv',  'hadcrut5'),
    ('NOAA',      '/glade/work/jrotondo/noaa_global.csv',      'noaa'),
]

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

# ============================================================
# STEP 1: PSN25 GRID + MASKS + REGIONS
# ============================================================
print("=" * 70, flush=True)
print("STEP 1: PSN25 grid, non-land mask, 6 Pistone regions", flush=True)
print("=" * 70, flush=True)

sample_sic = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
psn_shape = psn_xx.shape

transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d = transformer.transform(psn_xx, psn_yy)

# Polar-stereographic area-weighting correction
phi    = np.deg2rad(psn_lat2d)
phi_ts = np.deg2rad(70.0)
k_ps   = (1.0 + np.sin(phi_ts)) / np.maximum(1.0 + np.sin(phi), 1e-9)
AREA_WEIGHT_PSN = (1.0 / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    OCEAN_PSN = globe.is_ocean(psn_lat2d, lon_pm)
except ImportError:
    print("  WARNING: global-land-mask not installed; land NOT excluded.", flush=True)
    OCEAN_PSN = np.ones(psn_shape, dtype=bool)

ARCTIC_60N_PSN = (psn_lat2d >= 60.0)
DOMAIN_PSN     = OCEAN_PSN & ARCTIC_60N_PSN
print(f"  Ocean ∧ ≥60°N cells on PSN25: {int(DOMAIN_PSN.sum()):,}", flush=True)

REGION_MASKS = {}
for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
    lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    lon_ok = ((lon_pm >= lmn) & (lon_pm <= lmx)
              if lmn <= lmx else (lon_pm >= lmn) | (lon_pm <= lmx))
    m = lat_ok & lon_ok & DOMAIN_PSN
    REGION_MASKS[code] = m
    print(f"    {code}  {rname:<38s}  {int(m.sum()):>6d} cells", flush=True)

# ============================================================
# STEP 2: REGRID INDICES
# ============================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg, dtype=np.float32)
    lon = np.deg2rad(lon_deg, dtype=np.float32)
    xyz = np.empty((lat.size, 3), dtype=np.float32)
    xyz[:, 0] = np.cos(lat) * np.cos(lon)
    xyz[:, 1] = np.cos(lat) * np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    return xyz

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d,
                        chunk_size=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    idx  = np.empty(src_xyz.shape[0], dtype=np.int32)
    tree = cKDTree(tgt_xyz)
    for i in range(0, src_xyz.shape[0], chunk_size):
        e = min(i + chunk_size, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx, tgt_lat2d.shape

def scatter_avg(arr_flat, idx, out_shape):
    accum = np.zeros(np.prod(out_shape), dtype=np.float32)
    count = np.zeros(np.prod(out_shape), dtype=np.float32)
    valid = np.isfinite(arr_flat)
    np.add.at(accum, idx[valid], arr_flat[valid])
    np.add.at(count, idx[valid], np.float32(1.0))
    out    = np.full(np.prod(out_shape), np.nan, dtype=np.float32)
    m      = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

print("\n  SIAL → PSN25 scatter index ...", flush=True)
sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

try:
    arctic_sial = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy").astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.0)

idx_sial_to_psn, _ = build_scatter_index(
    sial_lat2d, sial_lon2d, psn_lat2d, psn_lon2d)

# APP-x → SIAL for September
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon
    appx_xyz       = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
    sial_xyz_tgt   = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
    _, idx_appx_to_sial = cKDTree(appx_xyz).query(sial_xyz_tgt, workers=-1)
    del appx_xyz, sial_xyz_tgt, appx_lat2d, appx_lon2d, appx_lat, appx_lon
    gc.collect()
    print("  APP-x → SIAL index built", flush=True)
except Exception as e:
    print(f"  WARNING: APP-x → SIAL mapping failed ({e})", flush=True)

# ============================================================
# STEP 3: KERNEL LOADER (handles both CAM5 and Huang layouts)
# ============================================================

# Cache regridded kernels to avoid rebuilding indices each month
_kernel_psn_cache = {}   # {kernel_name: {month_num: 2d array}}

def load_kernel_psn_month(kname, mn):
    """
    Load the per-month kernel for kernel config `kname`, regrid to PSN25.
    Result is cached in memory so the scatter index is built only once
    per kernel dataset (not once per month-loop iteration).
    """
    if kname in _kernel_psn_cache and mn in _kernel_psn_cache[kname]:
        return _kernel_psn_cache[kname][mn]

    cfg = KERNEL_CONFIGS[kname]
    if not os.path.exists(cfg['file']):
        print(f"  [WARN] Kernel file missing: {cfg['file']}", flush=True)
        return None

    try:
        with xr.open_dataset(cfg['file']) as ds:
            # Flexible time/month dimension
            tdim = cfg['time_dim']
            if tdim not in ds.dims:
                # Fall back: pick whichever size-12 dim exists
                tdim = next((d for d, s in ds.dims.items() if s == 12), None)
                if tdim is None:
                    raise KeyError(f"No size-12 dimension found in {cfg['file']}")

            # Robust variable lookup: try exact name, then case-insensitive
            vname = cfg['var']
            if vname not in ds:
                vname = next((v for v in ds.data_vars
                               if v.lower() == cfg['var'].lower()), None)
                if vname is None:
                    raise KeyError(
                        f"Variable '{cfg['var']}' not found. "
                        f"Available: {list(ds.data_vars)}")

            # Robust coordinate lookup (handles 'lat' or 'latitude')
            lat_name = 'latitude' if 'latitude' in ds else 'lat'
            lon_name = 'longitude' if 'longitude' in ds else 'lon'
            
            k_lat = ds[lat_name].values
            k_lon = ds[lon_name].values
            k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)

            # month index: CAM5 and Huang both store Jan=0 … Dec=11
            month_idx = mn - 1
            k_raw = (ds[vname].isel({tdim: month_idx})
                     .values.astype(np.float32))

        # Apply scale and take absolute value.
        # Sign convention: IS = −Δα/ΔT > 0, we want SIAF = IS × |K| > 0
        k_raw = np.abs(k_raw) * cfg['scale']

        # Build scatter index from kernel lat/lon → PSN25 (done once,
        # then reused for all months of the same kernel).
        if kname not in _kernel_psn_cache:
            _kernel_psn_cache[kname] = {}
            log_mem(f"  Building scatter idx for {kname}")
            # Store index on the cache so we don't rebuild each month
            _kernel_psn_cache[kname]['_idx'] = build_scatter_index(
                k_lat2d, k_lon2d, psn_lat2d, psn_lon2d)[0]
            log_mem(f"  Scatter idx for {kname} done")

        idx_k = _kernel_psn_cache[kname]['_idx']
        k_psn = scatter_avg(k_raw.ravel(), idx_k, psn_shape).astype(np.float32)
        _kernel_psn_cache[kname][mn] = k_psn

        del k_raw, k_lat2d, k_lon2d; gc.collect()
        return k_psn

    except Exception as e:
        print(f"  [ERROR] Failed to load {kname} kernel month {mn}: {e}",
              flush=True)
        return None

# ============================================================
# STEP 4: ΔT (multi-source)
# ============================================================
def trend_dT(years, vals, y0, y1):
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5: return np.nan
    return stats.linregress(years[mask], vals[mask]).slope * (y1 - y0)

def _load_t_dataset(kind, path):
    if not os.path.exists(path): return None, None
    try:
        if kind == 'gistemp':
            df = (pd.read_csv(path, skiprows=1, na_values='***')
                    .assign(val=lambda d: pd.to_numeric(d['J-D'], errors='coerce')))
            return df['Year'].values.astype(float), df['val'].values.astype(float)
        if kind == 'cowtan_way':
            df = pd.read_csv(path, sep=r'\s+', header=None,
                              names=['year','anomaly','unc1','unc2','unc3'])
            return df['year'].values.astype(float), df['anomaly'].values.astype(float)
        if kind == 'hadcrut5':
            df   = pd.read_csv(path)
            vcol = [c for c in df.columns
                    if 'anom' in c.lower() or 'mean' in c.lower()]
            return (df.iloc[:, 0].values.astype(float),
                    df[vcol[0]].values.astype(float) if vcol
                    else df.iloc[:, 1].values.astype(float))
        if kind == 'noaa':
            df = pd.read_csv(path, comment='#')
            return (df.iloc[:, 0].values.astype(float),
                    df.iloc[:, 1].values.astype(float))
    except Exception:
        return None, None
    return None, None

def delta_T_4source(y0, y1):
    vals = []
    for (lab, path, kind) in T_SOURCES:
        yy, vv = _load_t_dataset(kind, path)
        if yy is not None:
            dT = trend_dT(yy, vv, y0, y1)
            if np.isfinite(dT): vals.append(dT)
    if not vals:
        return (0.64 if y1 <= 2014 else 0.84), 0.10
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

DELTA_T_CMIP6, DELTA_T_CMIP6_STD = delta_T_4source(TREND_START_CMIP6, TREND_END_CMIP6)
DELTA_T_OBS,   DELTA_T_OBS_STD   = delta_T_4source(TREND_START_OBS,   TREND_END_OBS)
print(f"\n  DELTA_T_CMIP6 = {DELTA_T_CMIP6:.4f} ± {DELTA_T_CMIP6_STD:.4f} K")
print(f"  DELTA_T_OBS   = {DELTA_T_OBS:.4f} ± {DELTA_T_OBS_STD:.4f} K")

# ============================================================
# STEP 5: ALBEDO ON SIAL + IS DECOMPOSITION
# ============================================================
def streaming_alb_sial(cache_dir, mname, years):
    use_appx  = (mname == 'September')
    alb_sum   = None
    alb_cnt   = None
    for y in years:
        p = (f"{cache_dir}/appx_alb_{mname}_{y}.npy"
             if use_appx else f"{cache_dir}/clara_alb_{mname}_{y}.npy")
        if not os.path.exists(p): continue

        raw = np.load(p).astype(np.float32)
        if use_appx and idx_appx_to_sial is not None:
            a = (raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
                 if raw.shape != sial_shape else raw)
        else:
            a = raw

        fin = np.isfinite(a)
        a[~fin] = 0.0
        if alb_sum is None:
            alb_sum = a;    alb_cnt = fin.astype(np.int16)
        else:
            alb_sum += a;   alb_cnt += fin.astype(np.int16)

    if alb_sum is None: return None
    v = alb_cnt > 0
    alb_sum[v] /= alb_cnt[v].astype(np.float32)
    alb_sum[~v]  = np.nan
    return alb_sum


def compute_IS_split_sial(cache_dir, mname, early_years, late_years, delta_t):
    alb_e = streaming_alb_sial(cache_dir, mname, early_years)
    alb_l = streaming_alb_sial(cache_dir, mname, late_years)
    if alb_e is None or alb_l is None: return None, None, None

    dalpha = (alb_l - alb_e).astype(np.float32)
    try:
        slope = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy").astype(np.float32)
        pval  = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy").astype(np.float32)
        sic_e = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy").astype(np.float32)
        sic_l = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy").astype(np.float32)
    except Exception:
        return None, None, None

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval)
                  & (slope < 0) & (pval < PVAL))
    stable_ice = (valid & ~changing
                  & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

    IS_ext = np.where(changing,   -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable_ice, -dalpha / delta_t, np.nan).astype(np.float32)

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


def regrid_sial_to_psn(arr_sial, fill_zero_in_arctic=True):
    if fill_zero_in_arctic:
        src = np.where(arctic_sial,
                       np.where(np.isfinite(arr_sial), arr_sial, 0.),
                       np.nan).astype(np.float32)
    else:
        src = arr_sial.astype(np.float32)
    return scatter_avg(src.ravel(), idx_sial_to_psn, psn_shape)


def region_mean_psn(arr_psn, rmask):
    v, w = arr_psn[rmask], AREA_WEIGHT_PSN[rmask]
    ok   = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0: return np.nan, 0.0
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))

# ============================================================
# STEP 6: DUAL-KERNEL PIPELINE
# ============================================================
def run_dual_kernel(period_label, cache_dir, early_years, late_years,
                    delta_t, delta_t_std):
    """
    For each month, compute IS once on SIAL, then apply EACH kernel
    in turn. Returns list of row-dicts with a 'kernel' column.
    """
    rows = []

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f"\n  {period_label} / {mname}", flush=True)

        # IS decomposition is kernel-independent — compute once
        IS_tot_s, IS_ext_s, IS_drk_s = compute_IS_split_sial(
            cache_dir, mname, early_years, late_years, delta_t)
        if IS_tot_s is None:
            print("    missing albedo/SIC cache — skip", flush=True)
            continue

        # Regrid IS to PSN25 (done once, shared across both kernels)
        IS_tot_psn      = regrid_sial_to_psn(IS_tot_s, fill_zero_in_arctic=True)
        IS_ext_psn      = regrid_sial_to_psn(IS_ext_s, fill_zero_in_arctic=True)
        IS_drk_psn      = regrid_sial_to_psn(IS_drk_s, fill_zero_in_arctic=True)
        IS_tot_psn_icepx= regrid_sial_to_psn(IS_tot_s, fill_zero_in_arctic=False)
        log_mem(f"  IS on PSN25 ({mname})")

        # ΔT uncertainty scaling factors
        def _sc(sigma_sign):
            if delta_t_std <= 0 or not np.isfinite(delta_t_std): return 1.0
            denom = delta_t + sigma_sign * delta_t_std
            return delta_t / denom if np.abs(denom) > 1e-6 else np.nan

        scale_hi, scale_lo = _sc(-1.0), _sc(+1.0)

        # ---- loop over kernels ----
        for kname, kcfg in KERNEL_CONFIGS.items():
            print(f"    kernel: {kname}", flush=True)

            k_psn = load_kernel_psn_month(kname, mn)
            if k_psn is None:
                print(f"      [SKIP] kernel unavailable", flush=True)
                continue

            # Pixel-wise SIAF = IS × |K|
            SIAF_tot = (IS_tot_psn * k_psn).astype(np.float32)
            SIAF_ext = (IS_ext_psn * k_psn).astype(np.float32)
            SIAF_drk = (IS_drk_psn * k_psn).astype(np.float32)

            # Regional aggregation
            for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
                rmask = REGION_MASKS[code]
                if rmask.sum() == 0: continue

                k_r,        _      = region_mean_psn(k_psn,          rmask)
                SIAF_r,     wsum_r = region_mean_psn(SIAF_tot,       rmask)
                SIAF_ext_r, _      = region_mean_psn(SIAF_ext,       rmask)
                SIAF_drk_r, _      = region_mean_psn(SIAF_drk,       rmask)
                IS_r,       _      = region_mean_psn(IS_tot_psn,     rmask)
                IS_ext_r,   _      = region_mean_psn(IS_ext_psn,     rmask)
                IS_drk_r,   _      = region_mean_psn(IS_drk_psn,     rmask)
                IS_ice_r,   _      = region_mean_psn(IS_tot_psn_icepx, rmask)

                if not np.isfinite(SIAF_r): continue

                rows.append(dict(
                    period       = period_label,
                    month        = mname,
                    month_num    = mn,
                    kernel       = kname,
                    kernel_desc  = kcfg['description'],
                    region_code  = code,
                    region       = rname,
                    Kernel_Mean  = k_r,
                    IS           = IS_r,
                    IS_ext       = IS_ext_r,
                    IS_drk       = IS_drk_r,
                    IS_icepx     = IS_ice_r,
                    SIAF         = SIAF_r,
                    SIAF_ext     = SIAF_ext_r,
                    SIAF_drk     = SIAF_drk_r,
                    SIAF_lo      = SIAF_r     * scale_lo,
                    SIAF_hi      = SIAF_r     * scale_hi,
                    SIAF_ext_lo  = SIAF_ext_r * scale_lo,
                    SIAF_ext_hi  = SIAF_ext_r * scale_hi,
                    SIAF_drk_lo  = SIAF_drk_r * scale_lo,
                    SIAF_drk_hi  = SIAF_drk_r * scale_hi,
                    area_weight  = wsum_r,
                    delta_T      = delta_t,
                    delta_T_std  = delta_t_std,
                    n_cells      = int(rmask.sum()),
                ))

            del SIAF_tot, SIAF_ext, SIAF_drk

        del IS_tot_s, IS_ext_s, IS_drk_s
        del IS_tot_psn, IS_ext_psn, IS_drk_psn, IS_tot_psn_icepx
        gc.collect()

    return rows

# ============================================================
# STEP 7: RUN BOTH PERIODS
# ============================================================
all_rows = []
PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, DELTA_T_CMIP6_STD),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   DELTA_T_OBS,   DELTA_T_OBS_STD),
]
for (pl, cd, ey, ly, dt, dt_std) in PERIOD_CONFIGS:
    print(f"\n{'='*70}\nPeriod {pl} (ΔT={dt:+.4f} K)\n{'='*70}", flush=True)
    all_rows.extend(run_dual_kernel(pl, cd, ey, ly, dt, dt_std))

# ============================================================
# STEP 8: OUTPUT CSVs
# ============================================================
if not all_rows:
    print("\nNo rows generated. Check kernel file paths.", flush=True)
    sys.exit(1)

df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nRegional results → {OUT_CSV}", flush=True)

# ---- Summary: one row per (period, month, kernel) ----
summary_rows = []
for (period, month, kname), g in df.groupby(['period', 'month', 'kernel']):
    W = g['area_weight'].values
    if W.sum() == 0: continue

    def _wavg(col):
        v  = g[col].values
        ok = np.isfinite(v) & (W > 0)
        return float(np.sum(v[ok] * W[ok]) / np.sum(W[ok])) if ok.sum() > 0 else np.nan

    s_a   = _wavg('SIAF')
    s_ext = _wavg('SIAF_ext')
    s_drk = _wavg('SIAF_drk')

    S_season = AREA_RATIO_60N
    S_annual = AREA_RATIO_60N * SEASON_FRAC

    summary_rows.append(dict(
        period           = period,
        month            = month,
        kernel           = kname,
        kernel_desc      = g['kernel_desc'].iloc[0],
        delta_T          = g['delta_T'].iloc[0],
        delta_T_std      = g['delta_T_std'].iloc[0],
        IS_arctic        = _wavg('IS'),
        IS_arctic_ext    = _wavg('IS_ext'),
        IS_arctic_drk    = _wavg('IS_drk'),
        IS_arctic_icepx  = _wavg('IS_icepx'),
        Kernel_arctic    = _wavg('Kernel_Mean'),
        SIAF_arctic      = s_a,
        SIAF_arctic_ext  = s_ext,
        SIAF_arctic_drk  = s_drk,
        SIAF_arctic_lo   = _wavg('SIAF_lo'),
        SIAF_arctic_hi   = _wavg('SIAF_hi'),
        SIAF_global_season     = s_a   * S_season,
        SIAF_global_season_ext = s_ext * S_season,
        SIAF_global_season_drk = s_drk * S_season,
        SIAF_global_annual     = s_a   * S_annual,
        SIAF_global_annual_ext = s_ext * S_annual,
        SIAF_global_annual_drk = s_drk * S_annual,
        SIAF_global_annual_lo  = _wavg('SIAF_lo') * S_annual,
        SIAF_global_annual_hi  = _wavg('SIAF_hi') * S_annual,
        n_regions        = int(len(g)),
    ))

# Seasonal means per (period, kernel)
_tmp = pd.DataFrame(summary_rows)
skip_cols = {'period', 'month', 'kernel', 'kernel_desc', 'n_regions'}
num_cols  = [c for c in _tmp.columns if c not in skip_cols]
for (period, kname), g in _tmp.groupby(['period', 'kernel']):
    row = dict(period=period, month='MEAN_Apr-Sep', kernel=kname,
               kernel_desc=g['kernel_desc'].iloc[0])
    for c in num_cols:
        row[c] = float(np.nanmean(g[c].values))
    row['n_regions'] = int(np.nanmean(g['n_regions'].values))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUM, index=False)
print(f"Summary        → {OUT_SUM}", flush=True)

# ---- Cross-kernel spread for each (period, month) ----
print("\n" + "=" * 70, flush=True)
print("DUAL-KERNEL COMPARISON  (Arctic-mean SIAF → global annual forcing)",
      flush=True)
print("=" * 70, flush=True)

pivot_rows = []
for (period, month), g in summary.groupby(['period', 'month']):
    row = dict(period=period, month=month)
    for kname in KERNEL_CONFIGS:
        sub = g[g['kernel'] == kname]
        if sub.empty: continue
        row[f'{kname}_SIAF_arctic']       = sub['SIAF_arctic'].values[0]
        row[f'{kname}_SIAF_global_annual']= sub['SIAF_global_annual'].values[0]
        row[f'{kname}_SIAF_ext_global']   = sub['SIAF_global_annual_ext'].values[0]
        row[f'{kname}_SIAF_drk_global']   = sub['SIAF_global_annual_drk'].values[0]

    # Kernel spread (absolute difference between the two)
    knames = list(KERNEL_CONFIGS.keys())
    if len(knames) == 2:
        k1, k2 = knames
        for col in ['SIAF_arctic', 'SIAF_global_annual']:
            v1 = row.get(f'{k1}_{col}', np.nan)
            v2 = row.get(f'{k2}_{col}', np.nan)
            row[f'spread_{col}'] = abs(v1 - v2) if (np.isfinite(v1) and np.isfinite(v2)) else np.nan

    pivot_rows.append(row)

pivot = pd.DataFrame(pivot_rows)

with pd.option_context('display.float_format', '{:.5f}'.format,
                        'display.max_columns', 20,
                        'display.width', 160):
    print(pivot.to_string(index=False), flush=True)

print(f"\nSpread column = |CAM5 − Huang_ERA5|  (kernel atmospheric-state uncertainty)")

log_mem("DONE")

# ============================================================
# CLI: python script.py --download
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                        help='Download Huang ERA5 kernel and exit')
    args, _ = parser.parse_known_args()
    if args.download:
        ok = download_era5_kernel()
        sys.exit(0 if ok else 1)'''

#!/usr/bin/env python
"""
diagnose_covariance_duel.py
-------------------------------------------------------
Audits the covariance identity:

    mean(RS × IS) = mean(RS) × mean(IS) + Cov(RS, IS)

over the Arctic ocean (≥60°N) for each melt season month.

Strategy: work entirely on the CERES 1°×1° lat/lon grid.
  - RS  : T07-inverted from CERES EBAF (same as your main scripts)
  - IS  : −ΔAlbedo / ΔT, with ΔAlbedo computed by remapping SIAL-cached
          CLARA-A3 (Apr–Aug) / APP-x (Sep) means to the CERES grid via
          nearest-neighbour, matching streaming_alb_mean() logic exactly.
  - mask: Arctic ocean pixels only (≥60°N, non-land), mirroring the
          fill_zero=True / fill_zero=False distinction from arctic_wmean().

Two statistics are computed at every valid pixel and then averaged:

  Method A — Pixel-wise   :  mean( RS_i × IS_i )   [includes Cov term]
  Method B — Bulk product :  mean(RS) × mean(IS)    [kills covariance]
  Covariance term         :  A − B

Outputs:
  siaf_covariance_duel.csv   — one row per (period, month)
"""

import numpy as np
import pandas as pd
import xarray as xr
import os, glob, gc
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

# ============================================================
# CONFIG  — adjust paths to match your environment
# ============================================================
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_CSV     = ('/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
               'siaf_covariance_duel.csv')

EARLY          = list(range(1982, 1992))
LATE_CMIP6     = list(range(2005, 2015))
LATE_OBS       = list(range(2015, 2025))
CERES_YRS_CMIP6 = list(range(2000, 2015))
CERES_YRS_OBS   = list(range(2000, 2025))

# ΔT values — reuse whatever your main scripts computed, or set directly
DELTA_T_CMIP6 = 0.6400   # replace with your computed value
DELTA_T_OBS   = 0.8400   # replace with your computed value

MONTHS      = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

# ============================================================
# T07 INVERSION  (identical to your main scripts)
# ============================================================
def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd  = np.where(rsds  > 1., rsds,  np.nan)
        a   = np.clip(rsus / Sd, 0., 1.)
        p   = rsds / S;  q = rsut / S;  den = q + p - 2.*p*a
        R   = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan),
                      0., 0.99)
        A   = np.clip(1. - p*(1.-a*R)
                      / np.where((1.-R) > 0.01, 1.-R, np.nan), 0., 0.99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S;  dc = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan),
                      0., 0.99)
        Rl  = np.clip(R - Rc, 0., 0.99)
        RS  = (S * (1.-A) * (1.-Rl)**2
               / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS


# ============================================================
# NEAREST-NEIGHBOUR REMAP HELPER
# ============================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg.ravel().astype(np.float32))
    lon = np.deg2rad(lon_deg.ravel().astype(np.float32))
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)]).astype(np.float32)


def build_nn_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d):
    """For each target pixel, find the index of the nearest source pixel."""
    tree = cKDTree(latlon_to_xyz(src_lat2d, src_lon2d))
    _, idx = tree.query(latlon_to_xyz(tgt_lat2d, tgt_lon2d), workers=-1)
    del tree; gc.collect()
    return idx   # shape: (tgt_lat2d.size,)


# ============================================================
# ONE-TIME SETUP  — SIAL grid + APP-x → SIAL → CERES index chain
# ============================================================
print("Setting up SIAL grid and regrid indices ...", flush=True)

# --- SIAL grid ---
sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values

if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

# --- CERES grid ---
with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
    ceres_lat1d = ds['lat'].values
    ceres_lon1d = ds['lon'].values
ceres_lon2d, ceres_lat2d = np.meshgrid(ceres_lon1d, ceres_lat1d)
ceres_shape = ceres_lat2d.shape

# --- SIAL → CERES nearest-neighbour index ---
# For each CERES pixel, which SIAL pixel is nearest?
print("  Building SIAL → CERES index ...", flush=True)
idx_sial_to_ceres = build_nn_index(sial_lat2d, sial_lon2d,
                                    ceres_lat2d, ceres_lon2d)
# result: array of length ceres_shape[0]*ceres_shape[1]

# --- APP-x → SIAL index (for September) ---
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f"{DIR_APPX}/**/*.nc", recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon

    idx_appx_to_sial = build_nn_index(appx_lat2d, appx_lon2d,
                                       sial_lat2d, sial_lon2d)
    del appx_lat2d, appx_lon2d, appx_lat, appx_lon; gc.collect()
    print("  APP-x → SIAL index built.", flush=True)
except Exception as e:
    print(f"  WARNING: APP-x → SIAL failed ({e}). September will be skipped.",
          flush=True)

# --- Arctic ocean mask on CERES grid ---
# Use global_land_mask if available, otherwise ocean = everywhere
try:
    from global_land_mask import globe
    lon_pm = np.where(ceres_lon2d > 180, ceres_lon2d - 360, ceres_lon2d)
    ocean_ceres = globe.is_ocean(ceres_lat2d, lon_pm)
except ImportError:
    print("  WARNING: global-land-mask not installed; land not excluded.", flush=True)
    ocean_ceres = np.ones(ceres_shape, dtype=bool)

arctic_ocean_ceres = (ceres_lat2d >= 60.0) & ocean_ceres
cos_w = np.cos(np.deg2rad(ceres_lat2d)).astype(np.float32)
print(f"  Arctic ocean CERES pixels: {arctic_ocean_ceres.sum():,}", flush=True)


# ============================================================
# ALBEDO MEAN ON SIAL (then remap to CERES)
# ============================================================
def albedo_mean_on_ceres(cache_dir, mname, years):
    """
    Streaming mean of albedo over `years` on the SIAL grid,
    then NN-remapped to the CERES 1° grid.

    Apr–Aug : CLARA-A3 cache files
    Sep     : APP-x cache files (remapped SIAL-shape first)

    Returns 2-D float32 array on the CERES grid, or None.
    """
    use_appx = (mname == 'September')
    alb_sum = None
    alb_cnt = None

    for y in years:
        prefix = 'appx' if use_appx else 'clara'
        path   = f"{cache_dir}/{prefix}_alb_{mname}_{y}.npy"
        if not os.path.exists(path):
            continue

        raw = np.load(path).astype(np.float32)

        # APP-x → SIAL remap if needed
        if use_appx:
            if idx_appx_to_sial is None:
                continue
            a_sial = (raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
                      if raw.shape != sial_shape else raw)
        else:
            a_sial = raw

        fin = np.isfinite(a_sial)
        a_sial[~fin] = 0.0

        if alb_sum is None:
            alb_sum = a_sial.copy()
            alb_cnt = fin.astype(np.int16)
        else:
            alb_sum += a_sial
            alb_cnt += fin.astype(np.int16)
        del raw, a_sial, fin

    if alb_sum is None:
        return None

    valid = alb_cnt > 0
    alb_sum[valid]  /= alb_cnt[valid].astype(np.float32)
    alb_sum[~valid]  = np.nan
    del alb_cnt

    # Remap SIAL → CERES via the pre-built NN index
    alb_ceres = alb_sum.ravel()[idx_sial_to_ceres].reshape(ceres_shape)
    del alb_sum; gc.collect()
    return alb_ceres


# ============================================================
# CERES RS MEAN (T07) ON CERES GRID
# ============================================================
def load_rs_ceres(years, mn):
    """Monthly-mean RS from CERES, T07-inverted, on the native CERES grid."""
    keys  = ['solar','toa_all','sfc_d_all','sfc_u_all',
             'toa_clr','sfc_d_clr','sfc_u_clr']
    sums  = {k: None for k in keys}
    count = 0

    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idxs  = [i for i, t in enumerate(times)
                 if t.year in years and t.month == mn]
        if not idxs:
            return None

        for i in idxs:
            sl   = ds.isel(time=i).load()
            arrs = [sl['solar_mon'].values.astype(np.float32),
                    sl['toa_sw_all_mon'].values.astype(np.float32),
                    sl['sfc_sw_down_all_mon'].values.astype(np.float32),
                    sl['sfc_sw_up_all_mon'].values.astype(np.float32),
                    sl['toa_sw_clr_c_mon'].values.astype(np.float32),
                    sl['sfc_sw_down_clr_c_mon'].values.astype(np.float32),
                    sl['sfc_sw_up_clr_c_mon'].values.astype(np.float32)]
            for k, a in zip(keys, arrs):
                sums[k] = a if count == 0 else sums[k] + a
            count += 1
            del sl

    if count == 0:
        return None
    for k in sums:
        sums[k] /= np.float32(count)

    RS = t07_invert(sums['solar'],    sums['toa_all'],   sums['sfc_d_all'],
                    sums['sfc_u_all'], sums['toa_clr'],   sums['sfc_d_clr'],
                    sums['sfc_u_clr']).astype(np.float32)
    del sums; gc.collect()
    return RS


# ============================================================
# COVARIANCE AUDIT FOR ONE PERIOD × MONTH
# ============================================================
def wmean(arr, mask, weights):
    """Area-weighted mean over mask, NaN-safe."""
    v = arr[mask]
    w = weights[mask]
    ok = np.isfinite(v) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def audit_month(cache_dir, mname, early_years, late_years,
                delta_t, ceres_years):
    # --- albedo means ---
    alb_e = albedo_mean_on_ceres(cache_dir, mname, early_years)
    alb_l = albedo_mean_on_ceres(cache_dir, mname, late_years)
    if alb_e is None or alb_l is None:
        print(f"  {mname}: missing albedo — skip", flush=True)
        return None

    # IS = −ΔAlpha / ΔT   (positive = brighter → warmer feedback)
    dalpha = (alb_l - alb_e).astype(np.float32)
    IS = (-dalpha / delta_t).astype(np.float32)
    del alb_e, alb_l, dalpha

    # --- RS on CERES grid ---
    RS = load_rs_ceres(ceres_years, mn=MONTHS[
        [MONTH_NAMES[m] for m in MONTHS].index(mname)])
    if RS is None:
        print(f"  {mname}: missing CERES RS — skip", flush=True)
        return None

    # --- mask: Arctic ocean pixels with valid IS and RS ---
    valid = arctic_ocean_ceres & np.isfinite(IS) & np.isfinite(RS)
    if valid.sum() == 0:
        print(f"  {mname}: no valid pixels — skip", flush=True)
        return None

    w = cos_w   # area weights (full 2D array)

    # Weighted means over valid pixels
    IS_mean  = wmean(IS,       valid, w)
    RS_mean  = wmean(RS,       valid, w)
    prod     = (IS * RS).astype(np.float32)
    SIAF_A   = wmean(prod,     valid, w)   # Method A: mean(IS × RS)
    SIAF_B   = IS_mean * RS_mean            # Method B: mean(IS) × mean(RS)
    cov_term = SIAF_A - SIAF_B              # Cov(IS, RS)

    pct_bias = (cov_term / SIAF_B * 100.) if SIAF_B != 0 else np.nan

    del IS, RS, prod; gc.collect()

    return dict(
        month          = mname,
        n_valid_px     = int(valid.sum()),
        IS_mean        = IS_mean,
        RS_mean        = RS_mean,
        SIAF_pixel_wise= SIAF_A,    # mean(IS_i × RS_i) — includes covariance
        SIAF_bulk      = SIAF_B,    # mean(IS) × mean(RS) — no covariance
        Cov_IS_RS      = cov_term,  # additive covariance term
        Cov_pct_of_bulk= pct_bias,  # Cov / SIAF_bulk × 100 %
    )


# ============================================================
# RUN BOTH PERIODS
# ============================================================
PERIOD_CONFIGS = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, CERES_YRS_CMIP6),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   DELTA_T_OBS,   CERES_YRS_OBS),
]

all_rows = []
for (label, cache_dir, early_yrs, late_yrs, delta_t, ceres_yrs) in PERIOD_CONFIGS:
    print(f"\n{'='*60}", flush=True)
    print(f"Period: {label}  (ΔT = {delta_t:.4f} K)", flush=True)
    print(f"{'='*60}", flush=True)
    for mn in MONTHS:
        mname = MONTH_NAMES[mn]
        row   = audit_month(cache_dir, mname, early_yrs, late_yrs,
                            delta_t, ceres_yrs)
        if row is not None:
            row['period'] = label
            all_rows.append(row)

# ============================================================
# PRINT + SAVE
# ============================================================
df = pd.DataFrame(all_rows)

# Reorder columns for readability
col_order = ['period', 'month', 'n_valid_px',
             'IS_mean', 'RS_mean',
             'SIAF_pixel_wise', 'SIAF_bulk',
             'Cov_IS_RS', 'Cov_pct_of_bulk']
df = df[col_order]

print("\n" + "="*80, flush=True)
print("COVARIANCE AUDIT:  mean(IS × RS)  vs  mean(IS) × mean(RS)", flush=True)
print("="*80, flush=True)
with pd.option_context('display.float_format', '{:+.5f}'.format,
                        'display.max_columns', 20,
                        'display.width', 160):
    print(df.to_string(index=False), flush=True)

print("\n  Cov_IS_RS    = SIAF_pixel_wise − SIAF_bulk", flush=True)
print("  Cov_pct      = Cov_IS_RS / SIAF_bulk × 100 %", flush=True)
print("  If |Cov_pct| << 5 %, the bulk decomposition is fine.", flush=True)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}", flush=True)