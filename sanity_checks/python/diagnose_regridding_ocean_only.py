#!/usr/bin/env python
"""
diagnose_regridding_ocean_only.py
---------------------------------
SIC-FREE variant of diagnose_regridding.py.

The original script split the Arctic into SIC-trend "extent-loss" (changing)
pixels and SIC-threshold "stable-ice-darkening" pixels.  That decomposition
is entirely removed here.

New rule:
    Ice Sensitivity  IS(x) = -dα(x) / ΔT   for every pixel where
        (a) the pixel is in the ≥60°N Arctic domain, AND
        (b) dα(x) is finite in the cached albedo maps.

The cached CLARA-A3 / APP-x albedo fields already have NaN over land and
over pixels with no valid ocean/ice observation.  Treating those NaNs as
"contributes zero" in the area-weighted ≥60°N mean (fill_zero=True) is
exactly the behaviour the user requested:

    "consider anywhere the albedo changes over the ocean domain 60 N and
     above and then set over land to be 0 when calculating"

Configurations preserved from the original script so this run can be
compared head-to-head with the SIC-based results:
    C0 : IS × RS on native SIAL grid
    C1 : IS → 1°,   RS → 1°,   multiply
    C3 : IS → 2°,   RS → 2°,   multiply
    C4 : IS → 5°,   RS → 5°,   multiply
    C5 : IS × RS on native SIAL, then regrid product → 1°
    C6 : IS → 1°,   RS = scalar ice-only mean, multiply

Only SIAF_tot / IS_mean / RS_mean are reported - there is no longer a
SIAF_ext / SIAF_drk split because the SIC-based split is gone.

Outputs (new file names so results don't collide with the SIC version):
    siaf_sensitivity_results_ocean_only.csv
    siaf_sensitivity_summary_ocean_only.csv
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

# ============================================================
# MEMORY AUDITOR
# ============================================================
def log_mem(tag=""):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem_kb / (1024.0 * 1024.0)
    print(f"[MEM] {tag:<45} | Peak RAM: {mem_gb:.3f} GB", flush=True)
    sys.stdout.flush()

log_mem("STARTING SCRIPT (ocean-only, SIC-free)")

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
OUT_CSV  = os.path.join(OUT_DIR, 'siaf_sensitivity_results_ocean_only.csv')
OUT_SUM  = os.path.join(OUT_DIR, 'siaf_sensitivity_summary_ocean_only.csv')
os.makedirs(OUT_DIR, exist_ok=True)

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

TREND_START_CMIP6 = 1982;  TREND_END_CMIP6 = 2014
TREND_START_OBS   = 1982;  TREND_END_OBS   = 2024

CERES_YRS_CMIP6 = list(range(2000, 2015))
CERES_YRS_OBS   = list(range(2000, 2025))

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

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

    fill_zero=True  (SIAF / IS-with-land-zero): NaN → 0, all ≥60°N in
                    denominator.  Pairs with AREA_RATIO_60N = 0.065.
                    This is the SIC-free default used for the "land = 0"
                    rule the user asked for.
    fill_zero=False (RS ice-pixel-only mean): NaN excluded from both
                    numerator and denominator.
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

# SIC-independent ocean mask (written by build_albedo_cache_ocean_only.py).
# If the new cache builder hasn't been run yet, fall back to arctic_sial
# alone and print a loud warning so the user knows the run is NOT truly
# SIC-independent.
_ocean_sial_path = f"{CACHE_CMIP6}/ocean_mask_sial.npy"
if os.path.exists(_ocean_sial_path):
    ocean_sial = np.load(_ocean_sial_path).astype(bool)
    try:
        with open(f"{CACHE_CMIP6}/ocean_mask_source.txt") as fh:
            _ocean_src = fh.read().strip()
    except Exception:
        _ocean_src = '(source file missing)'
    print(f"  Loaded SIC-independent ocean mask "
          f"({int(ocean_sial.sum()):,} ocean pixels; source={_ocean_src})",
          flush=True)
    # Intersect ≥60°N with ocean to get the final SIC-independent domain.
    arctic_sial = arctic_sial & ocean_sial
else:
    print("  WARNING: ocean_mask_sial.npy NOT FOUND - run "
          "build_albedo_cache_ocean_only.py first for a truly SIC-free "
          "domain.  Falling back to arctic_60N_sial alone.", flush=True)

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

# APP-x → SIAL nearest-neighbour mapping (unchanged, needed for September)
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
# STEP 2: CERES RS LOADER  (unchanged from original)
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

    for gname, (glat, glon) in GRIDS.items():
        log_mem(f"  -> Regrid CERES → {gname}")
        glon2d, glat2d = np.meshgrid(glon, glat)
        idx, shp = build_scatter_index(clat2d, clon2d, glat2d, glon2d)
        rs_month[gname] = scatter_avg(RS.ravel(), idx, shp).astype(np.float32)
        del idx, glon2d, glat2d

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
# STEP 3: ΔT (unchanged)
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
# STEP 4: ALBEDO LOADER (unchanged from original)
# ============================================================
def streaming_alb_mean(cache_dir, mname, years):
    """Per-pixel streaming mean of surface albedo over `years`."""
    alb_sum = None
    alb_cnt = None
    use_appx = (mname == 'September')

    for y in years:
        if use_appx:
            if idx_appx_to_sial is None:
                continue
            appx_path = f"{cache_dir}/appx_alb_{mname}_{y}.npy"
            if not os.path.exists(appx_path):
                continue
            appx_raw = np.load(appx_path).astype(np.float32)
            a = appx_raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
            del appx_raw
        else:
            clara_path = f"{cache_dir}/clara_alb_{mname}_{y}.npy"
            if not os.path.exists(clara_path):
                continue
            a = np.load(clara_path).astype(np.float32)

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

    valid = alb_cnt > 0
    alb_sum[valid] /= alb_cnt[valid].astype(np.float32)
    alb_sum[~valid] = np.nan
    del alb_cnt
    return alb_sum

# ============================================================
# STEP 5: COMPUTE ALL CONFIGS FOR ONE MONTH × PERIOD  (SIC-FREE)
# ============================================================
def run_configs(cache_dir, early_years, late_years, delta_t, rs_maps_dict, mn):
    """
    SIC-free version.

    IS_nat(x) = -dα(x) / ΔT  wherever dα is finite AND the pixel is in
    the arctic_60N_sial ≥60°N domain.  No SIC trend test, no SIC
    threshold, no extent/darkening split.

    The cached albedo maps are already NaN over land and over pixels
    with no valid ocean/ice observation, so "finite dα" selects the
    ocean-domain-with-data pixels as the user requested.

    For every downstream SIAF area-weighted mean we use fill_zero=True,
    which treats land / data-gap pixels as zero contribution while still
    dividing by the full ≥60°N area.  That implements "set over land to
    be 0 when calculating".
    """
    mname = MONTH_NAMES[mn]
    log_mem(f"run_configs {mname} (ocean-only)")

    mean_e = streaming_alb_mean(cache_dir, mname, early_years)
    mean_l = streaming_alb_mean(cache_dir, mname, late_years)
    if mean_e is None or mean_l is None:
        print(f"  {mname}: missing albedo — skip", flush=True)
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dalpha = (mean_l - mean_e).astype(np.float32)
    del mean_e, mean_l

    # ------------------------------------------------------------------
    # SIC-FREE IS MASK:
    #   Any Arctic (≥60°N) pixel with finite dα → counts.
    #   Everything else (land, no-data, <60°N) → NaN  →  zero in mean.
    # ------------------------------------------------------------------
    ocean_changing = arctic_sial & np.isfinite(dalpha)
    IS_tot_nat     = np.where(ocean_changing, -dalpha / delta_t, np.nan
                              ).astype(np.float32)

    n_total  = int(arctic_sial.sum())
    n_active = int(ocean_changing.sum())
    frac = 100.0 * n_active / max(n_total, 1)
    print(f"    ocean-changing pixels: {n_active:,} / {n_total:,} "
          f"Arctic ({frac:.1f}% of ≥60°N domain)", flush=True)

    del dalpha, ocean_changing
    gc.collect()
    log_mem(f"  IS computed")

    # Zero-fill IS within the Arctic domain BEFORE regridding to coarser grids
    # so that land/no-data pixels contribute 0 to the scatter average.
    IS_tot_for_rg = np.where(arctic_sial,
                             np.where(np.isfinite(IS_tot_nat), IS_tot_nat, 0.),
                             np.nan).astype(np.float32)

    rs  = rs_maps_dict[mn]
    results = {}

    # ------------------------------------------------------------------
    # C0: IS × RS on native SIAL grid
    # SIAF mean: fill_zero=True (land = 0)
    # RS mean:   ice-pixel-only (fill_zero=False, diagnostic only)
    # ------------------------------------------------------------------
    log_mem("  -> C0")
    if rs is not None and 'sial_nn' in rs:
        RS_sial = rs['sial_nn']

        SIAF_tot_nat = (IS_tot_nat * RS_sial).astype(np.float32)

        lat2d_s = (np.tile(sial_lat1d[:, None], (1, sial_lon2d.shape[1]))
                   if sial_lat1d.ndim == 1 else sial_lat2d)
        w_s     = np.cos(np.deg2rad(lat2d_s))
        domain_60n = lat2d_s >= 60.
        active     = np.isfinite(IS_tot_nat) & domain_60n

        area_60n_sum = np.sum(w_s[domain_60n])
        area_act_sum = np.sum(w_s[active])

        # Land-as-zero mean over all ≥60°N (the user's requested rule)
        arr_z = np.where(np.isnan(SIAF_tot_nat), 0., SIAF_tot_nat)
        mean_60n = float(np.sum(arr_z[domain_60n] * w_s[domain_60n])
                         / area_60n_sum)
        global_forcing_60n = mean_60n * 0.065

        # Diagnostic: active-pixel-only mean (for comparison)
        if area_act_sum > 0:
            mean_act = float(np.sum(SIAF_tot_nat[active] * w_s[active])
                             / area_act_sum)
            act_ratio = (area_act_sum / area_60n_sum) * 0.065
            global_forcing_act = mean_act * act_ratio
        else:
            mean_act = np.nan; act_ratio = np.nan; global_forcing_act = np.nan

        print(f"\n      [MATH CHECK - {mname} C0, SIC-FREE]")
        print(f"      All-60N (land=0) Mean : {mean_60n:.4f} W/m2 "
              f"* 0.06500 Area = {global_forcing_60n:.5f} W/m2 global forcing")
        print(f"      Active-only Mean      : {mean_act:.4f} W/m2 "
              f"* {act_ratio:.5f} Area = {global_forcing_act:.5f} W/m2 global forcing")

        def _sial_mean(arr, fill_zero):
            if fill_zero:
                a = np.where(np.isnan(arr), 0., arr)
                return float(np.sum(a[domain_60n] * w_s[domain_60n])
                             / area_60n_sum)
            else:
                valid = domain_60n & np.isfinite(arr)
                if not valid.any(): return np.nan
                return float(np.sum(arr[valid] * w_s[valid])
                             / np.sum(w_s[valid]))

        results['C0'] = dict(
            SIAF_tot=_sial_mean(SIAF_tot_nat, fill_zero=True),
            IS_mean =_sial_mean(IS_tot_nat,    fill_zero=False),
            RS_mean =_sial_mean(np.where(np.isfinite(IS_tot_nat), RS_sial, np.nan),
                                fill_zero=False),
            n_active_pixels=n_active,
            frac_active_pct=frac,
        )
        del SIAF_tot_nat
        gc.collect()
    else:
        results['C0'] = None

    # ------------------------------------------------------------------
    # C1, C3, C4: IS and RS both regridded, product there.
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
        IS_tot_g      = regrid_2d(IS_tot_for_rg, idx, shp)
        IS_tot_g_diag = regrid_2d(IS_tot_nat,    idx, shp)
        del idx, glon2d, glat2d

        SIAF_tot = (IS_tot_g * RS_g).astype(np.float32)

        results[cname] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot,     glat, fill_zero=True),
            IS_mean =arctic_wmean(IS_tot_g_diag, glat, fill_zero=False),
            RS_mean =arctic_wmean(
                np.where(np.isfinite(IS_tot_g_diag), RS_g, np.nan),
                glat, fill_zero=False),
            n_active_pixels=n_active,
            frac_active_pct=frac,
        )
        del IS_tot_g, IS_tot_g_diag, SIAF_tot
        gc.collect()

    # ------------------------------------------------------------------
    # C5: IS × RS on native SIAL, regrid product to 1°
    # ------------------------------------------------------------------
    log_mem("  -> C5")
    if rs is not None and 'sial_nn' in rs:
        RS_sial = rs['sial_nn']

        SIAF_tot_p = (IS_tot_nat * RS_sial).astype(np.float32)
        SIAF_tot_p_rg = np.where(arctic_sial,
                                  np.where(np.isfinite(SIAF_tot_p), SIAF_tot_p, 0.),
                                  np.nan).astype(np.float32)
        del SIAF_tot_p

        glat1, glon1 = GRIDS['1deg']
        glon1d2d, glat1d2d = np.meshgrid(glon1, glat1)
        idx1, shp1 = build_scatter_index(sial_lat2d, sial_lon2d, glat1d2d, glon1d2d)

        SIAF_tot_1d = regrid_2d(SIAF_tot_p_rg, idx1, shp1)
        IS_1d_diag  = regrid_2d(IS_tot_nat, idx1, shp1)
        RS_1d       = regrid_2d(np.where(np.isfinite(IS_tot_nat), RS_sial, np.nan),
                                idx1, shp1)
        del idx1, glon1d2d, glat1d2d, SIAF_tot_p_rg

        results['C5'] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot_1d, glat1, fill_zero=True),
            IS_mean =arctic_wmean(IS_1d_diag,  glat1, fill_zero=False),
            RS_mean =arctic_wmean(RS_1d,        glat1, fill_zero=False),
            n_active_pixels=n_active,
            frac_active_pct=frac,
        )
        del SIAF_tot_1d, IS_1d_diag, RS_1d
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
        IS_tot_1d_diag = regrid_2d(IS_tot_nat,    idx1, shp1)
        del idx1, glon1d2d, glat1d2d

        RS_scalar = arctic_wmean(
            np.where(np.isfinite(IS_tot_1d_diag), RS_1d_full, np.nan),
            glat1, fill_zero=False)

        SIAF_tot_s = (IS_tot_1d * RS_scalar).astype(np.float32)

        results['C6'] = dict(
            SIAF_tot=arctic_wmean(SIAF_tot_s,      glat1, fill_zero=True),
            IS_mean =arctic_wmean(IS_tot_1d_diag,  glat1, fill_zero=False),
            RS_mean =RS_scalar,
            n_active_pixels=n_active,
            frac_active_pct=frac,
        )
        del IS_tot_1d, IS_tot_1d_diag, SIAF_tot_s
    else:
        results['C6'] = None

    del IS_tot_nat, IS_tot_for_rg
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
log_mem("Starting summary")

df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nRaw results → {OUT_CSV}", flush=True)

summary_rows = []
for (period, cname), grp in df.groupby(['period', 'config']):
    summary_rows.append(dict(
        period=period, config=cname,
        SIAF_tot=grp['SIAF_tot'].mean(),
        IS_mean =grp['IS_mean'].mean(),
        RS_mean =grp['RS_mean'].mean(),
        mean_frac_active_pct=grp['frac_active_pct'].mean(),
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

log_mem("DONE")