"""
fig8_donohoe_recreation_pval.py
--------------------------------
Main SIAF figure and uncertainty computation using p-value pixel
classification (p < 0.05) instead of ΔSIC threshold.

IS formulation:
  IS_extent = -(α_late - α_early) / ΔT   on significant-trend pixels
  IS_dark   = -(α_late - α_early) / ΔT   on stable ice pixels
  Both use directly observed albedo change — no ΔSIC proxy needed.

Uncertainty: same framework as Donohoe et al. 2020 —
  RS: bootstrap 100 estimates from 9-yr CERES subsets
  IS: ΔT propagation (DELTA_T_STD / DELTA_T × IS)
  Combined in quadrature via scalar Welford over RS × IS boot combinations.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import glob, os

# =============================================================================
# CONFIG
# =============================================================================

output_dir     = '/glade/work/jrotondo/ncc_paper/plots/'
CERES_FILE     = ('/glade/work/skygale/_variables/albedo/'
                  'CERES_EBAF_Edition4.2.1_200003-202504.nc')
sens_cache_dir = '/glade/work/jrotondo/siaf_extended/sensitivity_cache/'
PLOT_DIR       = f'{output_dir}plots_pval/'
os.makedirs(PLOT_DIR, exist_ok=True)

PVAL           = 0.05
MONTH_NAMES    = {4:'April', 5:'May', 6:'June',
                  7:'July',  8:'August', 9:'September'}
MONTH_NUMS     = [4, 5, 6, 7, 8, 9]
MONTHS_TO_ANALYZE = {v: k for k, v in MONTH_NAMES.items()}

EARLY_YEARS    = list(range(1982, 1992))
LATE_YEARS     = list(range(2015, 2025))
DELTA_T        = 0.707
DELTA_T_STD    = 0.092
AREA_RATIO_60N = 0.0330

N_BOOT_RS      = 100
N_BOOT_IS      = 100
CERES_RESAMP   = 9
SEED           = 42

SIC_ICE_THRESHOLD = 0.15

# =============================================================================
# HELPERS
# =============================================================================

def make_circle_boundary(ax, n=100):
    theta = np.linspace(0, 2*np.pi, n)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

def smooth_field(data_2d, sigma=0.7):
    valid   = np.isfinite(data_2d)
    tmp     = np.where(valid, data_2d, 0.)
    w       = gaussian_filter(valid.astype(float), sigma=sigma)
    blurred = gaussian_filter(tmp, sigma=sigma)
    out     = np.where(w > 0, blurred / w, np.nan)
    return np.where(valid, out, np.nan)

def prep(data_2d, lon_1d, lat_1d, apply_smooth=True):
    d = data_2d.copy()
    if apply_smooth:
        d = smooth_field(d, sigma=0.7)
    data_cyc, lon_cyc = add_cyclic_point(d, coord=lon_1d)
    lon2d, lat2d      = np.meshgrid(lon_cyc, lat_1d)
    return data_cyc, lon2d, lat2d

def arctic_wmean(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid])
            if valid.any() else np.nan)

def arctic_wsum(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return np.nansum(arr2d[valid] * w[valid])

def arctic_total_weight(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return np.nansum(w[valid])

def nanmean2(a, b):
    out    = np.full_like(a, np.nan)
    both   = np.isfinite(a) & np.isfinite(b)
    a_only = np.isfinite(a) & ~np.isfinite(b)
    b_only = np.isfinite(b) & ~np.isfinite(a)
    out[both]   = (a[both] + b[both]) / 2.0
    out[a_only] = a[a_only]
    out[b_only] = b[b_only]
    return out

def amjjas_mean(maps_by_month):
    arrays = [maps_by_month[m] for m in MONTH_NUMS
              if m in maps_by_month and maps_by_month[m] is not None]
    return np.nanmean(arrays, axis=0) if arrays else None

# =============================================================================
# STEP 0: LOAD CACHED GRID METADATA
# =============================================================================

print("Loading cached grid metadata...")

sial_shape      = tuple(np.load(f"{sens_cache_dir}/sial_shape.npy"))
appx_shape      = tuple(np.load(f"{sens_cache_dir}/appx_shape.npy"))
arctic_60N_sial = np.load(f"{sens_cache_dir}/arctic_60N_sial.npy")
arctic_60N_appx = np.load(f"{sens_cache_dir}/arctic_60N_appx.npy")

climo_nan_clara = {MONTH_NAMES[m]: np.load(
    f"{sens_cache_dir}/climo_nan_clara_{MONTH_NAMES[m]}.npy")
    for m in MONTH_NUMS}
climo_nan_appx  = {MONTH_NAMES[m]: np.load(
    f"{sens_cache_dir}/climo_nan_appx_{MONTH_NAMES[m]}.npy")
    for m in MONTH_NUMS}

sic_early_sial = {}; sic_late_sial = {}
sic_early_appx = {}; sic_late_appx = {}
for m in MONTH_NUMS:
    mn = MONTH_NAMES[m]
    sic_early_sial[mn] = np.load(f"{sens_cache_dir}/sic_early_sial_{mn}.npy")
    sic_late_sial[mn]  = np.load(f"{sens_cache_dir}/sic_late_sial_{mn}.npy")
    p_ea = f"{sens_cache_dir}/sic_early_appx_{mn}.npy"
    p_la = f"{sens_cache_dir}/sic_late_appx_{mn}.npy"
    sic_early_appx[mn] = np.load(p_ea) if os.path.exists(p_ea) else None
    sic_late_appx[mn]  = np.load(p_la) if os.path.exists(p_la) else None

sic_slope_sial = {}; sic_pval_sial = {}
sic_slope_appx = {}; sic_pval_appx = {}
for m in MONTH_NUMS:
    mn = MONTH_NAMES[m]
    sic_slope_sial[mn] = np.load(
        f"{sens_cache_dir}/sic_trend_slope_sial_{mn}.npy")
    sic_pval_sial[mn]  = np.load(
        f"{sens_cache_dir}/sic_trend_pval_sial_{mn}.npy")
    p_sl = f"{sens_cache_dir}/sic_trend_slope_appx_{mn}.npy"
    p_pv = f"{sens_cache_dir}/sic_trend_pval_appx_{mn}.npy"
    sic_slope_appx[mn] = np.load(p_sl) if os.path.exists(p_sl) else None
    sic_pval_appx[mn]  = np.load(p_pv) if os.path.exists(p_pv) else None

clara_alb = {MONTH_NAMES[m]: {} for m in MONTH_NUMS}
appx_alb  = {MONTH_NAMES[m]: {} for m in MONTH_NUMS}
for m in MONTH_NUMS:
    mn = MONTH_NAMES[m]
    for year in EARLY_YEARS + LATE_YEARS:
        p_c = f"{sens_cache_dir}/clara_alb_{mn}_{year}.npy"
        p_a = f"{sens_cache_dir}/appx_alb_{mn}_{year}.npy"
        if os.path.exists(p_c): clara_alb[mn][year] = np.load(p_c)
        if os.path.exists(p_a): appx_alb[mn][year]  = np.load(p_a)

print("  All cached data loaded.")

# =============================================================================
# STEP 1: BUILD CERES GRID AND REGRIDDERS
# =============================================================================

print("\nBuilding CERES grid and regridders...")

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

sample_sial_f = sorted(glob.glob(
    "/glade/derecho/scratch/jrotondo/clara_a3_obs/*.nc"))[0]
with xr.open_dataset(sample_sial_f) as ds:
    sial_lat1d = ds['lat'].values; sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d_g, sial_lat2d_g = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d_g, sial_lon2d_g = sial_lat1d, sial_lon1d
sial_xyz = latlon_to_xyz(sial_lat2d_g.ravel(), sial_lon2d_g.ravel())

from netCDF4 import Dataset as NC4Dataset
sample_appx_f = sorted(glob.glob(
    "/glade/derecho/scratch/jrotondo/APP-x_observations/**/*.nc",
    recursive=True))[0]
with NC4Dataset(sample_appx_f, 'r') as ds:
    alat = np.ma.filled(
        np.ma.asarray(ds.variables['latitude'][:]), np.nan).astype(float)
    alon = np.ma.filled(
        np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
if alat.ndim == 1:
    appx_lon2d_g, appx_lat2d_g = np.meshgrid(alon, alat)
else:
    appx_lat2d_g, appx_lon2d_g = alat, alon
appx_xyz = latlon_to_xyz(appx_lat2d_g.ravel(), appx_lon2d_g.ravel())

ds_cg = xr.open_dataset(CERES_FILE, chunks={})
ceres_lat = ds_cg['lat'].values; ceres_lon = ds_cg['lon'].values
ds_cg.close()
ceres_lon2d, ceres_lat2d = np.meshgrid(ceres_lon, ceres_lat)
ceres_shape  = ceres_lat2d.shape
lat2d_g      = ceres_lat2d
lat          = ceres_lat
lon          = ceres_lon
ceres_xyz    = latlon_to_xyz(ceres_lat2d.ravel(), ceres_lon2d.ravel())
ceres_tree   = cKDTree(ceres_xyz)
arctic_ceres = (ceres_lat2d >= 60.0)

_, sial_to_ceres_idx = ceres_tree.query(sial_xyz, workers=-1)
_, appx_to_ceres_idx = ceres_tree.query(appx_xyz, workers=-1)

def regrid_to_ceres(arr, idx):
    flat  = arr.ravel()
    accum = np.zeros(np.prod(ceres_shape))
    count = np.zeros(np.prod(ceres_shape))
    valid = np.isfinite(flat)
    np.add.at(accum, idx[valid], flat[valid])
    np.add.at(count, idx[valid], 1)
    out = np.full(np.prod(ceres_shape), np.nan)
    m   = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(ceres_shape)

def regrid_sial_to_ceres(arr): return regrid_to_ceres(arr, sial_to_ceres_idx)
def regrid_appx_to_ceres(arr): return regrid_to_ceres(arr, appx_to_ceres_idx)

# =============================================================================
# STEP 2: BOOTSTRAP RS FROM CERES
# =============================================================================

print(f"\nBootstrapping RS ({N_BOOT_RS} estimates, {CERES_RESAMP}-yr subsets)...")

ds_ceres    = xr.open_dataset(CERES_FILE)
ds_ceres    = ds_ceres.sel(time=(ds_ceres.time.dt.year >= 2000) &
                                  (ds_ceres.time.dt.year <= 2024))
ceres_times = pd.DatetimeIndex(ds_ceres.time.values)
ceres_df    = pd.DataFrame({'year': ceres_times.year,
                             'month': ceres_times.month},
                           index=np.arange(len(ceres_times)))
ceres_years = np.arange(2000, 2024)
rng         = np.random.default_rng(SEED)

def invert_T07_2d(S, SW_up, SW_d_sfc, SW_u_sfc):
    S_s  = S.where(S > 1.)
    Sd_s = SW_d_sfc.where(SW_d_sfc > 1.)
    a    = (SW_u_sfc / Sd_s).clip(0., 1.)
    p    = SW_d_sfc / S_s
    q    = SW_up    / S_s
    den  = q + p - 2.*p*a
    R    = ((q - p*a) / den.where(np.abs(den) > 1e-6)).clip(0., 0.99)
    A    = (1. - p*(1.-a*R) /
            (1.-R).where((1.-R) > 0.01)).clip(0., 0.99)
    return R, A, a

RS_boot_maps   = {}
RS_central     = {}
RS_2sig        = {}
RS_arctic_2sig = {}

for month_num in MONTH_NUMS:
    boot_maps = []; boot_scalars = []
    for _ in range(N_BOOT_RS):
        yr_sel  = rng.choice(ceres_years, size=CERES_RESAMP, replace=True)
        idx_sel = ceres_df[(ceres_df['year'].isin(yr_sel)) &
                            (ceres_df['month'] == month_num)].index.tolist()
        if not idx_sel: continue
        sel    = ds_ceres.isel(time=idx_sel)
        S_b    = sel['solar_mon'].mean('time')
        SWup_b = sel['toa_sw_all_mon'].mean('time')
        SWds_b = sel['sfc_sw_down_all_mon'].mean('time')
        SWus_b = sel['sfc_sw_up_all_mon'].mean('time')
        SWup_c = sel['toa_sw_clr_c_mon'].mean('time')
        SWds_c = sel['sfc_sw_down_clr_c_mon'].mean('time')
        SWus_c = sel['sfc_sw_up_clr_c_mon'].mean('time')
        R_all, A_all, alpha = invert_T07_2d(S_b, SWup_b, SWds_b, SWus_b)
        R_clr, _, _         = invert_T07_2d(S_b, SWup_c, SWds_c, SWus_c)
        R_cloud = (R_all - R_clr).clip(0., 0.99)
        S_safe  = S_b.where(S_b > 1.)
        RS_b    = (S_safe * (1.-A_all) * (1.-R_cloud)**2
                   / (1.-alpha*R_cloud)**2).values
        boot_maps.append(RS_b)
        boot_scalars.append(arctic_wmean(
            np.where(lat2d_g >= 60., RS_b, np.nan), lat))

    RS_boot_maps[month_num]   = np.stack(boot_maps, axis=0)
    RS_central[month_num]     = np.nanmean(RS_boot_maps[month_num], axis=0)
    RS_2sig[month_num]        = 2. * np.nanstd(RS_boot_maps[month_num], axis=0)
    RS_arctic_2sig[month_num] = 2. * np.nanstd(boot_scalars)
    print(f"  Month {month_num}: RS Arctic mean = "
          f"{np.nanmean(boot_scalars):.2f} ± {RS_arctic_2sig[month_num]:.2f} "
          f"W m⁻² per unit α")

ds_ceres.close()

RS_amjjas      = amjjas_mean(RS_central)
RS_2sig_amjjas = amjjas_mean(RS_2sig)

# =============================================================================
# STEP 3: COMPUTE IS FROM CACHED ALBEDO MAPS + p-VALUE CLASSIFICATION
# =============================================================================

print(f"\nComputing IS via p-value classification (p < {PVAL})...")

IS_central     = {}
IS_2sig        = {}
IS_boot_maps   = {}

IS_ext_central = {}
IS_ext_boots   = {}

IS_drk_central = {}
IS_drk_boots   = {}

for month_num in MONTH_NUMS:
    mn = MONTH_NAMES[month_num]

    # ── Epoch-mean albedo maps ────────────────────────────────────────────────
    yrs_c = sorted(clara_alb[mn].keys())
    ae_c  = [clara_alb[mn][y] for y in yrs_c if y in EARLY_YEARS]
    al_c  = [clara_alb[mn][y] for y in yrs_c if y in LATE_YEARS]
    dalpha_clara = (np.nanmean(al_c, axis=0) - np.nanmean(ae_c, axis=0)
                    if (ae_c and al_c) else None)

    yrs_a = sorted(appx_alb[mn].keys())
    ae_a  = [appx_alb[mn][y] for y in yrs_a if y in EARLY_YEARS]
    al_a  = [appx_alb[mn][y] for y in yrs_a if y in LATE_YEARS]
    dalpha_appx = (np.nanmean(al_a, axis=0) - np.nanmean(ae_a, axis=0)
                   if (ae_a and al_a) else None)

    # ── CLARA-A3 pixel masks ──────────────────────────────────────────────────
    valid_c    = arctic_60N_sial & ~climo_nan_clara[mn]
    slope_c    = sic_slope_sial[mn]; pval_c = sic_pval_sial[mn]
    changing_c = (valid_c & np.isfinite(slope_c) & np.isfinite(pval_c)
                  & (slope_c < 0) & (pval_c < PVAL))
    sic_max_c  = np.fmax(sic_early_sial[mn], sic_late_sial[mn])
    stice_c    = valid_c & ~changing_c & (sic_max_c >= SIC_ICE_THRESHOLD)

    # Negate dalpha so IS is positive (albedo decrease = positive sensitivity)
    IS_ext_c = (np.where(changing_c, -dalpha_clara / DELTA_T, np.nan)
                if dalpha_clara is not None else np.full(sial_shape, np.nan))
    IS_drk_c = (np.where(stice_c,   -dalpha_clara / DELTA_T, np.nan)
                if dalpha_clara is not None else np.full(sial_shape, np.nan))

    # ── APP-x pixel masks ─────────────────────────────────────────────────────
    if (dalpha_appx is not None
            and sic_slope_appx[mn] is not None
            and sic_pval_appx[mn]  is not None
            and sic_early_appx[mn] is not None
            and sic_late_appx[mn]  is not None):
        valid_a    = arctic_60N_appx & ~climo_nan_appx[mn]
        slope_a    = sic_slope_appx[mn]; pval_a = sic_pval_appx[mn]
        changing_a = (valid_a & np.isfinite(slope_a) & np.isfinite(pval_a)
                      & (slope_a < 0) & (pval_a < PVAL))
        sic_max_a  = np.fmax(sic_early_appx[mn], sic_late_appx[mn])
        stice_a    = valid_a & ~changing_a & (sic_max_a >= SIC_ICE_THRESHOLD)
        # Negate dalpha so IS is positive
        IS_ext_a   = np.where(changing_a, -dalpha_appx / DELTA_T, np.nan)
        IS_drk_a   = np.where(stice_a,   -dalpha_appx / DELTA_T, np.nan)
    else:
        IS_ext_a = np.full(appx_shape, np.nan)
        IS_drk_a = np.full(appx_shape, np.nan)

    # ── Regrid and nanmean ────────────────────────────────────────────────────
    IS_ext_comb = nanmean2(regrid_sial_to_ceres(IS_ext_c),
                           regrid_appx_to_ceres(IS_ext_a))
    IS_drk_comb = nanmean2(regrid_sial_to_ceres(IS_drk_c),
                           regrid_appx_to_ceres(IS_drk_a))

    IS_tot_comb = np.where(
        np.isfinite(IS_ext_comb) & np.isfinite(IS_drk_comb),
        IS_ext_comb + IS_drk_comb,
        np.where(np.isfinite(IS_ext_comb), IS_ext_comb,
        np.where(np.isfinite(IS_drk_comb), IS_drk_comb, np.nan))
    )

    IS_ext_central[month_num] = IS_ext_comb
    IS_drk_central[month_num] = IS_drk_comb
    IS_central[month_num]     = IS_tot_comb

    IS_dT_std     = np.abs(IS_tot_comb) * (DELTA_T_STD / DELTA_T)
    IS_ext_dT_std = np.abs(IS_ext_comb) * (DELTA_T_STD / DELTA_T)
    IS_drk_dT_std = np.abs(IS_drk_comb) * (DELTA_T_STD / DELTA_T)

    IS_2sig[month_num]      = 2. * IS_dT_std
    IS_boot_maps[month_num] = np.stack(
        [IS_tot_comb + rng.normal(0., IS_dT_std)
         for _ in range(N_BOOT_IS)], axis=0)
    IS_ext_boots[month_num] = np.stack(
        [IS_ext_comb + rng.normal(0., IS_ext_dT_std)
         for _ in range(N_BOOT_IS)], axis=0)
    IS_drk_boots[month_num] = np.stack(
        [IS_drk_comb + rng.normal(0., IS_drk_dT_std)
         for _ in range(N_BOOT_IS)], axis=0)

    am_tot = arctic_wmean(np.where(lat2d_g >= 60., IS_tot_comb, np.nan), lat)
    am_ext = arctic_wmean(np.where(lat2d_g >= 60., IS_ext_comb, np.nan), lat)
    am_drk = arctic_wmean(np.where(lat2d_g >= 60., IS_drk_comb, np.nan), lat)
    print(f"  Month {month_num}: IS_total = {am_tot:+.5f}  "
          f"IS_ext = {am_ext:+.5f}  IS_drk = {am_drk:+.5f}  "
          f"check = {am_ext+am_drk:+.5f}  "
          f"n_ext={changing_c.sum():,}  n_drk={stice_c.sum():,}")

IS_amjjas      = amjjas_mean(IS_central)
IS_2sig_amjjas = amjjas_mean(IS_2sig)

# =============================================================================
# STEP 4: SIAF CONVOLUTION WITH SCALAR WELFORD UNCERTAINTY
# =============================================================================

print("\nConvolving RS × IS — scalar Welford uncertainty...")

SIAF_central         = {}
SIAF_2sig            = {}
SIAF_arctic_2sig     = {}

SIAF_ext_central     = {}
SIAF_ext_arctic_2sig = {}

SIAF_drk_central     = {}
SIAF_drk_arctic_2sig = {}

for month_num in MONTH_NUMS:
    RS_boots = RS_boot_maps[month_num]
    IS_tot_b = IS_boot_maps[month_num]
    IS_ext_b = IS_ext_boots[month_num]
    IS_drk_b = IS_drk_boots[month_num]

    IS_c = IS_central[month_num]
    RS_c = RS_central[month_num]

    SIAF_central[month_num]     = IS_c                     * RS_c
    SIAF_ext_central[month_num] = IS_ext_central[month_num] * RS_c
    SIAF_drk_central[month_num] = IS_drk_central[month_num] * RS_c

    n_pix  = 0
    M2_pix = np.zeros_like(IS_c, dtype=np.float64)
    mu_pix = np.zeros_like(IS_c, dtype=np.float64)

    n_tot = 0; mu_tot = 0.0; M2_tot = 0.0
    n_ext = 0; mu_ext = 0.0; M2_ext = 0.0
    n_drk = 0; mu_drk = 0.0; M2_drk = 0.0

    for i in range(N_BOOT_RS):
        for j in range(N_BOOT_IS):
            siaf_tot = RS_boots[i] * IS_tot_b[j]
            valid    = np.isfinite(siaf_tot)
            n_pix   += 1
            delta    = np.where(valid, siaf_tot - mu_pix, 0.)
            mu_pix   = np.where(valid, mu_pix + delta / n_pix, mu_pix)
            M2_pix   = np.where(valid, M2_pix + delta*(siaf_tot - mu_pix), M2_pix)
            sc = arctic_wmean(siaf_tot, lat)
            if np.isfinite(sc):
                n_tot += 1; d = sc - mu_tot
                mu_tot += d/n_tot; M2_tot += d*(sc-mu_tot)

            siaf_ext = RS_boots[i] * IS_ext_b[j]
            sc = arctic_wmean(siaf_ext, lat)
            if np.isfinite(sc):
                n_ext += 1; d = sc - mu_ext
                mu_ext += d/n_ext; M2_ext += d*(sc-mu_ext)

            siaf_drk = RS_boots[i] * IS_drk_b[j]
            sc = arctic_wmean(siaf_drk, lat)
            if np.isfinite(sc):
                n_drk += 1; d = sc - mu_drk
                mu_drk += d/n_drk; M2_drk += d*(sc-mu_drk)

    std_pix     = np.where(n_pix > 1,
                           np.sqrt(np.maximum(M2_pix, 0.) / (n_pix-1)), np.nan)
    siaf_dT_pix = np.abs(SIAF_central[month_num]) * (DELTA_T_STD / DELTA_T)
    SIAF_2sig[month_num] = 2. * np.sqrt(std_pix**2 + siaf_dT_pix**2)

    std_tot     = np.sqrt(M2_tot/(n_tot-1)) if n_tot > 1 else np.nan
    mean_sc_tot = arctic_wmean(SIAF_central[month_num], lat)
    dT_tot      = abs(mean_sc_tot) * (DELTA_T_STD / DELTA_T)
    SIAF_arctic_2sig[month_num] = 2. * np.sqrt(std_tot**2 + dT_tot**2)

    std_ext     = np.sqrt(M2_ext/(n_ext-1)) if n_ext > 1 else np.nan
    mean_sc_ext = arctic_wmean(SIAF_ext_central[month_num], lat)
    dT_ext      = abs(mean_sc_ext) * (DELTA_T_STD / DELTA_T)
    SIAF_ext_arctic_2sig[month_num] = 2. * np.sqrt(std_ext**2 + dT_ext**2)

    std_drk     = np.sqrt(M2_drk/(n_drk-1)) if n_drk > 1 else np.nan
    mean_sc_drk = arctic_wmean(SIAF_drk_central[month_num], lat)
    dT_drk      = abs(mean_sc_drk) * (DELTA_T_STD / DELTA_T)
    SIAF_drk_arctic_2sig[month_num] = 2. * np.sqrt(std_drk**2 + dT_drk**2)

    print(f"  Month {month_num}: SIAF_tot = {mean_sc_tot:+.3f} ± "
          f"{SIAF_arctic_2sig[month_num]:.3f}  "
          f"ext = {mean_sc_ext:+.3f} ± {SIAF_ext_arctic_2sig[month_num]:.3f}  "
          f"drk = {mean_sc_drk:+.3f} ± {SIAF_drk_arctic_2sig[month_num]:.3f}")

SIAF_amjjas      = amjjas_mean(SIAF_central)
SIAF_2sig_amjjas = amjjas_mean(SIAF_2sig)

# =============================================================================
# STEP 5: GLOBAL SIAF
# =============================================================================

global_siaf     = 0.0
global_siaf_ext = 0.0
global_siaf_drk = 0.0

for month_num in MONTH_NUMS:
    scale    = (1/12) * AREA_RATIO_60N
    siaf_m   = np.where(lat2d_g >= 60., SIAF_central[month_num],     np.nan)
    siaf_ext = np.where(lat2d_g >= 60., SIAF_ext_central[month_num], np.nan)
    siaf_drk = np.where(lat2d_g >= 60., SIAF_drk_central[month_num], np.nan)
    W = arctic_total_weight(siaf_m, lat)
    global_siaf     += arctic_wsum(siaf_m,   lat) / W * scale
    global_siaf_ext += arctic_wsum(siaf_ext, lat) / W * scale
    global_siaf_drk += arctic_wsum(siaf_drk, lat) / W * scale

global_siaf_2sig     = (np.mean([SIAF_arctic_2sig[m]     for m in MONTH_NUMS])
                        * (6/12) * AREA_RATIO_60N)
global_siaf_ext_2sig = (np.mean([SIAF_ext_arctic_2sig[m] for m in MONTH_NUMS])
                        * (6/12) * AREA_RATIO_60N)
global_siaf_drk_2sig = (np.mean([SIAF_drk_arctic_2sig[m] for m in MONTH_NUMS])
                        * (6/12) * AREA_RATIO_60N)

siaf_arctic_mean = global_siaf / ((6/12) * AREA_RATIO_60N)

print(f"\nGlobal SIAF total : {global_siaf:+.4f} ± {global_siaf_2sig:.4f} W m⁻² K⁻¹")
print(f"Global SIAF ext   : {global_siaf_ext:+.4f} ± {global_siaf_ext_2sig:.4f} W m⁻² K⁻¹")
print(f"Global SIAF dark  : {global_siaf_drk:+.4f} ± {global_siaf_drk_2sig:.4f} W m⁻² K⁻¹")
print(f"Dark fraction     : {global_siaf_drk/global_siaf*100:.1f}%")

# =============================================================================
# STEP 6: SAVE UNCERTAINTY CSV
# =============================================================================

print("\nSaving uncertainty CSV...")

rows = []
for month_num in MONTH_NUMS:
    mname = MONTH_NAMES[month_num]
    scale = (1/12) * AREA_RATIO_60N

    siaf_m   = np.where(lat2d_g >= 60., SIAF_central[month_num],     np.nan)
    siaf_ext = np.where(lat2d_g >= 60., SIAF_ext_central[month_num], np.nan)
    siaf_drk = np.where(lat2d_g >= 60., SIAF_drk_central[month_num], np.nan)
    W = arctic_total_weight(siaf_m, lat)

    rows.append({
        'month'               : mname,
        'RS_mean_Wm2_unit_alb': arctic_wmean(
            np.where(lat2d_g >= 60., RS_central[month_num], np.nan), lat),
        'RS_2sig_Wm2_unit_alb': RS_arctic_2sig[month_num],
        'RS_mean_Wm2_per001'  : arctic_wmean(
            np.where(lat2d_g >= 60., RS_central[month_num]/100., np.nan), lat),
        'RS_2sig_Wm2_per001'  : RS_arctic_2sig[month_num] / 100.,
        'SIAF_ext_global'     : arctic_wsum(siaf_ext, lat) / W * scale,
        'SIAF_ext_2sig_global': SIAF_ext_arctic_2sig[month_num] * scale,
        'SIAF_drk_global'     : arctic_wsum(siaf_drk, lat) / W * scale,
        'SIAF_drk_2sig_global': SIAF_drk_arctic_2sig[month_num] * scale,
        'SIAF_tot_global'     : arctic_wsum(siaf_m,   lat) / W * scale,
        'SIAF_tot_2sig_global': SIAF_arctic_2sig[month_num] * scale,
        'SIAF_arctic_mean'    : arctic_wmean(siaf_m, lat),
        'SIAF_arctic_2sig'    : SIAF_arctic_2sig[month_num],
    })

rows.append({
    'month'               : 'AMJJAS',
    'RS_mean_Wm2_unit_alb': arctic_wmean(
        np.where(lat2d_g >= 60., RS_amjjas, np.nan), lat),
    'RS_2sig_Wm2_unit_alb': np.mean([RS_arctic_2sig[m] for m in MONTH_NUMS]),
    'RS_mean_Wm2_per001'  : arctic_wmean(
        np.where(lat2d_g >= 60., RS_amjjas/100., np.nan), lat),
    'RS_2sig_Wm2_per001'  : np.mean([RS_arctic_2sig[m] for m in MONTH_NUMS]) / 100.,
    'SIAF_ext_global'     : global_siaf_ext,
    'SIAF_ext_2sig_global': global_siaf_ext_2sig,
    'SIAF_drk_global'     : global_siaf_drk,
    'SIAF_drk_2sig_global': global_siaf_drk_2sig,
    'SIAF_tot_global'     : global_siaf,
    'SIAF_tot_2sig_global': global_siaf_2sig,
    'SIAF_arctic_mean'    : siaf_arctic_mean,
    'SIAF_arctic_2sig'    : np.mean([SIAF_arctic_2sig[m] for m in MONTH_NUMS]),
})

df_unc   = pd.DataFrame(rows)
csv_path = os.path.join(PLOT_DIR, 'uncertainty_per_month_for_slides_pval.csv')
df_unc.to_csv(csv_path, index=False, float_format='%.5f')
print(f"Saved: {csv_path}")
print(df_unc[['month', 'SIAF_ext_global', 'SIAF_ext_2sig_global',
              'SIAF_drk_global', 'SIAF_drk_2sig_global',
              'SIAF_tot_global', 'SIAF_tot_2sig_global']].to_string(index=False))

# =============================================================================
# STEP 7: PLOT — Donohoe Fig. 8 style
# =============================================================================

print("\nPlotting Fig. 8 recreation (p-value classification)...")

RS_amjjas_plot      = RS_amjjas      / 100.
RS_2sig_amjjas_plot = RS_2sig_amjjas / 100.

proj     = ccrs.NorthPolarStereo(central_longitude=-35)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 14), facecolor='white')
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.18, wspace=0.08,
                        top=0.93, bottom=0.04,
                        left=0.02, right=0.98)

panels = [
    (RS_amjjas_plot,        'Radiative Sensitivity',
     'RdYlBu_r', 0,     2.5,  'W m⁻² per 0.01 albedo decrease', 'A', False),
    (RS_2sig_amjjas_plot,   'Uncertainty in Radiative Sensitivity',
     'RdYlBu_r', 0,     0.25, '2σ RS  (W m⁻² per 0.01)',         'B', False),
    (IS_amjjas,             'Observed Ice Sensitivity',
     'RdBu_r',  -0.35,  0.35, 'IS — albedo change per K',        'C', True),
    (IS_2sig_amjjas,        'Uncertainty in Ice Sensitivity',
     'RdYlBu_r', 0,     0.2,  '2σ IS  (albedo per K)',            'D', False),
    (SIAF_amjjas,           'Sea Ice Albedo Feedback',
     'RdBu_r',  -45,    45,   'SIAF  (W m⁻² K⁻¹)',               'E', True),
    (SIAF_2sig_amjjas,      'Uncertainty in SIAF',
     'RdYlBu_r', 0,     30,   '2σ SIAF  (W m⁻² K⁻¹)',            'F', False),
]

for idx, (data, title, cmap, vmin, vmax, units, plbl, do_smooth) in \
        enumerate(panels):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col], projection=proj)
    ax.set_facecolor('#c8dff0')
    make_circle_boundary(ax)
    ax.set_extent([-180, 180, 60, 90], crs=data_crs)

    plot_data = np.where(lat2d_g >= 60., data, np.nan)
    data_cyc, lon2d_c, lat2d_c = prep(plot_data, lon, lat,
                                       apply_smooth=do_smooth)
    plot_cyc = np.where(lat2d_c >= 60., data_cyc, np.nan)

    im = ax.pcolormesh(lon2d_c, lat2d_c, plot_cyc, transform=data_crs,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', zorder=2)
    ax.add_feature(cfeature.LAND,      facecolor='#dbd9d0', zorder=3)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#444444',
                   linewidth=0.5, zorder=4)
    gl = ax.gridlines(crs=data_crs, draw_labels=False,
                      linewidth=0.35, color='#888888',
                      alpha=0.5, linestyle='--', zorder=5)
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator([60, 70, 80])

    ax.text(0.03, 0.97, plbl, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, lw=0))
    ax.set_title(title, fontsize=9.5, fontweight='bold',
                 color='#111111', pad=4)

    am = arctic_wmean(np.where(lat2d_g >= 60., data, np.nan), lat)
    ax.text(0.5, 0.02, f'Arctic mean: {am:+.3f} {units}',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='bottom',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0))

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.03, fraction=0.046, shrink=0.88)
    cbar.ax.tick_params(labelsize=7, colors='#333333')
    cbar.set_label(units, fontsize=7, color='#333333')
    cbar.outline.set_edgecolor('#aaaaaa')
    cbar.outline.set_linewidth(0.5)

fpath = os.path.join(PLOT_DIR, f'fig8_pval{int(PVAL*100):02d}_AMJJAS.png')
fig.savefig(fpath, dpi=250, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print(f"\nSaved: {fpath}")

# =============================================================================
# STEP 8: SUMMARY
# =============================================================================

from datetime import datetime

summary_lines = [
    "=" * 70,
    f"Fig. 8 Recreation — p-value classification (p < {PVAL})",
    f"Run completed: {datetime.now().isoformat()}",
    f"CLARA-A3 + APP-x  |  AMJJAS  |  EARLY {EARLY_YEARS[0]}-{EARLY_YEARS[-1]}"
    f"  LATE {LATE_YEARS[0]}-{LATE_YEARS[-1]}",
    "=" * 70,
    "",
    f"{'Month':<12} {'RS_mean':>10} {'RS_2sig':>10} "
    f"{'IS_mean':>10} {'SIAF_mean':>12} "
    f"{'SIAF_2sig':>10} {'SIAF_ext_2sig':>14} {'SIAF_drk_2sig':>14}",
    "-" * 100,
]

arctic = lat2d_g >= 60.
for month_num in MONTH_NUMS:
    mname = MONTH_NAMES[month_num]
    summary_lines.append(
        f"{mname:<12} "
        f"{arctic_wmean(np.where(arctic, RS_central[month_num]/100., np.nan), lat):>10.3f} "
        f"{RS_arctic_2sig[month_num]/100.:>10.3f} "
        f"{arctic_wmean(np.where(arctic, IS_central[month_num], np.nan), lat):>10.5f} "
        f"{arctic_wmean(np.where(arctic, SIAF_central[month_num], np.nan), lat):>12.4f} "
        f"{SIAF_arctic_2sig[month_num]:>10.4f} "
        f"{SIAF_ext_arctic_2sig[month_num]:>14.4f} "
        f"{SIAF_drk_arctic_2sig[month_num]:>14.4f}"
    )

summary_lines += [
    "",
    "AMJJAS Global SIAF:",
    f"  Total : {global_siaf:+.4f} ± {global_siaf_2sig:.4f} W m⁻² K⁻¹",
    f"  Extent: {global_siaf_ext:+.4f} ± {global_siaf_ext_2sig:.4f} W m⁻² K⁻¹",
    f"  Dark  : {global_siaf_drk:+.4f} ± {global_siaf_drk_2sig:.4f} W m⁻² K⁻¹",
    f"  Dark% : {global_siaf_drk/global_siaf*100:.1f}%",
    f"  Donohoe 2020 reference: 0.16 ± 0.04 W m⁻² K⁻¹",
    f"\nPlot: {fpath}",
    f"CSV : {csv_path}",
    "=" * 70,
]

for line in summary_lines:
    print(line)

txt_path = os.path.join(PLOT_DIR, f'summary_pval{int(PVAL*100):02d}.txt')
with open(txt_path, 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print(f"Summary: {txt_path}")