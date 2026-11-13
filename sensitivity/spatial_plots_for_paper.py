# %%
# =============================================================================
# STANDALONE: 6-panel per-month polar maps (p-val fields)
# Loads all cached data and computes RS, IS, SIAF from scratch.
# =============================================================================

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
import glob, os

# =============================================================================
# CONFIG
# =============================================================================

output_dir     = '/glade/work/jrotondo/ncc_paper/plots/'
CERES_FILE     = ('/glade/work/skygale/_variables/albedo/'
                  'CERES_EBAF_Edition4.2.1_200003-202504.nc')
sens_cache_dir = '/glade/work/jrotondo/siaf_extended/sensitivity_cache/'
clara_dir      = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
appx_dir       = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
PLOT_DIR       = os.path.join(output_dir, 'plots_pval_6panel/')
os.makedirs(PLOT_DIR, exist_ok=True)

PVAL              = 0.05
MONTH_NAMES       = {4:'April', 5:'May', 6:'June',
                     7:'July',  8:'August', 9:'September'}
MONTH_NUMS        = [4, 5, 6, 7, 8, 9]
EARLY_YEARS       = list(range(1982, 1992))
LATE_YEARS        = list(range(2015, 2025))
DELTA_T           = 0.909
DELTA_T_STD       = 0.034
AREA_RATIO_60N    = 0.0330
SIC_ICE_THRESHOLD = 0.15
N_BOOT_RS         = 100
N_BOOT_IS         = 100
CERES_RESAMP      = 9
SEED              = 42

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

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

def regrid_to_ceres(arr, idx, shape):
    flat  = arr.ravel()
    accum = np.zeros(np.prod(shape))
    count = np.zeros(np.prod(shape))
    valid = np.isfinite(flat)
    np.add.at(accum, idx[valid], flat[valid])
    np.add.at(count, idx[valid], 1)
    out = np.full(np.prod(shape), np.nan)
    m   = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(shape)

# =============================================================================
# LOAD CACHED GRID METADATA
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
sic_slope_sial = {}; sic_pval_sial = {}
sic_slope_appx = {}; sic_pval_appx = {}

for m in MONTH_NUMS:
    mn = MONTH_NAMES[m]
    sic_early_sial[mn] = np.load(f"{sens_cache_dir}/sic_early_sial_{mn}.npy")
    sic_late_sial[mn]  = np.load(f"{sens_cache_dir}/sic_late_sial_{mn}.npy")
    sic_slope_sial[mn] = np.load(f"{sens_cache_dir}/sic_trend_slope_sial_{mn}.npy")
    sic_pval_sial[mn]  = np.load(f"{sens_cache_dir}/sic_trend_pval_sial_{mn}.npy")
    for tag, store in [('sic_early_appx',      sic_early_appx),
                       ('sic_late_appx',       sic_late_appx),
                       ('sic_trend_slope_appx', sic_slope_appx),
                       ('sic_trend_pval_appx',  sic_pval_appx)]:
        p = f"{sens_cache_dir}/{tag}_{mn}.npy"
        store[mn] = np.load(p) if os.path.exists(p) else None

clara_alb = {MONTH_NAMES[m]: {} for m in MONTH_NUMS}
appx_alb  = {MONTH_NAMES[m]: {} for m in MONTH_NUMS}
for m in MONTH_NUMS:
    mn = MONTH_NAMES[m]
    for year in EARLY_YEARS + LATE_YEARS:
        p_c = f"{sens_cache_dir}/clara_alb_{mn}_{year}.npy"
        p_a = f"{sens_cache_dir}/appx_alb_{mn}_{year}.npy"
        if os.path.exists(p_c): clara_alb[mn][year] = np.load(p_c)
        if os.path.exists(p_a): appx_alb[mn][year]  = np.load(p_a)

print("  Done.")

# =============================================================================
# BUILD CERES GRID AND REGRIDDERS
# =============================================================================

print("Building CERES grid and regridders...")

sample_sial_f = sorted(glob.glob(f"{clara_dir}/*.nc"))[0]
with xr.open_dataset(sample_sial_f) as ds:
    sial_lat1d = ds['lat'].values; sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d_g, sial_lat2d_g = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d_g, sial_lon2d_g = sial_lat1d, sial_lon1d
sial_xyz = latlon_to_xyz(sial_lat2d_g.ravel(), sial_lon2d_g.ravel())

sample_appx_f = sorted(glob.glob(f"{appx_dir}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx_f, 'r') as ds:
    alat = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    alon = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
if alat.ndim == 1:
    appx_lon2d_g, appx_lat2d_g = np.meshgrid(alon, alat)
else:
    appx_lat2d_g, appx_lon2d_g = alat, alon
appx_xyz = latlon_to_xyz(appx_lat2d_g.ravel(), appx_lon2d_g.ravel())

with xr.open_dataset(CERES_FILE, chunks={}) as ds_cg:
    lat = ds_cg['lat'].values; lon = ds_cg['lon'].values
ceres_lon2d, ceres_lat2d = np.meshgrid(lon, lat)
ceres_shape = ceres_lat2d.shape
lat2d_g     = ceres_lat2d
ceres_xyz   = latlon_to_xyz(ceres_lat2d.ravel(), ceres_lon2d.ravel())
ceres_tree  = cKDTree(ceres_xyz)

_, sial_to_ceres_idx = ceres_tree.query(sial_xyz, workers=-1)
_, appx_to_ceres_idx = ceres_tree.query(appx_xyz, workers=-1)

def regrid_sial_to_ceres(arr):
    return regrid_to_ceres(arr, sial_to_ceres_idx, ceres_shape)
def regrid_appx_to_ceres(arr):
    return regrid_to_ceres(arr, appx_to_ceres_idx, ceres_shape)

print("  Done.")

# =============================================================================
# BOOTSTRAP RS FROM CERES
# =============================================================================

print(f"Bootstrapping RS ({N_BOOT_RS} estimates)...")

ds_ceres    = xr.open_dataset(CERES_FILE)
ds_ceres    = ds_ceres.sel(time=(ds_ceres.time.dt.year >= 2000) &
                                  (ds_ceres.time.dt.year <= 2024))
ceres_times = pd.DatetimeIndex(ds_ceres.time.values)
ceres_df    = pd.DataFrame({'year':  ceres_times.year,
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
        R_clr, _,     _     = invert_T07_2d(S_b, SWup_c, SWds_c, SWus_c)
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
          f"{np.nanmean(boot_scalars):.2f} ± {RS_arctic_2sig[month_num]:.2f}")

ds_ceres.close()

# =============================================================================
# COMPUTE IS
# =============================================================================

print(f"Computing IS (p < {PVAL})...")

IS_central     = {}
IS_ext_central = {}
IS_drk_central = {}
IS_2sig        = {}
IS_boot_maps   = {}
IS_ext_boots   = {}
IS_drk_boots   = {}

for month_num in MONTH_NUMS:
    mn = MONTH_NAMES[month_num]

    ae_c = [clara_alb[mn][y] for y in EARLY_YEARS if y in clara_alb[mn]]
    al_c = [clara_alb[mn][y] for y in LATE_YEARS  if y in clara_alb[mn]]
    dalpha_clara = (np.nanmean(al_c, axis=0) - np.nanmean(ae_c, axis=0)
                    if (ae_c and al_c) else None)

    ae_a = [appx_alb[mn][y] for y in EARLY_YEARS if y in appx_alb[mn]]
    al_a = [appx_alb[mn][y] for y in LATE_YEARS  if y in appx_alb[mn]]
    dalpha_appx = (np.nanmean(al_a, axis=0) - np.nanmean(ae_a, axis=0)
                   if (ae_a and al_a) else None)

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

    IS_ext_comb = nanmean2(regrid_sial_to_ceres(IS_ext_c),
                           regrid_appx_to_ceres(IS_ext_a))
    IS_drk_comb = nanmean2(regrid_sial_to_ceres(IS_drk_c),
                           regrid_appx_to_ceres(IS_drk_a))
    IS_tot_comb = nanmean2(IS_ext_comb, IS_drk_comb)

    IS_ext_central[month_num] = IS_ext_comb
    IS_drk_central[month_num] = IS_drk_comb
    IS_central[month_num]     = IS_tot_comb

    IS_dT_std     = np.abs(IS_tot_comb) * (DELTA_T_STD / DELTA_T)
    IS_ext_dT_std = np.abs(IS_ext_comb) * (DELTA_T_STD / DELTA_T)
    IS_drk_dT_std = np.abs(IS_drk_comb) * (DELTA_T_STD / DELTA_T)
    IS_2sig[month_num]      = 2. * IS_dT_std
    IS_boot_maps[month_num] = np.stack(
        [IS_tot_comb + rng.normal(0., IS_dT_std) for _ in range(N_BOOT_IS)], axis=0)
    IS_ext_boots[month_num] = np.stack(
        [IS_ext_comb + rng.normal(0., IS_ext_dT_std) for _ in range(N_BOOT_IS)], axis=0)
    IS_drk_boots[month_num] = np.stack(
        [IS_drk_comb + rng.normal(0., IS_drk_dT_std) for _ in range(N_BOOT_IS)], axis=0)

    print(f"  Month {month_num}: n_ext={changing_c.sum():,}  n_drk={stice_c.sum():,}")

# =============================================================================
# CONVOLVE RS × IS → SIAF
# =============================================================================

print("Convolving RS × IS...")

SIAF_central         = {}
SIAF_ext_central     = {}
SIAF_drk_central     = {}
SIAF_arctic_2sig     = {}
SIAF_ext_arctic_2sig = {}
SIAF_drk_arctic_2sig = {}

for month_num in MONTH_NUMS:
    RS_boots = RS_boot_maps[month_num]
    IS_tot_b = IS_boot_maps[month_num]
    IS_ext_b = IS_ext_boots[month_num]
    IS_drk_b = IS_drk_boots[month_num]
    IS_c     = IS_central[month_num]
    RS_c     = RS_central[month_num]

    SIAF_central[month_num]     = IS_c                      * RS_c
    SIAF_ext_central[month_num] = IS_ext_central[month_num] * RS_c
    SIAF_drk_central[month_num] = IS_drk_central[month_num] * RS_c

    n_tot = 0; mu_tot = 0.0; M2_tot = 0.0
    n_ext = 0; mu_ext = 0.0; M2_ext = 0.0
    n_drk = 0; mu_drk = 0.0; M2_drk = 0.0

    for i in range(N_BOOT_RS):
        for j in range(N_BOOT_IS):
            for (boots_IS, n, mu, M2, tag) in [
                (IS_tot_b, n_tot, mu_tot, M2_tot, 'tot'),
                (IS_ext_b, n_ext, mu_ext, M2_ext, 'ext'),
                (IS_drk_b, n_drk, mu_drk, M2_drk, 'drk'),
            ]:
                sc = arctic_wmean(RS_boots[i] * boots_IS[j], lat)
                if np.isfinite(sc):
                    n += 1; d = sc - mu; mu += d/n; M2 += d*(sc-mu)
                if tag == 'tot':   n_tot, mu_tot, M2_tot = n, mu, M2
                elif tag == 'ext': n_ext, mu_ext, M2_ext = n, mu, M2
                else:              n_drk, mu_drk, M2_drk = n, mu, M2

    for (n, mu, M2, central_dict, store) in [
        (n_tot, mu_tot, M2_tot, SIAF_central,     SIAF_arctic_2sig),
        (n_ext, mu_ext, M2_ext, SIAF_ext_central, SIAF_ext_arctic_2sig),
        (n_drk, mu_drk, M2_drk, SIAF_drk_central, SIAF_drk_arctic_2sig),
    ]:
        std     = np.sqrt(M2/(n-1)) if n > 1 else np.nan
        mean_sc = arctic_wmean(
            np.where(lat2d_g >= 60., central_dict[month_num], np.nan), lat)
        dT      = abs(mean_sc) * (DELTA_T_STD / DELTA_T)
        store[month_num] = 2. * np.sqrt(std**2 + dT**2)

    print(f"  Month {month_num}: SIAF = "
          f"{arctic_wmean(np.where(lat2d_g>=60., SIAF_central[month_num], np.nan), lat):+.3f} "
          f"± {SIAF_arctic_2sig[month_num]:.3f}")

# =============================================================================
# PLOT
# =============================================================================

print("Plotting...")

PLOT_SPECS_MONTHLY = [
    dict(title='IS — extent pixels (sig. neg. SIC trend, p < 0.05)',
         src=IS_ext_central,
         cmap='RdYlBu_r', vmin=-0.10,   vmax=0.10, 
         units='albedo per K', smooth=True,
         fname='IS_extent_pval_6panel.png'),
    dict(title='IS — darkening pixels (stable ice)',
         src=IS_drk_central,
         cmap='RdBu_r',   vmin=-0.10,  vmax=0.10,   # diverging: can go either way
         units='albedo per K', smooth=True,
         fname='IS_dark_pval_6panel.png'),
    dict(title='IS — total  (extent + dark)',
         src=IS_central,
         cmap='RdBu_r',   vmin=-0.15,  vmax=0.15,   # diverging: dark can dominate locally
         units='albedo per K', smooth=True,
         fname='IS_total_pval_6panel.png'),
    dict(title='Radiative Sensitivity',
         src=RS_central,
         cmap='YlOrRd',   vmin=0.0,   vmax=300.0,   # always positive by construction
         units='W m⁻² per unit α', smooth=False,
         fname='RS_pval_6panel.png'),
    dict(title='SIAF  =  IS_total × RS',
         src=SIAF_central,
         cmap='RdBu_r',   vmin=-40,    vmax=40,      # diverging: follows IS_total
         units='W m⁻² K⁻¹', smooth=True,
         fname='SIAF_pval_6panel.png'),
    dict(title='SIAF — extent pixels',
         src=SIAF_ext_central,
         cmap='RdYlBu_r', vmin=-40,   vmax=40, 
         units='W m⁻² K⁻¹', smooth=True,
         fname='SIAF_extent_pval_6panel.png'),
    dict(title='SIAF — darkening pixels',
         src=SIAF_drk_central,
         cmap='RdBu_r',   vmin=-40,    vmax=40,      # diverging: darkening can go either way
         units='W m⁻² K⁻¹', smooth=True,
         fname='SIAF_dark_pval_6panel.png'),
]

proj_p   = ccrs.NorthPolarStereo(central_longitude=-35)
data_crs = ccrs.PlateCarree()

for spec in PLOT_SPECS_MONTHLY:
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 11),
        subplot_kw={'projection': proj_p},
        facecolor='white'
    )
    fig.patch.set_facecolor('white')
    fig.suptitle(spec['title'], fontsize=13, fontweight='bold', y=0.97)
    axes = axes.ravel()

    for i, month_num in enumerate(MONTH_NUMS):
        ax    = axes[i]
        mname = MONTH_NAMES[month_num]
        raw   = spec['src'].get(month_num)

        ax.set_facecolor('#cce5f0')
        make_circle_boundary(ax)
        ax.set_extent([-180, 180, 60, 90], crs=data_crs)

        if raw is None:
            ax.set_title(mname, color='black', fontsize=10)
            continue

        plot_raw  = np.where(lat2d_g >= 60., raw, np.nan)
        data_cyc, lon2d_c, lat2d_c = prep(plot_raw, lon, lat,
                                           apply_smooth=spec['smooth'])
        data_plot = np.where(lat2d_c >= 60., data_cyc, np.nan)

        im = ax.pcolormesh(lon2d_c, lat2d_c, data_plot,
                           transform=data_crs,
                           cmap=spec['cmap'],
                           vmin=spec['vmin'], vmax=spec['vmax'],
                           shading='auto', zorder=2)

        ax.add_feature(cfeature.LAND,      facecolor='#d0cfc4', zorder=3)
        ax.add_feature(cfeature.COASTLINE, edgecolor='#555555',
                       linewidth=0.5, zorder=4)

        gl = ax.gridlines(crs=data_crs, draw_labels=False,
                          linewidth=0.4, color='#888888',
                          alpha=0.6, linestyle='--', zorder=5)
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator([60, 70, 80])

        mean_val = arctic_wmean(plot_raw, lat)
        ax.set_title(f"{mname}\nmean = {mean_val:+.4f} {spec['units']}",
                     color='black', fontsize=9.5, pad=4)

        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                            pad=0.04, fraction=0.046, shrink=0.85)
        cbar.ax.tick_params(labelsize=7, colors='black')
        cbar.set_label(spec['units'], color='black', fontsize=8)
        cbar.outline.set_edgecolor('#aaaaaa')
        cbar.outline.set_linewidth(0.5)

    for j in range(len(MONTH_NUMS), 6):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(PLOT_DIR, spec['fname'])
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"  Saved: {fpath}")

print("Done.")