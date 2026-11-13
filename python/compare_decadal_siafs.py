#!/usr/bin/env python
"""
diagnose_siaf_period_comparison.py
-----------------------------------
Compares SIAF between:
  (A) CMIP6:   early=1982-1991  late=2005-2014
  (B) Obs-era: early=1982-1991  late=2015-2024

ΔT: OLS trend restricted to each analysis period.
RS: Dynamically computed from CERES EBAF via T07 Inversion to 1°x1° common grid.
IS: Partitioned natively, then regridded to 1°x1° common grid before multiplication.

Memory: no KD-trees during loop, 1 month at a time.
Plots: Cartopy-powered publication-ready 1°x1° maps.
Gap-Filling: Falls back to APP-x (spatially regridded to SIAL) for missing CLARA data, 
             then uses Ordinary Kriging (via pixel indices) for remaining NaNs.
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc, warnings
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pykrige.ok import OrdinaryKriging
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG & COMMON GRID
# =============================================================================

DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_CSV     = '/glade/work/jrotondo/ncc_paper/plots/siaf_period_comparison.csv'
PLOT_DIR    = '/glade/work/jrotondo/ncc_paper/plots/spatial_maps/'
CERES_FILE  = '/glade/derecho/scratch/jrotondo/ceres_data_updated/CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc'

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))

TREND_START_CMIP6 = 1982;  TREND_END_CMIP6 = 2014
TREND_START_OBS   = 1982;  TREND_END_OBS   = 2024

CERES_YRS_CMIP6   = list(range(2000, 2015))
CERES_YRS_OBS     = list(range(2000, 2025))

PVAL              = 0.05
SIC_ICE_THRESHOLD = 0.80
AREA_RATIO_60N    = 0.0330

MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

# Common 1x1 Grid Definition
COMMON_LAT = np.arange(-90., 91.,  1.)
COMMON_LON = np.arange(  0., 360., 1.)
CM_LON2D, CM_LAT2D = np.meshgrid(COMMON_LON, COMMON_LAT)


# =============================================================================
# GEOMETRY & REGRIDDING HELPERS
# =============================================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def regrid_scatter_fn(arr, src_lat2d, src_lon2d, tgt_lat, tgt_lon):
    """Scatter-average: each src pixel votes into its nearest tgt cell."""
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    _, idx  = cKDTree(tgt_xyz).query(src_xyz, workers=-1)
    
    flat  = arr.ravel()
    accum = np.zeros(tgt_lat2d.size)
    count = np.zeros(tgt_lat2d.size)
    valid = np.isfinite(flat)
    
    np.add.at(accum, idx[valid], flat[valid])
    np.add.at(count, idx[valid], 1)
    
    out = np.full(tgt_lat2d.size, np.nan)
    m   = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(tgt_lat2d.shape)

def arctic_wmean_1d(arr2d, lat_1d):
    """Cosine-weighted Arctic mean (>=60N) for regular lat/lon grids."""
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    if not valid.any(): return np.nan
    return float(np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid]))

# =============================================================================
# STEP 1: ΔT
# =============================================================================

def trend_dT(years, vals, y0, y1):
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5:
        return np.nan
    res = stats.linregress(years[mask], vals[mask])
    return res.slope * (y1 - y0)

print("=" * 65)
print("Computing ΔT (OLS trend restricted to each analysis period)")
print(f"  CMIP6: {TREND_START_CMIP6}–{TREND_END_CMIP6}  span={TREND_END_CMIP6-TREND_START_CMIP6} yr")
print(f"  Obs  : {TREND_START_OBS}–{TREND_END_OBS}  span={TREND_END_OBS-TREND_START_OBS} yr")
print("=" * 65)

dT_cmip6_vals = []; dT_obs_vals = []

try:
    df = pd.read_csv('/glade/work/jrotondo/cowtan_way.txt', sep=r'\s+',
                     header=None, names=['year','anomaly','unc1','unc2','unc3'])
    y = df['year'].values.astype(float); v = df['anomaly'].values.astype(float)
    c = trend_dT(y, v, TREND_START_CMIP6, TREND_END_CMIP6)
    o = trend_dT(y, v, TREND_START_OBS,   TREND_END_OBS)
    dT_cmip6_vals.append(c); dT_obs_vals.append(o)
    print(f"  Cowtan&Way : dT_cmip6={c:.3f} K   dT_obs={o:.3f} K")
except Exception as e: print(f"  Cowtan&Way : FAILED ({e})")

try:
    df = pd.read_csv('/glade/work/jrotondo/gistemp_global.csv',
                     skiprows=1, na_values='***')
    df['Year'] = df['Year'].astype(int)
    df['J-D']  = pd.to_numeric(df['J-D'], errors='coerce')
    y = df['Year'].values.astype(float); v = df['J-D'].values.astype(float)
    c = trend_dT(y, v, TREND_START_CMIP6, TREND_END_CMIP6)
    o = trend_dT(y, v, TREND_START_OBS,   TREND_END_OBS)
    dT_cmip6_vals.append(c); dT_obs_vals.append(o)
    print(f"  GISTEMP    : dT_cmip6={c:.3f} K   dT_obs={o:.3f} K")
except Exception as e: print(f"  GISTEMP    : FAILED ({e})")

try:
    ds  = xr.open_dataset('/glade/work/jrotondo/ncep_air_sig995.nc')
    t   = ds['air']; w = np.cos(np.deg2rad(t.lat))
    ta  = t.weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean().load()
    y   = ta.time.dt.year.values.astype(float)
    v   = ta.values.squeeze().astype(float)
    c   = trend_dT(y, v, TREND_START_CMIP6, TREND_END_CMIP6)
    o   = trend_dT(y, v, TREND_START_OBS,   TREND_END_OBS)
    dT_cmip6_vals.append(c); dT_obs_vals.append(o)
    print(f"  NCEP       : dT_cmip6={c:.3f} K   dT_obs={o:.3f} K")
    ds.close()
except Exception as e: print(f"  NCEP       : FAILED ({e})")

try:
    files = sorted(glob.glob('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc'))
    ds  = xr.open_mfdataset(files, combine='by_coords')
    t   = ds['VAR_2T']; w = np.cos(np.deg2rad(t.latitude))
    ta  = t.weighted(w).mean(dim=['latitude','longitude']).resample(time='1YE').mean().load()
    y   = ta.time.dt.year.values.astype(float)
    v   = ta.values.squeeze().astype(float)
    c   = trend_dT(y, v, TREND_START_CMIP6, TREND_END_CMIP6)
    o   = trend_dT(y, v, TREND_START_OBS,   TREND_END_OBS)
    dT_cmip6_vals.append(c); dT_obs_vals.append(o)
    print(f"  ERA5       : dT_cmip6={c:.3f} K   dT_obs={o:.3f} K")
    ds.close()
except Exception as e: print(f"  ERA5       : FAILED ({e})")

DELTA_T_CMIP6     = float(np.mean(dT_cmip6_vals))
DELTA_T_OBS       = float(np.mean(dT_obs_vals))

print(f"\n  ΔT CMIP6: {DELTA_T_CMIP6:.4f} K")
print(f"  ΔT Obs  : {DELTA_T_OBS:.4f} K")


# =============================================================================
# STEP 2: SIAL GRID METADATA & APP-X REGRIDDING
# =============================================================================

print("\n" + "=" * 65)
print("Loading SIAL grid metadata and building APP-x mapping...")
print("=" * 65)
sial_shape   = tuple(np.load(f"{CACHE_CMIP6}/sial_shape.npy"))
arctic_sial  = np.load(f"{CACHE_CMIP6}/arctic_60N_sial.npy")

try:
    sample_sial = sorted(glob.glob(f"{DIR_SIAL}/*.nc"))[0]
    with xr.open_dataset(sample_sial) as ds:
        sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
    if sial_lat1d.ndim == 1:
        sial_lon, sial_lat = np.meshgrid(sial_lon1d, sial_lat1d)
    else:
        sial_lat, sial_lon = sial_lat1d, sial_lon1d
    sial_xyz = latlon_to_xyz(sial_lat.ravel(), sial_lon.ravel())

    dir_appx = '/glade/derecho/scratch/jrotondo/APP-x_observations'
    sample_appx = glob.glob(f"{dir_appx}/**/*.nc", recursive=True)[0]
    with Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:], np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon, appx_lat = np.meshgrid(appx_lon, appx_lat)
    appx_xyz = latlon_to_xyz(appx_lat.ravel(), appx_lon.ravel())

    print("  Computing KDTree for APP-x to SIAL regridding...")
    tree_appx = cKDTree(appx_xyz)
    _, idx_appx_to_sial = tree_appx.query(sial_xyz, workers=-1)
except Exception as e:
    print(f"  WARNING: Setup failed. ({e})")
    sial_lat, sial_lon, idx_appx_to_sial, sial_xyz = None, None, None, None


# =============================================================================
# STEP 3: DYNAMIC CERES RS TO COMMON 1x1 GRID
# =============================================================================

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt>1., rsdt, np.nan)
        Sd = np.where(rsds>1., rsds, np.nan)
        a  = np.clip(rsus/Sd, 0., 1.)
        p  = rsds/S; q = rsut/S; den = q+p-2.*p*a
        R  = np.clip((q-p*a)/np.where(np.abs(den)>1e-6,den,np.nan), 0., 0.99)
        A  = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>0.01,1.-R,np.nan), 0., 0.99)
        Sc = np.where(rsdscs>1.,rsdscs,np.nan)
        ac = np.clip(rsuscs/Sc,0.,1.)
        pc = rsdscs/S; qc = rsutcs/S; dc = qc+pc-2.*pc*ac
        Rc = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan), 0., 0.99)
        Rl = np.clip(R-Rc, 0., 0.99)
        RS = S*(1.-A)*(1.-Rl)**2 / np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan)
    return RS

def compute_obs_RS_1x1(years):
    print(f"  Processing CERES RS {min(years)}-{max(years)} to 1x1 Grid...")
    ds = xr.open_dataset(CERES_FILE)
    times = pd.DatetimeIndex(ds.time.values)
    
    clon2d, clat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    ceres_xyz = latlon_to_xyz(clat2d.ravel(), clon2d.ravel())
    
    cm_xyz = latlon_to_xyz(CM_LAT2D.ravel(), CM_LON2D.ravel())
    _, idx = cKDTree(cm_xyz).query(ceres_xyz, workers=-1)

    def _scatter(arr):
        flat = arr.ravel()
        accum = np.zeros(COMMON_LAT.size * COMMON_LON.size)
        count = np.zeros(COMMON_LAT.size * COMMON_LON.size)
        valid = np.isfinite(flat)
        np.add.at(accum, idx[valid], flat[valid])
        np.add.at(count, idx[valid], 1)
        out = np.full(COMMON_LAT.size * COMMON_LON.size, np.nan)
        m = count > 0; out[m] = accum[m]/count[m]
        return out.reshape(len(COMMON_LAT), len(COMMON_LON))

    rs_maps = {}
    for mn in MONTH_NUMS:
        mask = np.array([t.year in years and t.month == mn for t in times])
        if not mask.any():
            rs_maps[mn] = None
            continue
        sel = ds.isel(time=mask)
        
        RS_raw = t07_invert(
            sel['solar_mon'].mean('time').values,
            sel['toa_sw_all_mon'].mean('time').values,
            sel['sfc_sw_down_all_mon'].mean('time').values,
            sel['sfc_sw_up_all_mon'].mean('time').values,
            sel['toa_sw_clr_c_mon'].mean('time').values,
            sel['sfc_sw_down_clr_c_mon'].mean('time').values,
            sel['sfc_sw_up_clr_c_mon'].mean('time').values
        )
        
        RS_1x1 = _scatter(RS_raw)
        rs_maps[mn] = RS_1x1
        
        am = arctic_wmean_1d(RS_1x1, COMMON_LAT)
        print(f"    {MONTH_NAMES[mn]}: {am:.2f} W m⁻² per unit α")
        
    ds.close()
    return rs_maps

print("\n" + "=" * 65)
print("Extracting dynamic CERES RS kernels (1°x1° Maps via T07)")
print("=" * 65)
print("CMIP6 Period:")
RS_MAPS_CMIP6 = compute_obs_RS_1x1(CERES_YRS_CMIP6)
print("Obs-era Period:")
RS_MAPS_OBS = compute_obs_RS_1x1(CERES_YRS_OBS)

# =============================================================================
# ALBEDO FALLBACK & GAP-FILLING
# =============================================================================

def load_and_fill_albedo(cache_dir, mname, year):
    clara_path = f"{cache_dir}/clara_alb_{mname}_{year}.npy"
    if not os.path.exists(clara_path): return None
    merged = np.load(clara_path).astype(float)
    
    appx_path = f"{cache_dir}/appx_alb_{mname}_{year}.npy"
    if os.path.exists(appx_path) and idx_appx_to_sial is not None:
        appx_native = np.load(appx_path).astype(float)
        appx_regridded = appx_native.ravel()[idx_appx_to_sial].reshape(sial_shape)
        merged = np.where(np.isnan(merged), appx_regridded, merged)
        del appx_native, appx_regridded
    
    missing_mask = np.isnan(merged) & arctic_sial
    if missing_mask.sum() == 0: return merged
    
    valid_mask = ~np.isnan(merged) & arctic_sial
    grid_y, grid_x = np.indices(merged.shape)
    
    train_y = grid_y[valid_mask].astype(float)
    train_x = grid_x[valid_mask].astype(float)
    train_val = merged[valid_mask]
    
    if len(train_val) > 2500:
        idx = np.random.choice(len(train_val), 2500, replace=False)
        train_y = train_y[idx]
        train_x = train_x[idx]
        train_val = train_val[idx]
        
    target_y = grid_y[missing_mask].astype(float)
    target_x = grid_x[missing_mask].astype(float)
    
    ok = OrdinaryKriging(train_x, train_y, train_val, variogram_model='spherical', enable_statistics=False)
    kriged_vals, _ = ok.execute('points', target_x, target_y)
    merged[missing_mask] = kriged_vals
    
    return merged

# =============================================================================
# SPATIAL PLOTTING HELPER
# =============================================================================

def plot_spatial_components_1x1(IS_ext, IS_drk, IS_tot, RS, month_name, period_name, out_dir):
    SIAF_ext = IS_ext * RS
    SIAF_drk = IS_drk * RS
    SIAF_tot = IS_tot * RS

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    proj = ccrs.NorthPolarStereo()
    transform = ccrs.PlateCarree()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white', subplot_kw={'projection': proj})
    fig.suptitle(f"{period_name} 1°x1° Components - {month_name}", fontsize=18, fontweight='bold', y=0.96)

    is_lim = float(np.nanpercentile(np.abs(IS_tot), 98)) if np.isfinite(IS_tot).any() else 0.5
    siaf_lim = float(np.nanpercentile(np.abs(SIAF_tot), 98)) if np.isfinite(SIAF_tot).any() else 50.0
    
    is_lim = max(is_lim, 0.1)
    siaf_lim = max(siaf_lim, 10.0)

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('none')
    
    titles = [["IS Extent", "IS Darkening", "IS Total"], ["SIAF Extent", "SIAF Darkening", "SIAF Total"]]
    labels = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
    data = [[IS_ext, IS_drk, IS_tot], [SIAF_ext, SIAF_drk, SIAF_tot]]
    limits = [is_lim, siaf_lim]
    units = ["K$^{-1}$", "W m$^{-2}$ K$^{-1}$"]

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    for row in range(2):
        vmax = limits[row]
        im = None
        for col in range(3):
            ax = axes[row, col]
            ax.set_extent([-180, 180, 60, 90], crs=transform)
            ax.set_boundary(circle, transform=ax.transAxes)
            
            ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
            ax.add_feature(cfeature.LAND, facecolor='#d9d9d9', edgecolor='none', zorder=1)
            ax.coastlines(resolution='50m', linewidth=0.5, color='black', zorder=4)
            
            plot_d = np.where(CM_LAT2D >= 58., data[row][col], np.nan)
            im = ax.pcolormesh(CM_LON2D, CM_LAT2D, plot_d, transform=transform,
                               cmap=cmap, vmin=-vmax, vmax=vmax, shading='nearest', zorder=3)
            
            ax.set_title(titles[row][col], fontsize=14, pad=10)
            ax.text(0.02, 0.98, labels[row][col], transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left', zorder=10)

        cbar = fig.colorbar(im, ax=axes[row, :], location='right', shrink=0.85, pad=0.03, extend='both')
        cbar.set_label(units[row], fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spatial_{period_name}_{month_name}_1x1.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# STEP 4: MONTH-BY-MONTH COMPUTATION
# =============================================================================

def compute_month(cache_dir, early_years, late_years, delta_t, rs_dict, mn, period_name):
    mname = MONTH_NAMES[mn]
    print(f"  Processing {mname}...")
    
    ae = [alb for y in early_years if (alb := load_and_fill_albedo(cache_dir, mname, y)) is not None]
    al = [alb for y in late_years if (alb := load_and_fill_albedo(cache_dir, mname, y)) is not None]

    if not ae or not al: return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dalpha  = np.nanmean(al, axis=0) - np.nanmean(ae, axis=0)

    slope  = np.load(f"{cache_dir}/sic_trend_slope_sial_{mname}.npy")
    pval   = np.load(f"{cache_dir}/sic_trend_pval_sial_{mname}.npy")
    sic_e  = np.load(f"{cache_dir}/sic_early_sial_{mname}.npy")
    sic_l  = np.load(f"{cache_dir}/sic_late_sial_{mname}.npy")

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval) & (slope < 0) & (pval < PVAL))
    stable_ice = (valid & ~changing & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))
    
    n_ext_raw = int(changing.sum())
    n_drk_raw = int(stable_ice.sum())
    da_ext = float(np.nanmean(dalpha[changing])) if n_ext_raw > 0 else np.nan
    da_drk = float(np.nanmean(dalpha[stable_ice])) if n_drk_raw > 0 else np.nan

    del slope, pval, sic_e, sic_l, valid; gc.collect()

    # Compute IS natively
    IS_ext_nat = np.where(changing,    -dalpha / delta_t, np.nan)
    IS_drk_nat = np.where(stable_ice,  -dalpha / delta_t, np.nan)
    IS_tot_nat = np.where(
        np.isfinite(IS_ext_nat) | np.isfinite(IS_drk_nat),
        np.where(np.isfinite(IS_ext_nat), IS_ext_nat, 0.) + np.where(np.isfinite(IS_drk_nat), IS_drk_nat, 0.),
        np.nan)

    # ZERO-FILL native missing pixels so ext and drk masks match tot
    IS_ext_nat = np.where(np.isfinite(IS_tot_nat), np.where(np.isfinite(IS_ext_nat), IS_ext_nat, 0.), np.nan)
    IS_drk_nat = np.where(np.isfinite(IS_tot_nat), np.where(np.isfinite(IS_drk_nat), IS_drk_nat, 0.), np.nan)

    # Regrid cleanly to 1x1
    IS_tot_1x1 = regrid_scatter_fn(IS_tot_nat, sial_lat, sial_lon, COMMON_LAT, COMMON_LON)
    IS_ext_1x1 = regrid_scatter_fn(IS_ext_nat, sial_lat, sial_lon, COMMON_LAT, COMMON_LON)
    IS_drk_1x1 = regrid_scatter_fn(IS_drk_nat, sial_lat, sial_lon, COMMON_LAT, COMMON_LON)

    # Get 1x1 RS Map
    RS_1x1 = rs_dict[mn]

    # Calculate 1x1 SIAF 
    SIAF_tot_1x1 = np.where(np.isfinite(IS_tot_1x1) & np.isfinite(RS_1x1), IS_tot_1x1 * RS_1x1, np.nan)
    SIAF_ext_1x1 = np.where(np.isfinite(IS_ext_1x1) & np.isfinite(RS_1x1), IS_ext_1x1 * RS_1x1, np.nan)
    SIAF_drk_1x1 = np.where(np.isfinite(IS_drk_1x1) & np.isfinite(RS_1x1), IS_drk_1x1 * RS_1x1, np.nan)

    # ZERO-FILL 1x1 SIAF components so they strictly add up post-multiplication
    SIAF_ext_1x1 = np.where(np.isfinite(SIAF_tot_1x1), np.where(np.isfinite(SIAF_ext_1x1), SIAF_ext_1x1, 0.), np.nan)
    SIAF_drk_1x1 = np.where(np.isfinite(SIAF_tot_1x1), np.where(np.isfinite(SIAF_drk_1x1), SIAF_drk_1x1, 0.), np.nan)

    plot_spatial_components_1x1(IS_ext_1x1, IS_drk_1x1, IS_tot_1x1, RS_1x1, mname, period_name, PLOT_DIR)

    # Calculate Cosine-Weighted 1D Means
    return dict(
        delta_t    = delta_t,
        IS_ext     = arctic_wmean_1d(IS_ext_1x1, COMMON_LAT),
        IS_drk     = arctic_wmean_1d(IS_drk_1x1, COMMON_LAT),
        IS_tot     = arctic_wmean_1d(IS_tot_1x1, COMMON_LAT),
        SIAF_ext   = arctic_wmean_1d(SIAF_ext_1x1, COMMON_LAT),
        SIAF_drk   = arctic_wmean_1d(SIAF_drk_1x1, COMMON_LAT),
        SIAF_tot   = arctic_wmean_1d(SIAF_tot_1x1, COMMON_LAT),
        RS_mean    = arctic_wmean_1d(np.where(np.isfinite(IS_tot_1x1), RS_1x1, np.nan), COMMON_LAT),
        n_ext_px   = n_ext_raw,
        n_drk_px   = n_drk_raw,
        dalpha_ext = da_ext,
        dalpha_drk = da_drk,
    )

# =============================================================================
# STEP 5: RUN
# =============================================================================

res_cmip6 = {}; res_obs = {}

print(f"\n--- CMIP6 (late 2005-2014, ΔT={DELTA_T_CMIP6:.4f} K) ---")
for mn in MONTH_NUMS:
    res_cmip6[mn] = compute_month(
        CACHE_CMIP6, EARLY, LATE_CMIP6, DELTA_T_CMIP6, RS_MAPS_CMIP6, mn, "CMIP6")
    if res_cmip6[mn]:
        r = res_cmip6[mn]
        print(f"  -> {MONTH_NAMES[mn]:<10} SIAF_tot={r['SIAF_tot']:+.4f}  ext={r['SIAF_ext']:+.4f}  drk={r['SIAF_drk']:+.4f}")

print(f"\n--- Obs-era (late 2015-2024, ΔT={DELTA_T_OBS:.4f} K) ---")
for mn in MONTH_NUMS:
    res_obs[mn] = compute_month(
        CACHE_OBS, EARLY, LATE_OBS, DELTA_T_OBS, RS_MAPS_OBS, mn, "Obs-era")
    if res_obs[mn]:
        r = res_obs[mn]
        print(f"  -> {MONTH_NAMES[mn]:<10} SIAF_tot={r['SIAF_tot']:+.4f}  ext={r['SIAF_ext']:+.4f}  drk={r['SIAF_drk']:+.4f}")

# =============================================================================
# STEP 6: COMPARISON TABLE
# =============================================================================

valid = [mn for mn in MONTH_NUMS if res_cmip6.get(mn) and res_obs.get(mn)]
amj   = lambda d, k: np.mean([d[m][k] for m in valid])

print("\n" + "=" * 115)
print("COMPARISON  (obs-era minus CMIP6)  — CLARA/APP-x/Krige merged")
print(f"  ΔT_cmip6={DELTA_T_CMIP6:.4f} K   ΔT_obs={DELTA_T_OBS:.4f} K")
print(f"{'Month':<11} {'SIAF_C':>8} {'SIAF_O':>8} {'DSIAF':>8} {'Dext':>8} {'Ddrk':>8} {'Dn_ext':>8} {'Dn_drk':>8} {'Dda_ext':>10} {'DDeltaT':>9}")
print("-" * 115)

for mn in valid:
    c = res_cmip6[mn]; o = res_obs[mn]
    print(f"{MONTH_NAMES[mn]:<11} {c['SIAF_tot']:>+8.4f} {o['SIAF_tot']:>+8.4f} {o['SIAF_tot']-c['SIAF_tot']:>+8.4f} "
          f"{o['SIAF_ext']-c['SIAF_ext']:>+8.4f} {o['SIAF_drk']-c['SIAF_drk']:>+8.4f} "
          f"{o['n_ext_px']-c['n_ext_px']:>+8,} {o['n_drk_px']-c['n_drk_px']:>+8,} "
          f"{o['dalpha_ext']-c['dalpha_ext']:>+10.5f} {o['delta_t']-c['delta_t']:>+9.4f}")

print("-" * 115)
print(f"{'AMJJAS':<11} {amj(res_cmip6,'SIAF_tot'):>+8.4f} {amj(res_obs,'SIAF_tot'):>+8.4f} "
      f"{amj(res_obs,'SIAF_tot')-amj(res_cmip6,'SIAF_tot'):>+8.4f} "
      f"{amj(res_obs,'SIAF_ext')-amj(res_cmip6,'SIAF_ext'):>+8.4f} "
      f"{amj(res_obs,'SIAF_drk')-amj(res_cmip6,'SIAF_drk'):>+8.4f}")

# =============================================================================
# STEP 7: GLOBAL SIAF
# =============================================================================

def global_siaf(res, months):
    scale = (len(months) / 12.) * AREA_RATIO_60N
    tot = np.mean([res[m]['SIAF_tot'] for m in months if res.get(m)])
    ext = np.mean([res[m]['SIAF_ext'] for m in months if res.get(m)])
    drk = np.mean([res[m]['SIAF_drk'] for m in months if res.get(m)])
    return tot * scale, ext * scale, drk * scale

g_tot_c, g_ext_c, g_drk_c = global_siaf(res_cmip6, valid)
g_tot_o, g_ext_o, g_drk_o = global_siaf(res_obs,   valid)

print("\n" + "=" * 65)
print("GLOBAL SIAF  (W m-2 K-1)")
print(f"  Scale: (6/12) × {AREA_RATIO_60N} = {(6/12)*AREA_RATIO_60N:.4f}")
print("-" * 65)
print(f"  {'':25} {'CMIP6':>10} {'Obs-era':>10} {'Delta':>10}")
print(f"  {'SIAF total':25} {g_tot_c:>+10.4f} {g_tot_o:>+10.4f} {g_tot_o-g_tot_c:>+10.4f}")
print(f"  {'SIAF extent':25} {g_ext_c:>+10.4f} {g_ext_o:>+10.4f} {g_ext_o-g_ext_c:>+10.4f}")
print(f"  {'SIAF darkening':25} {g_drk_c:>+10.4f} {g_drk_o:>+10.4f} {g_drk_o-g_drk_c:>+10.4f}")
print("=" * 65)

# =============================================================================
# STEP 8: CSV
# =============================================================================

rows = []
for mn in valid:
    c = res_cmip6[mn]; o = res_obs[mn]
    rows.append({
        'month'           : MONTH_NAMES[mn],
        'RS_cmip6'        : round(c['RS_mean'], 2),
        'RS_obs'          : round(o['RS_mean'], 2),
        'delta_t_cmip6'   : round(c['delta_t'],   4),
        'delta_t_obs'     : round(o['delta_t'],   4),
        'SIAF_tot_cmip6'  : round(c['SIAF_tot'],  4),
        'SIAF_tot_obs'    : round(o['SIAF_tot'],  4),
        'delta_SIAF_tot'  : round(o['SIAF_tot']  - c['SIAF_tot'], 4),
        'SIAF_ext_cmip6'  : round(c['SIAF_ext'],  4),
        'SIAF_ext_obs'    : round(o['SIAF_ext'],  4),
        'SIAF_drk_cmip6'  : round(c['SIAF_drk'],  4),
        'SIAF_drk_obs'    : round(o['SIAF_drk'],  4),
        'IS_tot_cmip6'    : round(c['IS_tot'],    5),
        'IS_tot_obs'      : round(o['IS_tot'],    5),
        'n_ext_cmip6'     : c['n_ext_px'],
        'n_ext_obs'       : o['n_ext_px'],
        'n_drk_cmip6'     : c['n_drk_px'],
        'n_drk_obs'       : o['n_drk_px'],
        'global_SIAF_tot_cmip6' : round(g_tot_c, 5),
        'global_SIAF_tot_obs'   : round(g_tot_o, 5),
    })

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")