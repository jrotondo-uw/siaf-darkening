import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime as dt_mod
from pyproj import Transformer
from scipy.spatial import cKDTree
import warnings

# Suppress division by zero / invalid warnings for empty grid cells
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
dir_appx     = "/glade/derecho/scratch/jrotondo/APP-x_observations"
dir_sial     = "/glade/derecho/scratch/jrotondo/clara_a3_obs"
dir_sic      = "/glade/derecho/scratch/jrotondo/SIC_data"
legacy_cache = "/glade/work/jrotondo/ice_edge_comparison_REDONE"
output_dir   = "./correlation_figures"

os.makedirs(output_dir, exist_ok=True)

SIC_ICE_THRESHOLD = 0.80
MONTHS = {'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9}
OVERLAP_YEARS = list(range(2012, 2025))

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# =============================================================================
# HELPERS & GRID SETUP
# =============================================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def parse_date(fname):
    try: return dt_mod.datetime.strptime(re.search(r'_d(\d{8})', fname).group(1), "%Y%m%d")
    except Exception: return None

print("Building grids and cKDTrees for pixel-level matching...")

# 1. SIAL (CLARA) Grid
sample_sial = sorted(glob.glob(f"{dir_sial}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d) if sial_lat1d.ndim == 1 else (sial_lon1d, sial_lat1d)
sial_shape = sial_lat2d.shape
sial_xyz = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())

# 2. APP-x Grid (Baseline grid for correlations)
sample_appx = glob.glob(f"{dir_appx}/**/*.nc", recursive=True)[0]
with Dataset(sample_appx, 'r') as ds:
    appx_lat = np.ma.filled(ds.variables['latitude'][:], np.nan).astype(float)
    appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat) if appx_lat.ndim == 1 else (appx_lon, appx_lat)
appx_shape = appx_lat2d.shape
appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())

# 3. PSN25 (SIC) Grid
sample_sic = glob.glob(f"{dir_sic}/sic_psn25_*.nc")[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x, sic_y = ds['x'].values, ds['y'].values
transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
sic_lon, sic_lat = transformer.transform(*np.meshgrid(sic_x, sic_y))
sic_xyz = latlon_to_xyz(sic_lat.ravel(), sic_lon.ravel())

# 4. Regridding Trees 
tree_sic = cKDTree(sic_xyz)
tree_sial = cKDTree(sial_xyz)

# To map SIC onto APP-x
_, idx_sic_to_appx = tree_sic.query(appx_xyz, workers=-1)
# To map CLARA onto APP-x for direct cell-to-cell comparison
_, idx_sial_to_appx = tree_sial.query(appx_xyz, workers=-1)

legacy_mask_psn = np.load(f"{legacy_cache}/arctic_60N_mask_psn25.npy")
arctic_60N_appx = legacy_mask_psn.ravel()[idx_sic_to_appx].reshape(appx_shape)

def regrid_sic_to_appx(sic_2d): return sic_2d.ravel()[idx_sic_to_appx].reshape(appx_shape)
def regrid_sial_to_appx(sial_2d): return sial_2d.ravel()[idx_sial_to_appx].reshape(appx_shape)

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================
def get_clara_map(year, month_num):
    files = glob.glob(f"{dir_sial}/*{year}*.nc")
    if not files: return None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        smonth = ds.sel(time=ds['time.month'] == month_num)
        if len(smonth.time) == 0: return None
        snow = smonth['black_sky_albedo_snow_mean'].mean(dim='time').values / 100.0
        reg = smonth['black_sky_albedo_mean'].mean(dim='time').values / 100.0
        return np.where(~np.isnan(snow), snow, reg)
    except Exception:
        return None

def get_appx_map(year, month_num, prefix):
    base = dir_appx.rstrip('/')
    search_paths = [f"{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc", f"{base}/{prefix}*d{year}{month_num:02d}*.nc"]
    files = sorted(list(set([f for p in search_paths for f in glob.glob(p)])))
    if not files: return None
    
    best_files = {}
    for f in files:
        obs_date = parse_date(f)
        if not obs_date: continue
        cdate_match = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
        cdate = cdate_match.group(1) if cdate_match else '00000000'
        obs_str = obs_date.strftime('%Y%m%d')
        if obs_str not in best_files or cdate > best_files[obs_str][1]:
            best_files[obs_str] = (f, cdate)

    daily_maps = []
    for _, (f, _) in best_files.items():
        try:
            with Dataset(f, 'r') as ds:
                var_name = next((v for v in ds.variables if 'albedo' in v.lower() and ds.variables[v].ndim >= 2), None)
                if not var_name: continue
                alb = ds.variables[var_name][0] if ds.variables[var_name].ndim == 3 else ds.variables[var_name][:]
                if hasattr(alb, 'filled'): alb = alb.filled(np.nan)
                alb = alb.astype(float)
                alb[alb == float(getattr(ds.variables[var_name], '_FillValue', appx_fill))] = np.nan
                alb[(alb < 0) | (alb > 1)] = np.nan
                daily_maps.append(alb)
        except Exception: continue
    return np.nanmean(daily_maps, axis=0) if daily_maps else None

def get_sic_map(year, month_num):
    files = glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc")
    maps = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                if pd.Timestamp(ds['time'].values[0]).month == month_num:
                    raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                    if raw.ndim > 2: raw = raw[0]
                    if np.nanmax(raw) > 1.5: raw /= 100.0
                    maps.append(np.where(np.isfinite(raw), raw, np.nan))
        except Exception: pass
    return np.nanmean(maps, axis=0) if maps else None

# Vectorized Pearson Correlation
def calc_pixel_correlation(x_stack, y_stack, min_years=5):
    """Calculate Pearson r per pixel along the time axis (axis 0)"""
    valid = np.isfinite(x_stack) & np.isfinite(y_stack)
    x = np.where(valid, x_stack, np.nan)
    y = np.where(valid, y_stack, np.nan)
    
    n = np.sum(valid, axis=0)
    
    mean_x = np.nansum(x, axis=0) / n
    mean_y = np.nansum(y, axis=0) / n
    
    dx = x - mean_x
    dy = y - mean_y
    
    cov = np.nansum(dx * dy, axis=0)
    var_x = np.nansum(dx**2, axis=0)
    var_y = np.nansum(dy**2, axis=0)
    
    r = cov / np.sqrt(var_x * var_y)
    r[n < min_years] = np.nan  # Mask pixels without enough overlapping years
    return r

# =============================================================================
# MAIN PROCESSING & PLOTTING
# =============================================================================
print("\nCalculating per-grid cell correlations...")

for month_name, month_num in MONTHS.items():
    print(f"\nProcessing {month_name}...")
    
    clara_stack, appx_stack, vppx_stack, sic_stack = [], [], [], []
    
    for year in OVERLAP_YEARS:
        sic = get_sic_map(year, month_num)
        clara = get_clara_map(year, month_num)
        appx = get_appx_map(year, month_num, "Polar-APP")
        vppx = get_appx_map(year, month_num, "Polar-VPP-X")
        
        if sic is not None and clara is not None and appx is not None and vppx is not None:
            # Regrid to common APP-x Grid
            sic_appx_grid = regrid_sic_to_appx(sic)
            clara_appx_grid = regrid_sial_to_appx(clara)
            
            # Mask out non-ice and non-Arctic domain beforehand
            ice_mask = arctic_60N_appx & np.isfinite(sic_appx_grid) & (sic_appx_grid >= SIC_ICE_THRESHOLD)
            
            clara_appx_grid[~ice_mask] = np.nan
            appx[~ice_mask] = np.nan
            vppx[~ice_mask] = np.nan
            
            clara_stack.append(clara_appx_grid)
            appx_stack.append(appx)
            vppx_stack.append(vppx)
            
    if not vppx_stack:
        print(f"No valid data arrays found for {month_name}.")
        continue
        
    clara_stack = np.array(clara_stack)
    appx_stack = np.array(appx_stack)
    vppx_stack = np.array(vppx_stack)
    
    # Calculate correlations
    r_appx_vppx = calc_pixel_correlation(appx_stack, vppx_stack)
    r_clara_vppx = calc_pixel_correlation(clara_stack, vppx_stack)
    
    # Positive diff = CLARA correlates better with VPP-x than APP-x does
    # Negative diff = APP-x correlates better with VPP-x
    r_diff = r_clara_vppx - r_appx_vppx 

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=200)
    fig.suptitle(f"{month_name} Per-Pixel Correlation with VPP-x (VIIRS)\nOverlapping Years (2012-2024)", fontweight='bold')
    
    # 1. APP-x vs VPP-x
    im0 = axes[0].imshow(r_appx_vppx, cmap='RdYlBu_r', vmin=-1, vmax=1)
    axes[0].set_title('APP-x vs VPP-x ($r$)')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. CLARA vs VPP-x
    im1 = axes[1].imshow(r_clara_vppx, cmap='RdYlBu_r', vmin=-1, vmax=1)
    axes[1].set_title('CLARA-A3 vs VPP-x ($r$)')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Difference
    im2 = axes[2].imshow(r_diff, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference ($r_{CLARA} - r_{APP}$)\n(Purple = APP Better, Orange = CLARA Better)')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'correlation_map_{month_name}.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")

print("\nProcessing Complete.")