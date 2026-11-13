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
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
dir_appx     = "/glade/derecho/scratch/jrotondo/APP-x_observations"
dir_sial     = "/glade/derecho/scratch/jrotondo/clara_a3_obs"
dir_sic      = "/glade/derecho/scratch/jrotondo/SIC_data"
legacy_cache = "/glade/work/jrotondo/ice_edge_comparison_REDONE"

SIC_ICE_THRESHOLD = 0.15       # only for optional sea-ice masking
MIN_VALID_YEARS   = 8          # min # of yearly points required per pixel
MONTHS            = {'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9}
OVERLAP_YEARS     = list(range(2012, 2025))

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# =============================================================================
# HELPERS
# =============================================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                            np.cos(lat)*np.sin(lon),
                            np.sin(lat)])

def parse_date(fname):
    try:
        return dt_mod.datetime.strptime(
            re.search(r'_d(\d{8})', fname).group(1), "%Y%m%d")
    except Exception:
        return None

def parse_utc(fname):
    base = os.path.basename(fname)
    m = re.search(r'_(0[0-9]{3}|1[0-9]{3}|2[0-3][0-9]{2})_d\d{8}', base)
    return m.group(1) if m else '0000'

def parse_cdate(fname):
    m = re.search(r'_c(\d{8})\.nc$', os.path.basename(fname))
    return m.group(1) if m else '00000000'

# =============================================================================
# GRID SETUP
# =============================================================================
print("Building grids and cKDTrees ...")

# ---- SIAL (CLARA-A3) grid ---------------------------------------------------
sample_sial = sorted(glob.glob(f"{dir_sial}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d, sial_lon1d = ds['lat'].values, ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lon2d, sial_lat2d = sial_lon1d, sial_lat1d
sial_shape = sial_lat2d.shape
sial_xyz   = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())

# ---- APP-x / VPP-x grid -----------------------------------------------------
sample_appx = glob.glob(f"{dir_appx}/**/*.nc", recursive=True)[0]
with Dataset(sample_appx, 'r') as ds:
    appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
    appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'],
                              '_FillValue', 9999.0))
if appx_lat.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
else:
    appx_lon2d, appx_lat2d = appx_lon, appx_lat
appx_shape = appx_lat2d.shape
appx_xyz   = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())

# ---- PSN25 SIC grid ---------------------------------------------------------
sample_sic = glob.glob(f"{dir_sic}/sic_psn25_*.nc")[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x, sic_y = ds['x'].values, ds['y'].values
transformer = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
sic_lon, sic_lat = transformer.transform(*np.meshgrid(sic_x, sic_y))
sic_xyz = latlon_to_xyz(sic_lat.ravel(), sic_lon.ravel())

# ---- Regridding trees -------------------------------------------------------
tree_sial = cKDTree(sial_xyz)
_, idx_sial_to_appx = tree_sial.query(appx_xyz, workers=-1)

tree_sic = cKDTree(sic_xyz)
_, idx_sic_to_appx = tree_sic.query(appx_xyz, workers=-1)

legacy_mask_psn = np.load(f"{legacy_cache}/arctic_60N_mask_psn25.npy")
arctic_60N_appx = legacy_mask_psn.ravel()[idx_sic_to_appx].reshape(appx_shape)

def clara_to_appx(clara_2d):
    return clara_2d.ravel()[idx_sial_to_appx].reshape(appx_shape)

def sic_to_appx(sic_2d):
    return sic_2d.ravel()[idx_sic_to_appx].reshape(appx_shape)

# =============================================================================
# DATA READERS
# =============================================================================
def get_clara_mean(year, month_num):
    files = glob.glob(f"{dir_sial}/*{year}*.nc")
    if not files: return None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        smonth = ds.sel(time=ds['time.month'] == month_num)
        if len(smonth.time) == 0: return None
        snow = smonth['black_sky_albedo_snow_mean'].mean(dim='time').values / 100.0
        reg  = smonth['black_sky_albedo_mean'].mean(dim='time').values / 100.0
        return np.where(~np.isnan(snow), snow, reg)
    except Exception: return None

def _load_appx_albedo(fpath):
    try:
        with Dataset(fpath, 'r') as ds:
            var_name = next((v for v in ds.variables
                             if 'albedo' in v.lower() and ds.variables[v].ndim >= 2), None)
            if var_name is None: return None
            alb_var = ds.variables[var_name]
            alb = alb_var[0] if alb_var.ndim == 3 else alb_var[:]
            if hasattr(alb, 'filled'): alb = alb.filled(np.nan)
            alb = alb.astype(float)
            alb[alb == float(getattr(alb_var, '_FillValue', appx_fill))] = np.nan
            alb[(alb < 0) | (alb > 1)] = np.nan
            return alb
    except Exception: return None

def get_appx_mean_both_utc(year, month_num, prefix):
    base = dir_appx.rstrip('/')
    search_paths = [
        f"{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc",
        f"{base}/{prefix}*d{year}{month_num:02d}*.nc",
    ]
    files = sorted(set(f for p in search_paths for f in glob.glob(p)))
    if not files: return None

    best = {}
    for f in files:
        d = parse_date(f)
        if d is None: continue
        key = (d.strftime('%Y%m%d'), parse_utc(f))
        cd  = parse_cdate(f)
        if key not in best or cd > best[key][1]: best[key] = (f, cd)

    daily_maps = []
    for (f, _) in best.values():
        m = _load_appx_albedo(f)
        if m is not None: daily_maps.append(m)

    return np.nanmean(daily_maps, axis=0) if daily_maps else None

def get_sic_mean(year, month_num):
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

# =============================================================================
# PIXEL-WISE PEARSON CORRELATION
# =============================================================================
def pixel_corr(a_stack, b_stack, min_valid=MIN_VALID_YEARS):
    valid = np.isfinite(a_stack) & np.isfinite(b_stack)
    n_map = valid.sum(axis=0)

    a = np.where(valid, a_stack, np.nan)
    b = np.where(valid, b_stack, np.nan)

    mean_a = np.nanmean(a, axis=0)
    mean_b = np.nanmean(b, axis=0)

    da = a - mean_a[None, :, :]
    db = b - mean_b[None, :, :]

    num   = np.nansum(da * db,      axis=0)
    den_a = np.nansum(da * da,      axis=0)
    den_b = np.nansum(db * db,      axis=0)
    den   = np.sqrt(den_a * den_b)

    r = np.where(den > 0, num / den, np.nan)
    r[n_map < min_valid] = np.nan
    return r, n_map

# =============================================================================
# MAIN PROCESSING
# =============================================================================
corr_results = {}   

for month_name, month_num in MONTHS.items():
    print(f"\n=== {month_name} ===")
    clara_stack, appx_stack, vppx_stack, sic_stack, years_kept = [], [], [], [], []

    for year in OVERLAP_YEARS:
        clara = get_clara_mean(year, month_num)
        appx  = get_appx_mean_both_utc(year, month_num, "Polar-APP")
        vppx  = get_appx_mean_both_utc(year, month_num, "Polar-VPP-X")
        sic   = get_sic_mean(year, month_num)

        if clara is None or appx is None or vppx is None:
            continue

        clara_on_appx = clara_to_appx(clara)
        sic_on_appx   = sic_to_appx(sic) if sic is not None else None

        clara_stack.append(clara_on_appx)
        appx_stack.append(appx)
        vppx_stack.append(vppx)
        sic_stack.append(sic_on_appx if sic_on_appx is not None else np.full(appx_shape, np.nan))
        years_kept.append(year)

    if len(vppx_stack) < MIN_VALID_YEARS:
        print(f"  Only {len(vppx_stack)} years usable -- skipping month.")
        continue

    clara_arr = np.array(clara_stack)
    appx_arr  = np.array(appx_stack)
    vppx_arr  = np.array(vppx_stack)
    sic_arr   = np.array(sic_stack)

    r_clara, n_clara = pixel_corr(clara_arr, vppx_arr)
    r_appx,  n_appx  = pixel_corr(appx_arr,  vppx_arr)

    # Masking
    r_clara[~arctic_60N_appx] = np.nan
    r_appx [~arctic_60N_appx] = np.nan

    ice_count = np.nansum(sic_arr >= SIC_ICE_THRESHOLD, axis=0)
    ice_ok    = ice_count >= (len(years_kept) / 2)
    
    r_clara_ice = np.where(ice_ok, r_clara, np.nan)
    r_appx_ice  = np.where(ice_ok, r_appx,  np.nan)

    # ---> NEW: Extract flattened 1D arrays for the scatter plots <---
    mask_2d = arctic_60N_appx & ice_ok
    mask_3d = np.broadcast_to(mask_2d, clara_arr.shape)

    # CLARA vs VPP-x 1D valid pairs
    c_valid = mask_3d & np.isfinite(clara_arr) & np.isfinite(vppx_arr)
    c_x = vppx_arr[c_valid]
    c_y = clara_arr[c_valid]

    # APP-x vs VPP-x 1D valid pairs
    a_valid = mask_3d & np.isfinite(appx_arr) & np.isfinite(vppx_arr)
    a_x = vppx_arr[a_valid]
    a_y = appx_arr[a_valid]

    corr_results[month_name] = dict(
        years=years_kept,
        r_clara=r_clara, r_appx=r_appx,
        r_clara_ice=r_clara_ice, r_appx_ice=r_appx_ice,
        n_clara=n_clara, n_appx=n_appx,
        scatter_c_x=c_x, scatter_c_y=c_y,  # Stored for scatter plot
        scatter_a_x=a_x, scatter_a_y=a_y   # Stored for scatter plot
    )

    print(f"  Years used: {years_kept}")
    print(f"  median r (CLARA vs VPP-x, Arctic)  = {np.nanmedian(r_clara):+.3f}")
    print(f"  median r (APP-x vs VPP-x, Arctic)  = {np.nanmedian(r_appx):+.3f}")

# =============================================================================
# SAVE NETCDF
# =============================================================================
if corr_results:
    out_nc = "/glade/derecho/scratch/jrotondo/pixel_corr_vs_vppx.nc"
    data_vars = {}
    for month_name, d in corr_results.items():
        tag = month_name.lower()
        data_vars[f"r_clara_vppx_{tag}"] = (('y', 'x'), d['r_clara'])
        data_vars[f"r_appx_vppx_{tag}"]  = (('y', 'x'), d['r_appx'])
        data_vars[f"r_clara_vppx_ice_{tag}"] = (('y', 'x'), d['r_clara_ice'])
        data_vars[f"r_appx_vppx_ice_{tag}"]  = (('y', 'x'), d['r_appx_ice'])

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={'latitude': (('y', 'x'), appx_lat2d), 'longitude': (('y', 'x'), appx_lon2d)}
    )
    try:
        out_ds.to_netcdf(out_nc)
        print(f"\nSaved correlation maps -> {out_nc}")
    except Exception as e:
        print(f"NetCDF save failed: {e}")

# =============================================================================
# PLOTTING 1: Correlation Maps 
# =============================================================================
if corr_results:
    months_plot = [m for m in MONTHS if m in corr_results]
    n_months    = len(months_plot)

    fig, axes = plt.subplots(2, n_months, figsize=(4.2 * n_months, 9), dpi=150)
    if n_months == 1: axes = axes[:, None]

    cmap = plt.get_cmap('RdBu_r')
    norm = plt.Normalize(vmin=-1, vmax=1)

    arctic_rows = np.where(arctic_60N_appx.any(axis=1))[0]
    r0, r1 = (arctic_rows.min(), arctic_rows.max() + 1) if len(arctic_rows) else (0, appx_shape[0])
    
    arctic_cols = np.where(arctic_60N_appx.any(axis=0))[0]
    c0, c1 = (arctic_cols.min(), arctic_cols.max() + 1) if len(arctic_cols) else (0, appx_shape[1])

    for col, m in enumerate(months_plot):
        d = corr_results[m]
        axes[0, col].imshow(d['r_clara_ice'][r0:r1, c0:c1], cmap=cmap, norm=norm, origin='upper')
        axes[0, col].set_title(f"{m}\nCLARA-A3 vs VPP-x")
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        axes[1, col].imshow(d['r_appx_ice'][r0:r1, c0:c1], cmap=cmap, norm=norm, origin='upper')
        axes[1, col].set_title(f"{m}\nAPP-x vs VPP-x")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    fig.subplots_adjust(right=0.90, top=0.92, bottom=0.06, wspace=0.05)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb  = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_label("Pearson r  (monthly means, 2012-2024)")
    fig.suptitle("Per-pixel temporal correlation vs VPP-x", fontsize=18, y=0.995)

    out_png = "/glade/derecho/scratch/jrotondo/pixel_corr_vs_vppx.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved Map Figure -> {out_png}")


# =============================================================================
# PLOTTING 2: Density Scatter Plots + Linear Regression
# =============================================================================
    print("\nGenerating Scatter Plot Figure...")
    fig_scat, axes_scat = plt.subplots(2, n_months, figsize=(5.5 * n_months, 10), dpi=150)
    if n_months == 1: axes_scat = axes_scat[:, None]

    def plot_density_scatter(ax, x, y, title, ylabel):
        if len(x) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            return None
        
        # 2D Histogram to avoid overplotting millions of points
        h = ax.hist2d(x, y, bins=120, range=[[0, 1], [0, 1]], cmap='inferno', cmin=1)
        
        # Calculate Regression
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        x_line = np.array([0, 1])
        y_line = slope * x_line + intercept
        
        # Plot styling
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='1:1')
        ax.plot(x_line, y_line, color='cyan', lw=2.5, 
                label=f'y = {slope:.2f}x {intercept:+.2f}\nR$^2$ = {r_val**2:.2f}')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('VPP-x Albedo', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        return h[3] # Return mappable for colorbar

    for col, m in enumerate(months_plot):
        d = corr_results[m]

        # Top row: CLARA vs VPP-x
        plot_density_scatter(axes_scat[0, col], d['scatter_c_x'], d['scatter_c_y'], 
                             f"{m}\nCLARA-A3 vs VPP-x", "CLARA-A3 Albedo")

        # Bottom row: APP-x vs VPP-x
        mappable = plot_density_scatter(axes_scat[1, col], d['scatter_a_x'], d['scatter_a_y'], 
                                        f"{m}\nAPP-x vs VPP-x", "APP-x Albedo")

    fig_scat.tight_layout()
    
    # Add shared colorbar for the density
    fig_scat.subplots_adjust(right=0.93)
    cax_scat = fig_scat.add_axes([0.94, 0.15, 0.015, 0.7])
    cb_scat = fig_scat.colorbar(mappable, cax=cax_scat)
    cb_scat.set_label("Pixel Count", fontsize=14)

    out_scat = "/glade/derecho/scratch/jrotondo/scatter_vs_vppx.png"
    plt.savefig(out_scat, dpi=200, bbox_inches='tight')
    print(f"Saved Scatter Figure -> {out_scat}")
