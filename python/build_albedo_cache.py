"""
build_albedo_cache.py
---------------------
Builds sensitivity-sweep albedo caches for TWO analysis periods using ONLY
APP-x (AVHRR) data to maintain strict sensor consistency:

  (1) Full observational era  : EARLY 1982-1991, LATE 2015-2024, TREND 1982-2024
  (2) CMIP6 historical period : EARLY 1982-1991, LATE 2005-2014, TREND 1982-2014

Outputs
-------
  /glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/
  /glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats as sp_stats
from pyproj import Transformer
from netCDF4 import Dataset as NC4Dataset
import glob, os, gc, re
import datetime as dt_mod

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS = {
    'obs': {
        'label'       : 'Full observational era (1982-2024)',
        'early_years' : list(range(1982, 1992)),   # 1982-1991
        'late_years'  : list(range(2015, 2025)),   # 2015-2024
        'trend_years' : list(range(1982, 2025)),   # 1982-2024
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/',
    },
    'cmip6': {
        'label'       : 'CMIP6 historical period (1982-2014)',
        'early_years' : list(range(1982, 1992)),   # 1982-1991
        'late_years'  : list(range(2005, 2015)),   # 2005-2014
        'trend_years' : list(range(1982, 2015)),   # 1982-2014
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/',
    },
}

# =============================================================================
# SHARED CONFIG
# =============================================================================

dir_sial        = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
dir_appx        = "/glade/derecho/scratch/jrotondo/APP-x_observations/"
dir_sic         = "/glade/derecho/scratch/jrotondo/SIC_data/"

SIC_ICE_THRESHOLD = 0.15
PENTAD_DAYS       = 5
MIN_ARCTIC_PX     = 100
AREA_RATIO_60N    = 0.065  # Planetary constant for 60N-90N vs global surface area

MONTHS_TO_ANALYZE = {
    'April'    : 4,
    'May'      : 5,
    'June'     : 6,
    'July'     : 7,
    'August'   : 8,
    'September': 9,
}

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

def parse_date_appx(fname):
    base = os.path.basename(fname)
    try:
        return dt_mod.datetime.strptime(base.split('_d')[1][:8], "%Y%m%d")
    except Exception:
        return None

def weighted_mean(field, mask, weights):
    w = weights[mask]
    v = field[mask]
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if good.sum() == 0:
        return np.nan
    return float(np.sum(v[good] * w[good]) / np.sum(w[good]))

def load_sic_day(filepath):
    with xr.open_dataset(filepath) as ds:
        t   = pd.Timestamp(ds['time'].values[0])
        raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
        if raw.ndim > 2:
            raw = raw[0]
        if np.nanmax(raw) > 1.5:
            raw /= 100.0
            
        # Strict physical bounds (0.0 to 1.0) to filter out land/coast flags
        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
    return t, raw

# =============================================================================
# STEP 0: BUILD GRIDS
# =============================================================================

print("=" * 70)
print("STEP 0: Building grids")
print("=" * 70)

# SIAL grid
sample_sial = sorted(glob.glob(f"{dir_sial}/*.nc"))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape
sial_xyz   = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())

sial_coslat = np.cos(np.deg2rad(sial_lat2d)).astype(float)
sial_coslat = np.where(sial_coslat > 0, sial_coslat, 0.0)

# APP-x grid
sample_appx = sorted(glob.glob(f"{dir_appx}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]), np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if appx_lat_raw.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
appx_shape = appx_lat2d.shape
appx_xyz   = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
appx_weights = np.ones(appx_shape, dtype=float)

# PSN25 (SIC) grid
sample_sic = sorted(glob.glob(f"{dir_sic}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values
    sic_y = ds['y'].values
sic_xx, sic_yy = np.meshgrid(sic_x, sic_y)
transformer     = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
sic_lon2d_psn, sic_lat2d_psn = transformer.transform(sic_xx, sic_yy)
sic_shape = sic_lat2d_psn.shape
sic_xyz   = latlon_to_xyz(sic_lat2d_psn.ravel(), sic_lon2d_psn.ravel())
sic_tree  = cKDTree(sic_xyz)

_, idx_sic_to_sial = sic_tree.query(sial_xyz, workers=-1)
_, idx_sic_to_appx = sic_tree.query(appx_xyz, workers=-1)

def regrid_sic_to_sial(sic_2d): return sic_2d.ravel()[idx_sic_to_sial].reshape(sial_shape)
def regrid_sic_to_appx(sic_2d): return sic_2d.ravel()[idx_sic_to_appx].reshape(appx_shape)
def regrid_psn_to_sial(psn_2d): return psn_2d.ravel()[idx_sic_to_sial].reshape(sial_shape)
def regrid_psn_to_appx(psn_2d): return psn_2d.ravel()[idx_sic_to_appx].reshape(appx_shape)

# Arctic masks (Dynamically Generated)
arctic_60N_mask_psn = (sic_lat2d_psn >= 60.0)

_, idx_60N_sial = sic_tree.query(sial_xyz, workers=-1)
arctic_60N_sial = arctic_60N_mask_psn.ravel()[idx_60N_sial].reshape(sial_shape)

_, idx_60N_appx = sic_tree.query(appx_xyz, workers=-1)
arctic_60N_appx = arctic_60N_mask_psn.ravel()[idx_60N_appx].reshape(appx_shape)

print(f"  SIAL grid : {sial_shape}")
print(f"  APP-x grid: {appx_shape}")
print(f"  PSN25 grid: {sic_shape}")

# =============================================================================
# MAIN LOOP
# =============================================================================

for exp_key, exp in EXPERIMENTS.items():
    EARLY_YEARS    = exp['early_years']
    LATE_YEARS     = exp['late_years']
    TREND_YEARS    = exp['trend_years']
    ALL_YEARS      = sorted(set(EARLY_YEARS + LATE_YEARS))
    sens_cache_dir = exp['cache_dir']
    os.makedirs(sens_cache_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp['label']}")
    print("=" * 70)

    np.save(f"{sens_cache_dir}/sial_shape.npy", np.array(sial_shape))
    np.save(f"{sens_cache_dir}/appx_shape.npy", np.array(appx_shape))
    np.save(f"{sens_cache_dir}/arctic_60N_sial.npy", arctic_60N_sial)
    np.save(f"{sens_cache_dir}/arctic_60N_appx.npy", arctic_60N_appx)
    np.save(f"{sens_cache_dir}/appx_fill.npy", np.array(appx_fill))
    np.save(f"{sens_cache_dir}/sial_coslat.npy", sial_coslat)
    with open(f"{sens_cache_dir}/area_ratio.txt", 'w') as fh: fh.write(str(AREA_RATIO_60N))

    # -------------------------------------------------------------------------
    # STEP 2: CLARA-A3
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading CLARA-A3 albedo maps...")
    for year in ALL_YEARS:
        print(f"  CLARA {year}...", end='', flush=True)
        cache_check = f"{sens_cache_dir}/clara_alb_{list(MONTHS_TO_ANALYZE.keys())[0]}_{year}.npy"
        if os.path.exists(cache_check):
            print(" (cached, skip)")
            continue

        sial_files = [f for f in sorted(glob.glob(f"{dir_sial}/*.nc")) if str(year) in f]
        if not sial_files:
            print(" skip (no files)")
            continue

        try:
            ds_sial = xr.open_mfdataset(sial_files, combine='by_coords')
            snow_alb = ds_sial['black_sky_albedo_snow_mean'] / 100.0
            reg_alb = ds_sial['black_sky_albedo_mean'] / 100.0
            sial_raw = xr.where(~np.isnan(snow_alb), snow_alb, reg_alb)
            sial_pentad = sial_raw.resample(time=f'{PENTAD_DAYS}D').mean().load()
            ds_sial.close()
        except Exception as e:
            print(f" skip ({e})")
            continue

        sic_daily = {}
        for f in sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc")):
            try:
                t, raw = load_sic_day(f)
                # FIX: Leave NaNs as NaNs to ensure land remains fully masked
                sic_daily[t.date()] = regrid_sic_to_sial(raw)
            except Exception:
                continue

        sic_pentad_dict = {}
        for ptime in sial_pentad.time.values:
            pd_date = pd.Timestamp(ptime).date()
            bin_days = [pd_date + dt_mod.timedelta(days=d) for d in range(PENTAD_DAYS)]
            maps = [sic_daily[d] for d in bin_days if d in sic_daily]
            if maps:
                sic_pentad_dict[pd_date] = np.nanmean(maps, axis=0)

        for month_name, month_num in MONTHS_TO_ANALYZE.items():
            valid_domain = arctic_60N_sial
            sial_month = sial_pentad.sel(time=sial_pentad['time.month'] == month_num)
            if len(sial_month.time) == 0: continue

            sial_mean = sial_month.mean(dim='time').values
            sic_maps = [sic_pentad_dict[pd.Timestamp(t).date()] for t in sial_month.time.values if pd.Timestamp(t).date() in sic_pentad_dict]
            if not sic_maps: continue
            sic_mean = np.nanmean(sic_maps, axis=0)

            sial_mean = np.where(valid_domain, sial_mean, np.nan)
            sic_mean = np.where(valid_domain, sic_mean, np.nan)
            valid_px = (np.isfinite(sial_mean) & np.isfinite(sic_mean) & valid_domain)
            if valid_px.sum() == 0: continue

            m_ice = valid_px & (sic_mean >= SIC_ICE_THRESHOLD)
            m_ocean = valid_px & (sic_mean < SIC_ICE_THRESHOLD)

            alb_ice = weighted_mean(sial_mean, m_ice, sial_coslat)
            alb_ocean = weighted_mean(sial_mean, m_ocean, sial_coslat)

            if np.isfinite(alb_ice) and np.isfinite(alb_ocean):
                np.save(f"{sens_cache_dir}/clara_alb_{month_name}_{year}.npy", np.where(valid_px, sial_mean, np.nan))
                np.save(f"{sens_cache_dir}/clara_dalpha_{month_name}_{year}.npy", np.array(alb_ice - alb_ocean))
        print(" ✓")
        del sial_pentad, sic_daily
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: APP-x (AVHRR strictly)
    # -------------------------------------------------------------------------
    print("\nStep 3: Loading APP-x albedo maps...")

    def load_appx_monthly(year, month_num):
        base = dir_appx.rstrip('/')
        prefix = "Polar-APP-X"
        
        # 1. Search for ALL files (both 0400 and 1400)
        search_paths = [
            f"{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc",
            f"{base}/{prefix}*d{year}{month_num:02d}*.nc"
        ]
        
        files = []
        for p in search_paths: 
            files.extend(glob.glob(p))
        files = sorted(list(set(files)))
        
        if not files:
            print(f"\n      [!] Data Missing: No APP-X files found for {year}-{month_num:02d}")
            return None
            
        # 2. Group files by Date, then by Overpass Time to keep the newest version of each
        files_by_date = {}
        for f in files:
            obs_date = parse_date_appx(f)
            if obs_date is None: continue
            
            obs_str = obs_date.strftime('%Y%m%d')
            
            m_time = re.search(r'_(\d{4})_d', os.path.basename(f))
            overpass = m_time.group(1) if m_time else 'unknown'
            
            m_cdate = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
            cdate = m_cdate.group(1) if m_cdate else '00000000'
            
            if obs_str not in files_by_date:
                files_by_date[obs_str] = {}
                
            if overpass not in files_by_date[obs_str] or cdate > files_by_date[obs_str][overpass][1]:
                files_by_date[obs_str][overpass] = (f, cdate)
                
        # 3. Load and average the daily passes
        monthly_daily_maps = []
        for obs_str, passes in sorted(files_by_date.items()):
            daily_passes = []
            
            # Load all available overpasses for this specific day
            for overpass, (f, cdate) in passes.items():
                try:
                    with NC4Dataset(f, 'r') as ds:
                        var_name = next((v for v in ds.variables if 'albedo' in v.lower() and ds.variables[v].ndim >= 2), None)
                        if not var_name: continue
                            
                        alb_var = ds.variables[var_name]
                        alb = alb_var[0] if alb_var.ndim == 3 else alb_var[:]
                        if hasattr(alb, 'filled'): alb = alb.filled(np.nan)
                        alb = alb.astype(float)
                        
                        file_fill = float(getattr(alb_var, '_FillValue', appx_fill))
                        alb[alb == file_fill] = np.nan
                        alb[(alb < 0) | (alb > 1)] = np.nan
                        daily_passes.append(alb)
                except Exception as e:
                    # Silently skip corrupted files
                    continue
            
            # If we successfully loaded at least one overpass for the day, average them
            if daily_passes:
                # nanmean merges 0400 and 1400. If 0400 is NaN, it defaults to 1400.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    daily_mean = np.nanmean(daily_passes, axis=0)
                monthly_daily_maps.append(daily_mean)
                
        if not monthly_daily_maps:
            print(f"\n      [!] Extraction Failed: Files exist, but valid arrays could not be extracted.")
            return None
            
        # 4. Create the final monthly mean from the daily averages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(monthly_daily_maps, axis=0)

    def load_sic_monthly_appx(year, month_num):
        maps = []
        files = sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc"))
        if not files:
            print(f"\n      [!] SIC Missing: No PSN25 sea ice files found for {year}")
            return None
            
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    if pd.Timestamp(ds['time'].values[0]).month != month_num:
                        continue
                        
                    raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                    if raw.ndim > 2:
                        raw = raw[0]
                        
                    if not np.any(np.isfinite(raw)):
                        continue
                    if np.nanmax(raw) > 1.5:
                        raw /= 100.0
                        
                    # Strict physical bounds (0.0 to 1.0) to filter out land/coast flags
                    raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
                    
                    if np.sum(np.isfinite(raw[arctic_60N_mask_psn])) < MIN_ARCTIC_PX:
                        continue
                    maps.append(regrid_sic_to_appx(raw))
            except Exception as e:
                print(f"\n      [!] SIC Extract Error in {os.path.basename(f)}: {e}")
                continue
                
        if not maps:
            print(f"\n      [!] SIC Masking Failed: SIC files found, but all arrays evaluated as empty over 60N domain.")
            return None
            
        return np.nanmean(maps, axis=0)

    for year in ALL_YEARS:
        print(f"  APP-x {year}...", end='', flush=True)
        found_any = False

        for month_name, month_num in MONTHS_TO_ANALYZE.items():
            cache_path = f"{sens_cache_dir}/appx_alb_{month_name}_{year}.npy"
            
            if os.path.exists(cache_path):
                found_any = True
                continue

            alb_mean = load_appx_monthly(year, month_num)
            sic_mean = load_sic_monthly_appx(year, month_num)
            
            if alb_mean is None or sic_mean is None:
                continue

            valid_domain = arctic_60N_appx
            alb_mean = np.where(valid_domain, alb_mean, np.nan)
            sic_mean = np.where(valid_domain, sic_mean, np.nan)
            
            valid_px = (np.isfinite(alb_mean) & np.isfinite(sic_mean) & valid_domain)
            
            if valid_px.sum() < MIN_ARCTIC_PX:
                print(f"\n      [!] Masking Failed: {month_name} {year} resulted in {valid_px.sum()} valid pixels (< {MIN_ARCTIC_PX}).")
                continue

            m_ice = valid_px & (sic_mean >= SIC_ICE_THRESHOLD)
            m_ocean = valid_px & (sic_mean < SIC_ICE_THRESHOLD)
            
            if m_ice.sum() < MIN_ARCTIC_PX or m_ocean.sum() < MIN_ARCTIC_PX:
                continue

            alb_ice = weighted_mean(alb_mean, m_ice, appx_weights)
            alb_ocean = weighted_mean(alb_mean, m_ocean, appx_weights)
            
            if np.isfinite(alb_ice) and np.isfinite(alb_ocean):
                np.save(cache_path, np.where(valid_px, alb_mean, np.nan))
                np.save(f"{sens_cache_dir}/appx_dalpha_{month_name}_{year}.npy", np.array(alb_ice - alb_ocean))
                found_any = True

        print(" ✓" if found_any else " skip")
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 4: Combined dalpha
    # -------------------------------------------------------------------------
    print("\nStep 4: Computing combined dalpha...")
    dalpha_rows = []
    for month_name in MONTHS_TO_ANALYZE:
        for year in ALL_YEARS:
            vals = []
            p_c = f"{sens_cache_dir}/clara_dalpha_{month_name}_{year}.npy"
            p_a = f"{sens_cache_dir}/appx_dalpha_{month_name}_{year}.npy"
            
            if os.path.exists(p_c): vals.append(float(np.load(p_c)))
            if os.path.exists(p_a): vals.append(float(np.load(p_a)))
            
            if vals: dalpha_rows.append({'month_name': month_name, 'year': year, 'dalpha': float(np.mean(vals)), 'n_datasets': len(vals)})

    df_dalpha = pd.DataFrame(dalpha_rows)
    df_dalpha.to_csv(f"{sens_cache_dir}/combined_dalpha.csv", index=False)
    print(f"  Saved combined_dalpha.csv ({len(df_dalpha)} rows)")

    # -------------------------------------------------------------------------
    # STEP 5: SIC Climatologies & Per-pixel OLS Trends
    # -------------------------------------------------------------------------
    print(f"\nStep 5: SIC Climatologies & Trends ({min(TREND_YEARS)}-{max(TREND_YEARS)})...")
    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        out_slope_sial = f"{sens_cache_dir}/sic_trend_slope_sial_{month_name}.npy"
        out_pval_sial = f"{sens_cache_dir}/sic_trend_pval_sial_{month_name}.npy"
        out_slope_appx = f"{sens_cache_dir}/sic_trend_slope_appx_{month_name}.npy"
        out_pval_appx = f"{sens_cache_dir}/sic_trend_pval_appx_{month_name}.npy"

        print(f"  {month_name} (month {month_num})...", end='', flush=True)
        year_sic_psn, valid_trend_years = [], []
        
        for year in TREND_YEARS:
            daily_maps = []
            for f in sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc")):
                try:
                    with xr.open_dataset(f) as ds:
                        if pd.Timestamp(ds['time'].values[0]).month != month_num: continue
                        raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                        if raw.ndim > 2: raw = raw[0]
                        if not np.any(np.isfinite(raw)): continue
                        if np.nanmax(raw) > 1.5: raw /= 100.0
                        
                        # Strict physical bounds to filter out land/coast flags in climos/trends
                        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
                        
                        if np.sum(np.isfinite(raw[arctic_60N_mask_psn])) < MIN_ARCTIC_PX: continue
                        daily_maps.append(raw)
                except Exception: continue
                
            if daily_maps:
                year_sic_psn.append(np.nanmean(daily_maps, axis=0))
                valid_trend_years.append(year)

        if len(valid_trend_years) < 10:
            print(f" skip (only {len(valid_trend_years)} years)")
            continue

        sic_stack = np.array(year_sic_psn)
        slope_psn, pval_psn = np.full(sic_shape, np.nan), np.full(sic_shape, np.nan)
        flat_sic = sic_stack.reshape(sic_stack.shape[0], -1)

        # 1. Identify indices for EARLY and LATE years
        early_indices = [i for i, y in enumerate(valid_trend_years) if y in EARLY_YEARS]
        late_indices  = [i for i, y in enumerate(valid_trend_years) if y in LATE_YEARS]
        
        # 2. Calculate mean SIC for both periods
        early_mean_sic = np.nanmean(flat_sic[early_indices, :], axis=0) if early_indices else np.zeros(flat_sic.shape[1])
        late_mean_sic  = np.nanmean(flat_sic[late_indices, :], axis=0) if late_indices else np.zeros(flat_sic.shape[1])
            
        # 3. Apply the >= 15% constraint to the Climatologies and Save them
        # Everything below 15% becomes NaN, isolating the true ice pack for that era.
        early_masked_psn = np.where(early_mean_sic >= SIC_ICE_THRESHOLD, early_mean_sic, np.nan)
        late_masked_psn  = np.where(late_mean_sic >= SIC_ICE_THRESHOLD, late_mean_sic, np.nan)
        
        # Save SIAL regridded climos
        np.save(f"{sens_cache_dir}/sic_early_sial_{month_name}.npy", regrid_psn_to_sial(early_masked_psn))
        np.save(f"{sens_cache_dir}/sic_late_sial_{month_name}.npy", regrid_psn_to_sial(late_masked_psn))
        
        # Save APP-x regridded climos
        np.save(f"{sens_cache_dir}/sic_early_appx_{month_name}.npy", regrid_psn_to_appx(early_masked_psn))
        np.save(f"{sens_cache_dir}/sic_late_appx_{month_name}.npy", regrid_psn_to_appx(late_masked_psn))

        # 4. Filter for the Ever-Ice Mask for trends
        good_px = np.where(
            (np.sum(np.isfinite(flat_sic), axis=0) >= 10) & 
            ((early_mean_sic >= SIC_ICE_THRESHOLD) | (late_mean_sic >= SIC_ICE_THRESHOLD))
        )[0]

        for px in good_px:
            y, mask = flat_sic[:, px], np.isfinite(flat_sic[:, px])
            if mask.sum() < 10: continue
            result = sp_stats.linregress(np.array(valid_trend_years, dtype=float)[mask], y[mask])
            slope_psn.ravel()[px], pval_psn.ravel()[px] = result.slope, result.pvalue

        # Save trend outputs
        np.save(out_slope_sial, regrid_psn_to_sial(slope_psn))
        np.save(out_pval_sial, regrid_psn_to_sial(pval_psn))
        np.save(out_slope_appx, regrid_psn_to_appx(slope_psn))
        np.save(out_pval_appx, regrid_psn_to_appx(pval_psn))
        
        n_sig = int(np.sum((pval_psn < 0.05) & (slope_psn < 0) & arctic_60N_mask_psn & np.isfinite(slope_psn)))
        print(f" ✓ ({n_sig} sig pixels)")
        
        del sic_stack, flat_sic
        gc.collect()

    print(f"\n  ✓ {exp['label']} complete → {sens_cache_dir}")

print("\n" + "=" * 70)
print("All experiments complete.")