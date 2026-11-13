"""
build_albedo_cache_ocean_only.py
--------------------------------
SIC-INDEPENDENT variant of build_albedo_cache.py.

Original cache used `np.isfinite(sic_mean)` as part of the per-pixel
validity check, so the saved albedo maps secretly inherited the PSN25
SIC land mask.  This version replaces that with an external, static
land-sea mask so the cached albedo domain is truly independent of any
sea-ice product:

    is_ocean(lat, lon)  →  True over open water and sea ice
                        →  False over land

Two ways the ocean mask can be sourced (controlled by OCEAN_MASK_SOURCE
at the top of the script):

  'global_land_mask'  → uses the `global-land-mask` pip package, which
                        ships a 1-arcminute Natural-Earth-derived land
                        grid as a bundled .npz.  pip install global-land-mask
  'appx_surface_type' → uses the APP-x cdr_surface_type variable from
                        a sample APP-x file (1 = water/ice, others = land
                        per the APP-x CDR convention).  No new pip deps.

If the chosen source fails, the script raises so you don't silently fall
back to a SIC-based mask.

Outputs that change vs the original cache builder
-------------------------------------------------
  ocean_mask_sial.npy   : True over ocean (incl. sea ice) on SIAL grid
  ocean_mask_appx.npy   : True over ocean (incl. sea ice) on APP-x grid
  clara_alb_{month}_{year}.npy : albedo masked by ocean & ≥60°N only
  appx_alb_{month}_{year}.npy  : albedo masked by ocean & ≥60°N only

Outputs that are NO LONGER WRITTEN (they were SIC-derived):
  sic_early_*, sic_late_*, sic_trend_slope_*, sic_trend_pval_*,
  clara_dalpha_*, appx_dalpha_*, combined_dalpha.csv

The diagnose_regridding_ocean_only.py script does not need any of those.
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
import glob, os, gc, re
import datetime as dt_mod

# =============================================================================
# OCEAN-MASK SOURCE
# =============================================================================
# Choose ONE.  If you don't have global-land-mask installed and don't want
# to pip-install on the cluster, switch to 'appx_surface_type'.
OCEAN_MASK_SOURCE = 'global_land_mask'   # or 'appx_surface_type'

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================
EXPERIMENTS = {
    'obs': {
        'label'       : 'Full observational era (1982-2024)',
        'early_years' : list(range(1982, 1992)),
        'late_years'  : list(range(2015, 2025)),
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/',
    },
    'cmip6': {
        'label'       : 'CMIP6 historical period (1982-2014)',
        'early_years' : list(range(1982, 1992)),
        'late_years'  : list(range(2005, 2015)),
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/',
    },
}

# =============================================================================
# SHARED CONFIG
# =============================================================================
dir_sial        = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
dir_appx        = "/glade/derecho/scratch/jrotondo/APP-x_observations/"

PENTAD_DAYS       = 5
MIN_ARCTIC_PX     = 100
AREA_RATIO_60N    = 0.065

MONTHS_TO_ANALYZE = {
    'April'    : 4,
    'May'      : 5,
    'June'     : 6,
    'July'     : 7,
    'August'   : 8,
    'September': 9,
}

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
sial_shape  = sial_lat2d.shape
sial_xyz    = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
sial_coslat = np.where(np.cos(np.deg2rad(sial_lat2d)) > 0,
                       np.cos(np.deg2rad(sial_lat2d)), 0.0).astype(float)

# APP-x grid
sample_appx = sorted(glob.glob(f"{dir_appx}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if appx_lat_raw.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
appx_shape   = appx_lat2d.shape
appx_xyz     = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
appx_weights = np.ones(appx_shape, dtype=float)

# Arctic ≥60°N masks (purely geometric, no SIC)
arctic_60N_sial = (sial_lat2d >= 60.0)
arctic_60N_appx = (appx_lat2d >= 60.0)

print(f"  SIAL grid : {sial_shape}")
print(f"  APP-x grid: {appx_shape}")

# =============================================================================
# STEP 0b: BUILD A SIC-INDEPENDENT OCEAN MASK ON BOTH GRIDS
# =============================================================================
print(f"\nBuilding ocean mask via OCEAN_MASK_SOURCE = '{OCEAN_MASK_SOURCE}'")

def _wrap_lon_to_pm180(lon2d):
    return np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

if OCEAN_MASK_SOURCE == 'global_land_mask':
    try:
        from global_land_mask import globe
    except ImportError as e:
        raise ImportError(
            "global-land-mask is not installed.  Install with\n"
            "    pip install global-land-mask\n"
            "or switch OCEAN_MASK_SOURCE to 'appx_surface_type'."
        ) from e

    sial_lon_pm = _wrap_lon_to_pm180(sial_lon2d)
    appx_lon_pm = _wrap_lon_to_pm180(appx_lon2d)
    ocean_sial  = globe.is_ocean(sial_lat2d, sial_lon_pm)
    ocean_appx  = globe.is_ocean(appx_lat2d, appx_lon_pm)

elif OCEAN_MASK_SOURCE == 'appx_surface_type':
    # APP-x cdr_surface_type convention (per CDR ATBD):
    #   1 = water,  2 = sea ice,  3 = bare land / desert,  4 = snow on land,
    #   5 = land ice, ...  i.e. ocean = (st in {1, 2}).
    # We grab a single APP-x file (preferably summer northern hemisphere)
    # to get the static surface_type → land/ocean mapping, then nearest-
    # neighbour resample to SIAL.  Surface type can change seasonally for
    # snow-on-land (4 → 3) but the water/sea-ice → land boundary itself
    # is static in the APP-x grid, which is all we need here.
    st = None
    for f in sorted(glob.glob(f"{dir_appx}/**/*.nc", recursive=True))[:200]:
        try:
            with NC4Dataset(f, 'r') as ds:
                if 'cdr_surface_type' not in ds.variables:
                    continue
                v = ds.variables['cdr_surface_type']
                arr = v[0] if v.ndim == 3 else v[:]
                if hasattr(arr, 'filled'):
                    arr = arr.filled(-1)
                st = np.asarray(arr).astype(np.int16)
                break
        except Exception:
            continue
    if st is None:
        raise RuntimeError(
            "Could not find an APP-x file containing 'cdr_surface_type'. "
            "Switch OCEAN_MASK_SOURCE to 'global_land_mask' instead."
        )
    # Ocean = water OR sea ice (the APP-x land/ocean boundary)
    ocean_appx = np.isin(st, [1, 2])

    # Resample APP-x → SIAL via NN
    appx_tree = cKDTree(appx_xyz)
    _, idx_appx_for_sial = appx_tree.query(sial_xyz, workers=-1)
    ocean_sial = ocean_appx.ravel()[idx_appx_for_sial].reshape(sial_shape)
    del appx_tree, idx_appx_for_sial

else:
    raise ValueError(f"Unknown OCEAN_MASK_SOURCE: {OCEAN_MASK_SOURCE!r}")

# Final domain masks: ocean AND ≥60°N
domain_sial = ocean_sial & arctic_60N_sial
domain_appx = ocean_appx & arctic_60N_appx

print(f"  SIAL  ocean+≥60°N pixels : {int(domain_sial.sum()):>9,} "
      f"({100.*domain_sial.sum()/arctic_60N_sial.sum():.1f}% of ≥60°N)")
print(f"  APP-x ocean+≥60°N pixels : {int(domain_appx.sum()):>9,} "
      f"({100.*domain_appx.sum()/arctic_60N_appx.sum():.1f}% of ≥60°N)")

# =============================================================================
# MAIN LOOP
# =============================================================================
for exp_key, exp in EXPERIMENTS.items():
    EARLY_YEARS    = exp['early_years']
    LATE_YEARS     = exp['late_years']
    ALL_YEARS      = sorted(set(EARLY_YEARS + LATE_YEARS))
    sens_cache_dir = exp['cache_dir']
    os.makedirs(sens_cache_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp['label']}  [SIC-INDEPENDENT DOMAIN]")
    print("=" * 70)

    # Save grid + masks
    np.save(f"{sens_cache_dir}/sial_shape.npy",       np.array(sial_shape))
    np.save(f"{sens_cache_dir}/appx_shape.npy",       np.array(appx_shape))
    np.save(f"{sens_cache_dir}/arctic_60N_sial.npy",  arctic_60N_sial)
    np.save(f"{sens_cache_dir}/arctic_60N_appx.npy",  arctic_60N_appx)
    np.save(f"{sens_cache_dir}/ocean_mask_sial.npy",  ocean_sial)
    np.save(f"{sens_cache_dir}/ocean_mask_appx.npy",  ocean_appx)
    np.save(f"{sens_cache_dir}/appx_fill.npy",        np.array(appx_fill))
    np.save(f"{sens_cache_dir}/sial_coslat.npy",      sial_coslat)
    with open(f"{sens_cache_dir}/area_ratio.txt", 'w') as fh:
        fh.write(str(AREA_RATIO_60N))
    with open(f"{sens_cache_dir}/ocean_mask_source.txt", 'w') as fh:
        fh.write(OCEAN_MASK_SOURCE + "\n")

    # -------------------------------------------------------------------------
    # STEP 2: CLARA-A3   (NO SIC; mask = ocean ∧ ≥60°N ∧ finite albedo)
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading CLARA-A3 albedo maps (SIC-free domain)...")
    for year in ALL_YEARS:
        print(f"  CLARA {year}...", end='', flush=True)

        all_cached = all(
            os.path.exists(f"{sens_cache_dir}/clara_alb_{m}_{year}.npy")
            for m in MONTHS_TO_ANALYZE
        )
        if all_cached:
            print(" (cached, skip)")
            continue

        sial_files = [f for f in sorted(glob.glob(f"{dir_sial}/*.nc"))
                      if str(year) in f]
        if not sial_files:
            print(" skip (no files)")
            continue

        try:
            ds_sial   = xr.open_mfdataset(sial_files, combine='by_coords')
            snow_alb  = ds_sial['black_sky_albedo_snow_mean'] / 100.0
            reg_alb   = ds_sial['black_sky_albedo_mean']      / 100.0
            sial_raw  = xr.where(~np.isnan(snow_alb), snow_alb, reg_alb)
            sial_pent = sial_raw.resample(time=f'{PENTAD_DAYS}D').mean().load()
            ds_sial.close()
        except Exception as e:
            print(f" skip ({e})")
            continue

        for month_name, month_num in MONTHS_TO_ANALYZE.items():
            cache_path = f"{sens_cache_dir}/clara_alb_{month_name}_{year}.npy"
            if os.path.exists(cache_path):
                continue

            sial_month = sial_pent.sel(time=sial_pent['time.month'] == month_num)
            if len(sial_month.time) == 0:
                continue

            sial_mean = sial_month.mean(dim='time').values
            # Apply ocean ∧ ≥60°N mask only - NO SIC
            sial_mean = np.where(domain_sial, sial_mean, np.nan)

            valid_px = np.isfinite(sial_mean) & domain_sial
            if valid_px.sum() < MIN_ARCTIC_PX:
                continue

            np.save(cache_path, np.where(valid_px, sial_mean, np.nan))
        print(" ✓")
        del sial_pent
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: APP-x   (NO SIC; mask = ocean ∧ ≥60°N ∧ finite albedo)
    # -------------------------------------------------------------------------
    print("\nStep 3: Loading APP-x albedo maps (SIC-free domain)...")

    def load_appx_monthly(year, month_num):
        base   = dir_appx.rstrip('/')
        prefix = "Polar-APP-X"
        search_paths = [
            f"{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc",
            f"{base}/{prefix}*d{year}{month_num:02d}*.nc",
        ]
        files = []
        for p in search_paths:
            files.extend(glob.glob(p))
        files = sorted(set(files))
        if not files:
            return None

        # Group by (date, overpass-time), keep newest c-date per (date, overpass)
        files_by_date = {}
        for f in files:
            obs_date = parse_date_appx(f)
            if obs_date is None:
                continue
            obs_str   = obs_date.strftime('%Y%m%d')
            m_time    = re.search(r'_(\d{4})_d', os.path.basename(f))
            overpass  = m_time.group(1) if m_time else 'unknown'
            m_cdate   = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
            cdate     = m_cdate.group(1) if m_cdate else '00000000'
            files_by_date.setdefault(obs_str, {})
            cur = files_by_date[obs_str].get(overpass)
            if cur is None or cdate > cur[1]:
                files_by_date[obs_str][overpass] = (f, cdate)

        monthly_daily_maps = []
        for obs_str, passes in sorted(files_by_date.items()):
            daily_passes = []
            for overpass, (f, cdate) in passes.items():
                try:
                    with NC4Dataset(f, 'r') as ds:
                        var_name = next((v for v in ds.variables
                                         if 'albedo' in v.lower()
                                         and ds.variables[v].ndim >= 2), None)
                        if not var_name:
                            continue
                        alb_var = ds.variables[var_name]
                        alb     = alb_var[0] if alb_var.ndim == 3 else alb_var[:]
                        if hasattr(alb, 'filled'):
                            alb = alb.filled(np.nan)
                        alb = alb.astype(float)
                        file_fill = float(getattr(alb_var, '_FillValue', appx_fill))
                        alb[alb == file_fill] = np.nan
                        alb[(alb < 0) | (alb > 1)] = np.nan
                        daily_passes.append(alb)
                except Exception:
                    continue
            if daily_passes:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    daily_mean = np.nanmean(daily_passes, axis=0)
                monthly_daily_maps.append(daily_mean)

        if not monthly_daily_maps:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(monthly_daily_maps, axis=0)

    for year in ALL_YEARS:
        print(f"  APP-x {year}...", end='', flush=True)
        found_any = False

        for month_name, month_num in MONTHS_TO_ANALYZE.items():
            cache_path = f"{sens_cache_dir}/appx_alb_{month_name}_{year}.npy"
            if os.path.exists(cache_path):
                found_any = True
                continue

            alb_mean = load_appx_monthly(year, month_num)
            if alb_mean is None:
                continue

            # Apply ocean ∧ ≥60°N mask only - NO SIC
            alb_mean = np.where(domain_appx, alb_mean, np.nan)

            valid_px = np.isfinite(alb_mean) & domain_appx
            if valid_px.sum() < MIN_ARCTIC_PX:
                continue

            np.save(cache_path, np.where(valid_px, alb_mean, np.nan))
            found_any = True

        print(" ✓" if found_any else " skip")
        gc.collect()

    print(f"\n  ✓ {exp['label']} complete → {sens_cache_dir}")

print("\n" + "=" * 70)
print("All experiments complete.  Domain is SIC-INDEPENDENT.")
print("=" * 70)