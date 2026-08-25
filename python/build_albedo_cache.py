"""
build_albedo_cache.py
---------------------
Builds sensitivity-sweep albedo caches for TWO analysis periods using ONLY
APP-x (AVHRR) data to maintain strict sensor consistency:

  (1) Full observational era  : TREND 1982-2024
  (2) CMIP6 historical period : TREND 1982-2014

Sensor assignment:
  September  → APP-x only
  Apr–Aug    → CLARA-A3 primary; APP-x fills pixels where CLARA-A3 is NaN

Albedo sensitivity is expressed as dα/dT (alpha K⁻¹) via direct OLS
regression of the per-pixel albedo against global annual mean temperature
for each year in the trend window.  This is robust to non-linear warming
because temperature is the explicit X variable rather than time.

Outputs
-------
  /glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/
  /glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/

  Key output files per experiment:
    merged_alb_dalphadT_{month}.npy  — per-pixel dα/dT [alpha K⁻¹]
    T_annual.npy                     — ensemble-mean T anomaly for each trend year
    T_annual_years.npy               — integer years matching T_annual
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
        'early_years' : list(range(1982, 1992)),   # kept for SIC climo only
        'late_years'  : list(range(2015, 2025)),   # kept for SIC climo only
        'trend_years' : list(range(1982, 2025)),
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/',
    },
    'cmip6': {
        'label'       : 'CMIP6 historical period (1982-2014)',
        'early_years' : list(range(1982, 1992)),
        'late_years'  : list(range(2005, 2015)),
        'trend_years' : list(range(1982, 2015)),
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/',
    },
}

# =============================================================================
# SHARED CONFIG
# =============================================================================

dir_sial        = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
dir_appx        = "/glade/derecho/scratch/jrotondo/APP-x_observations/"
dir_sic         = "/glade/derecho/scratch/jrotondo/SIC_data/"

# Temperature data paths (same sources as siaf_analysis.py)
GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

SIC_ICE_THRESHOLD = 0.15
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

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# =============================================================================
# TEMPERATURE ENSEMBLE
# =============================================================================

def build_T_annual_ensemble(trend_years):
    """
    Load annual global-mean temperature from all available sources, center
    each on the trend-period mean (so OLS slopes are comparable across
    sources regardless of whether T is absolute or anomaly), then return
    the ensemble mean as a {year: T_anomaly} dict.

    Centering does not affect OLS slopes — only the regression intercept.
    """
    raw_sources = {}   # name → {year: value}

    # GISTEMP (anomaly, deg C)
    try:
        df = pd.read_csv(GISTEMP_PATH, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        raw_sources['GISTEMP'] = {
            int(r['Year']): float(r['J-D'])
            for _, r in df.iterrows() if np.isfinite(r['J-D'])
        }
        print(f'  T: GISTEMP loaded ({len(raw_sources["GISTEMP"])} years)')
    except Exception as e:
        print(f'  T: GISTEMP skipped ({e})')

    # Cowtan & Way (anomaly, deg C)
    try:
        df = pd.read_csv(COWTANWAY_PATH, sep=r'\s+', header=None,
                         names=['year', 'anomaly', 'u1', 'u2', 'u3'])
        raw_sources['CowtanWay'] = {
            int(r['year']): float(r['anomaly'])
            for _, r in df.iterrows() if np.isfinite(r['anomaly'])
        }
        print(f'  T: Cowtan-Way loaded ({len(raw_sources["CowtanWay"])} years)')
    except Exception as e:
        print(f'  T: Cowtan-Way skipped ({e})')

    # NCEP (absolute, K — will be centered)
    try:
        ds = xr.open_dataset(NCEP_PATH)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat', 'lon']).resample(time='1YE').mean()
        raw_sources['NCEP'] = {
            int(yr): float(v)
            for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
            if np.isfinite(v)
        }
        print(f'  T: NCEP loaded ({len(raw_sources["NCEP"])} years)')
    except Exception as e:
        print(f'  T: NCEP skipped ({e})')

    # ERA5 (K — will be centered)
    try:
        files = sorted(glob.glob(ERA5_GLOB))
        if files:
            ds = xr.open_mfdataset(files, combine='by_coords')
            w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
            ta = ds['VAR_2T'].weighted(w).mean(
                dim=['latitude', 'longitude']).resample(time='1YE').mean()
            raw_sources['ERA5'] = {
                int(yr): float(v)
                for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
                if np.isfinite(v)
            }
            print(f'  T: ERA5 loaded ({len(raw_sources["ERA5"])} years)')
    except Exception as e:
        print(f'  T: ERA5 skipped ({e})')

    if not raw_sources:
        raise RuntimeError('No temperature data sources found.')

    # Center each source on the trend-period mean so they are comparable
    centered = {}
    for name, yr_dict in raw_sources.items():
        common = [y for y in trend_years if y in yr_dict]
        if len(common) < 10:
            print(f'  T: {name} has only {len(common)} trend-period years — skipped')
            continue
        mean_val = float(np.mean([yr_dict[y] for y in common]))
        centered[name] = {y: v - mean_val for y, v in yr_dict.items()}

    if not centered:
        raise RuntimeError('No T source has enough trend-period coverage.')

    # Ensemble mean per year
    ensemble = {}
    for y in trend_years:
        vals = [v[y] for v in centered.values() if y in v]
        if vals:
            ensemble[y] = float(np.mean(vals))

    missing = [y for y in trend_years if y not in ensemble]
    if missing:
        print(f'  T: WARNING — no T data for years: {missing}')
    print(f'  T: ensemble built from {list(centered.keys())}, '
          f'{len(ensemble)}/{len(trend_years)} trend years covered')
    return ensemble

# =============================================================================
# GRID / REGRID HELPERS
# =============================================================================

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
        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
    return t, raw

# =============================================================================
# VECTORIZED OLS: per-pixel regression of a 3D stack against a 1D predictor
# =============================================================================

def compute_pixel_ols_slopes(stk, x_vec):
    """
    Per-pixel OLS slope of stk ~ x_vec.

    Works for any 1-D predictor (time, temperature, etc.).  NaN pixels are
    excluded per-column so different pixels can have different missing years.

    Parameters
    ----------
    stk   : (T, H, W) float array — dependent variable (e.g. albedo)
    x_vec : (T,) float array      — predictor (e.g. annual T anomaly)

    Returns
    -------
    slope_map : (H, W) float64; NaN where fewer than 10 valid pairs
    """
    T, H, W = stk.shape
    flat    = stk.reshape(T, -1).astype(np.float64)
    N       = flat.shape[1]
    x       = x_vec.astype(np.float64).reshape(-1, 1)   # (T, 1)

    valid   = np.isfinite(flat)                          # (T, N)
    n_valid = valid.sum(axis=0).astype(np.float64)       # (N,)
    flat_z  = np.where(valid, flat, 0.0)
    x_bc    = np.broadcast_to(x, (T, N))                # (T, N) read-only
    x_z     = np.where(valid, x_bc, 0.0)

    y_mean  = flat_z.sum(0) / np.maximum(n_valid, 1)
    x_mean  = x_z.sum(0)   / np.maximum(n_valid, 1)

    flat_c  = np.where(valid, flat  - y_mean, 0.0)
    x_c     = np.where(valid, x_bc  - x_mean, 0.0)

    num     = (x_c * flat_c).sum(0)
    den     = (x_c ** 2).sum(0)

    out     = np.full(N, np.nan, np.float64)
    good    = (n_valid >= 10) & (den > 0)
    out[good] = num[good] / den[good]
    return out.reshape(H, W)

# =============================================================================
# STEP 0: BUILD GRIDS
# =============================================================================

print("=" * 70)
print("STEP 0: Building grids")
print("=" * 70)

# SIAL (CLARA-A3) grid
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
sial_coslat = np.cos(np.deg2rad(sial_lat2d)).astype(float)
sial_coslat = np.where(sial_coslat > 0, sial_coslat, 0.0)

# APP-x grid
sample_appx = sorted(glob.glob(f"{dir_appx}/**/*.nc", recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill    = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if appx_lat_raw.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
appx_shape   = appx_lat2d.shape
appx_xyz     = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())
appx_weights = np.ones(appx_shape, dtype=float)

# PSN25 (SIC) grid
sample_sic = sorted(glob.glob(f"{dir_sic}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values
    sic_y = ds['y'].values
sic_xx, sic_yy   = np.meshgrid(sic_x, sic_y)
transformer       = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
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

# APP-x → SIAL index (for sensor merge in Step 4)
appx_tree = cKDTree(appx_xyz)
_, idx_appx_to_sial = appx_tree.query(sial_xyz, workers=-1)

def regrid_appx_to_sial(appx_2d):
    return appx_2d.ravel()[idx_appx_to_sial].reshape(sial_shape)

# Arctic masks (60 N+)
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
    ALL_YEARS      = sorted(TREND_YEARS)
    sens_cache_dir = exp['cache_dir']
    os.makedirs(sens_cache_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp['label']}")
    print("=" * 70)

    np.save(f"{sens_cache_dir}/sial_shape.npy",      np.array(sial_shape))
    np.save(f"{sens_cache_dir}/appx_shape.npy",      np.array(appx_shape))
    np.save(f"{sens_cache_dir}/arctic_60N_sial.npy", arctic_60N_sial)
    np.save(f"{sens_cache_dir}/arctic_60N_appx.npy", arctic_60N_appx)
    np.save(f"{sens_cache_dir}/appx_fill.npy",       np.array(appx_fill))
    np.save(f"{sens_cache_dir}/sial_coslat.npy",     sial_coslat)
    with open(f"{sens_cache_dir}/area_ratio.txt", 'w') as fh:
        fh.write(str(AREA_RATIO_60N))

    # ------------------------------------------------------------------
    # Build global annual temperature ensemble for this trend window
    # ------------------------------------------------------------------
    print("\nLoading temperature ensemble ...")
    T_annual_dict = build_T_annual_ensemble(TREND_YEARS)

    # Save for use by siaf_analysis.py bootstrap
    T_years_arr = np.array([y for y in TREND_YEARS if y in T_annual_dict], dtype=int)
    T_vals_arr  = np.array([T_annual_dict[y] for y in T_years_arr], dtype=np.float64)
    np.save(f"{sens_cache_dir}/T_annual.npy",       T_vals_arr)
    np.save(f"{sens_cache_dir}/T_annual_years.npy", T_years_arr)
    print(f"  Saved T_annual.npy  ({len(T_years_arr)} years, "
          f"range [{T_vals_arr.min():.3f}, {T_vals_arr.max():.3f}] K)")

    # -------------------------------------------------------------------------
    # STEP 2: CLARA-A3 — save per-year masked albedo maps for ALL trend years
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading CLARA-A3 albedo maps...")
    for year in ALL_YEARS:
        print(f"  CLARA {year}...", end='', flush=True)

        if all(os.path.exists(f"{sens_cache_dir}/clara_alb_{mn}_{year}.npy")
               for mn in MONTHS_TO_ANALYZE):
            print(" (cached, skip)")
            continue

        sial_files = [f for f in sorted(glob.glob(f"{dir_sial}/*.nc")) if str(year) in f]
        if not sial_files:
            print(" skip (no files)")
            continue

        try:
            ds_sial     = xr.open_mfdataset(sial_files, combine='by_coords')
            snow_alb    = ds_sial['black_sky_albedo_snow_mean'] / 100.0
            reg_alb     = ds_sial['black_sky_albedo_mean'] / 100.0
            sial_raw    = xr.where(~np.isnan(snow_alb), snow_alb, reg_alb)
            sial_pentad = sial_raw.resample(time=f'{PENTAD_DAYS}D').mean().load()
            ds_sial.close()
        except Exception as e:
            print(f" skip ({e})")
            continue

        sic_daily = {}
        for f in sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc")):
            try:
                t, raw = load_sic_day(f)
                sic_daily[t.date()] = regrid_sic_to_sial(raw)
            except Exception:
                continue

        sic_pentad_dict = {}
        for ptime in sial_pentad.time.values:
            pd_date  = pd.Timestamp(ptime).date()
            bin_days = [pd_date + dt_mod.timedelta(days=d) for d in range(PENTAD_DAYS)]
            maps     = [sic_daily[d] for d in bin_days if d in sic_daily]
            if maps:
                sic_pentad_dict[pd_date] = np.nanmean(maps, axis=0)

        for month_name, month_num in MONTHS_TO_ANALYZE.items():
            cache_path = f"{sens_cache_dir}/clara_alb_{month_name}_{year}.npy"
            if os.path.exists(cache_path):
                continue

            sial_month = sial_pentad.sel(time=sial_pentad['time.month'] == month_num)
            if len(sial_month.time) == 0:
                continue

            sial_mean = sial_month.mean(dim='time').values
            sic_maps  = [sic_pentad_dict[pd.Timestamp(t).date()]
                         for t in sial_month.time.values
                         if pd.Timestamp(t).date() in sic_pentad_dict]
            if not sic_maps:
                continue
            sic_mean = np.nanmean(sic_maps, axis=0)

            valid_domain = arctic_60N_sial
            sial_mean    = np.where(valid_domain, sial_mean, np.nan)
            sic_mean     = np.where(valid_domain, sic_mean,  np.nan)
            valid_px     = np.isfinite(sial_mean) & np.isfinite(sic_mean) & valid_domain

            if valid_px.sum() < MIN_ARCTIC_PX:
                continue

            np.save(cache_path, np.where(valid_px, sial_mean, np.nan))

        print(" ✓")
        del sial_pentad, sic_daily
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: APP-x — save per-year masked albedo maps for ALL trend years
    # -------------------------------------------------------------------------
    print("\nStep 3: Loading APP-x albedo maps...")

    def load_appx_monthly(year, month_num):
        base   = dir_appx.rstrip('/')
        prefix = "Polar-APP-X"
        search_paths = [
            f"{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc",
            f"{base}/{prefix}*d{year}{month_num:02d}*.nc",
        ]
        files = sorted(list(set(f for p in search_paths for f in glob.glob(p))))
        if not files:
            return None

        files_by_date = {}
        for f in files:
            obs_date = parse_date_appx(f)
            if obs_date is None:
                continue
            obs_str  = obs_date.strftime('%Y%m%d')
            m_time   = re.search(r'_(\d{4})_d', os.path.basename(f))
            overpass = m_time.group(1) if m_time else 'unknown'
            m_cdate  = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
            cdate    = m_cdate.group(1) if m_cdate else '00000000'
            files_by_date.setdefault(obs_str, {})
            if overpass not in files_by_date[obs_str] or \
               cdate > files_by_date[obs_str][overpass][1]:
                files_by_date[obs_str][overpass] = (f, cdate)

        monthly_daily_maps = []
        for obs_str, passes in sorted(files_by_date.items()):
            daily_passes = []
            for overpass, (f, cdate) in passes.items():
                try:
                    with NC4Dataset(f, 'r') as ds:
                        var_name = next(
                            (v for v in ds.variables
                             if 'albedo' in v.lower() and ds.variables[v].ndim >= 2),
                            None)
                        if not var_name:
                            continue
                        alb_var   = ds.variables[var_name]
                        alb       = alb_var[0] if alb_var.ndim == 3 else alb_var[:]
                        if hasattr(alb, 'filled'):
                            alb = alb.filled(np.nan)
                        alb       = alb.astype(float)
                        file_fill = float(getattr(alb_var, '_FillValue', appx_fill))
                        alb[alb == file_fill] = np.nan
                        alb[(alb < 0) | (alb > 1)] = np.nan
                        daily_passes.append(alb)
                except Exception:
                    continue
            if daily_passes:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    monthly_daily_maps.append(np.nanmean(daily_passes, axis=0))

        if not monthly_daily_maps:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(monthly_daily_maps, axis=0)

    def load_sic_monthly_appx(year, month_num):
        maps  = []
        files = sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc"))
        if not files:
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
                    raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
                    if np.sum(np.isfinite(raw[arctic_60N_mask_psn])) < MIN_ARCTIC_PX:
                        continue
                    maps.append(regrid_sic_to_appx(raw))
            except Exception:
                continue
        return np.nanmean(maps, axis=0) if maps else None

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
            alb_mean     = np.where(valid_domain, alb_mean, np.nan)
            sic_mean     = np.where(valid_domain, sic_mean, np.nan)
            valid_px     = np.isfinite(alb_mean) & np.isfinite(sic_mean) & valid_domain

            if valid_px.sum() < MIN_ARCTIC_PX:
                continue

            np.save(cache_path, np.where(valid_px, alb_mean, np.nan))
            found_any = True

        print(" ✓" if found_any else " skip")
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 4: Merge sensors + compute per-pixel dα/dT via direct α~T regression
    #
    # Sensor assignment:
    #   September → APP-x only (regridded to SIAL)
    #   Apr–Aug   → CLARA-A3 primary; APP-x fills NaN pixels
    #
    # For each pixel, OLS regresses the merged monthly albedo time series
    # against the global annual mean temperature anomaly for the same years.
    # The slope is dα/dT [alpha K⁻¹] — negative where warming causes darkening.
    #
    # Output: merged_alb_dalphadT_{month}.npy  (SIAL grid)
    # -------------------------------------------------------------------------
    print("\nStep 4: Merging sensors and computing per-pixel dα/dT (α~T regression)...")

    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        out_path = f"{sens_cache_dir}/merged_alb_dalphadT_{month_name}.npy"
        if os.path.exists(out_path):
            print(f"  {month_name}: (cached, skip)")
            continue

        use_appx_only = (month_name == 'September')
        yearly_maps   = {}

        for year in TREND_YEARS:
            p_c = f"{sens_cache_dir}/clara_alb_{month_name}_{year}.npy"
            p_a = f"{sens_cache_dir}/appx_alb_{month_name}_{year}.npy"

            if use_appx_only:
                if not os.path.exists(p_a):
                    continue
                yearly_maps[year] = regrid_appx_to_sial(
                    np.load(p_a).astype(np.float64))
            else:
                clara_map, appx_map = None, None
                if os.path.exists(p_c):
                    clara_map = np.load(p_c).astype(np.float64)
                if os.path.exists(p_a):
                    appx_map  = regrid_appx_to_sial(np.load(p_a).astype(np.float64))

                if clara_map is not None and appx_map is not None:
                    merged = np.where(np.isfinite(clara_map), clara_map, appx_map)
                elif clara_map is not None:
                    merged = clara_map
                elif appx_map is not None:
                    merged = appx_map
                else:
                    continue
                yearly_maps[year] = merged

        # Align years: only keep years with BOTH an albedo map AND a T value
        valid_years = sorted(y for y in yearly_maps if y in T_annual_dict)
        if len(valid_years) < 10:
            print(f"  {month_name}: skip (only {len(valid_years)} years with "
                  f"both albedo and T data)")
            continue

        stk   = np.stack([yearly_maps[y] for y in valid_years], axis=0)  # (T, H, W)
        T_vec = np.array([T_annual_dict[y] for y in valid_years])          # (T,)

        # Per-pixel OLS: α ~ T_global
        dalpha_dT = compute_pixel_ols_slopes(stk, T_vec)   # [alpha K⁻¹], negative = feedback
        np.save(out_path, dalpha_dT)

        n_neg = int(np.sum((dalpha_dT < 0) & arctic_60N_sial & np.isfinite(dalpha_dT)))
        print(f"  {month_name}: {len(valid_years)} years matched with T, "
              f"dα/dT range [{np.nanmin(dalpha_dT):.5f}, {np.nanmax(dalpha_dT):.5f}] K⁻¹, "
              f"{n_neg} pixels with negative slope (positive feedback)")

        del stk, yearly_maps
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 5: SIC Climatologies & Per-pixel OLS Trends (unchanged)
    # -------------------------------------------------------------------------
    print(f"\nStep 5: SIC Climatologies & Trends ({min(TREND_YEARS)}-{max(TREND_YEARS)})...")
    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        out_slope_sial = f"{sens_cache_dir}/sic_trend_slope_sial_{month_name}.npy"
        out_pval_sial  = f"{sens_cache_dir}/sic_trend_pval_sial_{month_name}.npy"
        out_slope_appx = f"{sens_cache_dir}/sic_trend_slope_appx_{month_name}.npy"
        out_pval_appx  = f"{sens_cache_dir}/sic_trend_pval_appx_{month_name}.npy"

        print(f"  {month_name} (month {month_num})...", end='', flush=True)
        year_sic_psn, valid_trend_years = [], []

        for year in TREND_YEARS:
            daily_maps = []
            for f in sorted(glob.glob(f"{dir_sic}/sic_psn25_{year}*.nc")):
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
                        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
                        if np.sum(np.isfinite(raw[arctic_60N_mask_psn])) < MIN_ARCTIC_PX:
                            continue
                        daily_maps.append(raw)
                except Exception:
                    continue
            if daily_maps:
                year_sic_psn.append(np.nanmean(daily_maps, axis=0))
                valid_trend_years.append(year)

        if len(valid_trend_years) < 10:
            print(f" skip (only {len(valid_trend_years)} years)")
            continue

        sic_stack = np.array(year_sic_psn)
        slope_psn = np.full(sic_shape, np.nan)
        pval_psn  = np.full(sic_shape, np.nan)
        flat_sic  = sic_stack.reshape(sic_stack.shape[0], -1)

        early_indices = [i for i, y in enumerate(valid_trend_years) if y in EARLY_YEARS]
        late_indices  = [i for i, y in enumerate(valid_trend_years) if y in LATE_YEARS]

        early_mean_sic = (np.nanmean(flat_sic[early_indices, :], axis=0)
                          if early_indices else np.zeros(flat_sic.shape[1]))
        late_mean_sic  = (np.nanmean(flat_sic[late_indices, :], axis=0)
                          if late_indices else np.zeros(flat_sic.shape[1]))

        early_masked_psn = np.where(early_mean_sic >= SIC_ICE_THRESHOLD, early_mean_sic, np.nan)
        late_masked_psn  = np.where(late_mean_sic  >= SIC_ICE_THRESHOLD, late_mean_sic,  np.nan)

        np.save(f"{sens_cache_dir}/sic_early_sial_{month_name}.npy", regrid_psn_to_sial(early_masked_psn))
        np.save(f"{sens_cache_dir}/sic_late_sial_{month_name}.npy",  regrid_psn_to_sial(late_masked_psn))
        np.save(f"{sens_cache_dir}/sic_early_appx_{month_name}.npy", regrid_psn_to_appx(early_masked_psn))
        np.save(f"{sens_cache_dir}/sic_late_appx_{month_name}.npy",  regrid_psn_to_appx(late_masked_psn))

        good_px = np.where(
            (np.sum(np.isfinite(flat_sic), axis=0) >= 10) &
            ((early_mean_sic >= SIC_ICE_THRESHOLD) | (late_mean_sic >= SIC_ICE_THRESHOLD))
        )[0]

        for px in good_px:
            y, mask = flat_sic[:, px], np.isfinite(flat_sic[:, px])
            if mask.sum() < 10:
                continue
            result = sp_stats.linregress(
                np.array(valid_trend_years, dtype=float)[mask], y[mask])
            slope_psn.ravel()[px] = result.slope
            pval_psn.ravel()[px]  = result.pvalue

        np.save(out_slope_sial, regrid_psn_to_sial(slope_psn))
        np.save(out_pval_sial,  regrid_psn_to_sial(pval_psn))
        np.save(out_slope_appx, regrid_psn_to_appx(slope_psn))
        np.save(out_pval_appx,  regrid_psn_to_appx(pval_psn))

        n_sig = int(np.sum(
            (pval_psn < 0.05) & (slope_psn < 0) &
            arctic_60N_mask_psn & np.isfinite(slope_psn)))
        print(f" ✓ ({n_sig} sig pixels)")

        del sic_stack, flat_sic
        gc.collect()

    print(f"\n  ✓ {exp['label']} complete → {sens_cache_dir}")

print("\n" + "=" * 70)
print("All experiments complete.")
