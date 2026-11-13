#!/usr/bin/env python
"""
albedo_sic_discrimination.py   (v3 — sensitivity_cache {obs,cmip6})
====================================================================
Validates CLARA-A3 and APP-x surface albedo as an ice detector against
NSIDC CDR sea-ice concentration (SIC), using the caches produced by
build_albedo_cache.py.

The original question was "are SIC >= 15% pixels systematically brighter?"
That test treats a pixel at SIC = 14% identical to one at SIC = 0%, and a
pixel at SIC = 15% identical to one at SIC = 100% — which inflates AUC
because the bulk of the Arctic is saturated (all-ice or all-ocean) and
the muddled MIZ is a thin fringe.  The SIAF extent-loss signal, however,
lives *entirely* inside that fringe.  So this script layers four
MIZ-aware diagnostics on top of the binary test.

────────────────────────────────────────────────────────────────────────
EXPERIMENTS  (matches build_albedo_cache.py)
────────────────────────────────────────────────────────────────────────
  obs   : full observational era  1982-2024  (early 82-91, late 15-24)
  cmip6 : CMIP6 historical period  1982-2014  (early 82-91, late 05-14)

Each experiment writes its own sub-tree under OUTPUT_ROOT.  Only the
EARLY + LATE years exist in the caches; there is no filled middle.

────────────────────────────────────────────────────────────────────────
DATA LAYOUT  (per experiment cache directory)
────────────────────────────────────────────────────────────────────────
  clara_alb_{MonthName}_{year}.npy        (SIAL grid)
  appx_alb_{MonthName}_{year}.npy         (APP-x native grid)
  sial_shape.npy, appx_shape.npy
  arctic_60N_sial.npy, arctic_60N_appx.npy
  sial_coslat.npy, appx_fill.npy, area_ratio.txt
  sic_early_sial_{MonthName}.npy, sic_late_sial_{MonthName}.npy
  sic_early_appx_{MonthName}.npy, sic_late_appx_{MonthName}.npy
  sic_trend_{slope,pval}_{sial,appx}_{MonthName}.npy

Raw SIC monthlies are still read from DIR_SIC for per-year SIC maps.

────────────────────────────────────────────────────────────────────────
METRICS  (per product × year × month × experiment)
────────────────────────────────────────────────────────────────────────
1. BINARY (the original test)
     Δα = mean_alb[SIC>=0.15]  −  mean_alb[SIC<0.15]
     Full-Arctic ROC, AUC, Youden-optimal albedo threshold.
2. SIC-STRATIFIED BINS
     Albedo mean / std / p10 / p90 inside six SIC classes.
3. ICE-EDGE-ZONE AUC
     Same ROC/AUC as (1) but restricted to pixels within EDGE_DILATE_PX
     of the SIC = 15% contour.
4. CONTINUOUS REGRESSION
     OLS fit   alpha ≈ a + b · SIC   across all valid Arctic pixels.
5. DISTANCE-TO-EDGE PROFILE
     Signed pixel-distance d from the SIC = 15% contour; mean albedo +
     IQR per 1-pixel bin, -30 to +30 px.

Extra PERIOD-level comparison (early vs late) using the climatological
SIC maps that live in the cache, so the ice-edge is defined from the
era's own climatology rather than from a single-year SIC map.

────────────────────────────────────────────────────────────────────────
OUTPUT (per experiment)
────────────────────────────────────────────────────────────────────────
CSVs:
    albedo_discrimination_yearly.csv
    albedo_discrimination_summary.csv
    albedo_discrimination_sic_bins.csv
    albedo_discrimination_edge_zone.csv
    albedo_discrimination_regression.csv
    albedo_discrimination_distance_profile.csv
    albedo_discrimination_period.csv        (early vs late, climatol. SIC edge)
Plots:
    roc_{month}.png
    boxplot_binary_{month}.png
    boxplot_sic_bins_{month}.png
    auc_summary.png
    delta_alpha_timeseries.png
    albedo_vs_sic_scatter_{month}.png
    distance_profile_{month}.png
    period_comparison_{month}.png           (new — early vs late, full + edge)
"""

import os, glob, gc, pickle
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy import ndimage, stats
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG  (must match build_albedo_cache.py)
# ═════════════════════════════════════════════════════════════════════
EXPERIMENTS = {
    'obs': {
        'label'       : 'Full observational era (1982-2024)',
        'early_years' : list(range(1982, 1992)),   # 1982-1991
        'late_years'  : list(range(2015, 2025)),   # 2015-2024
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/',
    },
    'cmip6': {
        'label'       : 'CMIP6 historical period (1982-2014)',
        'early_years' : list(range(1982, 1992)),   # 1982-1991
        'late_years'  : list(range(2005, 2015)),   # 2005-2014
        'cache_dir'   : '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/',
    },
}

# which experiments to run (order matters only for logging)
EXPS_TO_RUN = ['obs', 'cmip6']

DIR_SIC         = "/glade/derecho/scratch/jrotondo/SIC_data/"
DIR_SIAL_SAMPLE = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
DIR_APPX_SAMPLE = "/glade/derecho/scratch/jrotondo/APP-x_observations/"
OUTPUT_ROOT     = "/glade/work/jrotondo/j_of_clim_dk_sea_ice_paper/output/albedo_disc/"

MONTHS_TO_ANALYZE = {
    'April':     4,
    'May':       5,
    'June':      6,
    'July':      7,
    'August':    8,
    'September': 9,
}
MONTH_ORDER = list(MONTHS_TO_ANALYZE.keys())

SIC_ICE_THRESHOLD = 0.15
LAT_MIN           = 60.0
MIN_ICE_PX        = 100
MIN_OCEAN_PX      = 100

# Ice-edge-zone: pixels within this many SIAL pixels of the 15% contour
EDGE_DILATE_PX = 5         # ≈ 70-140 km edge-zone band on SIAL
DIST_BIN_EDGES = np.arange(-30, 31, 1)   # distance profile bins (pixels)

# SIC bins for stratified analysis   (lower_inclusive, upper_exclusive, label)
SIC_BINS = [
    (0.00, 0.05, 'open_0-5'),
    (0.05, 0.15, 'trace_5-15'),
    (0.15, 0.30, 'MIZ-lo_15-30'),
    (0.30, 0.50, 'MIZ-hi_30-50'),
    (0.50, 0.80, 'trans_50-80'),
    (0.80, 1.01, 'cons_80-100'),
]

# Fine ROC grid
ROC_THRESHOLDS = np.linspace(0.0, 0.9, 181)

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ═════════════════════════════════════════════════════════════════════
# GRID HELPERS
# ═════════════════════════════════════════════════════════════════════
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg).astype(np.float32)
    lon = np.deg2rad(lon_deg).astype(np.float32)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)]).astype(np.float32)


def build_nn_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d):
    """Return index array `idx` such that src.ravel()[idx].reshape(tgt_shape)
       is `src` nearest-neighbour-remapped onto `tgt`."""
    tree = cKDTree(latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel()))
    _, idx = tree.query(latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel()),
                         workers=-1)
    del tree; gc.collect()
    return idx


# ═════════════════════════════════════════════════════════════════════
# STEP 1 — SIAL GRID + REGRID INDICES  (SIC→SIAL, APPx→SIAL)
#          Built once; shared across both experiments.
# ═════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1:  SIAL / APP-x / PSN25 grids + NN regrid indices")
print("=" * 72)

sial_sample = sorted(glob.glob(f"{DIR_SIAL_SAMPLE}/*.nc"))[0]
with xr.open_dataset(sial_sample) as ds:
    _lat = ds['lat'].values
    _lon = ds['lon'].values
if _lat.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(_lon, _lat)
else:
    sial_lat2d, sial_lon2d = _lat, _lon
sial_shape  = sial_lat2d.shape
sial_arctic = (sial_lat2d >= LAT_MIN)
print(f"  SIAL grid: {sial_shape}  (Arctic px = {int(sial_arctic.sum()):,})")

# SIC (PSN25) → SIAL
sic_sample = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sic_sample) as ds:
    psn_x = ds['x'].values; psn_y = ds['y'].values
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
_tfm           = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d = _tfm.transform(psn_xx, psn_yy)
psn_shape = psn_lat2d.shape
print(f"  SIC PSN25 grid: {psn_shape}")
print("  Building SIC → SIAL NN index ...", flush=True)
idx_sic_to_sial = build_nn_index(psn_lat2d, psn_lon2d, sial_lat2d, sial_lon2d)

# APP-x native → SIAL  (so per-year APP-x cache can be remapped at load time)
idx_appx_to_sial = None
appx_shape       = None
try:
    sample_appx = (sorted(glob.glob(f"{DIR_APPX_SAMPLE}/**/*.nc", recursive=True))
                    or sorted(glob.glob(f"{DIR_APPX_SAMPLE}/*.nc")))[0]
    with Dataset(sample_appx, 'r') as ds:
        appx_lat_raw = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon_raw = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat_raw.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
    else:
        appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
    appx_shape = appx_lat2d.shape
    print(f"  APP-x grid: {appx_shape}")
    print("  Building APP-x → SIAL NN index ...", flush=True)
    idx_appx_to_sial = build_nn_index(appx_lat2d, appx_lon2d,
                                       sial_lat2d, sial_lon2d)
except Exception as e:
    print(f"  WARNING: APP-x grid read failed ({e}) — "
           f"will assume cache already on SIAL shape.")


# ═════════════════════════════════════════════════════════════════════
# STEP 2 — CORE METRIC FUNCTIONS  (work on SIAL-gridded 2-D fields)
# ═════════════════════════════════════════════════════════════════════
def roc_auc_youden(alb_ice, alb_ocean, thresholds=ROC_THRESHOLDS):
    tpr = np.array([np.mean(alb_ice   >= t) for t in thresholds])
    fpr = np.array([np.mean(alb_ocean >= t) for t in thresholds])
    order = np.argsort(fpr)
    auc   = float(np.trapz(tpr[order], fpr[order]))
    J     = tpr - fpr
    i_y   = int(np.argmax(J))
    return tpr, fpr, auc, float(thresholds[i_y]), float(J[i_y])


def build_edge_zone_mask(sic2d, threshold=SIC_ICE_THRESHOLD, dilate_px=EDGE_DILATE_PX):
    """Boolean mask of pixels within `dilate_px` of the SIC=threshold contour."""
    is_ice = (sic2d >= threshold) & np.isfinite(sic2d)
    is_ocn = (sic2d <  threshold) & np.isfinite(sic2d)
    edge = (is_ice & ndimage.binary_dilation(is_ocn, iterations=1)) | \
           (is_ocn & ndimage.binary_dilation(is_ice, iterations=1))
    return ndimage.binary_dilation(edge, iterations=dilate_px)


def signed_distance_to_edge(sic2d, threshold=SIC_ICE_THRESHOLD):
    """Signed pixel-distance to the SIC=threshold contour.
       Negative inside ice, positive in ocean."""
    valid  = np.isfinite(sic2d)
    is_ice = (sic2d >= threshold) & valid
    d_out  = ndimage.distance_transform_edt(~is_ice)
    d_in   = ndimage.distance_transform_edt(is_ice)
    d = np.where(is_ice, -d_in, d_out)
    d = np.where(valid, d, np.nan)
    return d


def discrimination_full(alb2d, sic2d, domain_mask):
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(sic2d)
    m_ice = valid & (sic2d >= SIC_ICE_THRESHOLD)
    m_ocn = valid & (sic2d <  SIC_ICE_THRESHOLD)
    if m_ice.sum() < MIN_ICE_PX or m_ocn.sum() < MIN_OCEAN_PX:
        return None
    aI, aO = alb2d[m_ice], alb2d[m_ocn]
    tpr, fpr, auc, thr, J = roc_auc_youden(aI, aO)
    return dict(
        n_ice=int(m_ice.sum()), n_ocean=int(m_ocn.sum()),
        mean_alb_ice=float(np.mean(aI)),   mean_alb_ocean=float(np.mean(aO)),
        std_alb_ice =float(np.std(aI)),    std_alb_ocean =float(np.std(aO)),
        delta_alpha =float(np.mean(aI) - np.mean(aO)),
        auc=auc, youden_thr=thr, youden_J=J,
        tpr=tpr, fpr=fpr,
        alb_ice_vals=aI, alb_ocn_vals=aO,
    )


def discrimination_edge_zone(alb2d, sic2d, domain_mask):
    zone  = build_edge_zone_mask(sic2d)
    valid = domain_mask & zone & np.isfinite(alb2d) & np.isfinite(sic2d)
    m_ice = valid & (sic2d >= SIC_ICE_THRESHOLD)
    m_ocn = valid & (sic2d <  SIC_ICE_THRESHOLD)
    if m_ice.sum() < MIN_ICE_PX or m_ocn.sum() < MIN_OCEAN_PX:
        return None
    aI, aO = alb2d[m_ice], alb2d[m_ocn]
    tpr, fpr, auc, thr, J = roc_auc_youden(aI, aO)
    return dict(
        n_ice=int(m_ice.sum()), n_ocean=int(m_ocn.sum()),
        mean_alb_ice=float(np.mean(aI)), mean_alb_ocean=float(np.mean(aO)),
        delta_alpha=float(np.mean(aI) - np.mean(aO)),
        auc=auc, youden_thr=thr, youden_J=J, tpr=tpr, fpr=fpr,
        n_edge_px=int(zone[domain_mask].sum()),
    )


def sic_bin_stats(alb2d, sic2d, domain_mask):
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(sic2d)
    out = []
    for lo, hi, label in SIC_BINS:
        m = valid & (sic2d >= lo) & (sic2d < hi)
        n = int(m.sum())
        if n < 25:
            out.append(dict(bin=label, lo=lo, hi=hi, n=n,
                            mean=np.nan, std=np.nan, p10=np.nan,
                            p50=np.nan, p90=np.nan))
            continue
        a = alb2d[m]
        out.append(dict(bin=label, lo=lo, hi=hi, n=n,
                        mean=float(np.mean(a)), std=float(np.std(a)),
                        p10=float(np.percentile(a, 10)),
                        p50=float(np.percentile(a, 50)),
                        p90=float(np.percentile(a, 90))))
    return out


def continuous_regression(alb2d, sic2d, domain_mask, max_n=200_000):
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(sic2d)
    a = alb2d[valid]; s = sic2d[valid]
    n = len(a)
    if n < 100:
        return None
    if n > max_n:
        rng = np.random.default_rng(0)
        pick = rng.choice(n, max_n, replace=False)
        a, s = a[pick], s[pick]
    lr = stats.linregress(s, a)
    return dict(
        n_used   = int(len(a)),
        slope    = float(lr.slope),
        intercept= float(lr.intercept),
        r        = float(lr.rvalue),
        r2       = float(lr.rvalue**2),
        se_slope = float(lr.stderr),
    )


def distance_profile(alb2d, sic2d, domain_mask):
    d     = signed_distance_to_edge(sic2d)
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(d)
    if valid.sum() < 500:
        return None
    dv    = d[valid]
    av    = alb2d[valid]
    means = []; p25s = []; p75s = []; counts = []
    for lo, hi in zip(DIST_BIN_EDGES[:-1], DIST_BIN_EDGES[1:]):
        m = (dv >= lo) & (dv < hi)
        n = int(m.sum())
        counts.append(n)
        if n < 10:
            means.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
        else:
            sub = av[m]
            means.append(float(np.mean(sub)))
            p25s .append(float(np.percentile(sub, 25)))
            p75s .append(float(np.percentile(sub, 75)))
    return dict(bin_centers=(DIST_BIN_EDGES[:-1] + 0.5),
                mean=np.array(means), p25=np.array(p25s),
                p75=np.array(p75s), n=np.array(counts))


# ═════════════════════════════════════════════════════════════════════
# STEP 3 — EXPERIMENT-LEVEL RUN
# ═════════════════════════════════════════════════════════════════════
def classify_period(year, early_years, late_years):
    if year in early_years: return 'early'
    if year in late_years:  return 'late'
    return 'other'


def load_albedo_cached(cache_dir, product, year, month_name):
    """Load cached monthly-mean albedo and remap onto the SIAL grid."""
    if product == 'CLARA-A3':
        path = f"{cache_dir}/clara_alb_{month_name}_{year}.npy"
    elif product == 'APP-x':
        path = f"{cache_dir}/appx_alb_{month_name}_{year}.npy"
    else:
        return None
    if not os.path.exists(path):
        return None
    raw = np.load(path).astype(np.float32)
    # APP-x cache is on native shape → NN-remap to SIAL
    if product == 'APP-x' and raw.shape != sial_shape and idx_appx_to_sial is not None:
        raw = raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
    raw[(raw < 0) | (raw > 1)] = np.nan
    return raw


def load_sic_month_on_sial(year, month_num):
    """Monthly-mean NSIDC SIC on the SIAL grid from raw NSIDC PSN25 files."""
    sic_files = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_{year}*.nc"))
    maps = []
    for f in sic_files:
        try:
            with xr.open_dataset(f) as ds:
                if pd.Timestamp(ds['time'].values[0]).month != month_num:
                    continue
                raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
        except Exception:
            continue
        if raw.ndim > 2: raw = raw[0]
        if not np.any(np.isfinite(raw)):
            continue
        if np.nanmax(raw) > 1.5:
            raw = raw / 100.0
        # Strict physical bounds (0.0–1.0) to reject land/coast flags
        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
        maps.append(raw)
    if not maps:
        return None
    sic_psn  = np.nanmean(maps, axis=0).astype(np.float32)
    sic_sial = sic_psn.ravel()[idx_sic_to_sial].reshape(sial_shape)
    return sic_sial


def load_climatology_sic_from_cache(cache_dir, period, month_name):
    """Load the SIAL-gridded era climatology SIC written by the build script.
       `period` is 'early' or 'late'."""
    path = f"{cache_dir}/sic_{period}_sial_{month_name}.npy"
    if not os.path.exists(path):
        return None
    sic = np.load(path).astype(np.float32)
    # Climatology was already ice-masked (< 15% → NaN) — restore zeros
    # for the "ocean" side so the ice-edge detector sees open water too.
    # Use the companion trend file is not strictly necessary — for edge
    # detection we just need a SIC field where open water = 0 or finite
    # < 0.15 and ice = finite >= 0.15.  The simplest robust choice is to
    # treat NaN-in-domain as ocean (0.0) so the edge contour is well-defined.
    return sic


def run_experiment(exp_key):
    exp          = EXPERIMENTS[exp_key]
    cache_dir    = exp['cache_dir']
    early_years  = exp['early_years']
    late_years   = exp['late_years']
    all_years    = sorted(set(early_years + late_years))

    out_dir = os.path.join(OUTPUT_ROOT, exp_key)
    plt_dir = os.path.join(out_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"EXPERIMENT:  {exp_key}   —  {exp['label']}")
    print(f"Cache dir :  {cache_dir}")
    print(f"Output    :  {out_dir}")
    print(f"Years     :  early {early_years[0]}-{early_years[-1]}, "
          f"late {late_years[0]}-{late_years[-1]}")
    print("=" * 72)

    # -- try to use pre-built 60N mask from cache; fall back to geometric
    arctic_sial = sial_arctic
    fpath = os.path.join(cache_dir, 'arctic_60N_sial.npy')
    if os.path.exists(fpath):
        try:
            tmp = np.load(fpath).astype(bool)
            if tmp.shape == sial_shape:
                arctic_sial = tmp
                print(f"  (using arctic_60N_sial.npy from cache, "
                      f"{int(arctic_sial.sum()):,} px)")
        except Exception:
            pass

    rows_yr, rows_bin, rows_edge, rows_reg, rows_period = [], [], [], [], []
    roc_store      = {m: {'CLARA-A3': {}, 'APP-x': {}} for m in MONTH_ORDER}
    roc_edge_store = {m: {'CLARA-A3': {}, 'APP-x': {}} for m in MONTH_ORDER}
    box_store      = {m: {'CLARA-A3': {'ice': [], 'ocean': []},
                          'APP-x'   : {'ice': [], 'ocean': []}} for m in MONTH_ORDER}
    dist_store     = {m: {'CLARA-A3': {'mean': [], 'p25': [], 'p75': [], 'n': []},
                          'APP-x'   : {'mean': [], 'p25': [], 'p75': [], 'n': []}}
                      for m in MONTH_ORDER}
    scat_store     = {m: {'CLARA-A3': {'sic': [], 'alb': []},
                          'APP-x'   : {'sic': [], 'alb': []}} for m in MONTH_ORDER}

    # =========================================================
    # (a) per-year discrimination
    # =========================================================
    for year in all_years:
        period = classify_period(year, early_years, late_years)
        print(f"  {year} ({period:5s})...", end='', flush=True)
        for mname, mnum in MONTHS_TO_ANALYZE.items():
            sic = load_sic_month_on_sial(year, mnum)
            if sic is None: continue

            for product in ['CLARA-A3', 'APP-x']:
                alb = load_albedo_cached(cache_dir, product, year, mname)
                if alb is None: continue

                r_full = discrimination_full(alb, sic, arctic_sial)
                if r_full is None: continue
                r_bins = sic_bin_stats(alb, sic, arctic_sial)
                r_edge = discrimination_edge_zone(alb, sic, arctic_sial)
                r_reg  = continuous_regression(alb, sic, arctic_sial)
                r_dist = distance_profile(alb, sic, arctic_sial)

                rows_yr.append(dict(
                    experiment=exp_key, product=product, year=year,
                    period=period, month=mname,
                    n_ice=r_full['n_ice'], n_ocean=r_full['n_ocean'],
                    mean_alb_ice=r_full['mean_alb_ice'],
                    mean_alb_ocean=r_full['mean_alb_ocean'],
                    std_alb_ice=r_full['std_alb_ice'],
                    std_alb_ocean=r_full['std_alb_ocean'],
                    delta_alpha=r_full['delta_alpha'],
                    auc=r_full['auc'],
                    youden_thr=r_full['youden_thr'],
                    youden_J=r_full['youden_J'],
                    auc_edge=(r_edge['auc'] if r_edge else np.nan),
                    youden_thr_edge=(r_edge['youden_thr'] if r_edge else np.nan),
                    delta_alpha_edge=(r_edge['delta_alpha'] if r_edge else np.nan),
                    n_edge_px=(r_edge['n_edge_px'] if r_edge else 0),
                    reg_slope=(r_reg['slope'] if r_reg else np.nan),
                    reg_intercept=(r_reg['intercept'] if r_reg else np.nan),
                    reg_r2=(r_reg['r2'] if r_reg else np.nan),
                    reg_r=(r_reg['r'] if r_reg else np.nan),
                ))
                for b in r_bins:
                    rows_bin.append(dict(experiment=exp_key, product=product,
                                         year=year, period=period,
                                         month=mname, **b))
                if r_edge:
                    rows_edge.append(dict(
                        experiment=exp_key, product=product, year=year,
                        period=period, month=mname,
                        n_edge_px=r_edge['n_edge_px'],
                        n_ice=r_edge['n_ice'], n_ocean=r_edge['n_ocean'],
                        mean_alb_ice=r_edge['mean_alb_ice'],
                        mean_alb_ocean=r_edge['mean_alb_ocean'],
                        delta_alpha=r_edge['delta_alpha'],
                        auc=r_edge['auc'],
                        youden_thr=r_edge['youden_thr'],
                        youden_J=r_edge['youden_J'],
                    ))
                if r_reg:
                    rows_reg.append(dict(experiment=exp_key, product=product,
                                         year=year, period=period,
                                         month=mname, **r_reg))

                # plotting stores
                roc_store[mname][product][year] = (r_full['tpr'], r_full['fpr'])
                if r_edge:
                    roc_edge_store[mname][product][year] = (r_edge['tpr'], r_edge['fpr'])
                box_store[mname][product]['ice']  .extend(r_full['alb_ice_vals'][:40_000].tolist())
                box_store[mname][product]['ocean'].extend(r_full['alb_ocn_vals'][:40_000].tolist())
                if r_dist:
                    for k in ('mean', 'p25', 'p75', 'n'):
                        dist_store[mname][product][k].append(r_dist[k])
                vmask = arctic_sial & np.isfinite(alb) & np.isfinite(sic)
                if vmask.sum() > 0:
                    a_flat, s_flat = alb[vmask], sic[vmask]
                    n = len(a_flat)
                    k = min(3000, n)
                    if k > 0:
                        rng  = np.random.default_rng(year * 100 + mnum)
                        pick = rng.choice(n, k, replace=False)
                        scat_store[mname][product]['alb'].extend(a_flat[pick].tolist())
                        scat_store[mname][product]['sic'].extend(s_flat[pick].tolist())
        print(" ✓")
        gc.collect()

    # =========================================================
    # (b) PERIOD-LEVEL comparison (early vs late) using cached
    #     climatological SIC edge — removes year-to-year SIC noise.
    # =========================================================
    print("\n  -- period-level early-vs-late analysis --")
    for mname in MONTH_ORDER:
        for period_label, period_years in [('early', early_years),
                                            ('late',  late_years)]:
            sic_clim = load_climatology_sic_from_cache(cache_dir, period_label, mname)
            if sic_clim is None:
                print(f"    {mname} {period_label}: no cached SIC clim, skip")
                continue
            # Treat NaN-in-Arctic as ocean (0.0) so the edge is well-defined
            sic_edge = np.where(arctic_sial & ~np.isfinite(sic_clim),
                                 0.0, sic_clim)

            for product in ['CLARA-A3', 'APP-x']:
                alb_stack = []
                for y in period_years:
                    a = load_albedo_cached(cache_dir, product, y, mname)
                    if a is not None:
                        alb_stack.append(a)
                if not alb_stack:
                    continue
                alb_mean = np.nanmean(np.array(alb_stack), axis=0)

                r_full = discrimination_full(alb_mean, sic_edge, arctic_sial)
                r_edge = discrimination_edge_zone(alb_mean, sic_edge, arctic_sial)
                r_reg  = continuous_regression(alb_mean, sic_edge, arctic_sial)
                r_dist = distance_profile(alb_mean, sic_edge, arctic_sial)

                rows_period.append(dict(
                    experiment=exp_key, product=product, month=mname,
                    period=period_label, n_years=len(alb_stack),
                    delta_alpha_full=(r_full['delta_alpha'] if r_full else np.nan),
                    auc_full        =(r_full['auc']         if r_full else np.nan),
                    youden_thr_full =(r_full['youden_thr']  if r_full else np.nan),
                    delta_alpha_edge=(r_edge['delta_alpha'] if r_edge else np.nan),
                    auc_edge        =(r_edge['auc']         if r_edge else np.nan),
                    reg_slope       =(r_reg['slope']        if r_reg  else np.nan),
                    reg_r2          =(r_reg['r2']           if r_reg  else np.nan),
                ))

                # Stash distance profile for plotting (keyed by period)
                if r_dist:
                    key = f"_period_{period_label}_{product}"
                    dist_store[mname].setdefault(key, r_dist)

    df_yr     = pd.DataFrame(rows_yr)
    df_bin    = pd.DataFrame(rows_bin)
    df_edge   = pd.DataFrame(rows_edge)
    df_reg    = pd.DataFrame(rows_reg)
    df_period = pd.DataFrame(rows_period)

    # =========================================================
    # (c) SAVE CSVs
    # =========================================================
    df_yr.to_csv    (os.path.join(out_dir, 'albedo_discrimination_yearly.csv'),     index=False)
    df_bin.to_csv   (os.path.join(out_dir, 'albedo_discrimination_sic_bins.csv'),   index=False)
    df_edge.to_csv  (os.path.join(out_dir, 'albedo_discrimination_edge_zone.csv'),  index=False)
    df_reg.to_csv   (os.path.join(out_dir, 'albedo_discrimination_regression.csv'), index=False)
    df_period.to_csv(os.path.join(out_dir, 'albedo_discrimination_period.csv'),     index=False)

    if df_yr.empty:
        print("  (no per-year rows — nothing else to plot)")
        return df_yr, df_period

    # Monthly summary across years (binary + edge-zone + regression)
    summary = (df_yr.groupby(['product', 'month'])
                    .agg(
                        delta_alpha_mean  =('delta_alpha', 'mean'),
                        delta_alpha_std   =('delta_alpha', 'std'),
                        mean_alb_ice_mean =('mean_alb_ice',   'mean'),
                        mean_alb_ocn_mean =('mean_alb_ocean', 'mean'),
                        auc_mean          =('auc', 'mean'),
                        auc_std           =('auc', 'std'),
                        youden_thr_mean   =('youden_thr', 'mean'),
                        auc_edge_mean     =('auc_edge', 'mean'),
                        auc_edge_std      =('auc_edge', 'std'),
                        delta_alpha_edge_mean=('delta_alpha_edge', 'mean'),
                        reg_slope_mean    =('reg_slope', 'mean'),
                        reg_intercept_mean=('reg_intercept', 'mean'),
                        reg_r2_mean       =('reg_r2', 'mean'),
                        n_years           =('year', 'count'),
                    ).reset_index())
    summary['month'] = pd.Categorical(summary['month'],
                                       categories=MONTH_ORDER, ordered=True)
    summary = summary.sort_values(['product', 'month'])
    summary.to_csv(os.path.join(out_dir, 'albedo_discrimination_summary.csv'),
                   index=False)

    # Distance-profile: flatten into a tidy csv
    dist_rows = []
    for mname, pdict in dist_store.items():
        for product in ['CLARA-A3', 'APP-x']:
            d = pdict.get(product, {})
            if not d or not d.get('mean'):
                continue
            means = np.nanmean(np.vstack(d['mean']), axis=0)
            p25   = np.nanmean(np.vstack(d['p25']),  axis=0)
            p75   = np.nanmean(np.vstack(d['p75']),  axis=0)
            ns    = np.nansum(np.vstack(d['n']),     axis=0)
            for c, mn, lo, hi, n in zip(DIST_BIN_EDGES[:-1] + 0.5,
                                          means, p25, p75, ns):
                dist_rows.append(dict(experiment=exp_key, product=product,
                                       month=mname,
                                       bin_center_px=float(c),
                                       mean_alb=float(mn) if np.isfinite(mn) else np.nan,
                                       p25_alb=float(lo) if np.isfinite(lo) else np.nan,
                                       p75_alb=float(hi) if np.isfinite(hi) else np.nan,
                                       n_pixels_total=int(n)))
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(out_dir, 'albedo_discrimination_distance_profile.csv'),
        index=False)

    # ROC arrays for future reuse
    with open(os.path.join(out_dir, 'roc_arrays.pkl'), 'wb') as f:
        pickle.dump({'roc_full': roc_store, 'roc_edge': roc_edge_store,
                     'thresholds': ROC_THRESHOLDS,
                     'experiment': exp_key}, f)

    # =========================================================
    # (d) CONSOLE SUMMARY
    # =========================================================
    print("\n" + "-" * 72)
    print(f"MONTHLY SUMMARY  ({exp_key})")
    print("-" * 72)
    for prod in ['CLARA-A3', 'APP-x']:
        sub = summary[summary['product'] == prod]
        if sub.empty:
            print(f"\n{prod}: no data"); continue
        print(f"\n{prod}")
        hdr = f"  {'Month':<10} {'Δα_full':>9} {'AUC_full':>9} "\
              f"{'AUC_edge':>9} {'Δα_edge':>9} {'slope':>7} {'R²':>6} {'thr':>6}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for _, r in sub.iterrows():
            print(f"  {str(r['month']):<10} "
                  f"{r['delta_alpha_mean']:>+9.4f} "
                  f"{r['auc_mean']:>9.3f} "
                  f"{r['auc_edge_mean']:>9.3f} "
                  f"{r['delta_alpha_edge_mean']:>+9.4f} "
                  f"{r['reg_slope_mean']:>7.3f} "
                  f"{r['reg_r2_mean']:>6.3f} "
                  f"{r['youden_thr_mean']:>6.3f}")

    # Period-level summary print
    if not df_period.empty:
        print("\n" + "-" * 72)
        print(f"PERIOD COMPARISON  ({exp_key})")
        print("-" * 72)
        tbl = df_period.pivot_table(index=['product', 'month'],
                                      columns='period',
                                      values=['delta_alpha_full', 'auc_full',
                                              'delta_alpha_edge', 'auc_edge',
                                              'reg_slope'])
        with pd.option_context('display.float_format', '{:+.3f}'.format,
                               'display.width', 140):
            print(tbl)

    # =========================================================
    # (e) PLOTS
    # =========================================================
    print("\n  generating plots ...")
    COLORS = {'CLARA-A3': '#1f77b4', 'APP-x': '#d62728'}

    # ---- (e1) ROC per month: FULL vs EDGE-ZONE, both products ----
    for mname in MONTH_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
        fig.suptitle(f"{exp_key} — {mname} — ROC: albedo as ice detector\n"
                      f"thin lines = yearly curves   thick = multi-year mean",
                      fontsize=10, fontweight='bold')
        for ax, prod_name in zip(axes, ['CLARA-A3', 'APP-x']):
            col  = COLORS[prod_name]
            full = roc_store[mname].get(prod_name, {})
            edge = roc_edge_store[mname].get(prod_name, {})
            tprs, fprs = [], []
            for _, (tpr, fpr) in sorted(full.items()):
                ax.plot(fpr, tpr, color=col, alpha=0.10, lw=0.6)
                tprs.append(tpr); fprs.append(fpr)
            if tprs:
                mT = np.nanmean(tprs, axis=0); mF = np.nanmean(fprs, axis=0)
                order = np.argsort(mF)
                mAUC = float(np.trapz(mT[order], mF[order]))
                ax.plot(mF, mT, color=col, lw=2.5,
                        label=f"Full Arctic (AUC={mAUC:.3f})")
            tprs_e, fprs_e = [], []
            for _, (tpr, fpr) in sorted(edge.items()):
                ax.plot(fpr, tpr, color=col, alpha=0.08, lw=0.5, ls='--')
                tprs_e.append(tpr); fprs_e.append(fpr)
            if tprs_e:
                mT = np.nanmean(tprs_e, axis=0); mF = np.nanmean(fprs_e, axis=0)
                order = np.argsort(mF)
                mAUC_e = float(np.trapz(mT[order], mF[order]))
                ax.plot(mF, mT, color=col, lw=2.5, ls='--',
                        label=f"Edge zone (AUC={mAUC_e:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', lw=0.6, label='No skill')
            ax.set_xlabel('FPR (false alarm)')
            ax.set_ylabel('TPR (detection)')
            ax.set_title(f"{prod_name}  —  {len(full)} years")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.25)
        plt.tight_layout()
        f = os.path.join(plt_dir, f"roc_{mname.lower()}.png")
        fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e2) Box-plots BINARY (original): ice vs ocean ----
    for mname in MONTH_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        fig.suptitle(f"{exp_key} — {mname} — "
                      f"Binary albedo: SIC≥15% vs SIC<15% (all years combined)",
                      fontsize=10, fontweight='bold')
        rng = np.random.default_rng(0)
        for ax, prod_name in zip(axes, ['CLARA-A3', 'APP-x']):
            col = COLORS[prod_name]
            ice = np.asarray(box_store[mname][prod_name]['ice'])
            ocn = np.asarray(box_store[mname][prod_name]['ocean'])
            if len(ice) == 0 or len(ocn) == 0:
                ax.set_title(f"{prod_name} — no data"); continue
            ice_s = rng.choice(ice, min(len(ice), 100_000), replace=False)
            ocn_s = rng.choice(ocn, min(len(ocn), 100_000), replace=False)
            bp = ax.boxplot([ocn_s, ice_s],
                             labels=['Ocean\nSIC<15%', 'Ice\nSIC≥15%'],
                             patch_artist=True,
                             medianprops=dict(color='k', lw=1.5),
                             flierprops=dict(marker='.', ms=1, alpha=0.08),
                             showfliers=True)
            bp['boxes'][0].set_facecolor('#AED6F1'); bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor(col);       bp['boxes'][1].set_alpha(0.7)
            delta = float(np.mean(ice_s) - np.mean(ocn_s))
            ax.set_ylabel('Surface albedo')
            ax.set_title(f"{prod_name}   Δα = {delta:+.3f}")
            ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(-0.02, 1.02)
        plt.tight_layout()
        f = os.path.join(plt_dir, f"boxplot_binary_{mname.lower()}.png")
        fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e3) SIC-stratified profile ----
    for mname in MONTH_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(f"{exp_key} — {mname} — Albedo stratified by SIC class",
                      fontsize=10, fontweight='bold')
        for ax, prod_name in zip(axes, ['CLARA-A3', 'APP-x']):
            sub = df_bin[(df_bin['product'] == prod_name) &
                          (df_bin['month']   == mname)]
            if sub.empty:
                ax.set_title(f"{prod_name} — no data"); continue
            grp = sub.groupby('bin', sort=False).agg(
                mean_mean=('mean', 'mean'),
                p10_mean =('p10',  'mean'),
                p90_mean =('p90',  'mean'),
                n_mean   =('n',    'mean'),
            ).reindex([b[2] for b in SIC_BINS])
            x    = np.arange(len(SIC_BINS))
            mean = grp['mean_mean'].values
            p10  = grp['p10_mean'].values
            p90  = grp['p90_mean'].values
            ax.errorbar(x, mean, yerr=[mean - p10, p90 - mean],
                        fmt='o-', lw=2, capsize=4,
                        color=COLORS[prod_name],
                        label=f"{prod_name}  mean ± (p10, p90)")
            ax.set_xticks(x)
            ax.set_xticklabels([b[2] for b in SIC_BINS], rotation=30, ha='right',
                                fontsize=8)
            ax.set_ylabel('Surface albedo')
            ax.set_title(f"{prod_name}")
            ax.grid(True, alpha=0.3); ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.15, color='gray', ls=':', lw=0.8, alpha=0.7)
            ax.legend(fontsize=9, loc='upper left')
        plt.tight_layout()
        f = os.path.join(plt_dir, f"boxplot_sic_bins_{mname.lower()}.png")
        fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e4) AUC summary bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    months_avail = [m for m in MONTH_ORDER if m in summary['month'].values]
    x            = np.arange(len(months_avail))
    for ax, (zone_label, col_mean, col_std) in zip(axes, [
            ('Full Arctic',  'auc_mean',      'auc_std'),
            ('Edge zone',    'auc_edge_mean', 'auc_edge_std'),
    ]):
        for offset, prod_name in zip([-0.2, 0.2], ['CLARA-A3', 'APP-x']):
            sub  = summary[summary['product'] == prod_name].set_index('month')
            vals = [float(sub.loc[m, col_mean]) if m in sub.index else np.nan
                    for m in months_avail]
            errs = [float(sub.loc[m, col_std])  if m in sub.index else 0.0
                    for m in months_avail]
            ax.bar(x + offset, vals, 0.4, color=COLORS[prod_name],
                   alpha=0.85, yerr=errs, capsize=3, label=prod_name,
                   error_kw={'lw': 1.0})
        ax.axhline(0.5, color='k', lw=0.8, ls=':', label='No skill')
        ax.axhline(0.9, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:3] for m in months_avail])
        ax.set_ylabel('AUC')
        ax.set_title(zone_label, fontsize=11)
        ax.set_ylim(0.4, 1.05); ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')
    fig.suptitle(f'{exp_key} — AUC: full Arctic vs ice-edge zone',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(plt_dir, 'auc_summary.png'),
                 dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e5) Δα time series ----
    fig, axes = plt.subplots(len(MONTH_ORDER), 1,
                             figsize=(12, 2.2 * len(MONTH_ORDER)), sharex=False)
    for i, mname in enumerate(MONTH_ORDER):
        ax = axes[i]
        for prod_name, col in COLORS.items():
            sub = df_yr[(df_yr['product'] == prod_name) &
                         (df_yr['month']   == mname)].sort_values('year')
            if sub.empty: continue
            ax.plot(sub['year'], sub['delta_alpha'], 'o-', color=col,
                    label=f"{prod_name} (full)", lw=1.3, ms=3)
            ax.plot(sub['year'], sub['delta_alpha_edge'], '^--', color=col,
                    label=f"{prod_name} (edge)", lw=1.0, ms=3, alpha=0.7)
        ax.axhline(0, color='k', lw=0.7, ls='--')
        # shade early / late periods
        ax.axvspan(early_years[0] - 0.5, early_years[-1] + 0.5,
                   alpha=0.08, color='blue')
        ax.axvspan(late_years[0]  - 0.5, late_years[-1]  + 0.5,
                   alpha=0.08, color='red')
        ax.set_ylabel('Δα', fontsize=8)
        ax.set_title(mname, fontsize=9, loc='left', pad=2)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, ncol=4, loc='upper right')
    axes[-1].set_xlabel('Year')
    fig.suptitle(f"{exp_key} — Δα = ice−ocean mean albedo  "
                  f"(solid=full, dashed=edge; shaded = early/late)",
                  fontsize=10, fontweight='bold', y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(plt_dir, 'delta_alpha_timeseries.png'),
                 dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e6) Albedo vs SIC hexbin with regression ----
    for mname in MONTH_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"{exp_key} — {mname} — Albedo vs SIC (continuous)",
                      fontsize=10, fontweight='bold')
        for ax, prod_name in zip(axes, ['CLARA-A3', 'APP-x']):
            s = np.asarray(scat_store[mname][prod_name]['sic'])
            a = np.asarray(scat_store[mname][prod_name]['alb'])
            if len(s) < 100:
                ax.set_title(f"{prod_name} — insufficient data"); continue
            hb = ax.hexbin(s, a, gridsize=40, mincnt=1, cmap='Blues',
                            bins='log')
            ax.set_xlabel('SIC')
            ax.set_ylabel('Surface albedo')
            lr = stats.linregress(s, a)
            xx = np.linspace(0, 1, 50)
            ax.plot(xx, lr.intercept + lr.slope * xx,
                    color=COLORS[prod_name], lw=2,
                    label=f"α = {lr.intercept:+.3f} + {lr.slope:+.3f}·SIC\n"
                           f"R² = {lr.rvalue**2:.3f}")
            ax.axhline(0.15, color='gray', ls=':', lw=0.7)
            ax.axvline(0.15, color='gray', ls=':', lw=0.7)
            ax.set_title(prod_name)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=9, loc='upper left')
            plt.colorbar(hb, ax=ax, shrink=0.75, label='log10(count)')
        plt.tight_layout()
        fig.savefig(os.path.join(plt_dir,
                                   f"albedo_vs_sic_scatter_{mname.lower()}.png"),
                     dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e7) Distance-to-edge profile (yearly-avg) ----
    px_km_label = "distance in SIAL pixels  (1 px ≈ 14-28 km at Arctic lat)"
    for mname in MONTH_ORDER:
        fig, ax = plt.subplots(figsize=(10, 5))
        for prod_name in ['CLARA-A3', 'APP-x']:
            d = dist_store[mname].get(prod_name, {})
            if not d or not d.get('mean'): continue
            means = np.nanmean(np.vstack(d['mean']), axis=0)
            p25   = np.nanmean(np.vstack(d['p25']),  axis=0)
            p75   = np.nanmean(np.vstack(d['p75']),  axis=0)
            bc    = DIST_BIN_EDGES[:-1] + 0.5
            ax.plot(bc, means, 'o-', color=COLORS[prod_name], lw=1.8,
                    label=prod_name)
            ax.fill_between(bc, p25, p75, color=COLORS[prod_name], alpha=0.18)
        ax.axvline(0, color='k', lw=0.8, ls='--', label='SIC=15% contour')
        ax.set_xlabel(px_km_label)
        ax.set_ylabel('Mean surface albedo\n(shaded = IQR across pixels)')
        ax.set_title(f"{exp_key} — {mname} — distance-to-edge albedo profile\n"
                      f"(negative = inside ice, positive = in ocean)",
                      fontsize=10, fontweight='bold')
        ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(plt_dir,
                                   f"distance_profile_{mname.lower()}.png"),
                     dpi=150, bbox_inches='tight'); plt.close()

    # ---- (e8) Period comparison (early vs late) using climatology SIC edge ----
    for mname in MONTH_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"{exp_key} — {mname} — early vs late distance-to-edge\n"
                      f"(SIC edge from that era's climatology)",
                      fontsize=10, fontweight='bold')
        for ax, prod_name in zip(axes, ['CLARA-A3', 'APP-x']):
            ax.axvline(0, color='k', lw=0.8, ls='--',
                        label='SIC=15% (climatology)')
            ax.set_title(prod_name)
            ax.set_xlabel(px_km_label)
            ax.set_ylabel('Mean surface albedo')
            ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)

            for period_label, pcol in [('early', '#1f77b4'),
                                        ('late',  '#d62728')]:
                key = f"_period_{period_label}_{prod_name}"
                rd = dist_store[mname].get(key)
                if rd is None: continue
                bc = DIST_BIN_EDGES[:-1] + 0.5
                ax.plot(bc, rd['mean'], 'o-', color=pcol, lw=1.8,
                        label=f"{period_label}")
                ax.fill_between(bc, rd['p25'], rd['p75'], color=pcol, alpha=0.18)
            ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(plt_dir,
                                   f"period_comparison_{mname.lower()}.png"),
                     dpi=150, bbox_inches='tight'); plt.close()

    print(f"  ✓ {exp_key} done   ({out_dir})")
    return df_yr, df_period


# ═════════════════════════════════════════════════════════════════════
# STEP 4 — RUN ALL EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════
all_yr, all_period = [], []
for exp_key in EXPS_TO_RUN:
    df_y, df_p = run_experiment(exp_key)
    all_yr.append(df_y)
    all_period.append(df_p)

# combined outputs across experiments (easier to diff obs vs cmip6)
if all_yr and any(not df.empty for df in all_yr):
    df_all = pd.concat(all_yr, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_ROOT,
                               'albedo_discrimination_yearly_ALL.csv'),
                   index=False)
if all_period and any(not df.empty for df in all_period):
    df_p_all = pd.concat(all_period, ignore_index=True)
    df_p_all.to_csv(os.path.join(OUTPUT_ROOT,
                                  'albedo_discrimination_period_ALL.csv'),
                     index=False)
    # Quick cross-experiment period print
    print("\n" + "=" * 72)
    print("CROSS-EXPERIMENT PERIOD COMPARISON  (AUC_edge, Δα_edge)")
    print("=" * 72)
    pivot = df_p_all.pivot_table(index=['product', 'month'],
                                   columns=['experiment', 'period'],
                                   values=['auc_edge', 'delta_alpha_edge'])
    with pd.option_context('display.float_format', '{:+.3f}'.format,
                           'display.width', 160):
        print(pivot)

print(f"\n✓  All experiments complete.  Root: {OUTPUT_ROOT}")