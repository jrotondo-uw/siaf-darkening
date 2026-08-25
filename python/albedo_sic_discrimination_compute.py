#!/usr/bin/env python
"""
albedo_sic_discrimination_compute.py
=====================================
Runs all discrimination metrics (binary, edge-zone, SIC-bins,
regression, distance-profile, period comparison) and saves:

  Per experiment under {OUTPUT_ROOT}/{exp}/
    albedo_discrimination_yearly.csv
    albedo_discrimination_sic_bins.csv
    albedo_discrimination_edge_zone.csv
    albedo_discrimination_regression.csv
    albedo_discrimination_distance_profile.csv
    albedo_discrimination_period.csv
    albedo_discrimination_summary.csv
    plot_data_{exp}.pkl          ← everything the notebook needs

  Combined across experiments:
    albedo_discrimination_yearly_ALL.csv
    albedo_discrimination_period_ALL.csv
"""

import os, glob, gc, pickle
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy import ndimage, stats
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ═════════════════════════════════════════════════════════════════════
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
EXPS_TO_RUN = ['obs', 'cmip6']

DIR_SIC         = "/glade/derecho/scratch/jrotondo/SIC_data/"
DIR_SIAL_SAMPLE = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
DIR_APPX_SAMPLE = "/glade/derecho/scratch/jrotondo/APP-x_observations/"
OUTPUT_ROOT     = "/glade/work/jrotondo/j_of_clim_dk_sea_ice_paper/output/albedo_disc/"

MONTHS_TO_ANALYZE = {'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9}
MONTH_ORDER = list(MONTHS_TO_ANALYZE.keys())

SIC_ICE_THRESHOLD = 0.15
LAT_MIN           = 60.0
MIN_ICE_PX        = 100
MIN_OCEAN_PX      = 100
EDGE_DILATE_PX    = 5
DIST_BIN_EDGES    = np.arange(-30, 31, 1)
SIC_BINS = [
    (0.00, 0.05, 'open_0-5'),
    (0.05, 0.15, 'trace_5-15'),
    (0.15, 0.30, 'MIZ-lo_15-30'),
    (0.30, 0.50, 'MIZ-hi_30-50'),
    (0.50, 0.80, 'trans_50-80'),
    (0.80, 1.01, 'cons_80-100'),
]
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
    tree = cKDTree(latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel()))
    _, idx = tree.query(latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel()),
                         workers=-1)
    del tree; gc.collect()
    return idx


# ═════════════════════════════════════════════════════════════════════
# STEP 1 — GRIDS + NN INDICES (built once, shared across experiments)
# ═════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1:  SIAL / APP-x / PSN25 grids + NN regrid indices")
print("=" * 72)

sial_sample = sorted(glob.glob(f"{DIR_SIAL_SAMPLE}/*.nc"))[0]
with xr.open_dataset(sial_sample) as ds:
    _lat = ds['lat'].values; _lon = ds['lon'].values
if _lat.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(_lon, _lat)
else:
    sial_lat2d, sial_lon2d = _lat, _lon
sial_shape  = sial_lat2d.shape
sial_arctic = (sial_lat2d >= LAT_MIN)
print(f"  SIAL grid: {sial_shape}  (Arctic px = {int(sial_arctic.sum()):,})")

sic_sample = sorted(glob.glob(f"{DIR_SIC}/sic_psn25_*.nc"))[0]
with xr.open_dataset(sic_sample) as ds:
    psn_x = ds['x'].values; psn_y = ds['y'].values
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
_tfm = Transformer.from_crs(PSN25_PROJ4, "EPSG:4326", always_xy=True)
psn_lon2d, psn_lat2d = _tfm.transform(psn_xx, psn_yy)
psn_shape = psn_lat2d.shape
print(f"  SIC PSN25 grid: {psn_shape}")
print("  Building SIC → SIAL NN index ...", flush=True)
idx_sic_to_sial = build_nn_index(psn_lat2d, psn_lon2d, sial_lat2d, sial_lon2d)

idx_appx_to_sial = None
appx_shape       = None
try:
    sample_appx = (sorted(glob.glob(f"{DIR_APPX_SAMPLE}/**/*.nc", recursive=True))
                    or sorted(glob.glob(f"{DIR_APPX_SAMPLE}/*.nc")))[0]
    with Dataset(sample_appx, 'r') as ds:
        appx_lat_raw = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon_raw = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat_raw.ndim == 1:
        appx_lon2d, appx_lat2d_g = np.meshgrid(appx_lon_raw, appx_lat_raw)
    else:
        appx_lat2d_g, appx_lon2d = appx_lat_raw, appx_lon_raw
    appx_shape = appx_lat2d_g.shape
    print(f"  APP-x grid: {appx_shape}")
    print("  Building APP-x → SIAL NN index ...", flush=True)
    idx_appx_to_sial = build_nn_index(appx_lat2d_g, appx_lon2d,
                                       sial_lat2d, sial_lon2d)
except Exception as e:
    print(f"  WARNING: APP-x grid read failed ({e}) — assuming cache on SIAL shape.")


# ═════════════════════════════════════════════════════════════════════
# STEP 2 — METRIC FUNCTIONS
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
    is_ice = (sic2d >= threshold) & np.isfinite(sic2d)
    is_ocn = (sic2d <  threshold) & np.isfinite(sic2d)
    edge = (is_ice & ndimage.binary_dilation(is_ocn, iterations=1)) | \
           (is_ocn & ndimage.binary_dilation(is_ice, iterations=1))
    return ndimage.binary_dilation(edge, iterations=dilate_px)


def signed_distance_to_edge(sic2d, threshold=SIC_ICE_THRESHOLD):
    valid  = np.isfinite(sic2d)
    is_ice = (sic2d >= threshold) & valid
    d_out  = ndimage.distance_transform_edt(~is_ice)
    d_in   = ndimage.distance_transform_edt(is_ice)
    d = np.where(is_ice, -d_in, d_out)
    return np.where(valid, d, np.nan)


def discrimination_full(alb2d, sic2d, domain_mask):
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(sic2d)
    m_ice = valid & (sic2d >= SIC_ICE_THRESHOLD)
    m_ocn = valid & (sic2d <  SIC_ICE_THRESHOLD)
    if m_ice.sum() < MIN_ICE_PX or m_ocn.sum() < MIN_OCEAN_PX:
        return None
    aI, aO = alb2d[m_ice], alb2d[m_ocn]
    tpr, fpr, auc, thr, J = roc_auc_youden(aI, aO)
    return dict(n_ice=int(m_ice.sum()), n_ocean=int(m_ocn.sum()),
                mean_alb_ice=float(np.mean(aI)), mean_alb_ocean=float(np.mean(aO)),
                std_alb_ice=float(np.std(aI)),   std_alb_ocean=float(np.std(aO)),
                delta_alpha=float(np.mean(aI) - np.mean(aO)),
                auc=auc, youden_thr=thr, youden_J=J,
                tpr=tpr, fpr=fpr,
                alb_ice_vals=aI, alb_ocn_vals=aO)


def discrimination_edge_zone(alb2d, sic2d, domain_mask):
    zone  = build_edge_zone_mask(sic2d)
    valid = domain_mask & zone & np.isfinite(alb2d) & np.isfinite(sic2d)
    m_ice = valid & (sic2d >= SIC_ICE_THRESHOLD)
    m_ocn = valid & (sic2d <  SIC_ICE_THRESHOLD)
    if m_ice.sum() < MIN_ICE_PX or m_ocn.sum() < MIN_OCEAN_PX:
        return None
    aI, aO = alb2d[m_ice], alb2d[m_ocn]
    tpr, fpr, auc, thr, J = roc_auc_youden(aI, aO)
    return dict(n_ice=int(m_ice.sum()), n_ocean=int(m_ocn.sum()),
                mean_alb_ice=float(np.mean(aI)), mean_alb_ocean=float(np.mean(aO)),
                delta_alpha=float(np.mean(aI) - np.mean(aO)),
                auc=auc, youden_thr=thr, youden_J=J, tpr=tpr, fpr=fpr,
                n_edge_px=int(zone[domain_mask].sum()))


def sic_bin_stats(alb2d, sic2d, domain_mask):
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(sic2d)
    out = []
    for lo, hi, label in SIC_BINS:
        m = valid & (sic2d >= lo) & (sic2d < hi)
        n = int(m.sum())
        if n < 25:
            out.append(dict(bin=label, lo=lo, hi=hi, n=n,
                            mean=np.nan, std=np.nan, p10=np.nan,
                            p50=np.nan,  p90=np.nan))
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
        pick = np.random.default_rng(0).choice(n, max_n, replace=False)
        a, s = a[pick], s[pick]
    lr = stats.linregress(s, a)
    return dict(n_used=int(len(a)), slope=float(lr.slope),
                intercept=float(lr.intercept), r=float(lr.rvalue),
                r2=float(lr.rvalue**2), se_slope=float(lr.stderr))


def distance_profile(alb2d, sic2d, domain_mask):
    d     = signed_distance_to_edge(sic2d)
    valid = domain_mask & np.isfinite(alb2d) & np.isfinite(d)
    if valid.sum() < 500:
        return None
    dv = d[valid]; av = alb2d[valid]
    means = []; p25s = []; p75s = []; counts = []
    for lo, hi in zip(DIST_BIN_EDGES[:-1], DIST_BIN_EDGES[1:]):
        m = (dv >= lo) & (dv < hi); n = int(m.sum()); counts.append(n)
        if n < 10:
            means.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
        else:
            sub = av[m]
            means.append(float(np.mean(sub)))
            p25s .append(float(np.percentile(sub, 25)))
            p75s .append(float(np.percentile(sub, 75)))
    return dict(bin_centers=(DIST_BIN_EDGES[:-1] + 0.5),
                mean=np.array(means), p25=np.array(p25s),
                p75=np.array(p75s),  n=np.array(counts))


# ═════════════════════════════════════════════════════════════════════
# STEP 3 — EXPERIMENT RUNNER  (saves CSVs + plot_data pkl; no plotting)
# ═════════════════════════════════════════════════════════════════════
def classify_period(year, early_years, late_years):
    if year in early_years: return 'early'
    if year in late_years:  return 'late'
    return 'other'


def load_albedo_cached(cache_dir, product, year, month_name):
    path = (f"{cache_dir}/clara_alb_{month_name}_{year}.npy" if product == 'CLARA-A3'
            else f"{cache_dir}/appx_alb_{month_name}_{year}.npy")
    if not os.path.exists(path):
        return None
    raw = np.load(path).astype(np.float32)
    if product == 'APP-x' and raw.shape != sial_shape and idx_appx_to_sial is not None:
        raw = raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
    raw[(raw < 0) | (raw > 1)] = np.nan
    return raw


def load_sic_month_on_sial(year, month_num):
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
        raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
        maps.append(raw)
    if not maps:
        return None
    sic_psn  = np.nanmean(maps, axis=0).astype(np.float32)
    return sic_psn.ravel()[idx_sic_to_sial].reshape(sial_shape)


def load_climatology_sic_from_cache(cache_dir, period, month_name):
    path = f"{cache_dir}/sic_{period}_sial_{month_name}.npy"
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)


def run_experiment(exp_key):
    exp         = EXPERIMENTS[exp_key]
    cache_dir   = exp['cache_dir']
    early_years = exp['early_years']
    late_years  = exp['late_years']
    all_years   = sorted(set(early_years + late_years))

    out_dir = os.path.join(OUTPUT_ROOT, exp_key)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"EXPERIMENT:  {exp_key}   —  {exp['label']}")
    print(f"Cache dir :  {cache_dir}")
    print(f"Output    :  {out_dir}")
    print(f"Years     :  early {early_years[0]}-{early_years[-1]}, "
          f"late {late_years[0]}-{late_years[-1]}")
    print("=" * 72)

    arctic_sial = sial_arctic
    fpath = os.path.join(cache_dir, 'arctic_60N_sial.npy')
    if os.path.exists(fpath):
        try:
            tmp = np.load(fpath).astype(bool)
            if tmp.shape == sial_shape:
                arctic_sial = tmp
                print(f"  (using arctic_60N_sial.npy, {int(arctic_sial.sum()):,} px)")
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

    # ── per-year discrimination ───────────────────────────────────────
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
                                         year=year, period=period, month=mname, **b))
                if r_edge:
                    rows_edge.append(dict(
                        experiment=exp_key, product=product, year=year,
                        period=period, month=mname,
                        n_edge_px=r_edge['n_edge_px'],
                        n_ice=r_edge['n_ice'], n_ocean=r_edge['n_ocean'],
                        mean_alb_ice=r_edge['mean_alb_ice'],
                        mean_alb_ocean=r_edge['mean_alb_ocean'],
                        delta_alpha=r_edge['delta_alpha'],
                        auc=r_edge['auc'], youden_thr=r_edge['youden_thr'],
                        youden_J=r_edge['youden_J'],
                    ))
                if r_reg:
                    rows_reg.append(dict(experiment=exp_key, product=product,
                                         year=year, period=period, month=mname, **r_reg))

                roc_store[mname][product][year]      = (r_full['tpr'], r_full['fpr'])
                if r_edge:
                    roc_edge_store[mname][product][year] = (r_edge['tpr'], r_edge['fpr'])
                box_store[mname][product]['ice']  .extend(r_full['alb_ice_vals'][:40_000].tolist())
                box_store[mname][product]['ocean'].extend(r_full['alb_ocn_vals'][:40_000].tolist())
                if r_dist:
                    for k in ('mean', 'p25', 'p75', 'n'):
                        dist_store[mname][product][k].append(r_dist[k])
                vmask = arctic_sial & np.isfinite(alb) & np.isfinite(sic)
                if vmask.sum() > 0:
                    a_f, s_f = alb[vmask], sic[vmask]
                    k = min(3000, len(a_f))
                    pick = np.random.default_rng(year * 100 + mnum).choice(
                        len(a_f), k, replace=False)
                    scat_store[mname][product]['alb'].extend(a_f[pick].tolist())
                    scat_store[mname][product]['sic'].extend(s_f[pick].tolist())
        print(" ✓")
        gc.collect()

    # ── period-level early-vs-late ────────────────────────────────────
    print("\n  -- period-level early-vs-late analysis --")
    for mname in MONTH_ORDER:
        for period_label, period_years in [('early', early_years), ('late', late_years)]:
            sic_clim = load_climatology_sic_from_cache(cache_dir, period_label, mname)
            if sic_clim is None:
                continue
            sic_edge = np.where(arctic_sial & ~np.isfinite(sic_clim), 0.0, sic_clim)

            for product in ['CLARA-A3', 'APP-x']:
                alb_stack = [a for y in period_years
                             for a in [load_albedo_cached(cache_dir, product, y, mname)]
                             if a is not None]
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
                    delta_alpha_full =(r_full['delta_alpha'] if r_full else np.nan),
                    auc_full         =(r_full['auc']         if r_full else np.nan),
                    youden_thr_full  =(r_full['youden_thr']  if r_full else np.nan),
                    delta_alpha_edge =(r_edge['delta_alpha'] if r_edge else np.nan),
                    auc_edge         =(r_edge['auc']         if r_edge else np.nan),
                    reg_slope        =(r_reg['slope']        if r_reg  else np.nan),
                    reg_r2           =(r_reg['r2']           if r_reg  else np.nan),
                ))
                if r_dist:
                    dist_store[mname][f"_period_{period_label}_{product}"] = r_dist

    df_yr     = pd.DataFrame(rows_yr)
    df_bin    = pd.DataFrame(rows_bin)
    df_edge   = pd.DataFrame(rows_edge)
    df_reg    = pd.DataFrame(rows_reg)
    df_period = pd.DataFrame(rows_period)

    # ── CSVs ─────────────────────────────────────────────────────────
    df_yr.to_csv    (os.path.join(out_dir, 'albedo_discrimination_yearly.csv'),     index=False)
    df_bin.to_csv   (os.path.join(out_dir, 'albedo_discrimination_sic_bins.csv'),   index=False)
    df_edge.to_csv  (os.path.join(out_dir, 'albedo_discrimination_edge_zone.csv'),  index=False)
    df_reg.to_csv   (os.path.join(out_dir, 'albedo_discrimination_regression.csv'), index=False)
    df_period.to_csv(os.path.join(out_dir, 'albedo_discrimination_period.csv'),     index=False)

    summary = pd.DataFrame()
    if not df_yr.empty:
        summary = (df_yr.groupby(['product', 'month'])
                        .agg(
                            delta_alpha_mean      =('delta_alpha',       'mean'),
                            delta_alpha_std       =('delta_alpha',       'std'),
                            mean_alb_ice_mean     =('mean_alb_ice',      'mean'),
                            mean_alb_ocn_mean     =('mean_alb_ocean',    'mean'),
                            auc_mean              =('auc',               'mean'),
                            auc_std               =('auc',               'std'),
                            youden_thr_mean       =('youden_thr',        'mean'),
                            auc_edge_mean         =('auc_edge',          'mean'),
                            auc_edge_std          =('auc_edge',          'std'),
                            delta_alpha_edge_mean =('delta_alpha_edge',  'mean'),
                            reg_slope_mean        =('reg_slope',         'mean'),
                            reg_intercept_mean    =('reg_intercept',     'mean'),
                            reg_r2_mean           =('reg_r2',            'mean'),
                            n_years               =('year',              'count'),
                        ).reset_index())
        summary['month'] = pd.Categorical(summary['month'],
                                           categories=MONTH_ORDER, ordered=True)
        summary = summary.sort_values(['product', 'month'])
        summary.to_csv(os.path.join(out_dir, 'albedo_discrimination_summary.csv'),
                       index=False)

    # ── distance profile CSV ──────────────────────────────────────────
    dist_rows = []
    for mname, pdict in dist_store.items():
        for product in ['CLARA-A3', 'APP-x']:
            d = pdict.get(product, {})
            if not d or not d.get('mean'): continue
            means = np.nanmean(np.vstack(d['mean']), axis=0)
            p25   = np.nanmean(np.vstack(d['p25']),  axis=0)
            p75   = np.nanmean(np.vstack(d['p75']),  axis=0)
            ns    = np.nansum (np.vstack(d['n']),    axis=0)
            for c, mn, lo, hi, n in zip(DIST_BIN_EDGES[:-1] + 0.5,
                                          means, p25, p75, ns):
                dist_rows.append(dict(experiment=exp_key, product=product,
                                       month=mname, bin_center_px=float(c),
                                       mean_alb=float(mn) if np.isfinite(mn) else np.nan,
                                       p25_alb=float(lo) if np.isfinite(lo) else np.nan,
                                       p75_alb=float(hi) if np.isfinite(hi) else np.nan,
                                       n_pixels_total=int(n)))
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(out_dir, 'albedo_discrimination_distance_profile.csv'),
        index=False)

    # ── console summary ───────────────────────────────────────────────
    if not summary.empty:
        print("\n" + "-" * 72)
        print(f"MONTHLY SUMMARY  ({exp_key})")
        print("-" * 72)
        for prod in ['CLARA-A3', 'APP-x']:
            sub = summary[summary['product'] == prod]
            if sub.empty: continue
            print(f"\n{prod}")
            print(f"  {'Month':<10} {'Δα_full':>9} {'AUC_full':>9} "
                  f"{'AUC_edge':>9} {'Δα_edge':>9} {'slope':>7} {'R²':>6} {'thr':>6}")
            print("  " + "-" * 65)
            for _, r in sub.iterrows():
                print(f"  {str(r['month']):<10} "
                      f"{r['delta_alpha_mean']:>+9.4f} "
                      f"{r['auc_mean']:>9.3f} "
                      f"{r['auc_edge_mean']:>9.3f} "
                      f"{r['delta_alpha_edge_mean']:>+9.4f} "
                      f"{r['reg_slope_mean']:>7.3f} "
                      f"{r['reg_r2_mean']:>6.3f} "
                      f"{r['youden_thr_mean']:>6.3f}")

    # ── save everything the notebook needs ────────────────────────────
    pkl_path = os.path.join(out_dir, f'plot_data_{exp_key}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'exp_key':        exp_key,
            'exp_label':      exp['label'],
            'early_years':    early_years,
            'late_years':     late_years,
            'out_dir':        out_dir,
            'roc_store':      roc_store,
            'roc_edge_store': roc_edge_store,
            'box_store':      box_store,
            'dist_store':     dist_store,
            'scat_store':     scat_store,
            'df_yr':          df_yr,
            'df_bin':         df_bin,
            'df_edge':        df_edge,
            'df_period':      df_period,
            'summary':        summary,
            'MONTH_ORDER':    MONTH_ORDER,
            'SIC_BINS':       SIC_BINS,
            'DIST_BIN_EDGES': DIST_BIN_EDGES,
            'ROC_THRESHOLDS': ROC_THRESHOLDS,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  ✓ {exp_key} done   →  {pkl_path}")
    return df_yr, df_period


# ═════════════════════════════════════════════════════════════════════
# STEP 4 — RUN
# ═════════════════════════════════════════════════════════════════════
all_yr, all_period = [], []
for exp_key in EXPS_TO_RUN:
    df_y, df_p = run_experiment(exp_key)
    all_yr.append(df_y); all_period.append(df_p)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
if any(not df.empty for df in all_yr):
    pd.concat(all_yr, ignore_index=True).to_csv(
        os.path.join(OUTPUT_ROOT, 'albedo_discrimination_yearly_ALL.csv'), index=False)
if any(not df.empty for df in all_period):
    df_p_all = pd.concat(all_period, ignore_index=True)
    df_p_all.to_csv(
        os.path.join(OUTPUT_ROOT, 'albedo_discrimination_period_ALL.csv'), index=False)

print(f"\n✓  All experiments complete.  Root: {OUTPUT_ROOT}")