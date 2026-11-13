'''#!/usr/bin/env python
"""
pistone_method_comparison.py
────────────────────────────────────────────────────────────────────────────
Compares three SIAF estimation methods side-by-side to show the impact of
the Pistone-correct regional averaging vs. the pixel-product approach:

  Method A — CERES, pixel-product (current code):
      SIAF_r = mean_r( IS_pixel × RS_pixel )

  Method B — CERES, Pistone-correct (product of regional means):
      SIAF_r = mean_r( IS_pixel ) × mean_r( RS_pixel )

  Method C — ERA5 Radiative Kernel (Huang & Huang 2023):
      SIAF_r = mean_r( IS_pixel ) × mean_r( K_pixel )
               where K is the ERA5 offline shortwave surface albedo kernel

Both CMIP6-era (1982–2014) and Obs-era (1982–2024) periods are computed.
A simple 1000-draw bootstrap propagates interannual albedo variability and
ΔT structural uncertainty (across GISTEMP, Cowtan & Way, NCEP, ERA5).

Outputs:
  pistone_comparison_p{pval}.png   — grouped stacked bar chart per p-value
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, gc
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

# ============================================================
# CONFIG  —  update paths to match your environment
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
KERNEL_FILE = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR  = 'SFC_all'
KERNEL_SCALE = 100.0      # unit conversion
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)

# Temperature dataset paths
GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

PISTONE_REGIONS = [
    ('EU', 60, 90,    0,   60),
    ('WA', 60, 90,   60,  120),
    ('WP', 60, 90,  120,  180),
    ('EP', 60, 90, -180, -120),
    ('NA', 60, 90, -120,  -60),
    ('AO', 60, 90,  -60,    0),
]

EARLY      = list(range(1982, 1992))
LATE_CMIP6 = list(range(2005, 2015))
LATE_OBS   = list(range(2015, 2025))
CERES_CMIP6 = list(range(2000, 2015))
CERES_OBS   = list(range(2000, 2025))

TREND_CMIP6 = (1982, 2014)
TREND_OBS   = (1982, 2024)

SIC_THRESH    = 0.15
AREA_RATIO    = 0.0347
SEASON_FRAC   = 6. / 12.
SCALE         = AREA_RATIO * SEASON_FRAC
PVAL_THRESHOLDS = [0.01, 0.05, 0.10]
N_BOOT        = 1000
BOOT_SEED     = 42

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

# ============================================================
# TEMPERATURE LOADERS
# ============================================================

def _load_gistemp(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        return df['Year'].values.astype(float), df['J-D'].values.astype(float)
    except: return None, None

def _load_cowtan(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None,
                         names=['year','anomaly','u1','u2','u3'])
        return df['year'].values.astype(float), df['anomaly'].values.astype(float)
    except: return None, None

def _load_ncep(path):
    if not os.path.exists(path): return None, None
    try:
        ds = xr.open_dataset(path)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except: return None, None

def _load_era5_t2m(pattern):
    files = sorted(glob.glob(pattern))
    if not files: return None, None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
        ta = ds['VAR_2T'].weighted(w).mean(dim=['latitude','longitude']).resample(time='1YE').mean()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except: return None, None

def build_T_sources():
    candidates = [
        ('GISTEMP',   _load_gistemp(GISTEMP_PATH)),
        ('CowtanWay', _load_cowtan(COWTANWAY_PATH)),
        ('NCEP',      _load_ncep(NCEP_PATH)),
        ('ERA5',      _load_era5_t2m(ERA5_GLOB)),
    ]
    sources = [(n, yy, vv) for n,(yy,vv) in candidates
               if yy is not None and np.isfinite(vv).sum() > 10]
    print(f'  Loaded {len(sources)} T-sources: {[s[0] for s in sources]}')
    if len(sources) < 2:
        raise RuntimeError('Need at least 2 T sources.')
    return sources

def trend_dT(years, vals, y0, y1):
    m = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if m.sum() < 5: return np.nan
    return stats.linregress(years[m], vals[m]).slope * (y1 - y0)

def compute_dT_samples(T_sources, y0, y1):
    vals = [trend_dT(yy, vv, y0, y1)
            for _, yy, vv in T_sources]
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=np.float32)
    if len(vals) == 0:
        default = 0.64 if y1 <= 2014 else 0.84
        return np.array([default], dtype=np.float32)
    print(f'  ΔT {y0}–{y1}: mean={vals.mean():+.4f}  '
          f'std={vals.std():.4f}  n={len(vals)}  samples={vals}')
    return vals

# ============================================================
# GRID / SPATIAL UTILITIES
# ============================================================

def latlon_to_xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat).ravel()), np.deg2rad(np.asarray(lon).ravel())
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo), np.sin(la)])

def build_scatter_index(src_lat, src_lon, tgt_lat, tgt_lon):
    tgt_xyz = latlon_to_xyz(tgt_lat, tgt_lon)
    src_xyz = latlon_to_xyz(src_lat, src_lon)
    _, idx  = cKDTree(tgt_xyz).query(src_xyz, workers=-1)
    return idx.astype(np.int32)

def build_nn_index(src_lat, src_lon, tgt_lat, tgt_lon):
    _, idx = cKDTree(latlon_to_xyz(src_lat, src_lon)).query(
        latlon_to_xyz(tgt_lat, tgt_lon), workers=-1)
    return idx.astype(np.int32)

def scatter_avg(arr_flat, idx, shape):
    n = int(np.prod(shape))
    acc = np.zeros(n, np.float32); cnt = np.zeros(n, np.float32)
    v = np.isfinite(arr_flat)
    np.add.at(acc, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(cnt, idx[v], np.float32(1.))
    out = np.full(n, np.nan, np.float32)
    m = cnt > 0; out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def regrid_sial_to_psn(arr, fill_zero=True):
    if fill_zero:
        src = np.where(ARCTIC_SIAL,
                       np.where(np.isfinite(arr), arr, np.float32(0.)), np.nan)
    else:
        src = arr
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)

def region_mean(arr_psn, mask):
    v, w = arr_psn[mask], AREA_W[mask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0: return np.nan, 0.
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))

# ============================================================
# DATA LOADERS
# ============================================================

_CERES_VARS = ['solar_mon','toa_sw_all_mon','sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon','toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon','sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt>1., rsdt, np.nan)
        Sd  = np.where(rsds>1., rsds, np.nan)
        a   = np.clip(rsus/Sd, 0., 1.)
        p   = rsds/S; q = rsut/S
        den = q + p - 2.*p*a
        R   = np.clip((q-p*a)/np.where(np.abs(den)>1e-6,den,np.nan),0.,.99)
        A   = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan),0.,.99)
        Sc  = np.where(rsdscs>1.,rsdscs,np.nan)
        ac  = np.clip(rsuscs/Sc,0.,1.)
        pc  = rsdscs/S; qc = rsutcs/S
        dc  = qc+pc-2.*pc*ac
        Rc  = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan),0.,.99)
        Rl  = np.clip(R-Rc,0.,.99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               /np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS.astype(np.float32)

def load_alb_yearly(cache_dir, mname, years):
    use_appx = (mname == 'September')
    result = []
    for y in years:
        p = (f'{cache_dir}/appx_alb_{mname}_{y}.npy' if use_appx
             else f'{cache_dir}/clara_alb_{mname}_{y}.npy')
        if not os.path.exists(p): continue
        raw = np.load(p).astype(np.float32)
        if use_appx and IDX_APPX_TO_SIAL is not None:
            a = (raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
                 if raw.shape != SIAL_SHAPE else raw.copy())
        else:
            a = raw.copy()
        result.append(a); del raw
    return result

def load_alb_mean(cache_dir, mname, years):
    stacks = load_alb_yearly(cache_dir, mname, years)
    if not stacks: return None
    s = np.nansum(np.stack(stacks, axis=0), axis=0)
    c = np.sum([np.isfinite(a) for a in stacks], axis=0).astype(np.float32)
    out = np.full(SIAL_SHAPE, np.nan, np.float32)
    v = c > 0; out[v] = s[v] / c[v]
    return out

def load_RS_ceres_mean(years, mn):
    sums = {k: None for k in _CERES_VARS}; cnt = 0
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idxs = [i for i,t in enumerate(times)
                if t.year in years and t.month == mn]
        if not idxs: return None
        for i in idxs:
            sl = ds.isel(time=i).load()
            for k in _CERES_VARS:
                a = sl[k].values.astype(np.float32)
                sums[k] = a if cnt == 0 else sums[k] + a
            cnt += 1; del sl
    if cnt == 0: return None
    for k in sums: sums[k] /= cnt
    RS = t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                    sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                    sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                    sums['sfc_sw_up_clr_c_mon'])
    gc.collect(); return RS

def load_RS_ceres_yearly(years, mn):
    result = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in years:
            idxs = [i for i,t in enumerate(times)
                    if t.year == y and t.month == mn]
            if not idxs: continue
            sums = {k: None for k in _CERES_VARS}; cnt = 0
            for i in idxs:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1; del sl
            if cnt == 0: continue
            for k in sums: sums[k] /= cnt
            result.append(t07_invert(
                sums['solar_mon'], sums['toa_sw_all_mon'],
                sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                sums['sfc_sw_up_clr_c_mon']))
    gc.collect(); return result

def load_sic_data(cache_dir, mname):
    try:
        return dict(
            slope = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval  = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except: return None

def load_kernel_psn(mn):
    """Load ERA5 kernel for month mn, regridded to PSN25."""
    with xr.open_dataset(KERNEL_FILE) as ds:
        # find the time dim (size 12)
        tdim = next((d for d,s in ds.dims.items() if s == 12), None)
        if tdim is None: raise KeyError('No size-12 dim in kernel file')
        vname = KERNEL_VAR
        if vname not in ds:
            vname = next((v for v in ds.data_vars
                          if v.lower() == KERNEL_VAR.lower()), None)
        lat_n = 'latitude' if 'latitude' in ds else 'lat'
        lon_n = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[vname].isel({tdim: mn-1}).values.astype(np.float32)
    k_raw = np.abs(k_raw) * KERNEL_SCALE
    idx_k = build_scatter_index(k_lat2d.ravel(), k_lon2d.ravel(),
                                 PSN_LAT.ravel(), PSN_LON.ravel())
    return scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)

# ============================================================
# IS COMPUTATION
# ============================================================

def compute_IS(ae, al, sic, dT, pval_thresh):
    """Returns (IS_tot, IS_ext, IS_drk) on SIAL grid."""
    dalpha  = al - ae
    slope   = sic['slope']; pv = sic['pval']
    sic_e   = sic['sic_e']; sic_l = sic['sic_l']
    valid   = ARCTIC_SIAL & np.isfinite(dalpha)
    chang   = (valid & np.isfinite(slope) & np.isfinite(pv)
               & (slope < 0) & (pv < pval_thresh))
    stable  = (valid & ~chang
               & (np.fmax(sic_e, sic_l) >= SIC_THRESH))
    IS_ext  = np.where(chang,  -dalpha/dT, np.nan).astype(np.float32)
    IS_drk  = np.where(stable, -dalpha/dT, np.nan).astype(np.float32)
    IS_tot  = np.where(
        np.isfinite(IS_ext) | np.isfinite(IS_drk),
        np.where(np.isfinite(IS_ext), IS_ext, 0.)
        + np.where(np.isfinite(IS_drk), IS_drk, 0.),
        np.nan).astype(np.float32)
    IS_ext  = np.where(np.isfinite(IS_tot),
                       np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
    IS_drk  = np.where(np.isfinite(IS_tot),
                       np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)
    return IS_tot, IS_ext, IS_drk

# ============================================================
# REGIONAL SIAF — THREE METHODS
# ============================================================

def siaf_pixel_product(IS_t_p, IS_x_p, IS_d_p, RS_psn):
    """
    Method A (current code): mean_region( IS_pixel × RS_pixel )
    """
    rows = []
    for code, *_ in PISTONE_REGIONS:
        rm = REGION_MASKS[code]
        if rm.sum() == 0: continue
        ST, w = region_mean(IS_t_p * RS_psn, rm)
        if not (np.isfinite(ST) and w > 0): continue
        SX, _ = region_mean(IS_x_p * RS_psn, rm)
        SD, _ = region_mean(IS_d_p * RS_psn, rm)
        rows.append([ST, SX, SD, w])
    if not rows: return None
    rows = np.array(rows)
    W = rows[:, 3]
    tot_w = np.sum(W)
    return [float(np.sum(rows[:,0]*W)/tot_w),
            float(np.sum(rows[:,1]*W)/tot_w),
            float(np.sum(rows[:,2]*W)/tot_w)]

def siaf_pistone_correct(IS_t_p, IS_x_p, IS_d_p, RS_psn):
    """
    Method B (Pistone-correct): mean_region(IS) × mean_region(RS)
    """
    rows = []
    for code, *_ in PISTONE_REGIONS:
        rm = REGION_MASKS[code]
        if rm.sum() == 0: continue
        IS_r, w = region_mean(IS_t_p, rm)
        if not (np.isfinite(IS_r) and w > 0): continue
        RS_r, _ = region_mean(RS_psn, rm)
        if not np.isfinite(RS_r): continue
        IX_r, _ = region_mean(IS_x_p, rm)
        ID_r, _ = region_mean(IS_d_p, rm)
        rows.append([IS_r*RS_r, IX_r*RS_r, ID_r*RS_r, w])
    if not rows: return None
    rows = np.array(rows)
    W = rows[:, 3]
    tot_w = np.sum(W)
    return [float(np.sum(rows[:,0]*W)/tot_w),
            float(np.sum(rows[:,1]*W)/tot_w),
            float(np.sum(rows[:,2]*W)/tot_w)]

# ============================================================
# BOOTSTRAP
# ============================================================

def _stack(arr_list):
    valid = [a for a in arr_list if a is not None]
    return np.stack(valid, axis=0).astype(np.float32) if valid else None

def _resamp(stack, rng):
    n = stack.shape[0]
    return np.nanmean(stack[rng.integers(0, n, size=n)], axis=0).astype(np.float32)

def bootstrap_month(early_stk, late_stk, rs_or_kernel_stk,
                    sic, dT_samples, rng, pval,
                    method='pixel'):
    """
    method: 'pixel' = mean(IS×RS)  |  'pistone' = mean(IS)×mean(RS)
    For kernel (method='pistone'), rs_or_kernel_stk is a single 2D
    array (no yearly variation), so we pass it as shape (1, H, W).
    """
    boot = np.full((N_BOOT, 3), np.nan, np.float64)
    n_t = len(dT_samples)
    for b in range(N_BOOT):
        ae  = _resamp(early_stk, rng)
        al  = _resamp(late_stk,  rng)
        dT  = float(dT_samples[rng.integers(0, n_t)])
        RS  = _resamp(rs_or_kernel_stk, rng)
        IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT, pval)
        if IS_t is None: continue
        IS_t_p = regrid_sial_to_psn(IS_t)
        IS_x_p = regrid_sial_to_psn(IS_x)
        IS_d_p = regrid_sial_to_psn(IS_d)
        fn = siaf_pixel_product if method == 'pixel' else siaf_pistone_correct
        r = fn(IS_t_p, IS_x_p, IS_d_p, RS)
        if r is not None:
            boot[b] = r
    return boot

# ============================================================
# ONE-TIME SETUP
# ============================================================
print('='*70)
print('SETUP')
print('='*70)

T_SOURCES = build_T_sources()

# SIAL grid
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sla = ds['lat'].values; slo = ds['lon'].values
if sla.ndim == 1:
    SIAL_LON, SIAL_LAT = np.meshgrid(slo, sla)
else:
    SIAL_LAT, SIAL_LON = sla, slo
SIAL_SHAPE = SIAL_LAT.shape

try:
    ARCTIC_SIAL = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)

# APP-x -> SIAL index
IDX_APPX_TO_SIAL = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        al = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        ao = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if al.ndim == 1:
        ao2, al2 = np.meshgrid(ao, al)
    else:
        al2, ao2 = al, ao
    IDX_APPX_TO_SIAL = build_nn_index(
        al2.ravel(), ao2.ravel(), SIAL_LAT.ravel(), SIAL_LON.ravel())
    print('  APP-x -> SIAL index built.')
except Exception as e:
    print(f'  WARNING: APP-x -> SIAL failed ({e})')

# CERES grid
with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
    c_la = _ds['lat'].values; c_lo = _ds['lon'].values
CERES_LON, CERES_LAT = np.meshgrid(c_lo, c_la)

# PSN25 grid
from pyproj import Transformer
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
PSN_SHAPE = psn_xx.shape
tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = tfm.transform(psn_xx, psn_yy)

phi_ts = np.deg2rad(70.)
k_ps = (1. + np.sin(phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W = (1. / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe as _globe
    lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    DOMAIN_PSN = (PSN_LAT >= 60.) & _globe.is_ocean(PSN_LAT, lp)
except ImportError:
    DOMAIN_PSN = (PSN_LAT >= 60.)
    print('  WARNING: global-land-mask not available; land not excluded.')

REGION_MASKS = {}
for code, lo, hi, lmn, lmx in PISTONE_REGIONS:
    lat_ok = (PSN_LAT >= lo) & (PSN_LAT <= hi)
    lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    lon_ok = ((lp >= lmn) & (lp <= lmx)
              if lmn <= lmx else (lp >= lmn) | (lp <= lmx))
    REGION_MASKS[code] = lat_ok & lon_ok & DOMAIN_PSN

print('  Building SIAL -> PSN25 scatter index ...')
IDX_SIAL_TO_PSN = build_scatter_index(
    SIAL_LAT.ravel(), SIAL_LON.ravel(),
    PSN_LAT.ravel(),  PSN_LON.ravel())
print('  Building CERES -> PSN25 scatter index ...')
IDX_CERES_TO_PSN = build_scatter_index(
    CERES_LAT.ravel(), CERES_LON.ravel(),
    PSN_LAT.ravel(),   PSN_LON.ravel())
print(f'  PSN25 domain: {DOMAIN_PSN.sum():,} ocean cells >= 60N')

# Pre-load kernels (one per month, static — no yearly stacks needed)
print('  Loading ERA5 kernels ...')
KERNEL_PSN = {}
for mn in MONTH_NUMS:
    try:
        KERNEL_PSN[mn] = load_kernel_psn(mn)
        print(f'    Month {mn} kernel loaded.')
    except Exception as e:
        print(f'    Month {mn} kernel FAILED: {e}')
        KERNEL_PSN[mn] = None

# ΔT samples
dT_samples_cmip6 = compute_dT_samples(T_SOURCES, *TREND_CMIP6)
dT_samples_obs   = compute_dT_samples(T_SOURCES, *TREND_OBS)

PERIOD_CFG = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, CERES_CMIP6, dT_samples_cmip6),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   CERES_OBS,   dT_samples_obs),
]

RNG_A = np.random.default_rng(BOOT_SEED)
RNG_B = np.random.default_rng(BOOT_SEED + 1)
RNG_C = np.random.default_rng(BOOT_SEED + 2)

# ============================================================
# MAIN LOOP — compute all three methods
# ============================================================
print('\n' + '='*70)
print('COMPUTING SIAF — three methods, two periods, three p-values')
print('='*70)

# Results storage: {(method_label, period, pval): [monthly boot arrays]}
BOOT = {}
POINT = {}   # {(method_label, period, pval): [monthly [tot,ext,drk]]}

METHOD_LABELS = ['A_pixel_product', 'B_pistone_correct', 'C_kernel_ERA5']

for ml in METHOD_LABELS:
    for period, *_ in PERIOD_CFG:
        for pval in PVAL_THRESHOLDS:
            BOOT[ (ml, period, pval)] = []
            POINT[(ml, period, pval)] = []

for period, cache_dir, early_yrs, late_yrs, ceres_yrs, dT_samps in PERIOD_CFG:
    print(f'\n--- Period: {period} ---')
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f'  {mname} ...', end=' ', flush=True)

        ae     = load_alb_mean(cache_dir, mname, early_yrs)
        al     = load_alb_mean(cache_dir, mname, late_yrs)
        sic    = load_sic_data(cache_dir, mname)
        RS_c   = load_RS_ceres_mean(ceres_yrs, mn)
        k_psn  = KERNEL_PSN.get(mn)

        # Yearly stacks for bootstrap
        early_stk = _stack(load_alb_yearly(cache_dir, mname, early_yrs))
        late_stk  = _stack(load_alb_yearly(cache_dir, mname, late_yrs))
        rs_c_yearly = load_RS_ceres_yearly(ceres_yrs, mn)
        rs_c_stk  = _stack([scatter_avg(r.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
                             for r in rs_c_yearly]) if rs_c_yearly else None

        # kernel "stack" is just the single array repeated (no interannual variation)
        k_stk = (np.stack([k_psn], axis=0).astype(np.float32)
                 if k_psn is not None else None)

        RS_psn = (scatter_avg(RS_c.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
                  if RS_c is not None else None)

        have_ceres  = (ae is not None and al is not None and sic is not None
                       and RS_psn is not None)
        have_kernel = (ae is not None and al is not None and sic is not None
                       and k_psn is not None)

        for pval in PVAL_THRESHOLDS:
            # ── Point estimates ──────────────────────────────────────────
            if have_ceres:
                dT = float(np.mean(dT_samps))
                IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT, pval)
                IS_t_p = regrid_sial_to_psn(IS_t)
                IS_x_p = regrid_sial_to_psn(IS_x)
                IS_d_p = regrid_sial_to_psn(IS_d)

                rA = siaf_pixel_product  (IS_t_p, IS_x_p, IS_d_p, RS_psn)
                rB = siaf_pistone_correct(IS_t_p, IS_x_p, IS_d_p, RS_psn)
                POINT[('A_pixel_product',   period, pval)].append(rA)
                POINT[('B_pistone_correct', period, pval)].append(rB)
            else:
                POINT[('A_pixel_product',   period, pval)].append(None)
                POINT[('B_pistone_correct', period, pval)].append(None)

            if have_kernel:
                dT = float(np.mean(dT_samps))
                IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT, pval)
                IS_t_p = regrid_sial_to_psn(IS_t)
                IS_x_p = regrid_sial_to_psn(IS_x)
                IS_d_p = regrid_sial_to_psn(IS_d)
                rC = siaf_pistone_correct(IS_t_p, IS_x_p, IS_d_p, k_psn)
                POINT[('C_kernel_ERA5', period, pval)].append(rC)
            else:
                POINT[('C_kernel_ERA5', period, pval)].append(None)

            # ── Bootstrap ────────────────────────────────────────────────
            if have_ceres and early_stk is not None and late_stk is not None and rs_c_stk is not None:
                bA = bootstrap_month(early_stk, late_stk, rs_c_stk,
                                     sic, dT_samps, RNG_A, pval, method='pixel')
                bB = bootstrap_month(early_stk, late_stk, rs_c_stk,
                                     sic, dT_samps, RNG_B, pval, method='pistone')
                BOOT[('A_pixel_product',   period, pval)].append(bA)
                BOOT[('B_pistone_correct', period, pval)].append(bB)
            else:
                BOOT[('A_pixel_product',   period, pval)].append(None)
                BOOT[('B_pistone_correct', period, pval)].append(None)

            if have_kernel and early_stk is not None and late_stk is not None:
                bC = bootstrap_month(early_stk, late_stk, k_stk,
                                     sic, dT_samps, RNG_C, pval, method='pistone')
                BOOT[('C_kernel_ERA5', period, pval)].append(bC)
            else:
                BOOT[('C_kernel_ERA5', period, pval)].append(None)

        del ae, al, sic, RS_c, RS_psn, early_stk, late_stk, rs_c_stk, rs_c_yearly
        gc.collect()
        print('done')

# ============================================================
# AGGREGATE TO SEASONAL MEANS
# ============================================================
print('\nAggregating to seasonal SIAF ...')

def seasonal_stats(point_list, boot_list):
    """
    point_list: list of monthly [tot,ext,drk] (or None)
    boot_list:  list of monthly (N_BOOT, 3) arrays (or None)
    Returns dict with tot/ext/drk and 2σ uncertainties.
    """
    valid_pt = [p for p in point_list if p is not None]
    valid_bt = [b for b in boot_list  if b is not None
                and not np.all(np.isnan(b))]

    if not valid_pt:
        return dict(tot=np.nan, ext=np.nan, drk=np.nan,
                    tot_2s=np.nan, ext_2s=np.nan, drk_2s=np.nan)

    pt_arr = np.array(valid_pt)                  # (n_months, 3)
    tot = float(np.nanmean(pt_arr[:, 0]))
    ext = float(np.nanmean(pt_arr[:, 1]))
    drk = float(np.nanmean(pt_arr[:, 2]))

    if valid_bt:
        seas_boot = np.nanmean(np.stack(valid_bt, axis=1), axis=1)  # (N_BOOT, 3)
        tot_2s = float(2 * np.nanstd(seas_boot[:, 0]))
        ext_2s = float(2 * np.nanstd(seas_boot[:, 1]))
        drk_2s = float(2 * np.nanstd(seas_boot[:, 2]))
    else:
        tot_2s = ext_2s = drk_2s = np.nan

    return dict(tot=tot, ext=ext, drk=drk,
                tot_2s=tot_2s, ext_2s=ext_2s, drk_2s=drk_2s)

RESULTS = {}  # {(method, period, pval): stats_dict}

for ml in METHOD_LABELS:
    for period, *_ in PERIOD_CFG:
        for pval in PVAL_THRESHOLDS:
            pt = POINT[(ml, period, pval)]
            bt = BOOT[ (ml, period, pval)]
            RESULTS[(ml, period, pval)] = seasonal_stats(pt, bt)

# Print summary table
for pval in PVAL_THRESHOLDS:
    print(f'\n--- p < {pval} ---')
    print(f"{'Method':<25} {'Period':<10} {'tot_glo±2σ':^22} "
          f"{'ext_glo':^12} {'drk_glo':^12} {'ext%':>6} {'drk%':>6}")
    for ml in METHOD_LABELS:
        for period, *_ in PERIOD_CFG:
            r = RESULTS[(ml, period, pval)]
            tg  = r['tot'] * SCALE
            eg  = r['ext'] * SCALE
            dg  = r['drk'] * SCALE
            ts  = r['tot_2s'] * SCALE
            ep  = r['ext']/r['tot']*100 if r['tot'] != 0 else np.nan
            dp  = r['drk']/r['tot']*100 if r['tot'] != 0 else np.nan
            print(f"{ml:<25} {period:<10} "
                  f"{tg:+.4f} ± {ts:.4f}  {eg:+.4f}  {dg:+.4f}  "
                  f"{ep:5.1f}%  {dp:5.1f}%")

# ============================================================
# BAR CHART
# ============================================================

METHOD_DISPLAY = {
    'A_pixel_product':   'CERES EBAF\nmean(IS × RS)',
    'B_pistone_correct': 'CERES EBAF\nmean(IS) × mean(RS)\n[Pistone-correct]',
    'C_kernel_ERA5':     'ERA5 Kernel\n(Huang & Huang 2023)',
}

COL_EXT = '#3A6BAF'
COL_DRK = '#D95F1A'
ALPHA   = {'CMIP6': 0.55, 'Obs-era': 1.00}
HATCHES = {'CMIP6': '////', 'Obs-era': ''}

def make_comparison_barplot(pval, out_path):
    periods  = ['CMIP6', 'Obs-era']
    methods  = METHOD_LABELS
    n        = len(methods)
    x        = np.arange(n)
    w        = 0.32
    offsets  = {'CMIP6': -(w/2 + 0.03), 'Obs-era': (w/2 + 0.03)}

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    legend_handles = []

    for period in periods:
        a, hat = ALPHA[period], HATCHES[period]
        for i, ml in enumerate(methods):
            r   = RESULTS[(ml, period, pval)]
            if not np.isfinite(r['tot']): continue
            tot = r['tot'] * SCALE
            ext = r['ext'] * SCALE
            drk = r['drk'] * SCALE
            err = r['tot_2s'] * SCALE
            xp  = x[i] + offsets[period]

            ax.bar(xp, ext, w, color=COL_EXT, alpha=a, hatch=hat,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.bar(xp, drk, w, bottom=ext, color=COL_DRK, alpha=a, hatch=hat,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.errorbar(xp, tot, yerr=err, fmt='none',
                        color='#222222', capsize=5,
                        linewidth=1.5, capthick=1.5, zorder=5)

            ep = ext/tot*100 if tot > 0 else 0
            dp = drk/tot*100 if tot > 0 else 0
            if ext > 0.002:
                ax.text(xp, ext/2, f'{ep:.0f}%', ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold', zorder=6)
            if drk > 0.002:
                ax.text(xp, ext + drk/2, f'{dp:.0f}%', ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold', zorder=6)

            # Period label above bar
            plbl = 'CMIP6\n(1982–2014)' if period == 'CMIP6' else 'Obs\n(1982–2024)'
            ax.text(xp, tot + err + 0.001, plbl, ha='center', va='bottom',
                    color='#555555', fontsize=7, style='italic')

        legend_handles.append(mpatches.Patch(
            facecolor=COL_EXT, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Extent ({period})'))
        legend_handles.append(mpatches.Patch(
            facecolor=COL_DRK, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Darkening ({period})'))

    # Bracket annotation showing A vs B difference
    sub_ax = ax.transData
    for period in periods:
        rA = RESULTS[('A_pixel_product',   period, pval)]
        rB = RESULTS[('B_pistone_correct', period, pval)]
        if np.isfinite(rA['tot']) and np.isfinite(rB['tot']):
            diff_pct = (rB['tot'] - rA['tot']) / rA['tot'] * 100
            mid_x    = (x[0]+offsets[period] + x[1]+offsets[period]) / 2
            y_top    = max(rA['tot'], rB['tot']) * SCALE + rA['tot_2s']*SCALE + 0.012
            ax.annotate(f'Δ = {diff_pct:+.1f}%',
                        xy=(mid_x, y_top), ha='center', va='bottom',
                        fontsize=8, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  ec='#AAAAAA', alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[ml] for ml in methods], fontsize=10.5)
    ax.set_ylabel('Global Annual SIAF  (W m⁻² K⁻¹)', fontsize=11)
    ax.set_title(
        f'Impact of Regional Averaging Method on SIAF  (p < {pval})\n'
        f'Method A vs B: mean(IS×RS) vs mean(IS)×mean(RS)',
        fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', colors='#444444')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, ncol=2, columnspacing=0.8,
              handlelength=1.4, handleheight=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_path}')

print('\nGenerating comparison bar charts ...')
for pval in PVAL_THRESHOLDS:
    out = os.path.join(OUT_DIR, f'pistone_comparison_p{pval}.png')
    make_comparison_barplot(pval, out)

print('\nDone.')'''

#!/usr/bin/env python
"""
siaf_pixelwise.py
─────────────────────────────────────────────────────────────────────────────
SIAF computed as a direct pixel-wise product IS × RS, area-weighted over
the Arctic Ocean domain (≥60N, non-land), with NO regional averaging step.

  SIAF_arctic = Σ_i [ IS_i × RS_i × w_i ] / Σ_i w_i
  SIAF_global = SIAF_arctic × AREA_RATIO × SEASON_FRAC

Two RS estimates:
  CERES  — T07-inverted shortwave flux  (Pistone et al. 2014 RS definition)
  Kernel — ERA5 offline kernel           (Huang & Huang 2023)

Two periods:
  CMIP6   1982–2014
  Obs-era 1982–2024

Three significance thresholds: p < 0.01, 0.05, 0.10

Bootstrap (N=1000) propagates:
  1. Interannual albedo variability     (yearly CLARA/APP-x stacks)
  2. Interannual CERES RS variability   (yearly T07 stacks)  [CERES only]
  3. ΔT structural uncertainty          (4 T-sources: GISTEMP, C&W, NCEP, ERA5)

Outputs:
  siaf_pixelwise_summary.csv
  siaf_pixelwise_p{pval}.png
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os, gc
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

# ============================================================
# CONFIG
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
KERNEL_FILE = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR  = 'SFC_all'
KERNEL_SCALE = 100.0
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
OUT_CSV     = os.path.join(OUT_DIR, 'siaf_pixelwise_summary.csv')
PLOT_OUT    = os.path.join(OUT_DIR, 'siaf_pixelwise_p{pval}.png')
os.makedirs(OUT_DIR, exist_ok=True)

GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

EARLY       = list(range(1982, 1992))
LATE_CMIP6  = list(range(2005, 2015))
LATE_OBS    = list(range(2015, 2025))
CERES_CMIP6 = list(range(2000, 2015))
CERES_OBS   = list(range(2000, 2025))
TREND_CMIP6 = (1982, 2014)
TREND_OBS   = (1982, 2024)

SIC_THRESH      = 0.15
AREA_RATIO      = 0.0347
SEASON_FRAC     = 6. / 12.
SCALE           = AREA_RATIO * SEASON_FRAC
PVAL_THRESHOLDS = [0.01, 0.05, 0.10]
N_BOOT          = 1000
BOOT_SEED       = 42

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

# ============================================================
# TEMPERATURE LOADERS
# ============================================================

def _load_gistemp(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        return df['Year'].values.astype(float), df['J-D'].values.astype(float)
    except: return None, None

def _load_cowtan(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None,
                         names=['year','anomaly','u1','u2','u3'])
        return df['year'].values.astype(float), df['anomaly'].values.astype(float)
    except: return None, None

def _load_ncep(path):
    if not os.path.exists(path): return None, None
    try:
        ds = xr.open_dataset(path)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except: return None, None

def _load_era5_t2m(pattern):
    files = sorted(glob.glob(pattern))
    if not files: return None, None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
        ta = ds['VAR_2T'].weighted(w).mean(
            dim=['latitude','longitude']).resample(time='1YE').mean()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except: return None, None

def build_T_sources():
    candidates = [
        ('GISTEMP',   _load_gistemp(GISTEMP_PATH)),
        ('CowtanWay', _load_cowtan(COWTANWAY_PATH)),
        ('NCEP',      _load_ncep(NCEP_PATH)),
        ('ERA5',      _load_era5_t2m(ERA5_GLOB)),
    ]
    sources = [(n, yy, vv) for n,(yy,vv) in candidates
               if yy is not None and np.isfinite(vv).sum() > 10]
    print(f'  Loaded {len(sources)} T-sources: {[s[0] for s in sources]}')
    if len(sources) < 2:
        raise RuntimeError('Need at least 2 T sources.')
    return sources

def trend_dT(years, vals, y0, y1):
    m = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if m.sum() < 5: return np.nan
    return stats.linregress(years[m], vals[m]).slope * (y1 - y0)

def compute_dT_samples(T_sources, y0, y1):
    vals = [trend_dT(yy, vv, y0, y1) for _, yy, vv in T_sources]
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=np.float32)
    if len(vals) == 0:
        default = 0.64 if y1 <= 2014 else 0.84
        return np.array([default], dtype=np.float32)
    print(f'  ΔT {y0}–{y1}: mean={vals.mean():+.4f}  '
          f'std={vals.std():.4f}  n={len(vals)}  samples={vals}')
    return vals

# ============================================================
# GRID UTILITIES
# ============================================================

def latlon_to_xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel())
    lo = np.deg2rad(np.asarray(lon).ravel())
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])

def build_scatter_index(src_lat, src_lon, tgt_lat, tgt_lon):
    _, idx = cKDTree(latlon_to_xyz(tgt_lat, tgt_lon)).query(
        latlon_to_xyz(src_lat, src_lon), workers=-1)
    return idx.astype(np.int32)

def build_nn_index(src_lat, src_lon, tgt_lat, tgt_lon):
    _, idx = cKDTree(latlon_to_xyz(src_lat, src_lon)).query(
        latlon_to_xyz(tgt_lat, tgt_lon), workers=-1)
    return idx.astype(np.int32)

def scatter_avg(arr_flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    v   = np.isfinite(arr_flat)
    np.add.at(acc, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(cnt, idx[v], np.float32(1.))
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def regrid_sial_to_psn(arr, fill_zero=True):
    src = (np.where(ARCTIC_SIAL,
                    np.where(np.isfinite(arr), arr, np.float32(0.)), np.nan)
           if fill_zero else arr)
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)

def domain_mean(field_psn):
    """
    Area-weighted mean of field_psn over DOMAIN_PSN.
    Returns scalar. NaN pixels are excluded.
    """
    v  = field_psn[DOMAIN_PSN]
    w  = AREA_W[DOMAIN_PSN]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))

# ============================================================
# DATA LOADERS
# ============================================================

_CERES_VARS = ['solar_mon','toa_sw_all_mon','sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon','toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon','sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt>1.,  rsdt,  np.nan)
        Sd  = np.where(rsds>1.,  rsds,  np.nan)
        a   = np.clip(rsus/Sd, 0., 1.)
        p   = rsds/S;  q  = rsut/S
        den = q + p - 2.*p*a
        R   = np.clip((q-p*a)/np.where(np.abs(den)>1e-6, den, np.nan), 0., .99)
        A   = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan), 0., .99)
        Sc  = np.where(rsdscs>1., rsdscs, np.nan)
        ac  = np.clip(rsuscs/Sc, 0., 1.)
        pc  = rsdscs/S;  qc = rsutcs/S
        dc  = qc+pc-2.*pc*ac
        Rc  = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan),0.,.99)
        Rl  = np.clip(R-Rc,0.,.99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               /np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS.astype(np.float32)

def _ceres_sums_to_RS(sums):
    return t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                      sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                      sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                      sums['sfc_sw_up_clr_c_mon'])

def load_alb_yearly(cache_dir, mname, years):
    use_appx = (mname == 'September')
    result   = []
    for y in years:
        p = (f'{cache_dir}/appx_alb_{mname}_{y}.npy' if use_appx
             else f'{cache_dir}/clara_alb_{mname}_{y}.npy')
        if not os.path.exists(p): continue
        raw = np.load(p).astype(np.float32)
        if use_appx and IDX_APPX_TO_SIAL is not None:
            a = (raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
                 if raw.shape != SIAL_SHAPE else raw.copy())
        else:
            a = raw.copy()
        result.append(a); del raw
    return result

def load_alb_mean(cache_dir, mname, years):
    stacks = load_alb_yearly(cache_dir, mname, years)
    if not stacks: return None
    s = np.nansum(np.stack(stacks,0), axis=0)
    c = np.sum([np.isfinite(a) for a in stacks], axis=0).astype(np.float32)
    out = np.full(SIAL_SHAPE, np.nan, np.float32)
    v = c > 0; out[v] = s[v] / c[v]
    return out

def load_RS_ceres_yearly(years, mn):
    result = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in years:
            idxs = [i for i,t in enumerate(times)
                    if t.year == y and t.month == mn]
            if not idxs: continue
            sums = {k: None for k in _CERES_VARS}; cnt = 0
            for i in idxs:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1; del sl
            if cnt == 0: continue
            for k in sums: sums[k] /= cnt
            result.append(_ceres_sums_to_RS(sums))
    gc.collect(); return result

def load_RS_ceres_mean(years, mn):
    yearly = load_RS_ceres_yearly(years, mn)
    if not yearly: return None
    stk = np.stack(yearly, 0)
    return np.nanmean(stk, axis=0).astype(np.float32)

def load_sic_data(cache_dir, mname):
    try:
        return dict(
            slope = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval  = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except: return None

def load_kernel_psn(mn):
    with xr.open_dataset(KERNEL_FILE) as ds:
        tdim  = next((d for d,s in ds.sizes.items() if s == 12), None)
        if tdim is None: raise KeyError('No size-12 dim in kernel file')
        vname = KERNEL_VAR
        if vname not in ds:
            vname = next((v for v in ds.data_vars
                          if v.lower() == KERNEL_VAR.lower()), None)
        lat_n = 'latitude' if 'latitude' in ds else 'lat'
        lon_n = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[vname].isel({tdim: mn-1}).values.astype(np.float32)
    k_raw = np.abs(k_raw) * KERNEL_SCALE
    idx_k = build_scatter_index(k_lat2d.ravel(), k_lon2d.ravel(),
                                 PSN_LAT.ravel(), PSN_LON.ravel())
    return scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)

# ============================================================
# IS COMPUTATION  (on SIAL grid)
# ============================================================

def compute_IS(ae, al, sic, dT, pval_thresh):
    dalpha = al - ae
    valid  = ARCTIC_SIAL & np.isfinite(dalpha)
    chang  = (valid & np.isfinite(sic['slope']) & np.isfinite(sic['pval'])
              & (sic['slope'] < 0) & (sic['pval'] < pval_thresh))
    stable = (valid & ~chang
              & (np.fmax(sic['sic_e'], sic['sic_l']) >= SIC_THRESH))

    IS_ext = np.where(chang,  -dalpha/dT, np.nan).astype(np.float32)
    IS_drk = np.where(stable, -dalpha/dT, np.nan).astype(np.float32)
    IS_tot = np.where(
        np.isfinite(IS_ext) | np.isfinite(IS_drk),
        np.where(np.isfinite(IS_ext), IS_ext, 0.)
        + np.where(np.isfinite(IS_drk), IS_drk, 0.),
        np.nan).astype(np.float32)
    IS_ext = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
    IS_drk = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)
    return IS_tot, IS_ext, IS_drk

# ============================================================
# PIXEL-WISE SIAF  (the whole point of this script)
# ============================================================

def siaf_pixelwise(IS_t_psn, IS_x_psn, IS_d_psn, RS_psn):
    """
    SIAF = area-weighted mean of IS × RS over DOMAIN_PSN.
    No regional aggregation. Returns [tot, ext, drk].
    """
    tot = domain_mean(IS_t_psn * RS_psn)
    ext = domain_mean(IS_x_psn * RS_psn)
    drk = domain_mean(IS_d_psn * RS_psn)
    return [tot, ext, drk]

# ============================================================
# BOOTSTRAP
# ============================================================

def _stack(arr_list):
    valid = [a for a in arr_list if a is not None]
    return np.stack(valid, 0).astype(np.float32) if valid else None

def _resamp(stk, rng):
    n = stk.shape[0]
    return np.nanmean(stk[rng.integers(0, n, size=n)], axis=0).astype(np.float32)

def bootstrap_month(early_stk, late_stk, rs_stk,
                    sic, dT_samples, rng, pval):
    """
    rs_stk shape: (n_years, H, W)  for CERES (interannual variation)
                  (1, H, W)        for kernel (static — no yearly variation)
    """
    boot  = np.full((N_BOOT, 3), np.nan, np.float64)
    n_t   = len(dT_samples)
    for b in range(N_BOOT):
        ae  = _resamp(early_stk, rng)
        al  = _resamp(late_stk,  rng)
        RS  = _resamp(rs_stk,    rng)
        dT  = float(dT_samples[rng.integers(0, n_t)])

        IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT, pval)
        IS_t_p = regrid_sial_to_psn(IS_t)
        IS_x_p = regrid_sial_to_psn(IS_x)
        IS_d_p = regrid_sial_to_psn(IS_d)

        r = siaf_pixelwise(IS_t_p, IS_x_p, IS_d_p, RS)
        if all(np.isfinite(v) for v in r):
            boot[b] = r
    return boot

# ============================================================
# ONE-TIME SETUP
# ============================================================
print('='*70)
print('SETUP')
print('='*70)

T_SOURCES = build_T_sources()

# SIAL grid
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sla = ds['lat'].values; slo = ds['lon'].values
if sla.ndim == 1:
    SIAL_LON, SIAL_LAT = np.meshgrid(slo, sla)
else:
    SIAL_LAT, SIAL_LON = sla, slo
SIAL_SHAPE = SIAL_LAT.shape

try:
    ARCTIC_SIAL = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)

# APP-x -> SIAL index
IDX_APPX_TO_SIAL = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        al = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        ao = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if al.ndim == 1:
        ao2, al2 = np.meshgrid(ao, al)
    else:
        al2, ao2 = al, ao
    IDX_APPX_TO_SIAL = build_nn_index(
        al2.ravel(), ao2.ravel(), SIAL_LAT.ravel(), SIAL_LON.ravel())
    print('  APP-x -> SIAL index built.')
except Exception as e:
    print(f'  WARNING: APP-x -> SIAL failed ({e})')

# CERES grid
with xr.open_dataset(CERES_FILE, engine='netcdf4') as _ds:
    c_la = _ds['lat'].values; c_lo = _ds['lon'].values
CERES_LON, CERES_LAT = np.meshgrid(c_lo, c_la)

# PSN25 grid
from pyproj import Transformer
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
PSN_SHAPE = psn_xx.shape
tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = tfm.transform(psn_xx, psn_yy)

# Area weights (polar stereographic correction)
phi_ts  = np.deg2rad(70.)
k_ps    = (1. + np.sin(phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W  = (1. / k_ps**2).astype(np.float32)

# Ocean domain mask
try:
    from global_land_mask import globe as _globe
    lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    DOMAIN_PSN = (PSN_LAT >= 60.) & _globe.is_ocean(PSN_LAT, lp)
except ImportError:
    DOMAIN_PSN = (PSN_LAT >= 60.)
    print('  WARNING: global-land-mask not available; land not excluded.')

print(f'  PSN25 domain: {DOMAIN_PSN.sum():,} ocean pixels >= 60N')

# Scatter indices
print('  Building SIAL -> PSN25 index ...')
IDX_SIAL_TO_PSN  = build_scatter_index(
    SIAL_LAT.ravel(), SIAL_LON.ravel(),
    PSN_LAT.ravel(),  PSN_LON.ravel())
print('  Building CERES -> PSN25 index ...')
IDX_CERES_TO_PSN = build_scatter_index(
    CERES_LAT.ravel(), CERES_LON.ravel(),
    PSN_LAT.ravel(),   PSN_LON.ravel())

# Pre-load kernels
print('  Loading ERA5 kernels ...')
KERNEL_PSN = {}
for mn in MONTH_NUMS:
    try:
        KERNEL_PSN[mn] = load_kernel_psn(mn)
        print(f'    Month {mn} kernel loaded.')
    except Exception as e:
        print(f'    Month {mn} kernel FAILED: {e}')
        KERNEL_PSN[mn] = None

# ΔT samples
dT_cmip6 = compute_dT_samples(T_SOURCES, *TREND_CMIP6)
dT_obs   = compute_dT_samples(T_SOURCES, *TREND_OBS)

PERIOD_CFG = [
    ('CMIP6',   CACHE_CMIP6, EARLY, LATE_CMIP6, CERES_CMIP6, dT_cmip6),
    ('Obs-era', CACHE_OBS,   EARLY, LATE_OBS,   CERES_OBS,   dT_obs),
]

RNG_CERES  = np.random.default_rng(BOOT_SEED)
RNG_KERNEL = np.random.default_rng(BOOT_SEED + 1)

# ============================================================
# MAIN LOOP
# ============================================================
print('\n' + '='*70)
print('PIXEL-WISE SIAF  — no regional averaging')
print('='*70)

# Storage: {(rs_type, period, pval): list of monthly boot arrays}
BOOT  = {}
POINT = {}
for rs in ['CERES', 'Kernel']:
    for period, *_ in PERIOD_CFG:
        for pval in PVAL_THRESHOLDS:
            BOOT[ (rs, period, pval)] = []
            POINT[(rs, period, pval)] = []

for period, cache_dir, early_yrs, late_yrs, ceres_yrs, dT_samps in PERIOD_CFG:
    print(f'\n--- Period: {period} ---')

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        print(f'  {mname} ...', end=' ', flush=True)

        ae      = load_alb_mean(cache_dir, mname, early_yrs)
        al      = load_alb_mean(cache_dir, mname, late_yrs)
        sic     = load_sic_data(cache_dir, mname)

        # CERES RS
        RS_c_mean   = load_RS_ceres_mean(ceres_yrs, mn)
        RS_c_yearly = load_RS_ceres_yearly(ceres_yrs, mn)
        RS_c_psn    = (scatter_avg(RS_c_mean.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
                       if RS_c_mean is not None else None)
        rs_c_stk    = (_stack([scatter_avg(r.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
                                for r in RS_c_yearly])
                       if RS_c_yearly else None)

        # Kernel RS
        k_psn = KERNEL_PSN.get(mn)
        k_stk = (np.stack([k_psn], 0).astype(np.float32)
                 if k_psn is not None else None)

        # Albedo stacks for bootstrap
        early_stk = _stack(load_alb_yearly(cache_dir, mname, early_yrs))
        late_stk  = _stack(load_alb_yearly(cache_dir, mname, late_yrs))

        have_ceres  = all(x is not None for x in [ae, al, sic, RS_c_psn])
        have_kernel = all(x is not None for x in [ae, al, sic, k_psn])
        have_boot   = (early_stk is not None and late_stk is not None)

        dT_mean = float(np.mean(dT_samps))

        for pval in PVAL_THRESHOLDS:
            # ── CERES point estimate ─────────────────────────────────────
            if have_ceres:
                IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT_mean, pval)
                IS_t_p = regrid_sial_to_psn(IS_t)
                IS_x_p = regrid_sial_to_psn(IS_x)
                IS_d_p = regrid_sial_to_psn(IS_d)
                POINT[('CERES', period, pval)].append(
                    siaf_pixelwise(IS_t_p, IS_x_p, IS_d_p, RS_c_psn))
            else:
                POINT[('CERES', period, pval)].append(None)

            # ── Kernel point estimate ────────────────────────────────────
            if have_kernel:
                IS_t, IS_x, IS_d = compute_IS(ae, al, sic, dT_mean, pval)
                IS_t_p = regrid_sial_to_psn(IS_t)
                IS_x_p = regrid_sial_to_psn(IS_x)
                IS_d_p = regrid_sial_to_psn(IS_d)
                POINT[('Kernel', period, pval)].append(
                    siaf_pixelwise(IS_t_p, IS_x_p, IS_d_p, k_psn))
            else:
                POINT[('Kernel', period, pval)].append(None)

            # ── Bootstrap ────────────────────────────────────────────────
            if have_ceres and have_boot and rs_c_stk is not None:
                BOOT[('CERES', period, pval)].append(
                    bootstrap_month(early_stk, late_stk, rs_c_stk,
                                    sic, dT_samps, RNG_CERES, pval))
            else:
                BOOT[('CERES', period, pval)].append(None)

            if have_kernel and have_boot:
                BOOT[('Kernel', period, pval)].append(
                    bootstrap_month(early_stk, late_stk, k_stk,
                                    sic, dT_samps, RNG_KERNEL, pval))
            else:
                BOOT[('Kernel', period, pval)].append(None)

        del ae, al, sic, RS_c_mean, RS_c_psn, rs_c_stk
        del RS_c_yearly, early_stk, late_stk
        gc.collect()
        print('done')

# ============================================================
# AGGREGATE: seasonal mean + bootstrap 2σ
# ============================================================
print('\nAggregating ...')

def seasonal_stats(point_list, boot_list):
    valid_pt = [p for p in point_list if p is not None]
    valid_bt = [b for b in boot_list
                if b is not None and not np.all(np.isnan(b))]
    if not valid_pt:
        return dict(tot=np.nan, ext=np.nan, drk=np.nan,
                    tot_2s=np.nan, ext_2s=np.nan, drk_2s=np.nan)
    pt   = np.nanmean(np.array(valid_pt), axis=0)
    tot, ext, drk = float(pt[0]), float(pt[1]), float(pt[2])
    if valid_bt:
        seas = np.nanmean(np.stack(valid_bt, 1), axis=1)  # (N_BOOT, 3)
        tot_2s = float(2 * np.nanstd(seas[:, 0]))
        ext_2s = float(2 * np.nanstd(seas[:, 1]))
        drk_2s = float(2 * np.nanstd(seas[:, 2]))
    else:
        tot_2s = ext_2s = drk_2s = np.nan
    return dict(tot=tot, ext=ext, drk=drk,
                tot_2s=tot_2s, ext_2s=ext_2s, drk_2s=drk_2s)

RESULTS = {}
for rs in ['CERES', 'Kernel']:
    for period, *_ in PERIOD_CFG:
        for pval in PVAL_THRESHOLDS:
            RESULTS[(rs, period, pval)] = seasonal_stats(
                POINT[(rs, period, pval)],
                BOOT[ (rs, period, pval)])

# ── Print table ──────────────────────────────────────────────────────────────
print()
for pval in PVAL_THRESHOLDS:
    print(f'\n{"="*70}')
    print(f'p < {pval}')
    print(f'{"="*70}')
    hdr = (f"{'RS method':<10} {'Period':<10} "
           f"{'tot_glo ± 2σ':^22} {'ext_glo':^12} "
           f"{'drk_glo':^12} {'ext%':>6} {'drk%':>6}")
    print(hdr)
    print('-'*len(hdr))
    for rs in ['CERES', 'Kernel']:
        for period, *_ in PERIOD_CFG:
            r   = RESULTS[(rs, period, pval)]
            tg  = r['tot'] * SCALE
            eg  = r['ext'] * SCALE
            dg  = r['drk'] * SCALE
            ts  = r['tot_2s'] * SCALE
            ep  = r['ext']/r['tot']*100 if r['tot'] else np.nan
            dp  = r['drk']/r['tot']*100 if r['tot'] else np.nan
            print(f"{rs:<10} {period:<10} "
                  f"{tg:+.4f} ± {ts:.4f}  {eg:+.4f}  {dg:+.4f}  "
                  f"{ep:5.1f}%  {dp:5.1f}%")

# Save CSV
rows = []
for rs in ['CERES', 'Kernel']:
    for period, *_ in PERIOD_CFG:
        for pval in PVAL_THRESHOLDS:
            r = RESULTS[(rs, period, pval)]
            rows.append(dict(
                rs_method = rs, period = period, pval = pval,
                SIAF_tot_arctic  = round(r['tot'],      5),
                SIAF_ext_arctic  = round(r['ext'],      5),
                SIAF_drk_arctic  = round(r['drk'],      5),
                SIAF_tot_global  = round(r['tot']*SCALE, 6),
                SIAF_ext_global  = round(r['ext']*SCALE, 6),
                SIAF_drk_global  = round(r['drk']*SCALE, 6),
                sigma2_tot_global= round(r['tot_2s']*SCALE, 6),
                sigma2_ext_global= round(r['ext_2s']*SCALE, 6),
                sigma2_drk_global= round(r['drk_2s']*SCALE, 6),
                ext_pct = round(r['ext']/r['tot']*100 if r['tot'] else np.nan, 1),
                drk_pct = round(r['drk']/r['tot']*100 if r['tot'] else np.nan, 1),
            ))
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'\nCSV -> {OUT_CSV}')

# ============================================================
# BAR CHART
# ============================================================

RS_DISPLAY = {
    'CERES':  'CERES EBAF\n(T07-inverted RS)',
    'Kernel': 'ERA5 Kernel\n(Huang & Huang 2023)',
}
COL_EXT = '#3A6BAF'
COL_DRK = '#D95F1A'
ALPHA   = {'CMIP6': 0.55, 'Obs-era': 1.00}
HATCHES = {'CMIP6': '////', 'Obs-era': ''}

def make_barplot(pval, out_path):
    rs_types = ['CERES', 'Kernel']
    periods  = ['CMIP6', 'Obs-era']
    n = len(rs_types)
    x = np.arange(n)
    w = 0.32
    offsets = {'CMIP6': -(w/2 + 0.03), 'Obs-era': (w/2 + 0.03)}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    legend_handles = []

    for period in periods:
        a, hat = ALPHA[period], HATCHES[period]
        for i, rs in enumerate(rs_types):
            r   = RESULTS[(rs, period, pval)]
            if not np.isfinite(r['tot']): continue
            tot = r['tot'] * SCALE
            ext = r['ext'] * SCALE
            drk = r['drk'] * SCALE
            err = r['tot_2s'] * SCALE
            xp  = x[i] + offsets[period]

            ax.bar(xp, ext, w, color=COL_EXT, alpha=a, hatch=hat,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.bar(xp, drk, w, bottom=ext, color=COL_DRK, alpha=a,
                   hatch=hat, edgecolor='white', linewidth=0.5, zorder=3)
            ax.errorbar(xp, tot, yerr=err, fmt='none', color='#222222',
                        capsize=5, linewidth=1.5, capthick=1.5, zorder=5)

            ep = ext/tot*100 if tot > 0 else 0
            dp = drk/tot*100 if tot > 0 else 0
            if ext > 0.002:
                ax.text(xp, ext/2, f'{ep:.0f}%', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold', zorder=6)
            if drk > 0.002:
                ax.text(xp, ext+drk/2, f'{dp:.0f}%', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold', zorder=6)

            plbl = 'CMIP6\n(1982–2014)' if period == 'CMIP6' else 'Obs\n(1982–2024)'
            ax.text(xp, tot + err + 0.001, plbl, ha='center', va='bottom',
                    color='#555555', fontsize=7.5, style='italic')

        legend_handles.append(mpatches.Patch(
            facecolor=COL_EXT, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Extent ({period})'))
        legend_handles.append(mpatches.Patch(
            facecolor=COL_DRK, alpha=a, hatch=hat, edgecolor='#555',
            label=f'Darkening ({period})'))

    ax.set_xticks(x)
    ax.set_xticklabels([RS_DISPLAY[rs] for rs in rs_types], fontsize=11)
    ax.set_ylabel('Global Annual SIAF  (W m⁻² K⁻¹)', fontsize=11)
    ax.set_title(
        f'SIAF: Pixel-wise IS × RS, area-weighted  (p < {pval})\n'
        f'No regional averaging — direct domain integration',
        fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', colors='#444444')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, ncol=2, columnspacing=0.8,
              handlelength=1.4, handleheight=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_path}')

print('\nGenerating bar charts ...')
for pval in PVAL_THRESHOLDS:
    make_barplot(pval, PLOT_OUT.format(pval=pval))

print('\nDone.')