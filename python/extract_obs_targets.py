#!/usr/bin/env python
"""
extract_obs_targets.py
──────────────────────
Calculates the isolated IS and RS means (and 2σ bootstraps) from observations
for the CMIP6 period (Method B2, p < 0.05) to be used as plotting targets.
(Memory-optimized version with ΔT structural uncertainty)
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob, os, gc
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG & PATHS
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CERES_FILE  = '/glade/derecho/scratch/jrotondo/ceres_data_updated/CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc'

EARLY       = list(range(1982, 1992))
LATE_CMIP6  = list(range(2005, 2015))
CERES_CMIP6 = list(range(2000, 2015))
TREND_START_CMIP6, TREND_END_CMIP6 = 1982, 2014

MONTH_NAMES = {4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
PVAL_THRESH = 0.05
SIC_ICE_THRESHOLD = 0.15
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

PISTONE_REGIONS = [
    ('EU', 'Europe_Barents',              60, 90,    0,   60),
    ('WA', 'WestAsia_KaraLaptev',         60, 90,   60,  120),
    ('WP', 'WestPacific_EastSiberian',    60, 90,  120,  180),
    ('EP', 'EastPacific_ChukchiBeaufort', 60, 90, -180, -120),
    ('NA', 'NorthAmerica_CdnArchBaffin',  60, 90, -120,  -60),
    ('AO', 'Atlantic_LabradorGreenland',  60, 90,  -60,    0),
]

T_SOURCES = [
    ('GISTEMP',   '/glade/work/jrotondo/gistemp_global.csv',   'gistemp'),
    ('CowtanWay', '/glade/work/jrotondo/cowtan_way.txt',       'cowtan_way'),
    ('HadCRUT5',  '/glade/work/jrotondo/hadcrut5_global.csv',  'hadcrut5'),
    ('NOAA',      '/glade/work/jrotondo/noaa_global.csv',      'noaa'),
]

# ============================================================
# GRID & DATA LOADERS
# ============================================================
def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tree    = cKDTree(tgt_xyz)
    idx     = np.empty(src_xyz.shape[0], dtype=np.int32)
    for i in range(0, src_xyz.shape[0], chunk):
        e = min(i + chunk, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx

def build_nn_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    tree    = cKDTree(src_xyz)
    idx     = np.empty(tgt_xyz.shape[0], dtype=np.int32)
    for i in range(0, tgt_xyz.shape[0], chunk):
        e = min(i + chunk, tgt_xyz.shape[0])
        _, idx[i:e] = tree.query(tgt_xyz[i:e], workers=1)
    del tree, src_xyz, tgt_xyz; gc.collect()
    return idx

def scatter_avg(arr_flat, idx, out_shape):
    n = int(np.prod(out_shape))
    accum = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    v = np.isfinite(arr_flat)
    np.add.at(accum, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(count, idx[v], np.float32(1.))
    out = np.full(n, np.nan, dtype=np.float32)
    m = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt > 1., rsdt, np.nan)
        Sd  = np.where(rsds > 1., rsds, np.nan)
        a   = np.clip(rsus / Sd, 0., 1.)
        p   = rsds / S;  q = rsut / S
        den = q + p - 2.*p*a
        R   = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan), 0., .99)
        A   = np.clip(1. - p*(1.-a*R) / np.where((1.-R) > .01, 1.-R, np.nan), 0., .99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S
        dc  = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan), 0., .99)
        Rl  = np.clip(R - Rc, 0., .99)
        RS  = (S * (1.-A) * (1.-Rl)**2 / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS

def compute_IS_from_alb(ae, al, sic_data, delta_t):
    dalpha = al.astype(np.float32) - ae.astype(np.float32)
    slope, pval_arr = sic_data['slope'], sic_data['pval']
    sic_e, sic_l = sic_data['sic_e'], sic_data['sic_l']
    
    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval_arr) & (slope < 0) & (pval_arr < PVAL_THRESH))
    stable_ice = (valid & ~changing & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))
    
    IS_ext = np.where(changing,   -dalpha / delta_t, np.nan)
    IS_drk = np.where(stable_ice, -dalpha / delta_t, np.nan)
    
    IS_tot = np.where(np.isfinite(IS_ext) | np.isfinite(IS_drk),
                      np.nan_to_num(IS_ext) + np.nan_to_num(IS_drk), np.nan)
    return IS_tot

def get_arctic_mean(arr_psn, reg_masks, weights):
    num = den = 0.
    for code, _, _, _, _, _ in PISTONE_REGIONS:
        rm = reg_masks[code]
        v, w = arr_psn[rm], weights[rm]
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if ok.sum() > 0:
            num += np.sum(v[ok] * w[ok])
            den += np.sum(w[ok])
    return float(num / den) if den > 0 else np.nan

# ============================================================
# MAIN EXTRACTION
# ============================================================
print("Setting up grids and loaders...", flush=True)

# Delta T Arrays
vals = []
for _, path, kind in T_SOURCES:
    if os.path.exists(path):
        try:
            if kind == 'gistemp':
                df = pd.read_csv(path, skiprows=1, na_values='***')
                yy, vv = df['Year'].values.astype(float), pd.to_numeric(df['J-D'], errors='coerce').values.astype(float)
            elif kind == 'cowtan_way':
                df = pd.read_csv(path, sep=r'\s+', header=None, names=['year','anomaly','u1','u2','u3'])
                yy, vv = df['year'].values.astype(float), df['anomaly'].values.astype(float)
            elif kind == 'hadcrut5':
                df = pd.read_csv(path)
                vc = next((c for c in df.columns if 'anom' in c.lower() or 'mean' in c.lower()), None)
                yy, vv = df.iloc[:,0].values.astype(float), (df[vc].values if vc else df.iloc[:,1].values)
            elif kind == 'noaa':
                df = pd.read_csv(path, comment='#')
                yy, vv = df.iloc[:,0].values.astype(float), df.iloc[:,1].values.astype(float)
                
            m = (yy >= TREND_START_CMIP6) & (yy <= TREND_END_CMIP6) & np.isfinite(vv)
            if m.sum() >= 5:
                vals.append(stats.linregress(yy[m], vv[m]).slope * (TREND_END_CMIP6 - TREND_START_CMIP6))
        except: pass

DELTA_T = float(np.mean(vals)) if vals else 0.64
DT_ARRAY = np.array(vals) if vals else np.array([0.64])

# SIAL Grid (SAFE LOAD)
with xr.open_dataset(sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]) as ds:
    s_lat = ds['lat'].values
    s_lon = ds['lon'].values
if s_lat.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(s_lon, s_lat)
else:
    sial_lat2d, sial_lon2d = s_lat, s_lon

try: arctic_sial = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except: arctic_sial = (sial_lat2d >= 60.)

# CERES Grid (SAFE LOAD)
with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
    c_lat = ds['lat'].values
    c_lon = ds['lon'].values
if c_lat.ndim == 1:
    ceres_lon2d, ceres_lat2d = np.meshgrid(c_lon, c_lat)
else:
    ceres_lat2d, ceres_lon2d = c_lat, c_lon

# APP-x -> SIAL (SAFE LOAD)
try:
    with NC4Dataset(sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0], 'r') as ds:
        ax_lat = np.ma.filled(ds.variables['latitude'][:], np.nan).astype(float)
        ax_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if ax_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(ax_lon, ax_lat)
    else:
        appx_lat2d, appx_lon2d = ax_lat, ax_lon
        
    idx_appx_to_sial = build_nn_index(appx_lat2d, appx_lon2d, sial_lat2d, sial_lon2d)
    del appx_lat2d, appx_lon2d, ax_lat, ax_lon; gc.collect()
except Exception as e: 
    print(f"Warning: APP-X setup failed ({e})")
    idx_appx_to_sial = None

# PSN25 Grid
with xr.open_dataset(sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]) as ds:
    psn_xx, psn_yy = np.meshgrid(ds['x'].values, ds['y'].values)
psn_lon2d, psn_lat2d = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True).transform(psn_xx, psn_yy)
AREA_WEIGHT_PSN = (1. / ((1. + np.sin(np.deg2rad(70.))) / np.maximum(1. + np.sin(np.deg2rad(psn_lat2d)), 1e-9))**2).astype(np.float32)

REGION_MASKS = {}
for code, _, lo, hi, lmn, lmx in PISTONE_REGIONS:
    lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
    lp = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    lon_ok = ((lp >= lmn) & (lp <= lmx) if lmn <= lmx else (lp >= lmn) | (lp <= lmx))
    try: from global_land_mask import globe as gm; ocean = gm.is_ocean(psn_lat2d, lp)
    except: ocean = np.ones(psn_lat2d.shape, dtype=bool)
    REGION_MASKS[code] = lat_ok & lon_ok & ocean & (psn_lat2d >= 60.)

idx_sial_to_psn = build_scatter_index(sial_lat2d, sial_lon2d, psn_lat2d, psn_lon2d)
idx_ceres_to_psn = build_scatter_index(ceres_lat2d, ceres_lon2d, psn_lat2d, psn_lon2d)

print("Setup Complete. Processing months sequentially to save memory...", flush=True)

rng = np.random.default_rng(BOOTSTRAP_SEED)
c_vars = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon', 'sfc_sw_up_all_mon', 
          'toa_sw_clr_c_mon', 'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

def _resample_mean(stack, rng_obj):
    n = stack.shape[0]
    return np.nanmean(stack[rng_obj.integers(0, n, size=n)], axis=0)

is_means = []
rs_means = []
boot_is_monthly = np.zeros((N_BOOTSTRAP, len(MONTH_NUMS)))
boot_rs_monthly = np.zeros((N_BOOTSTRAP, len(MONTH_NUMS)))

for idx_m, mn in enumerate(MONTH_NUMS):
    mname = MONTH_NAMES[mn]
    print(f"  -> Extracting {mname}...", flush=True)
    
    # Load Early Albedo
    ea_list = []
    for y in EARLY:
        p = f'{CACHE_CMIP6}/appx_alb_{mname}_{y}.npy' if mn==9 else f'{CACHE_CMIP6}/clara_alb_{mname}_{y}.npy'
        if os.path.exists(p): 
            r = np.load(p)
            ea_list.append(r.ravel()[idx_appx_to_sial].reshape(sial_lat2d.shape) if mn==9 and idx_appx_to_sial is not None else r)
    ea_stack = np.stack(ea_list) if ea_list else None
    
    # Load Late Albedo
    la_list = []
    for y in LATE_CMIP6:
        p = f'{CACHE_CMIP6}/appx_alb_{mname}_{y}.npy' if mn==9 else f'{CACHE_CMIP6}/clara_alb_{mname}_{y}.npy'
        if os.path.exists(p): 
            r = np.load(p)
            la_list.append(r.ravel()[idx_appx_to_sial].reshape(sial_lat2d.shape) if mn==9 and idx_appx_to_sial is not None else r)
    la_stack = np.stack(la_list) if la_list else None
            
    # Load SIC Data
    sic_data = dict(
        slope = np.load(f'{CACHE_CMIP6}/sic_trend_slope_sial_{mname}.npy'),
        pval  = np.load(f'{CACHE_CMIP6}/sic_trend_pval_sial_{mname}.npy'),
        sic_e = np.load(f'{CACHE_CMIP6}/sic_early_sial_{mname}.npy'),
        sic_l = np.load(f'{CACHE_CMIP6}/sic_late_sial_{mname}.npy')
    )
    
    # Load CERES RS
    rs_list = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in CERES_CMIP6:
            idxs = [i for i, t in enumerate(times) if t.year == y and t.month == mn]
            if idxs:
                sums = {k: np.zeros(ceres_lat2d.shape, dtype=np.float32) for k in c_vars}
                for i in idxs:
                    sl = ds.isel(time=i).load()
                    for k in c_vars: sums[k] += sl[k].values
                for k in sums: sums[k] /= len(idxs)
                
                # Correctly mapping explicit variable names to t07_invert
                rs_list.append(t07_invert(
                    sums['solar_mon'],
                    sums['toa_sw_all_mon'],
                    sums['sfc_sw_down_all_mon'],
                    sums['sfc_sw_up_all_mon'],
                    sums['toa_sw_clr_c_mon'],
                    sums['sfc_sw_down_clr_c_mon'],
                    sums['sfc_sw_up_clr_c_mon']
                ))
                
    rs_stack = np.stack(rs_list) if rs_list else None
    
    if ea_stack is None or la_stack is None or rs_stack is None:
        print(f"     Skipping {mname} due to missing data.")
        is_means.append(np.nan)
        rs_means.append(np.nan)
        boot_is_monthly[:, idx_m] = np.nan
        boot_rs_monthly[:, idx_m] = np.nan
        continue

    # --- 1. Calculate True Mean for this month ---
    ae = np.nanmean(ea_stack, axis=0)
    al = np.nanmean(la_stack, axis=0)
    rs_mean = np.nanmean(rs_stack, axis=0)
    
    IS_tot = compute_IS_from_alb(ae, al, sic_data, DELTA_T)
    IS_tot_p = scatter_avg(np.where(arctic_sial, np.nan_to_num(IS_tot), np.nan).ravel(), idx_sial_to_psn, psn_lat2d.shape)
    RS_psn   = scatter_avg(rs_mean.ravel(), idx_ceres_to_psn, psn_lat2d.shape)
    
    is_means.append(get_arctic_mean(IS_tot_p, REGION_MASKS, AREA_WEIGHT_PSN))
    rs_means.append(get_arctic_mean(RS_psn, REGION_MASKS, AREA_WEIGHT_PSN))

    # --- 2. Bootstrap for this month ---
    for b in range(N_BOOTSTRAP):
        ae_b = _resample_mean(ea_stack, rng)
        al_b = _resample_mean(la_stack, rng)
        rs_b = _resample_mean(rs_stack, rng)
        
        dt_b = rng.choice(DT_ARRAY)
        
        IS_tot_b = compute_IS_from_alb(ae_b, al_b, sic_data, dt_b)
        IS_tot_p_b = scatter_avg(np.where(arctic_sial, np.nan_to_num(IS_tot_b), np.nan).ravel(), idx_sial_to_psn, psn_lat2d.shape)
        RS_psn_b   = scatter_avg(rs_b.ravel(), idx_ceres_to_psn, psn_lat2d.shape)
        
        boot_is_monthly[b, idx_m] = get_arctic_mean(IS_tot_p_b, REGION_MASKS, AREA_WEIGHT_PSN)
        boot_rs_monthly[b, idx_m] = get_arctic_mean(RS_psn_b, REGION_MASKS, AREA_WEIGHT_PSN)
        
    # Flush memory
    del ea_list, la_list, rs_list, ea_stack, la_stack, rs_stack, sic_data
    gc.collect()

# --- Combine Monthly Results ---
OBS_IS_MEAN = np.nanmean(is_means)
OBS_RS_MEAN = np.nanmean(rs_means)

boot_is = np.nanmean(boot_is_monthly, axis=1)
boot_rs = np.nanmean(boot_rs_monthly, axis=1)

OBS_IS_2SIGMA = 2 * np.nanstd(boot_is)
OBS_RS_2SIGMA = 2 * np.nanstd(boot_rs)

print("\n" + "="*50)
print(" PASTE THIS INTO plot_is_rs_uncertainty.py")
print("="*50)
print(f"OBS_IS_MEAN   = {OBS_IS_MEAN:.5f}")
print(f"OBS_IS_2SIGMA = {OBS_IS_2SIGMA:.5f}")
print(f"OBS_RS_MEAN   = {OBS_RS_MEAN:.5f}")
print(f"OBS_RS_2SIGMA = {OBS_RS_2SIGMA:.5f}")
print("="*50)
