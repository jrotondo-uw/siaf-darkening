#!/usr/bin/env python
"""
compute_rs_ice_mean.py
──────────────────────
Compute the AMJJAS-mean RS (radiative sensitivity) averaged only over
ice pixels, for both CERES and ERA5 Kernel methods.

Uses pre-computed caches only — no bootstrap, no albedo loading.
Runs in < 1 minute.
"""

import numpy as np
import xarray as xr
import glob, os
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer

# ── CONFIG (match siaf_obs_analysis.py) ───────────────────────────────────────
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
KERNEL_FILE = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR  = 'SFC_all'
KERNEL_SCALE = 100.0

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

SIC_THRESH = 0.15
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

CERES_CMIP6 = list(range(2000, 2015))
CERES_OBS   = list(range(2000, 2025))

PERIOD_CFG = [
    ('CMIP6',   CACHE_CMIP6, CERES_CMIP6),
    ('Obs-era', CACHE_OBS,   CERES_OBS),
]

# ── GRID UTILITIES ────────────────────────────────────────────────────────────
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

def domain_mean(field, domain, weights):
    v  = field[domain]
    w  = weights[domain]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))

# ── SETUP: SIAL grid ──────────────────────────────────────────────────────────
print('Setting up grids ...')
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
except FileNotFoundError:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)

# ── SETUP: PSN25 grid ─────────────────────────────────────────────────────────
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
PSN_SHAPE = psn_xx.shape
tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = tfm.transform(psn_xx, psn_yy)

phi_ts = np.deg2rad(70.)
k_ps   = (1. + np.sin(phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W = (1. / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe as _globe
    lp = np.where(PSN_LON > 180, PSN_LON - 360, PSN_LON)
    DOMAIN_PSN = (PSN_LAT >= 60.) & _globe.is_ocean(PSN_LAT, lp)
except ImportError:
    DOMAIN_PSN = (PSN_LAT >= 60.)
    print('  WARNING: global-land-mask not available.')

# ── SETUP: regrid indices ─────────────────────────────────────────────────────
print('Building regrid indices ...')
IDX_SIAL_TO_PSN = build_scatter_index(
    SIAL_LAT.ravel(), SIAL_LON.ravel(),
    PSN_LAT.ravel(),  PSN_LON.ravel())

with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
    c_la = ds['lat'].values; c_lo = ds['lon'].values
CERES_LON, CERES_LAT = np.meshgrid(c_lo, c_la)
IDX_CERES_TO_PSN = build_scatter_index(
    CERES_LAT.ravel(), CERES_LON.ravel(),
    PSN_LAT.ravel(),   PSN_LON.ravel())

# ── CERES RS loader ───────────────────────────────────────────────────────────
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

def load_RS_ceres_mean_psn(years, mn):
    """Mean CERES RS for given years/month, regridded to PSN25."""
    import gc, pandas as pd
    maps = []
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for y in years:
            idxs = [i for i,t in enumerate(times)
                    if t.year == y and t.month == mn]
            if not idxs:
                continue
            sums = {k: None for k in _CERES_VARS}; cnt = 0
            for i in idxs:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1
            if cnt == 0:
                continue
            for k in sums:
                sums[k] /= cnt
            rs = t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                            sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                            sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                            sums['sfc_sw_up_clr_c_mon'])
            maps.append(rs)
    if not maps:
        return None
    mean_rs = np.nanmean(np.stack(maps, 0), axis=0).astype(np.float32)
    return scatter_avg(mean_rs.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)

# ── ERA5 Kernel loader ────────────────────────────────────────────────────────
def load_kernel_psn(mn):
    with xr.open_dataset(KERNEL_FILE) as ds:
        tdim  = next((d for d,s in ds.sizes.items() if s == 12), None)
        lat_n = 'latitude' if 'latitude' in ds else 'lat'
        lon_n = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_n].values; k_lon = ds[lon_n].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[KERNEL_VAR].isel({tdim: mn-1}).values.astype(np.float32)
    k_raw = np.abs(k_raw) * KERNEL_SCALE
    idx_k = build_scatter_index(k_lat2d.ravel(), k_lon2d.ravel(),
                                 PSN_LAT.ravel(), PSN_LON.ravel())
    return scatter_avg(k_raw.ravel(), idx_k, PSN_SHAPE)

# ── Ice mask: where SIC ≥ threshold in either early or late period ────────────
def load_ice_mask_psn(cache_dir, mname):
    """Returns PSN25 boolean mask: True where ice present (SIC ≥ SIC_THRESH)."""
    try:
        sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32)
        sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32)
    except FileNotFoundError:
        return None
    has_ice_sial = (ARCTIC_SIAL &
                    (np.fmax(sic_e, sic_l) >= SIC_THRESH) &
                    np.isfinite(sic_e) & np.isfinite(sic_l))
    # Regrid boolean to PSN25 (scatter_avg → > 0 means at least one SIAL pixel had ice)
    ice_psn = scatter_avg(has_ice_sial.astype(np.float32).ravel(),
                           IDX_SIAL_TO_PSN, PSN_SHAPE)
    return (ice_psn > 0) & DOMAIN_PSN

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print('Computing ice-masked RS ...\n')

results = {}   # (rs_method, period) -> list of monthly RS means

for period, cache_dir, ceres_yrs in PERIOD_CFG:
    for rs_method in ['CERES', 'Kernel']:
        monthly_rs = []
        print(f'  {period} | {rs_method}')
        for mn in MONTH_NUMS:
            mname = MONTH_NAMES[mn]

            # Load RS on PSN25
            if rs_method == 'CERES':
                rs_psn = load_RS_ceres_mean_psn(ceres_yrs, mn)
            else:
                try:
                    rs_psn = load_kernel_psn(mn)
                except Exception as e:
                    print(f'    {mname}: kernel failed ({e})')
                    rs_psn = None

            if rs_psn is None:
                print(f'    {mname}: no RS — skipped')
                monthly_rs.append(np.nan)
                continue

            # Ice mask for this month/period
            ice_mask = load_ice_mask_psn(cache_dir, mname)
            if ice_mask is None:
                print(f'    {mname}: no SIC cache — skipped')
                monthly_rs.append(np.nan)
                continue

            # RS mean over ice pixels only
            rs_ice = domain_mean(rs_psn, ice_mask, AREA_W)
            monthly_rs.append(rs_ice)
            print(f'    {mname}: RS_ice = {rs_ice:.2f} W m⁻² albedo⁻¹')

        results[(rs_method, period)] = monthly_rs

# ── REPORT ────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('AMJJAS-mean RS over ice pixels')
print('='*60)

print(f"\n{'Method':<10}  {'Period':<10}  {'RS_ice (W m⁻² albedo⁻¹)':>26}")
print('-'*50)

rs_for_avg = {}
for period in ['CMIP6', 'Obs-era']:
    for rs_method in ['CERES', 'Kernel']:
        monthly = results[(rs_method, period)]
        finite  = [v for v in monthly if np.isfinite(v)]
        if not finite:
            print(f'{rs_method:<10}  {period:<10}  no data')
            continue
        seas_mean = float(np.mean(finite))
        monthly_str = '  '.join(f'{v:.1f}' for v in monthly)
        print(f'{rs_method:<10}  {period:<10}  {seas_mean:.2f}')
        print(f'           monthly: [{monthly_str}]')
        rs_for_avg[(rs_method, period)] = seas_mean

# ── Average CERES + Kernel for each period ────────────────────────────────────
print('\n' + '-'*50)
for period in ['CMIP6', 'Obs-era']:
    c = rs_for_avg.get(('CERES', period), np.nan)
    k = rs_for_avg.get(('Kernel', period), np.nan)
    if np.isfinite(c) and np.isfinite(k):
        avg    = (c + k) / 2.
        spread = abs(c - k) / 2.
        print(f'\n{period}:')
        print(f'  CERES  = {c:.2f}')
        print(f'  Kernel = {k:.2f}')
        print(f'  Mean   = {avg:.2f} ± {spread:.2f} W m⁻² albedo⁻¹  (±inter-method spread)')
        print(f'  → For text: {avg:.0f} ± {spread:.0f} W m⁻² per unit albedo')
