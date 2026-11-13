#!/usr/bin/env python
"""
plot_siaf_rolling.py
────────────────────
Computes and plots SIAF as a function of time using a trailing 5-year
rolling late-epoch window against a fixed early baseline (1982–1991).

For each window ending at year t (t = 1986 … 2024):
    late  = [t-4, t-3, t-2, t-1, t]          (5-year trailing mean)
    early = 1982–1991                          (fixed)
    IS    = -(α_late − α_early) / ΔT
    ΔT    = mean(T_late) − mean(T_early)       (epoch difference, not OLS)
    SIAF  = IS × RS                            (RS = fixed CERES 2000–2024 clim)

RS is fixed to the full CERES climatology so that the time series reflects
only how the albedo sensitivity IS evolves — not noise in RS.

Bootstrap uncertainty (N=500 per window, 2σ band):
    1. Albedo interannual — resample years within each 5-yr window
    2. ΔT structural      — draw one T-source at random per iteration

Method: B2 (CERES EBAF Regional, PSN25 Pistone regions)

Outputs
-------
    siaf_rolling_timeseries.png   — main figure
    siaf_rolling_timeseries.csv   — data table
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, gc
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

# ── Paths (match diagnose_siaf_unified.py) ────────────────────────────────────
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)

GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

# ── Constants ─────────────────────────────────────────────────────────────────
EARLY_YEARS    = list(range(1982, 1992))   # fixed early baseline
WINDOW         = 5                         # trailing window length (years)
FIRST_LATE_END = 2000                      # start after early period fully clears
LAST_LATE_END  = 2024
CERES_CLIM_YRS = list(range(2000, 2025))   # fixed RS climatology period
MONTH_NUMS     = [4, 5, 6, 7, 8, 9]
MONTH_NAMES    = {4:'April', 5:'May', 6:'June',
                  7:'July',  8:'August', 9:'September'}
SIC_ICE_THRESHOLD = 0.15
PVAL_THRESH    = 0.05
AREA_RATIO_OCEAN  = 0.0347
SEASON_FRAC       = 6. / 12.
SCALE             = AREA_RATIO_OCEAN * SEASON_FRAC
N_BOOTSTRAP    = 500
BOOTSTRAP_SEED = 42
PSN25_PROJ4    = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
                  '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
PISTONE_REGIONS = [
    ('EU', 60, 90,    0,   60),
    ('WA', 60, 90,   60,  120),
    ('WP', 60, 90,  120,  180),
    ('EP', 60, 90, -180, -120),
    ('NA', 60, 90, -120,  -60),
    ('AO', 60, 90,  -60,    0),
]

# ── Helpers (copied from diagnose_siaf_unified.py) ────────────────────────────

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tree = cKDTree(tgt_xyz)
    idx  = np.empty(src_xyz.shape[0], dtype=np.int32)
    for i in range(0, src_xyz.shape[0], chunk):
        e = min(i + chunk, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree; gc.collect()
    return idx, tgt_lat2d.shape

def scatter_avg(arr_flat, idx, out_shape):
    n     = int(np.prod(out_shape))
    accum = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    v     = np.isfinite(arr_flat)
    np.add.at(accum, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(count, idx[v], np.float32(1.))
    out   = np.full(n, np.nan, dtype=np.float32)
    m     = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

def regrid_sial_to_psn(arr):
    src = np.where(arctic_sial,
                   np.where(np.isfinite(arr), arr, np.float32(0.)),
                   np.nan).astype(np.float32)
    return scatter_avg(src.ravel(), idx_sial_to_psn, psn_shape)

def region_mean_psn(arr_psn, rmask):
    v, w = arr_psn[rmask], AREA_WEIGHT_PSN[rmask]
    ok   = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan, 0.
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd = np.where(rsds  > 1., rsds,  np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds / S;  q = rsut / S
        den = q + p - 2.*p*a
        R   = np.clip((q-p*a)/np.where(np.abs(den)>1e-6, den, np.nan), 0., .99)
        A   = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan), 0., .99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S
        dc  = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan), 0., .99)
        Rl  = np.clip(R - Rc, 0., .99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               / np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS

# ── Temperature loaders ────────────────────────────────────────────────────────

def _load_gistemp():
    if not os.path.exists(GISTEMP_PATH): return None, None
    df = pd.read_csv(GISTEMP_PATH, skiprows=1, na_values='***')
    df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
    return df['Year'].values.astype(float), df['J-D'].values.astype(float)

def _load_cowtan():
    if not os.path.exists(COWTANWAY_PATH): return None, None
    df = pd.read_csv(COWTANWAY_PATH, sep=r'\s+', header=None,
                     names=['year','anomaly','u1','u2','u3'])
    return df['year'].values.astype(float), df['anomaly'].values.astype(float)

def _load_ncep():
    if not os.path.exists(NCEP_PATH): return None, None
    ds = xr.open_dataset(NCEP_PATH)
    t  = ds['air']
    w  = np.cos(np.deg2rad(t.lat))
    ta = t.weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
    ds.close()
    return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)

def _load_era5():
    files = sorted(glob.glob(ERA5_GLOB))
    if not files: return None, None
    ds = xr.open_mfdataset(files, combine='by_coords')
    t  = ds['VAR_2T']
    w  = np.cos(np.deg2rad(t.latitude))
    ta = t.weighted(w).mean(dim=['latitude','longitude']).resample(time='1YE').mean()
    ds.close()
    return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)

def build_T_sources():
    sources = []
    for name, loader in [('GISTEMP',_load_gistemp), ('CowtanWay',_load_cowtan),
                          ('NCEP',_load_ncep), ('ERA5',_load_era5)]:
        try:
            yy, vv = loader()
            if yy is not None and np.isfinite(vv).sum() > 10:
                sources.append((name, yy, vv))
                print(f'  T-source {name}: {int(yy.min())}–{int(yy.max())}')
        except Exception as e:
            print(f'  T-source {name} failed: {e}')
    return sources

def compute_dT_samples_ols(t_end):
    """
    ΔT = OLS_slope(1982 → t_end) × (t_end − 1982).
    Identical method to diagnose_siaf_unified.py (trend_dT).
    Ensures rolling values converge to ~0.21 at t_end=2014 and ~0.16 at 2024.
    """
    vals = []
    y0 = 1982.
    for name, yy, vv in T_SOURCES:
        m = (yy >= y0) & (yy <= t_end) & np.isfinite(vv)
        if m.sum() < 5:
            continue
        dT = stats.linregress(yy[m], vv[m]).slope * (t_end - y0)
        if np.isfinite(dT) and dT > 0:
            vals.append(float(dT))
    return np.array(vals, dtype=np.float32) if vals else np.array([0.7], dtype=np.float32)

# ── Albedo loader ──────────────────────────────────────────────────────────────

def load_alb_yearly_stack(mname, years, use_appx=False):
    stack = []
    for y in years:
        prefix = 'appx' if use_appx else 'clara'
        p = f'{CACHE_OBS}/{prefix}_alb_{mname}_{y}.npy'
        if not os.path.exists(p):
            continue
        raw = np.load(p).astype(np.float32)
        if use_appx and idx_appx_to_sial is not None:
            raw = raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
        stack.append(raw)
    return np.stack(stack, axis=0) if stack else None

def load_alb_mean(mname, years):
    """Mean over years, preferring CLARA, falling back to APP-x for Sept."""
    use_appx = (mname == 'September')
    stack = load_alb_yearly_stack(mname, years, use_appx=use_appx)
    if stack is None:
        return None
    return np.nanmean(stack, axis=0).astype(np.float32)

# ── CERES RS fixed climatology ─────────────────────────────────────────────────
_CERES_VARS = ['solar_mon','toa_sw_all_mon','sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon','toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon','sfc_sw_up_clr_c_mon']

def load_RS_ceres_clim(years, mn):
    sums = {k: None for k in _CERES_VARS}
    cnt  = 0
    with xr.open_dataset(CERES_FILE, engine='netcdf4') as ds:
        times = pd.DatetimeIndex(ds.time.values)
        for i, t in enumerate(times):
            if t.year in years and t.month == mn:
                sl = ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a = sl[k].values.astype(np.float32)
                    sums[k] = a if cnt == 0 else sums[k] + a
                cnt += 1;  del sl
    if cnt == 0:
        return None
    for k in sums:
        sums[k] /= np.float32(cnt)
    return t07_invert(*[sums[k] for k in _CERES_VARS]).astype(np.float32)

# ── IS / SIAF computation ──────────────────────────────────────────────────────

def compute_IS_SIAF(ae, al, sic_data, delta_t, RS_psn):
    """
    Returns (SIAF_tot, SIAF_ext, SIAF_drk) as Pistone-region weighted means,
    or None if computation fails.
    """
    if sic_data is None or delta_t == 0 or not np.isfinite(delta_t):
        return None, None, None

    dalpha   = al.astype(np.float32) - ae.astype(np.float32)
    slope    = sic_data['slope']
    pval_arr = sic_data['pval']
    sic_e    = sic_data['sic_e']
    sic_l    = sic_data['sic_l']

    valid      = arctic_sial & np.isfinite(dalpha)
    changing   = (valid & np.isfinite(slope) & np.isfinite(pval_arr)
                  & (slope < 0) & (pval_arr < PVAL_THRESH))
    stable_ice = (valid & ~changing
                  & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

    IS_ext = np.where(changing,   -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk = np.where(stable_ice, -dalpha / delta_t, np.nan).astype(np.float32)
    IS_tot = np.where(
        np.isfinite(IS_ext) | np.isfinite(IS_drk),
        np.where(np.isfinite(IS_ext), IS_ext, 0.)
        + np.where(np.isfinite(IS_drk), IS_drk, 0.),
        np.nan).astype(np.float32)

    IS_ext = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
    IS_drk = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)

    # Regrid to PSN25
    IT_p = regrid_sial_to_psn(IS_tot)
    IX_p = regrid_sial_to_psn(IS_ext)
    ID_p = regrid_sial_to_psn(IS_drk)

    # Pistone region weighted mean using pre-computed masks
    reg_siaf, reg_wts = [], []
    for rm in REGION_MASKS_PSN:
        if rm.sum() == 0:
            continue
        ST, w = region_mean_psn(IT_p * RS_psn, rm)
        if not (np.isfinite(ST) and w > 0):
            continue
        SX, _ = region_mean_psn(IX_p * RS_psn, rm)
        SD, _ = region_mean_psn(ID_p * RS_psn, rm)
        reg_siaf.append([ST, SX, SD]);  reg_wts.append(w)

    if not reg_wts:
        return None, None, None

    wts = np.array(reg_wts, dtype=np.float64)
    arr = np.array(reg_siaf, dtype=np.float64)
    wmean = (np.sum(arr * wts[:, None], axis=0) / np.sum(wts)).tolist()
    return wmean[0], wmean[1], wmean[2]

# ── ONE-TIME SETUP ─────────────────────────────────────────────────────────────
print('Setting up grids...', flush=True)

T_SOURCES = build_T_sources()

# SIAL grid
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape

try:
    arctic_sial = np.load(f'{CACHE_OBS}/arctic_60N_sial.npy').astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.)

# APP-x → SIAL index
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon
    # Tree from APP-x, queried with SIAL → one entry per SIAL pixel
    # so raw.ravel()[idx_appx_to_sial].reshape(sial_shape) works correctly
    tree = cKDTree(latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel()))
    _, idx_appx_to_sial = tree.query(
        latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel()), workers=-1)
    del tree; gc.collect()
except Exception as e:
    print(f'  APP-x→SIAL index failed: {e}')

# PSN25 grid
from pyproj import Transformer
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)
psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
psn_shape = psn_xx.shape
tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
psn_lon2d, psn_lat2d = tfm.transform(psn_xx, psn_yy)

phi_ts          = np.deg2rad(70.)
k_ps            = (1. + np.sin(phi_ts)) / np.maximum(
                      1. + np.sin(np.deg2rad(psn_lat2d)), 1e-9)
AREA_WEIGHT_PSN = (1. / k_ps**2).astype(np.float32)

try:
    from global_land_mask import globe as _globe
    lp        = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    OCEAN_PSN = (psn_lat2d >= 60.) & _globe.is_ocean(psn_lat2d, lp)
except ImportError:
    OCEAN_PSN = psn_lat2d >= 60.
DOMAIN_PSN = OCEAN_PSN & (psn_lat2d >= 60.)

# Pre-compute Pistone region masks once
REGION_MASKS_PSN = []
for code, lo, hi, lmn, lmx in PISTONE_REGIONS:
    lp     = np.where(psn_lon2d > 180, psn_lon2d - 360., psn_lon2d)
    lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
    lon_ok = ((lp >= lmn) & (lp <= lmx)
              if lmn <= lmx else (lp >= lmn) | (lp <= lmx))
    REGION_MASKS_PSN.append(lat_ok & lon_ok & DOMAIN_PSN)

with xr.open_dataset(
        sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]) as _ds:
    ceres_lat1d = xr.open_dataset(CERES_FILE, engine='netcdf4')['lat'].values
    ceres_lon1d = xr.open_dataset(CERES_FILE, engine='netcdf4')['lon'].values
ceres_lon2d, ceres_lat2d = np.meshgrid(ceres_lon1d, ceres_lat1d)

idx_sial_to_psn,  _ = build_scatter_index(
    sial_lat2d,  sial_lon2d,  psn_lat2d, psn_lon2d)
idx_ceres_to_psn, _ = build_scatter_index(
    ceres_lat2d, ceres_lon2d, psn_lat2d, psn_lon2d)

# Fixed CERES RS climatology (2000–2024)
print('Building fixed CERES RS climatology...', flush=True)
RS_PSN_CLIM = {}   # keyed by month number
for mn in MONTH_NUMS:
    RS_c = load_RS_ceres_clim(CERES_CLIM_YRS, mn)
    if RS_c is not None:
        RS_PSN_CLIM[mn] = scatter_avg(RS_c.ravel(), idx_ceres_to_psn, psn_shape)
        print(f'  Month {mn} RS clim loaded', flush=True)

# Fixed SIC data from obs cache
def load_sic(mname):
    try:
        return dict(
            slope=np.load(f'{CACHE_OBS}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval =np.load(f'{CACHE_OBS}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e=np.load(f'{CACHE_OBS}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l=np.load(f'{CACHE_OBS}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except Exception:
        return None

# Fixed early-period albedo
AE_EARLY = {}   # keyed by month name
for mn in MONTH_NUMS:
    mname = MONTH_NAMES[mn]
    ae = load_alb_mean(mname, EARLY_YEARS)
    if ae is not None:
        AE_EARLY[mname] = ae

# ── MAIN ROLLING LOOP ──────────────────────────────────────────────────────────
rng = np.random.default_rng(BOOTSTRAP_SEED)
rows = []

target_years = list(range(FIRST_LATE_END, LAST_LATE_END + 1))
print(f'\nComputing rolling SIAF for {len(target_years)} windows...', flush=True)

for t_end in target_years:
    late_years = list(range(t_end - WINDOW + 1, t_end + 1))
    dT_samples = compute_dT_samples_ols(t_end)
    dT_mean    = float(np.mean(dT_samples))

    if not np.isfinite(dT_mean) or dT_mean <= 0:
        print(f'  {t_end}: skip (dT = {dT_mean:.3f})', flush=True)
        continue

    month_siaf_tot, month_siaf_ext, month_siaf_drk = [], [], []
    month_boot = []   # list of (N_BOOTSTRAP,3) per month

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        ae = AE_EARLY.get(mname)
        if ae is None:
            continue
        al = load_alb_mean(mname, late_years)
        if al is None:
            continue
        sic = load_sic(mname)
        if sic is None:
            continue
        RS_psn = RS_PSN_CLIM.get(mn)
        if RS_psn is None:
            continue

        # Point estimate
        ST, SX, SD = compute_IS_SIAF(ae, al, sic, dT_mean, RS_psn)
        if ST is not None:
            month_siaf_tot.append(ST)
            month_siaf_ext.append(SX)
            month_siaf_drk.append(SD)

        # Bootstrap
        use_appx = (mname == 'September')
        ae_stk = load_alb_yearly_stack(mname, EARLY_YEARS, use_appx=use_appx)
        al_stk = load_alb_yearly_stack(mname, late_years,  use_appx=use_appx)
        if ae_stk is None or al_stk is None:
            month_boot.append(None)
            continue

        boot_mn = np.full((N_BOOTSTRAP, 3), np.nan)
        n_e, n_l, n_t = ae_stk.shape[0], al_stk.shape[0], len(dT_samples)
        for b in range(N_BOOTSTRAP):
            ae_b = np.nanmean(ae_stk[rng.integers(0, n_e, size=n_e)], axis=0)
            al_b = np.nanmean(al_stk[rng.integers(0, n_l, size=n_l)], axis=0)
            dT_b = float(dT_samples[rng.integers(0, n_t)])
            if dT_b <= 0:
                continue
            bT, bX, bD = compute_IS_SIAF(ae_b, al_b, sic, dT_b, RS_psn)
            if bT is not None:
                boot_mn[b] = [bT, bX, bD]
        month_boot.append(boot_mn)

    if not month_siaf_tot:
        print(f'  {t_end}: no valid months', flush=True)
        continue

    # Seasonal mean (simple mean over months)
    seas_tot = float(np.nanmean(month_siaf_tot)) * SCALE
    seas_ext = float(np.nanmean(month_siaf_ext)) * SCALE
    seas_drk = float(np.nanmean(month_siaf_drk)) * SCALE

    # Bootstrap seasonal mean and 2σ
    valid_boots = [b for b in month_boot if b is not None and not np.all(np.isnan(b))]
    if valid_boots:
        seas_b = np.nanmean(np.stack(valid_boots, axis=1), axis=1)  # (N_BOOT, 3)
        tot_2s = float(2 * np.nanstd(seas_b[:, 0])) * SCALE
        ext_2s = float(2 * np.nanstd(seas_b[:, 1])) * SCALE
        drk_2s = float(2 * np.nanstd(seas_b[:, 2])) * SCALE
    else:
        tot_2s = ext_2s = drk_2s = np.nan

    rows.append(dict(
        year      = t_end,
        late_start= t_end - WINDOW + 1,
        dT_mean   = round(dT_mean, 4),
        SIAF_tot  = round(seas_tot, 6),
        SIAF_ext  = round(seas_ext, 6),
        SIAF_drk  = round(seas_drk, 6),
        sigma2_tot= round(tot_2s,   6),
        sigma2_ext= round(ext_2s,   6),
        sigma2_drk= round(drk_2s,   6),
    ))
    print(f'  {t_end} (late {t_end-WINDOW+1}–{t_end}): '
          f'SIAF={seas_tot:+.4f} ± {tot_2s:.4f}  '
          f'(ext={seas_ext:+.4f}, drk={seas_drk:+.4f})', flush=True)
    gc.collect()

# ── SAVE CSV ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_out = os.path.join(OUT_DIR, 'siaf_rolling_timeseries.csv')
df.to_csv(csv_out, index=False)
print(f'\nCSV → {csv_out}')

# ── FIGURE ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.patch.set_facecolor('#FAFAFA')

yr  = df['year'].values
tot = df['SIAF_tot'].values
ext = df['SIAF_ext'].values
drk = df['SIAF_drk'].values
e_t = df['sigma2_tot'].values
e_x = df['sigma2_ext'].values
e_d = df['sigma2_drk'].values

COL_TOT = '#1a1a2e'
COL_EXT = '#3A6BAF'
COL_DRK = '#D95F1A'

# ── Panel 1: total SIAF with 2σ band + stacked components ──
ax = axes[0]
ax.set_facecolor('#FAFAFA')

# 2σ uncertainty band on total
ax.fill_between(yr, tot - e_t, tot + e_t,
                color=COL_TOT, alpha=0.15, label='Total SIAF ±2σ')

# Stacked area: extent + darkening
ax.fill_between(yr, 0, ext,
                color=COL_EXT, alpha=0.55, label='Extent')
ax.fill_between(yr, ext, ext + drk,
                color=COL_DRK, alpha=0.55, label='Darkening')

# Total line on top
ax.plot(yr, tot, color=COL_TOT, lw=2.0, zorder=5, label='Total SIAF')

# Reference lines for the two main analysis epochs
for yref, lbl in [(2014, 'CMIP6 era end'), (2024, 'Obs era end')]:
    if yref in yr:
        ax.axvline(yref, color='#888888', lw=1.0, ls=':', zorder=2)
        ax.text(yref + 0.3, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0.03,
                lbl, fontsize=8, color='#666666', va='top', rotation=90)

ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
ax.set_ylabel('Global Annual SIAF\n(W m⁻² K⁻¹)', fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, loc='upper left', ncol=2)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

# ── Panel 2: extent fraction + ΔT ──
ax2 = axes[1]
ax2_r = ax2.twinx()
ax2.set_facecolor('#FAFAFA')

ext_frac = np.where(tot > 0, ext / tot, np.nan)
ax2.plot(yr, ext_frac * 100, color=COL_EXT, lw=2.0,
         label='Extent fraction (%)')
ax2.fill_between(yr, ext_frac * 100 - (e_x / tot) * 100,
                      ext_frac * 100 + (e_x / tot) * 100,
                 color=COL_EXT, alpha=0.15)
ax2.set_ylabel('Extent fraction of SIAF (%)', fontsize=11, color=COL_EXT)
ax2.tick_params(axis='y', colors=COL_EXT)
ax2.set_ylim(0, 110)

ax2_r.plot(yr, df['dT_mean'].values, color='#888800', lw=1.5,
           ls='--', alpha=0.7, label='ΔT (K)')
ax2_r.set_ylabel('ΔT  (late − early,  K)', fontsize=10, color='#888800')
ax2_r.tick_params(axis='y', colors='#888800')

lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=9,
           framealpha=0.9, loc='lower right')

ax2.yaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)

ax2.set_xlabel(f'Year (end of trailing {WINDOW}-year late window)', fontsize=11)

fig.suptitle(
    f'Sea Ice–Albedo Feedback: {WINDOW}-Year Rolling Window\n'
    f'(late epoch = trailing {WINDOW} years;  ΔT = OLS trend×years, 1982→t;  '
    f'RS = CERES 2000–2024 climatology;  p < {PVAL_THRESH})',
    fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
fig_out = os.path.join(OUT_DIR, 'siaf_rolling_timeseries.png')
fig.savefig(fig_out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f'Figure → {fig_out}')

'''#!/usr/bin/env python
"""
fill_cache_gap_years.py
───────────────────────
Builds CLARA-A3 and APP-x albedo cache files for years 2000–2004,
which are not covered by either sensitivity_cache_obs (2015–2024 late)
or sensitivity_cache_cmip6 (2005–2014 late).

Saves into CACHE_OBS so the rolling SIAF script finds them automatically
(it searches CACHE_OBS before CACHE_CMIP6).

Only albedo files are needed — SIC trend/climo .npy files already exist
in both caches and the rolling script reuses them unchanged.

Run time: ~10–20 min (mostly I/O bound on CLARA-A3 and APP-x files).
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
from netCDF4 import Dataset as NC4Dataset
import glob, os, gc, re
import datetime as dt_mod

# ── Paths (match build_albedo_cache.py) ───────────────────────────────────────
DIR_SIAL   = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX   = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC    = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_OUT  = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'   # write here

# Years to fill in (gap between CMIP6 late-epoch end and obs late-epoch start)
GAP_YEARS  = list(range(2000, 2005))   # 2000, 2001, 2002, 2003, 2004

MONTHS_TO_ANALYZE = {
    'April'    : 4,
    'May'      : 5,
    'June'     : 6,
    'July'     : 7,
    'August'   : 8,
    'September': 9,
}

SIC_ICE_THRESHOLD = 0.15
PENTAD_DAYS       = 5
MIN_ARCTIC_PX     = 100

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

os.makedirs(CACHE_OUT, exist_ok=True)

# ── Grid helpers (identical to build_albedo_cache.py) ─────────────────────────

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
    w = weights[mask];  v = field[mask]
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

# ── Build grids ────────────────────────────────────────────────────────────────
print('Building grids...', flush=True)

# SIAL
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values
    sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape   = sial_lat2d.shape
sial_coslat  = np.cos(np.deg2rad(sial_lat2d)).astype(float)
sial_coslat  = np.where(sial_coslat > 0, sial_coslat, 0.0)

# APP-x
sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
    appx_fill    = float(getattr(ds.variables['cdr_surface_albedo'], '_FillValue', 9999.0))
if appx_lat_raw.ndim == 1:
    appx_lon2d, appx_lat2d = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    appx_lat2d, appx_lon2d = appx_lat_raw, appx_lon_raw
appx_shape   = appx_lat2d.shape
appx_weights = np.ones(appx_shape, dtype=float)

# PSN25 (for SIC regridding)
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values
    sic_y = ds['y'].values
sic_xx, sic_yy = np.meshgrid(sic_x, sic_y)
transformer     = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
sic_lon2d_psn, sic_lat2d_psn = transformer.transform(sic_xx, sic_yy)
sic_shape = sic_lat2d_psn.shape
sic_xyz   = latlon_to_xyz(sic_lat2d_psn.ravel(), sic_lon2d_psn.ravel())
sic_tree  = cKDTree(sic_xyz)

sial_xyz = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())

_, idx_sic_to_sial = sic_tree.query(sial_xyz, workers=-1)
_, idx_sic_to_appx = sic_tree.query(appx_xyz, workers=-1)

def regrid_sic_to_sial(sic_2d): return sic_2d.ravel()[idx_sic_to_sial].reshape(sial_shape)
def regrid_sic_to_appx(sic_2d): return sic_2d.ravel()[idx_sic_to_appx].reshape(appx_shape)

# Arctic 60N mask
arctic_60N_mask_psn = (sic_lat2d_psn >= 60.0)
_, idx_60N_sial     = sic_tree.query(sial_xyz, workers=-1)
arctic_60N_sial     = arctic_60N_mask_psn.ravel()[idx_60N_sial].reshape(sial_shape)
_, idx_60N_appx     = sic_tree.query(appx_xyz, workers=-1)
arctic_60N_appx     = arctic_60N_mask_psn.ravel()[idx_60N_appx].reshape(appx_shape)

print(f'  SIAL: {sial_shape}  APP-x: {appx_shape}  PSN25: {sic_shape}', flush=True)


# ── CLARA-A3 for gap years ─────────────────────────────────────────────────────
print(f'\nStep 1: CLARA-A3 albedo for {GAP_YEARS}', flush=True)

for year in GAP_YEARS:
    print(f'  CLARA {year}...', end='', flush=True)

    # Skip if all months already cached
    all_cached = all(
        os.path.exists(f'{CACHE_OUT}/clara_alb_{mn}_{year}.npy')
        for mn in MONTHS_TO_ANALYZE
    )
    if all_cached:
        print(' (all cached, skip)')
        continue

    sial_files = [f for f in sorted(glob.glob(f'{DIR_SIAL}/*.nc')) if str(year) in f]
    if not sial_files:
        print(' skip (no CLARA files)')
        continue

    try:
        ds_sial   = xr.open_mfdataset(sial_files, combine='by_coords')
        snow_alb  = ds_sial['black_sky_albedo_snow_mean'] / 100.0
        reg_alb   = ds_sial['black_sky_albedo_mean'] / 100.0
        sial_raw  = xr.where(~np.isnan(snow_alb), snow_alb, reg_alb)
        sial_pent = sial_raw.resample(time=f'{PENTAD_DAYS}D').mean().load()
        ds_sial.close()
    except Exception as e:
        print(f' skip ({e})')
        continue

    # Load SIC daily for this year
    sic_daily = {}
    for f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}*.nc')):
        try:
            t, raw = load_sic_day(f)
            sic_daily[t.date()] = regrid_sic_to_sial(raw)
        except Exception:
            continue

    sic_pent = {}
    for ptime in sial_pent.time.values:
        pd_date  = pd.Timestamp(ptime).date()
        bin_days = [pd_date + dt_mod.timedelta(days=d) for d in range(PENTAD_DAYS)]
        maps     = [sic_daily[d] for d in bin_days if d in sic_daily]
        if maps:
            sic_pent[pd_date] = np.nanmean(maps, axis=0)

    saved = 0
    for month_name, month_num in MONTHS_TO_ANALYZE.items():
        out_path = f'{CACHE_OUT}/clara_alb_{month_name}_{year}.npy'
        if os.path.exists(out_path):
            saved += 1
            continue

        sial_month = sial_pent.sel(time=sial_pent['time.month'] == month_num)
        if len(sial_month.time) == 0:
            continue

        sial_mean = sial_month.mean(dim='time').values
        sic_maps  = [sic_pent[pd.Timestamp(t).date()]
                     for t in sial_month.time.values
                     if pd.Timestamp(t).date() in sic_pent]
        if not sic_maps:
            continue
        sic_mean = np.nanmean(sic_maps, axis=0)

        valid    = arctic_60N_sial
        sial_out = np.where(valid, sial_mean, np.nan)
        sic_out  = np.where(valid, sic_mean,  np.nan)
        valid_px = np.isfinite(sial_out) & np.isfinite(sic_out) & valid
        if valid_px.sum() < MIN_ARCTIC_PX:
            continue

        m_ice   = valid_px & (sic_out >= SIC_ICE_THRESHOLD)
        m_ocean = valid_px & (sic_out <  SIC_ICE_THRESHOLD)
        alb_ice   = weighted_mean(sial_out, m_ice,   sial_coslat)
        alb_ocean = weighted_mean(sial_out, m_ocean, sial_coslat)

        if np.isfinite(alb_ice) and np.isfinite(alb_ocean):
            np.save(out_path, np.where(valid_px, sial_out, np.nan))
            np.save(f'{CACHE_OUT}/clara_dalpha_{month_name}_{year}.npy',
                    np.array(alb_ice - alb_ocean))
            saved += 1

    print(f' ✓ ({saved}/{len(MONTHS_TO_ANALYZE)} months)', flush=True)
    del sial_pent, sic_daily
    gc.collect()


# ── APP-x for September gap years ─────────────────────────────────────────────
# (CLARA is used for Apr–Aug; APP-x only needed for September)
print(f'\nStep 2: APP-x albedo (September only) for {GAP_YEARS}', flush=True)

def load_appx_monthly(year, month_num):
    base   = DIR_APPX.rstrip('/')
    prefix = 'Polar-APP-X'
    files  = []
    for p in [f'{base}/{year}/{prefix}*d{year}{month_num:02d}*.nc',
              f'{base}/{prefix}*d{year}{month_num:02d}*.nc']:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        return None

    files_by_date = {}
    for f in files:
        obs_date = parse_date_appx(f)
        if obs_date is None:
            continue
        obs_str = obs_date.strftime('%Y%m%d')
        m_time  = re.search(r'_(\d{4})_d', os.path.basename(f))
        overpass = m_time.group(1) if m_time else 'unknown'
        m_cdate  = re.search(r'_c(\d{8})\.nc$', os.path.basename(f))
        cdate    = m_cdate.group(1) if m_cdate else '00000000'
        if obs_str not in files_by_date:
            files_by_date[obs_str] = {}
        if overpass not in files_by_date[obs_str] or \
                cdate > files_by_date[obs_str][overpass][1]:
            files_by_date[obs_str][overpass] = (f, cdate)

    monthly_maps = []
    for obs_str, passes in sorted(files_by_date.items()):
        daily = []
        for overpass, (f, cdate) in passes.items():
            try:
                with NC4Dataset(f, 'r') as ds:
                    vname = next((v for v in ds.variables
                                  if 'albedo' in v.lower()
                                  and ds.variables[v].ndim >= 2), None)
                    if not vname:
                        continue
                    alb_var = ds.variables[vname]
                    alb     = alb_var[0] if alb_var.ndim == 3 else alb_var[:]
                    if hasattr(alb, 'filled'):
                        alb = alb.filled(np.nan)
                    alb = alb.astype(float)
                    file_fill = float(getattr(alb_var, '_FillValue', appx_fill))
                    alb[alb == file_fill] = np.nan
                    alb[(alb < 0) | (alb > 1)] = np.nan
                    daily.append(alb)
            except Exception:
                continue
        if daily:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                monthly_maps.append(np.nanmean(daily, axis=0))

    if not monthly_maps:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(monthly_maps, axis=0)


def load_sic_monthly_appx(year, month_num):
    maps = []
    for f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}*.nc')):
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


for year in GAP_YEARS:
    month_name, month_num = 'September', 9
    out_path = f'{CACHE_OUT}/appx_alb_{month_name}_{year}.npy'

    if os.path.exists(out_path):
        print(f'  APP-x Sep {year}: (cached, skip)', flush=True)
        continue

    print(f'  APP-x Sep {year}...', end='', flush=True)
    alb_mean = load_appx_monthly(year, month_num)
    sic_mean = load_sic_monthly_appx(year, month_num)

    if alb_mean is None or sic_mean is None:
        print(' skip (missing data)', flush=True)
        continue

    valid    = arctic_60N_appx
    alb_out  = np.where(valid, alb_mean, np.nan)
    sic_out  = np.where(valid, sic_mean, np.nan)
    valid_px = np.isfinite(alb_out) & np.isfinite(sic_out) & valid
    if valid_px.sum() < MIN_ARCTIC_PX:
        print(f' skip ({valid_px.sum()} valid px)', flush=True)
        continue

    m_ice   = valid_px & (sic_out >= SIC_ICE_THRESHOLD)
    m_ocean = valid_px & (sic_out <  SIC_ICE_THRESHOLD)
    if m_ice.sum() < MIN_ARCTIC_PX or m_ocean.sum() < MIN_ARCTIC_PX:
        print(' skip (insufficient ice/ocean px)', flush=True)
        continue

    alb_ice   = weighted_mean(alb_out, m_ice,   appx_weights)
    alb_ocean = weighted_mean(alb_out, m_ocean, appx_weights)
    if np.isfinite(alb_ice) and np.isfinite(alb_ocean):
        np.save(out_path, np.where(valid_px, alb_out, np.nan))
        np.save(f'{CACHE_OUT}/appx_dalpha_{month_name}_{year}.npy',
                np.array(alb_ice - alb_ocean))
        print(' ✓', flush=True)
    else:
        print(' skip (non-finite albedo stats)', flush=True)
    gc.collect()


# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Gap-fill complete. Files written to:', flush=True)
print(f'  {CACHE_OUT}')
print('\nFiles created:')
for year in GAP_YEARS:
    for month_name in MONTHS_TO_ANALYZE:
        prefix = 'appx' if month_name == 'September' else 'clara'
        p = f'{CACHE_OUT}/{prefix}_alb_{month_name}_{year}.npy'
        status = '✓' if os.path.exists(p) else '✗ MISSING'
        print(f'  {status}  {os.path.basename(p)}')'''