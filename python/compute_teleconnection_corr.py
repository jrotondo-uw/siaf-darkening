'''#!/usr/bin/env python
"""
compute_teleconnection_corr.py
──────────────────────────────
Tests whether PDO and AO indices explain the decadal wiggle in the
forced α-T relationship and the interannual residuals.

Data sources (download to GLADE before running):
  AO  monthly: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii
  PDO monthly: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat

Expected column formats:
  AO  : year  jan feb mar apr may jun jul aug sep oct nov dec
  PDO : year  jan feb mar apr may jun jul aug sep oct nov dec  annual
"""

import numpy as np
import pandas as pd
import pickle, os
from scipy import stats
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL_PATH = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/dalpha_dt_evolution.pkl'
AO_PATH  = '/glade/work/jrotondo/ao_monthly.txt'
PDO_PATH = '/glade/work/jrotondo/pdo_monthly.txt'
OUT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
os.makedirs(OUT_DIR, exist_ok=True)

MONTH_NUMS = [4, 5, 6, 7, 8, 9]   # AMJJAS
LP_WIN     = 9

# ── LOAD PICKLE ───────────────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as fh:
    d = pickle.load(fh)

yr_arr   = d['yr_arr']
alb_arr  = d['alb_arr']
T_arr    = d['T_arr']
alb_high = d['alb_high']
T_high   = d['T_high']
alb_low  = d['alb_low']
T_low    = d['T_low']
lo_slope = d['lo_slope']
hi_slope = d['hi_slope']
full_slope = d['full_slope']
full_ic    = d['full_ic']

edge = LP_WIN // 2
sl_e = slice(edge, len(yr_arr) - edge)


# ── OLS HELPER ────────────────────────────────────────────────────────────────
def ols(x, y):
    return stats.linregress(x, y)


# ── LOAD INDEX ────────────────────────────────────────────────────────────────
def load_noaa_monthly_index(path, missing_vals=(-99.99, -9.99, 999.0, -999.0)):
    """
    Parse NOAA monthly index files.  Handles two formats automatically:
      Wide : year  jan feb mar apr may jun jul aug sep oct nov dec  [annual]
      Long : year  month  value   (one row per month, NOAA CPC AO style)
    Returns dict {year: np.array of 12 monthly values (NaN where missing)}.
    """
    def _clean(v):
        try:
            fv = float(v)
            return np.nan if any(abs(fv - mv) < 0.01 for mv in missing_vals) else fv
        except ValueError:
            return np.nan

    # Peek at first data line to detect format
    fmt = None
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            fmt = 'wide' if len(parts) >= 13 else 'long'
            break

    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                yr = int(parts[0])
            except ValueError:
                continue
            if fmt == 'wide':
                if len(parts) < 13:
                    continue
                rows[yr] = np.array([_clean(parts[i]) for i in range(1, 13)])
            else:  # long format: year month value
                try:
                    mn = int(parts[1])
                except ValueError:
                    continue
                if not (1 <= mn <= 12):
                    continue
                if yr not in rows:
                    rows[yr] = np.full(12, np.nan)
                rows[yr][mn - 1] = _clean(parts[2])
    return rows


def amjjas_mean(monthly_dict, years):
    """AMJJAS (months 4-9, indices 3-8) mean for each year."""
    out = {}
    for y in years:
        if y not in monthly_dict:
            continue
        vals = monthly_dict[y][3:9]   # Apr=idx3 ... Sep=idx8
        if np.isfinite(vals).sum() >= 3:
            out[y] = float(np.nanmean(vals))
    return out


# ── READ AND ALIGN ─────────────────────────────────────────────────────────────
indices = {}

for name, path in [('AO', AO_PATH), ('PDO', PDO_PATH)]:
    if not os.path.exists(path):
        print(f'  WARNING: {name} file not found at {path} -- skipping')
        continue
    monthly = load_noaa_monthly_index(path)
    seasonal = amjjas_mean(monthly, yr_arr.tolist())

    idx_arr = np.array([seasonal.get(int(y), np.nan) for y in yr_arr])
    # Demean over available years
    finite_mask = np.isfinite(idx_arr)
    if finite_mask.sum() < 10:
        print(f'  WARNING: {name} has too few valid years -- skipping')
        continue
    idx_arr -= np.nanmean(idx_arr)
    indices[name] = idx_arr
    print(f'  {name}: {finite_mask.sum()} AMJJAS years loaded')

if not indices:
    print('\nNo index files found.  Download them first:\n'
          '  AO:  https://www.cpc.ncep.noaa.gov/products/precip/CWlink/'
          'daily_ao_index/monthly.ao.index.b50.current.ascii\n'
          '  PDO: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/'
          'ersst.v5.pdo.dat\n'
          'Save as ao_monthly.txt and pdo_monthly.txt in /glade/work/jrotondo/')
    raise SystemExit(0)

# Low-pass filter each index (same 9-yr filter as T and α)
indices_low  = {k: uniform_filter1d(v, size=LP_WIN, mode='nearest')
                for k, v in indices.items() if np.isfinite(v).all()}
indices_high = {k: v - indices_low[k]
                for k, v in indices.items() if k in indices_low}


# ── CORRELATIONS ──────────────────────────────────────────────────────────────
print('\n── Correlations with full-period albedo ────────────────────────────────')
for name, idx in indices.items():
    mask = np.isfinite(idx)
    r, p = stats.pearsonr(idx[mask], alb_arr[mask])
    print(f'  {name} vs α_raw:       r = {r:+.3f}  p = {p:.4f}')

print('\n── Interannual (high-pass) residuals ───────────────────────────────────')
for name, idx in indices_high.items():
    mask = np.isfinite(idx[sl_e])
    r, p = stats.pearsonr(idx[sl_e][mask], alb_high[sl_e][mask])
    print(f'  {name}_high vs α_high: r = {r:+.3f}  p = {p:.4f}')
    r2, p2 = stats.pearsonr(idx[sl_e][mask], T_high[sl_e][mask])
    print(f'  {name}_high vs T_high: r = {r2:+.3f}  p = {p2:.4f}')

# Residuals from the low-pass regression (the "decadal wiggle" in panel e)
lo_ic  = float(np.mean(alb_low) - lo_slope * np.mean(T_low))
resid_low = alb_low - (lo_slope * T_low + lo_ic)

print('\n── Decadal wiggle (residuals of forced regression) ─────────────────────')
for name, idx in indices_low.items():
    mask = np.isfinite(idx[sl_e])
    r, p = stats.pearsonr(idx[sl_e][mask], resid_low[sl_e][mask])
    print(f'  {name}_low vs resid_low: r = {r:+.3f}  p = {p:.4f}')

# Test whether the wiggle is just secular nonlinearity (correlates with time)
r_t, p_t = stats.pearsonr(yr_arr[sl_e], resid_low[sl_e])
print(f'\n── Is the wiggle just time (nonlinear β)? ──────────────────────────────')
print(f'  time vs resid_low:  r = {r_t:+.3f}  p = {p_t:.4f}')
# Quadratic fit: if R² much larger with t² than with t, it's nonlinear
t_c = yr_arr[sl_e] - yr_arr[sl_e].mean()
X_quad = np.column_stack([t_c, t_c**2, np.ones(len(t_c))])
coef_q, _, _, _ = np.linalg.lstsq(X_quad, resid_low[sl_e], rcond=None)
resid_q = resid_low[sl_e] - X_quad @ coef_q
r2_quad = 1. - np.var(resid_q) / np.var(resid_low[sl_e])
print(f'  Quadratic fit R²:   {r2_quad:.3f}  '
      f'(high = wiggle is temporal nonlinearity, not a climate mode)')

# ── PARTIAL REGRESSION: α = β_T*T + β_AO*AO + β_PDO*PDO + ε ─────────────────
print('\n── Multiple regression (raw annual, all indices) ───────────────────────')
names_present = list(indices.keys())
col_list = [T_arr] + [indices[n] for n in names_present]
col_names = ['T'] + names_present

# Only use years where all series are finite
stack = np.column_stack(col_list)
ok = np.all(np.isfinite(stack), axis=1)
X = np.column_stack([stack[ok], np.ones(ok.sum())])
y = alb_arr[ok]
coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ coefs
ss_res = np.sum((y - y_hat)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_multi = 1. - ss_res / ss_tot
print(f'  Predictors: {col_names} + intercept')
for nm, c in zip(col_names + ['intercept'], coefs):
    print(f'    {nm:12s}  coef = {c:+.6f}')
print(f'  R² = {r2_multi:.3f}  (cf. T-only R² = {d["full_r"]**2:.3f})')

# Variance in α explained by each index alone (controlling for T)
print('\n── Partial R² for each index after removing T ──────────────────────────')
T_ok = T_arr[ok]
alb_ok = alb_arr[ok]
# residuals of α on T
_, _, _, _, _ = ols(T_ok, alb_ok)
alb_resid_T = alb_ok - (ols(T_ok, alb_ok).slope * T_ok + ols(T_ok, alb_ok).intercept)
for nm in names_present:
    idx_ok = indices[nm][ok]
    r_part, p_part = stats.pearsonr(idx_ok, alb_resid_T)
    print(f'  {nm} vs α|T:  partial r = {r_part:+.3f}  '
          f'partial R² = {r_part**2:.3f}  p = {p_part:.4f}')


# ── FIGURE ────────────────────────────────────────────────────────────────────
nidx = len(indices)
fig, axes = plt.subplots(2, max(nidx, 1) + 1,
                         figsize=(5 * (nidx + 1), 8),
                         facecolor='white')
if axes.ndim == 1:
    axes = axes[np.newaxis, :]

fig.suptitle('PDO / AO vs Arctic Albedo Residuals  (AMJJAS)', fontsize=12,
             fontweight='bold')

COLORS = {'AO': '#e07b00', 'PDO': '#007b5e'}

for col, (name, idx) in enumerate(indices.items()):
    clr = COLORS.get(name, '#444')

    # Row 0: index time series + low-pass
    ax = axes[0, col]
    idx_lp = indices_low.get(name, np.full_like(idx, np.nan))
    ax.fill_between(yr_arr, idx, idx_lp, color=clr, alpha=0.18)
    ax.plot(yr_arr, idx,    color='#aaa', lw=0.9, alpha=0.7, label='Raw')
    ax.plot(yr_arr, idx_lp, color=clr,   lw=2.1, label=f'{LP_WIN}-yr low-pass')
    ax.axhline(0, color='#ccc', lw=0.7, ls=':')
    ax.set_title(f'{name} index (AMJJAS)', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel(f'{name} anomaly')
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # Row 1: scatter of index_high vs alb_high
    ax = axes[1, col]
    idx_hp = indices_high.get(name, np.full_like(idx, np.nan))
    mask = np.isfinite(idx_hp[sl_e])
    sc = ax.scatter(idx_hp[sl_e][mask], alb_high[sl_e][mask],
                    c=yr_arr[sl_e][mask], cmap='plasma',
                    s=40, edgecolors='white', lw=0.4, zorder=3)
    res = ols(idx_hp[sl_e][mask], alb_high[sl_e][mask])
    if abs(res.rvalue) > 0.05:
        xr = np.array([idx_hp[sl_e][mask].min(), idx_hp[sl_e][mask].max()])
        ax.plot(xr, res.slope * xr + res.intercept, color=clr, lw=1.8)
    ax.axhline(0, color='#ccc', lw=0.6, ls=':')
    ax.axvline(0, color='#ccc', lw=0.6, ls=':')
    ax.text(0.05, 0.95,
            f'r = {res.rvalue:+.3f}  p = {res.pvalue:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(fc='white', ec=clr, alpha=0.2, pad=3))
    ax.set_xlabel(f'{name} anomaly (interannual)')
    ax.set_ylabel('Δα interannual (high-pass)')
    ax.set_title(f'{name}_high vs α_high', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Year', shrink=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# Last column: α residuals on T coloured by low-pass index
for row, (name, idx_lp) in enumerate(indices_low.items()):
    if row >= 2:
        break
    ax = axes[row, nidx]
    clr = COLORS.get(name, '#444')
    sc = ax.scatter(T_low[sl_e], resid_low[sl_e],
                    c=idx_lp[sl_e], cmap='RdBu_r',
                    s=40, edgecolors='white', lw=0.4, zorder=3)
    ax.axhline(0, color='#ccc', lw=0.7, ls=':')
    mask_lp = np.isfinite(idx_lp[sl_e])
    res_lp = ols(idx_lp[sl_e][mask_lp], resid_low[sl_e][mask_lp])
    ax.text(0.05, 0.95,
            f'{name}_low vs resid\nr = {res_lp.rvalue:+.3f}  p = {res_lp.pvalue:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(fc='white', ec=clr, alpha=0.2, pad=3))
    ax.set_xlabel('ΔT forced (low-pass, K)')
    ax.set_ylabel('α residual from forced regression')
    ax.set_title(f'Decadal wiggle coloured by {name}', fontweight='bold')
    cb = plt.colorbar(sc, ax=ax, label=f'{name} (low-pass)', shrink=0.8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# Hide any unused axes
for r in range(2):
    for c in range(nidx + 1):
        if r >= 1 and c >= nidx and len(indices_low) == 1:
            axes[r, c].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'teleconnection_corr.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved -> {out_path}')

#!/usr/bin/env python
"""
compute_sie_rs_teleconnection.py
────────────────────────────────
Computes annual AMJJAS Arctic SIE (from NSIDC CDR PSN25 SIC files),
domain-mean RS from two sources:
  • CERES EBAF T07-inversion          (2000-2024)
  • ERA5 Huang kernel × CLARA-A3/APP-x albedo  (1982-2024)
Then tests how both relate to the AO and PDO indices.

SIE : sum of grid-cell area where cdr_seaice_conc >= 15%  (Mkm²)
RS  : domain-mean surface solar absorption (W/m²)
"""

import gc, glob, os, warnings
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from pyproj import Transformer
import xarray as xr
import pandas as pd
from netCDF4 import Dataset as NC4Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DIR_SIC    = '/glade/derecho/scratch/jrotondo/SIC_data/'
CERES_FILE = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
              'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
DIR_SIAL   = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX   = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
CACHE_OBS  = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
KERNEL_FILE  = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR   = 'SFC_all'
KERNEL_SCALE = 100.0

AO_PATH  = '/glade/work/jrotondo/ao_monthly.txt'
PDO_PATH = '/glade/work/jrotondo/pdo_monthly.txt'

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/sie_rs_teleconnection/'
os.makedirs(OUT_DIR, exist_ok=True)

YEARS_SIE   = list(range(1982, 2025))
YEARS_CERES = list(range(2000, 2025))
YEARS_ERA5  = list(range(1982, 2025))
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
ARCTIC_LAT_MIN = 60.
SIC_ICE_THRESH = 0.15
LP_WIN         = 9

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ── PSN25 GRID ────────────────────────────────────────────────────────────────
print('Setting up PSN25 grid ...')
_sic_sample = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(_sic_sample) as _ds:
    _px = _ds['x'].values.astype(np.float64)
    _py = _ds['y'].values.astype(np.float64)
_pxx, _pyy = np.meshgrid(_px, _py)
PSN_SHAPE  = _pxx.shape
_tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = _tfm.transform(_pxx, _pyy)
del _pxx, _pyy

_phi_ts = np.deg2rad(70.)
_k_ps   = (1. + np.sin(_phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W       = (1. / _k_ps**2).astype(np.float32)
CELL_AREA_KM2 = (25.**2) * AREA_W   # km² per PSN25 cell

DOMAIN_PSN = (PSN_LAT >= ARCTIC_LAT_MIN)
print(f'  PSN_SHAPE={PSN_SHAPE}  domain pixels >= {ARCTIC_LAT_MIN}N: '
      f'{DOMAIN_PSN.sum():,}  total area: '
      f'{float(np.sum(CELL_AREA_KM2[DOMAIN_PSN]))/1e6:.1f} Mkm²')

# ── SIAL (CLARA-A3) GRID ─────────────────────────────────────────────────────
print('Setting up SIAL grid ...')
_sial_sample = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(_sial_sample) as _ds:
    _sla = _ds['lat'].values; _slo = _ds['lon'].values
if _sla.ndim == 1:
    SIAL_LON, SIAL_LAT = np.meshgrid(_slo, _sla)
else:
    SIAL_LAT, SIAL_LON = _sla, _slo
SIAL_SHAPE = SIAL_LAT.shape
try:
    ARCTIC_SIAL = np.load(f'{CACHE_OBS}/arctic_60N_sial.npy').astype(bool)
except FileNotFoundError:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)
print(f'  SIAL_SHAPE={SIAL_SHAPE}  Arctic mask: {ARCTIC_SIAL.sum():,} pixels')

# APP-x → SIAL nearest-neighbour index
IDX_APPX_TO_SIAL = None
try:
    _appx_sample = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(_appx_sample, 'r') as _ds:
        _al = np.ma.filled(_ds.variables['latitude'][:],  np.nan).astype(float)
        _ao = np.ma.filled(_ds.variables['longitude'][:], np.nan).astype(float)
    if _al.ndim == 1:
        _ao2, _al2 = np.meshgrid(_ao, _al)
    else:
        _al2, _ao2 = _al, _ao
    def _xyz(lat, lon):
        la = np.deg2rad(np.asarray(lat).ravel().astype(np.float64))
        lo = np.deg2rad(np.asarray(lon).ravel().astype(np.float64))
        return np.column_stack([np.cos(la)*np.cos(lo),
                                 np.cos(la)*np.sin(lo),
                                 np.sin(la)])
    _, IDX_APPX_TO_SIAL = cKDTree(_xyz(_al2, _ao2)).query(
        _xyz(SIAL_LAT, SIAL_LON), workers=-1)
    IDX_APPX_TO_SIAL = IDX_APPX_TO_SIAL.astype(np.int32)
    print('  APP-x → SIAL index built.')
except Exception as _e:
    print(f'  WARNING: APP-x → SIAL index failed ({_e})')

# Define _xyz at module level (may not have been created if APP-x failed)
def _xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel().astype(np.float64))
    lo = np.deg2rad(np.asarray(lon).ravel().astype(np.float64))
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])

# SIAL → PSN25 index
print('  Building SIAL → PSN25 index ...')
_, IDX_SIAL_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
    _xyz(SIAL_LAT, SIAL_LON), workers=-1)
IDX_SIAL_TO_PSN = IDX_SIAL_TO_PSN.astype(np.int32)

# CERES → PSN25 nearest-neighbour index
print('  Building CERES → PSN25 index ...')
with xr.open_dataset(CERES_FILE) as _ds:
    _cla = _ds['lat'].values; _clo = _ds['lon'].values
_clon2d, _clat2d = np.meshgrid(_clo, _cla)
_, IDX_CERES_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
    _xyz(_clat2d, _clon2d), workers=-1)
del _clat2d, _clon2d


# ── GRID UTILITIES ────────────────────────────────────────────────────────────
def scatter_avg(flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32); cnt = np.zeros(n, np.float32)
    ok  = np.isfinite(flat)
    np.add.at(acc, idx[ok], flat[ok].astype(np.float32))
    np.add.at(cnt, idx[ok], 1.)
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0; out[m] = acc[m] / cnt[m]
    return out.reshape(shape)

def regrid_sial_to_psn(arr):
    src = np.where(ARCTIC_SIAL,
                   np.where(np.isfinite(arr), arr, np.float32(0.)),
                   np.nan)
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)

def domain_mean(arr):
    v = arr[DOMAIN_PSN]; w = AREA_W[DOMAIN_PSN]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.sum() else np.nan


# ── SIC LOADING ───────────────────────────────────────────────────────────────
def _find_sic_var(ds):
    preferred = ['cdr_seaice_conc', 'seaice_conc', 'siconc',
                 'sea_ice_concentration', 'sic', 'ice_conc']
    for name in preferred:
        if name in ds.data_vars:
            return name
    for v in ds.data_vars:
        vl = v.lower()
        if ('conc' in vl or 'sic' in vl) and not any(
                x in vl for x in ('flag', 'source', 'stdev', 'spatial',
                                   'interp', 'bt_', 'nt_')):
            return v
    return None

def load_psn_sic(year, month):
    files = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}{month:02d}*.nc'))
    if not files:
        return None
    try:
        with xr.open_dataset(files[0]) as ds:
            vname = _find_sic_var(ds)
            if vname is None:
                return None
            arr = ds[vname].values.squeeze().astype(np.float32)
        if float(np.nanmax(arr)) > 1.5:
            arr = arr / 100.
        arr = np.where((arr >= 0.) & (arr <= 1.), arr, np.nan)
        return arr
    except Exception as e:
        print(f'  WARNING: load_psn_sic({year},{month}): {e}')
        return None


# ── CERES RS (T07) ────────────────────────────────────────────────────────────
_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt > 1., rsdt, np.nan)
        Sd = np.where(rsds > 1., rsds, np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds/S;  q  = rsut/S
        d  = q + p - 2.*p*a
        R  = np.clip((q-p*a) / np.where(np.abs(d)>1e-6, d, np.nan), 0., .99)
        A  = np.clip(1.-p*(1.-a*R) / np.where((1.-R)>.01, 1.-R, np.nan), 0., .99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs/S;  qc = rsutcs/S
        dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc-pc*ac) / np.where(np.abs(dc)>1e-6, dc, np.nan), 0., .99)
        Rl = np.clip(R - Rc, 0., .99)
        RS = (S*(1.-A)*(1.-Rl)**2
              / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS.astype(np.float32)

def load_ceres_rs_domainmean(year, month):
    with xr.open_dataset(CERES_FILE) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idxs  = [i for i, t in enumerate(times)
                 if t.year == year and t.month == month]
        if not idxs: return np.nan
        sums = {k: None for k in _CERES_VARS}; cnt = 0
        for i in idxs:
            sl = ds.isel(time=i).load()
            for k in _CERES_VARS:
                v = sl[k].values.astype(np.float32)
                sums[k] = v if cnt == 0 else sums[k] + v
            cnt += 1
    if cnt == 0: return np.nan
    for k in sums: sums[k] /= cnt
    rs = t07_invert(sums['solar_mon'], sums['toa_sw_all_mon'],
                    sums['sfc_sw_down_all_mon'], sums['sfc_sw_up_all_mon'],
                    sums['toa_sw_clr_c_mon'], sums['sfc_sw_down_clr_c_mon'],
                    sums['sfc_sw_up_clr_c_mon'])
    rs_psn = scatter_avg(rs.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)
    return domain_mean(rs_psn)


# ── ERA5 KERNEL RS (Huang & Huang, kernel × CLARA-A3/APP-x albedo) ────────────
print('\nLoading ERA5 Huang kernels ...')
KERNEL_PSN = {}
try:
    with xr.open_dataset(KERNEL_FILE) as _kds:
        _tdim = next((d for d, s in _kds.sizes.items() if s == 12), None)
        _vname = KERNEL_VAR
        if _vname not in _kds:
            _vname = next((v for v in _kds.data_vars
                           if v.lower() == KERNEL_VAR.lower()), None)
        _lat_n = 'latitude' if 'latitude' in _kds else 'lat'
        _lon_n = 'longitude' if 'longitude' in _kds else 'lon'
        _k_lat = _kds[_lat_n].values; _k_lon = _kds[_lon_n].values
        _k_lon2d, _k_lat2d = np.meshgrid(_k_lon, _k_lat)
        _, _idx_k = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
            _xyz(_k_lat2d, _k_lon2d), workers=-1)
        for _mn in MONTH_NUMS:
            _k_raw = np.abs(_kds[_vname].isel({_tdim: _mn-1}).values
                            .astype(np.float32)) * KERNEL_SCALE
            KERNEL_PSN[_mn] = scatter_avg(_k_raw.ravel(), _idx_k, PSN_SHAPE)
    print(f'  Kernels loaded for months: {list(KERNEL_PSN.keys())}')
    print(f'  Kernel domain mean (Apr): {domain_mean(KERNEL_PSN[4]):.1f} W/m²')
except Exception as _e:
    print(f'  WARNING: kernel load failed ({_e}) — ERA5 RS will be skipped')

def load_obs_albedo(year, month):
    """Load merged CLARA-A3/APP-x albedo on SIAL grid."""
    mname = MONTH_NAMES[month]
    if month == 9:
        p = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
        if not os.path.exists(p):
            return None
        raw = np.load(p).astype(np.float32)
        if IDX_APPX_TO_SIAL is not None and raw.shape != SIAL_SHAPE:
            raw = raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
        return raw
    p_c = f'{CACHE_OBS}/clara_alb_{mname}_{year}.npy'
    p_a = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
    clara = np.load(p_c).astype(np.float32) if os.path.exists(p_c) else None
    if os.path.exists(p_a):
        raw_a = np.load(p_a).astype(np.float32)
        if IDX_APPX_TO_SIAL is not None and raw_a.shape != SIAL_SHAPE:
            raw_a = raw_a.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
        appx = raw_a
    else:
        appx = None
    if clara is not None and appx is not None:
        return np.where(np.isfinite(clara), clara, appx)
    return clara if clara is not None else appx

def load_era5_rs_domainmean(year, month):
    """
    RS = domain_mean(k_psn × (1 − α))
    k_psn = |∂RS/∂α| × SCALE ≈ rsds (surface downwelling SW).
    Absorbed solar = rsds × (1 − α_surface), so RS = k_psn × (1 − α).
    """
    k_psn = KERNEL_PSN.get(month)
    if k_psn is None:
        return np.nan
    alb_sial = load_obs_albedo(year, month)
    if alb_sial is None:
        return np.nan
    alb_psn = regrid_sial_to_psn(alb_sial)
    one_minus_alb = np.where(np.isfinite(alb_psn),
                             1. - np.clip(alb_psn, 0., 1.), np.nan)
    return domain_mean(k_psn * one_minus_alb)


# ── DIAGNOSTIC: SIC variable check ───────────────────────────────────────────
print('\nSIC variable check (sample files) ...')
for _yr, _mn in [(1984, 4), (1995, 9), (2010, 9)]:
    _files = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{_yr}{_mn:02d}*.nc'))
    if not _files:
        print(f'  {_yr}-{_mn:02d}: no file'); continue
    with xr.open_dataset(_files[0]) as _ds:
        _vn = _find_sic_var(_ds)
        if _vn:
            _raw = _ds[_vn].values.squeeze()
            _mx  = float(np.nanmax(_raw))
            _arr = _raw/100. if _mx > 1.5 else _raw
            _arr = np.where((_arr>=0.)&(_arr<=1.), _arr, np.nan)
            _nice = int(np.sum(_arr >= SIC_ICE_THRESH))
            print(f'  {_yr}-{_mn:02d}: var={_vn!r}  max_raw={_mx:.1f}  '
                  f'n(>=0.15)={_nice:,}  '
                  f'SIE_this_month≈{_nice*625/1e6:.2f} Mkm²')
        else:
            print(f'  {_yr}-{_mn:02d}: NO SIC VAR. All vars: {list(_ds.data_vars)}')


# ── COMPUTE SIE TIME SERIES ───────────────────────────────────────────────────
print('\nComputing AMJJAS SIE (1982-2024) ...')
sie_by_year   = {}
n_months_used = {}

for year in YEARS_SIE:
    vals = []
    for mn in MONTH_NUMS:
        sic = load_psn_sic(year, mn)
        if sic is None or sic.shape != PSN_SHAPE:
            continue
        ice_mask = (sic >= SIC_ICE_THRESH) & np.isfinite(sic)
        sie_mkm2 = float(np.sum(CELL_AREA_KM2[ice_mask])) / 1e6
        vals.append(sie_mkm2)
    if vals:
        sie_by_year[year] = float(np.mean(vals))
        n_months_used[year] = len(vals)
    print(f'  {year}: SIE={sie_by_year.get(year, np.nan):.2f} Mkm²  '
          f'(n_months={n_months_used.get(year, 0)})')
gc.collect()


# ── COMPUTE CERES RS TIME SERIES (2000-2024) ──────────────────────────────────
print('\nComputing AMJJAS CERES RS (2000-2024) ...')
rs_ceres_by_year = {}
for year in YEARS_CERES:
    vals = [load_ceres_rs_domainmean(year, mn) for mn in MONTH_NUMS]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        rs_ceres_by_year[year] = float(np.mean(vals))
    print(f'  {year}: RS(CERES)={rs_ceres_by_year.get(year, np.nan):.2f} W/m²')
gc.collect()


# ── COMPUTE ERA5 KERNEL RS TIME SERIES (1982-2024) ────────────────────────────
rs_era5_by_year = {}
if KERNEL_PSN:
    print('\nComputing AMJJAS ERA5 kernel RS (1982-2024) ...')
    for year in YEARS_ERA5:
        vals = [load_era5_rs_domainmean(year, mn) for mn in MONTH_NUMS]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            rs_era5_by_year[year] = float(np.mean(vals))
        print(f'  {year}: RS(ERA5-kernel)={rs_era5_by_year.get(year, np.nan):.2f} W/m²')
    gc.collect()
else:
    print('\nSkipping ERA5 kernel RS (kernel not loaded).')


# ── LOAD AO / PDO ─────────────────────────────────────────────────────────────
def _load_index(path, missing=(-99.99, -9.99, 999., -999.)):
    def _clean(v):
        try:
            fv = float(v)
            return np.nan if any(abs(fv-m)<0.01 for m in missing) else fv
        except ValueError: return np.nan
    fmt = None
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3: continue
            try: int(p[0])
            except ValueError: continue
            fmt = 'wide' if len(p) >= 13 else 'long'
            break
    rows = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3: continue
            try: yr = int(p[0])
            except ValueError: continue
            if fmt == 'wide':
                if len(p) < 13: continue
                rows[yr] = np.array([_clean(p[i]) for i in range(1, 13)])
            else:
                try: mn = int(p[1])
                except ValueError: continue
                if not (1 <= mn <= 12): continue
                if yr not in rows: rows[yr] = np.full(12, np.nan)
                rows[yr][mn-1] = _clean(p[2])
    return rows

def amjjas_mean(monthly_dict, years):
    out = {}
    for y in years:
        if y not in monthly_dict: continue
        vals = monthly_dict[y][3:9]
        if np.isfinite(vals).sum() >= 3:
            out[y] = float(np.nanmean(vals))
    return out

yr_sie  = np.array(sorted(sie_by_year.keys()))
sie_arr = np.array([sie_by_year[y] for y in yr_sie])
n_mon_arr = np.array([n_months_used[y] for y in yr_sie])

yr_ceres  = np.array(sorted(rs_ceres_by_year.keys()))
rs_ceres  = np.array([rs_ceres_by_year[y] for y in yr_ceres])
yr_era5   = np.array(sorted(rs_era5_by_year.keys()))
rs_era5   = np.array([rs_era5_by_year[y] for y in yr_era5])

indices = {}
for name, path in [('AO', AO_PATH), ('PDO', PDO_PATH)]:
    if not os.path.exists(path):
        print(f'  WARNING: {name} file not found'); continue
    monthly = _load_index(path)
    seasonal = amjjas_mean(monthly, yr_sie.tolist())
    idx = np.array([seasonal.get(int(y), np.nan) for y in yr_sie])
    idx -= np.nanmean(idx[np.isfinite(idx)])
    indices[name] = idx
    print(f'  {name}: {np.isfinite(idx).sum()} AMJJAS years')


# ── LOW-PASS / HIGH-PASS ──────────────────────────────────────────────────────
edge = LP_WIN // 2
sl_e = slice(edge, len(yr_sie) - edge)

def lp_hp(arr):
    if not np.isfinite(arr).all():
        return np.full_like(arr, np.nan), np.full_like(arr, np.nan)
    lp = uniform_filter1d(arr, size=LP_WIN, mode='nearest')
    return lp, arr - lp

sie_lp, sie_hp = lp_hp(sie_arr)
idx_lp = {}; idx_hp = {}
for name, idx in indices.items():
    if np.isfinite(idx).all():
        idx_lp[name], idx_hp[name] = lp_hp(idx)

ceres_lp = ceres_hp = None
if len(rs_ceres) > LP_WIN:
    # Align CERES LP/HP to sie year axis for overlay
    ceres_lp, ceres_hp = lp_hp(rs_ceres)

era5_lp = era5_hp = None
if len(rs_era5) > LP_WIN:
    era5_lp, era5_hp = lp_hp(rs_era5)


# ── PRINT DIAGNOSTICS ─────────────────────────────────────────────────────────
print('\n── n_months used per year ──────────────────────────────────────────────')
for yr, n in zip(yr_sie, n_mon_arr):
    print(f'  {yr}: {n} months')

if len(yr_era5) > 0 and len(yr_ceres) > 0:
    overlap = np.intersect1d(yr_era5, yr_ceres)
    if len(overlap) >= 5:
        era5_ovlp  = np.array([rs_era5_by_year[y]  for y in overlap])
        ceres_ovlp = np.array([rs_ceres_by_year[y] for y in overlap])
        r_ov, p_ov = stats.pearsonr(era5_ovlp, ceres_ovlp)
        print(f'\n── ERA5 kernel RS vs CERES RS (2000-2024 overlap) ──────────────────────')
        print(f'  ERA5 mean={np.mean(era5_ovlp):.2f}  CERES mean={np.mean(ceres_ovlp):.2f} W/m²')
        print(f'  r={r_ov:+.3f}  p={p_ov:.4f}  (variability correlation)')


# ── STATS ─────────────────────────────────────────────────────────────────────
def ols(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return None
    return stats.linregress(x[mask], y[mask])

def pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

print('\n── SIE trend (1982-2024) ───────────────────────────────────────────────')
res = ols(yr_sie.astype(float), sie_arr)
if res:
    print(f'  slope = {res.slope*1e3:.1f} ×10³ km²/yr  '
          f'r={res.rvalue:.3f}  p={res.pvalue:.4f}')
    print(f'  total change = {res.slope*(yr_sie[-1]-yr_sie[0]):.2f} Mkm²')

print('\n── CERES RS trend (2000-2024) ──────────────────────────────────────────')
res_rc = ols(yr_ceres.astype(float), rs_ceres)
if res_rc:
    print(f'  slope = {res_rc.slope:.3f} W/m²/yr  '
          f'r={res_rc.rvalue:.3f}  p={res_rc.pvalue:.4f}')

if len(rs_era5) > 5:
    print('\n── ERA5 kernel RS trend (1982-2024) ────────────────────────────────────')
    res_re = ols(yr_era5.astype(float), rs_era5)
    if res_re:
        print(f'  slope = {res_re.slope:.4f} W/m²/yr  '
              f'r={res_re.rvalue:.3f}  p={res_re.pvalue:.4f}')

print('\n── SIE vs AO / PDO ────────────────────────────────────────────────────')
sie_dt = sie_arr - np.polyval(np.polyfit(yr_sie.astype(float), sie_arr, 1),
                               yr_sie.astype(float))
for name, idx in indices.items():
    mask = np.isfinite(idx)
    r, p = stats.pearsonr(idx[mask], sie_arr[mask])
    print(f'  {name} vs SIE (raw):          r={r:+.3f}  p={p:.4f}  {pstar(p)}')
    if name in idx_hp:
        m2 = np.isfinite(idx_hp[name][sl_e])
        r2, p2 = stats.pearsonr(idx_hp[name][sl_e][m2], sie_hp[sl_e][m2])
        print(f'  {name}_high vs SIE_high:     r={r2:+.3f}  p={p2:.4f}  {pstar(p2)}')
        m3 = np.isfinite(idx_lp[name][sl_e])
        r3, p3 = stats.pearsonr(idx_lp[name][sl_e][m3], sie_lp[sl_e][m3])
        print(f'  {name}_low  vs SIE_low:      r={r3:+.3f}  p={p3:.4f}  {pstar(p3)}')
    r_dt, p_dt = stats.pearsonr(idx[mask], sie_dt[mask])
    print(f'  {name} vs SIE (detrended):    r={r_dt:+.3f}  p={p_dt:.4f}  {pstar(p_dt)}')

def _rs_corr(yr_rs, rs_arr_local, label):
    print(f'\n── {label} vs AO / PDO ─────────────────────────────────────────────')
    rs_dt = rs_arr_local - np.polyval(
        np.polyfit(yr_rs.astype(float), rs_arr_local, 1), yr_rs.astype(float))
    for name, idx in indices.items():
        idx_al = np.array([indices[name][np.where(yr_sie==y)[0][0]]
                           for y in yr_rs if y in yr_sie])
        mask = np.isfinite(idx_al) & np.isfinite(rs_arr_local)
        if mask.sum() < 5: continue
        r, p = stats.pearsonr(idx_al[mask], rs_arr_local[mask])
        print(f'  {name} vs {label} (raw):      r={r:+.3f}  p={p:.4f}  {pstar(p)}')
        r_dt, p_dt = stats.pearsonr(idx_al[mask], rs_dt[mask])
        print(f'  {name} vs {label} (detrended):r={r_dt:+.3f}  p={p_dt:.4f}  {pstar(p_dt)}')

if len(rs_ceres) > 5:
    _rs_corr(yr_ceres, rs_ceres, 'CERES RS')
if len(rs_era5) > 5:
    _rs_corr(yr_era5, rs_era5, 'ERA5 kernel RS')

print('\n── SIE vs RS ───────────────────────────────────────────────────────────')
sie_ceres_yrs = np.intersect1d(yr_sie, yr_ceres)
sie_ovlp = np.array([sie_by_year[y] for y in sie_ceres_yrs])
rs_c_ovlp = np.array([rs_ceres_by_year[y] for y in sie_ceres_yrs])
r_sr, p_sr = stats.pearsonr(sie_ovlp, rs_c_ovlp)
print(f'  SIE vs CERES RS (2000-2024): r={r_sr:+.3f}  p={p_sr:.4f}  {pstar(p_sr)}')
if len(rs_era5) > 5:
    sie_era5_yrs = np.intersect1d(yr_sie, yr_era5)
    sie_e_ovlp = np.array([sie_by_year[y] for y in sie_era5_yrs])
    rs_e_ovlp  = np.array([rs_era5_by_year[y] for y in sie_era5_yrs])
    r_se, p_se = stats.pearsonr(sie_e_ovlp, rs_e_ovlp)
    print(f'  SIE vs ERA5 RS (1982-2024):  r={r_se:+.3f}  p={p_se:.4f}  {pstar(p_se)}')


# ── FIGURE ────────────────────────────────────────────────────────────────────
C_SIE = '#1d5fa8'; C_RC = '#b5000a'; C_RE = '#2ca02c'
C_AO = '#e07b00'; C_PDO = '#007b5e'

fig = plt.figure(figsize=(16, 13), facecolor='white')
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

# ── (a) SIE ──────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :2])
ax.fill_between(yr_sie, sie_arr, sie_lp, color=C_SIE, alpha=0.15)
ax.plot(yr_sie, sie_arr, color='#aaa', lw=0.9, alpha=0.7)
ax.plot(yr_sie, sie_lp,  color=C_SIE,  lw=2.2, label='Low-pass (forced)')
res_sie = ols(yr_sie.astype(float), sie_arr)
if res_sie:
    yt = res_sie.slope*yr_sie + (np.mean(sie_arr)-res_sie.slope*np.mean(yr_sie))
    ax.plot(yr_sie, yt, color=C_SIE, lw=1.2, ls='--', alpha=0.7,
            label=f'Trend: {res_sie.slope*1e3:.0f}×10³ km²/yr {pstar(res_sie.pvalue)}')
sparse = n_mon_arr < 6
if sparse.any():
    ax.scatter(yr_sie[sparse], sie_arr[sparse], marker='x', color='gray',
               s=50, zorder=5, label='< 6 months')
ax.set_ylabel('AMJJAS SIE  (Mkm²)')
ax.set_title('(a) Arctic SIE (AMJJAS, 1982–2024)', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(yr_sie[0]-1, yr_sie[-1]+1)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# ── (b) RS — CERES and ERA5 kernel on same axes ───────────────────────────────
ax = fig.add_subplot(gs[0, 2:])
if len(rs_era5) > 0:
    ax.plot(yr_era5, rs_era5, color=C_RE, lw=1.5, alpha=0.8,
            label='ERA5 kernel×α (1982–2024)')
    if era5_lp is not None:
        ax.plot(yr_era5, era5_lp, color=C_RE, lw=2.4, alpha=0.9)
    res_re = ols(yr_era5.astype(float), rs_era5)
    if res_re:
        yt = res_re.slope*yr_era5 + (np.mean(rs_era5)-res_re.slope*np.mean(yr_era5))
        ax.plot(yr_era5, yt, color=C_RE, lw=1.2, ls='--',
                label=f'ERA5 trend: {res_re.slope:.3f} W/m²/yr {pstar(res_re.pvalue)}')

ax2 = ax.twinx()
ax2.plot(yr_ceres, rs_ceres, 'D-', color=C_RC, lw=1.8, ms=5,
         markeredgecolor='white', label='CERES T07 (2000–2024)')
if ceres_lp is not None:
    ax2.plot(yr_ceres, rs_ceres + (ceres_lp - ceres_lp), color=C_RC, lw=2.4, alpha=0.7)
res_rc2 = ols(yr_ceres.astype(float), rs_ceres)
if res_rc2:
    yt = res_rc2.slope*yr_ceres + (np.mean(rs_ceres)-res_rc2.slope*np.mean(yr_ceres))
    ax2.plot(yr_ceres, yt, color=C_RC, lw=1.2, ls='--',
             label=f'CERES trend: {res_rc2.slope:.3f} W/m²/yr {pstar(res_rc2.pvalue)}')
ax.set_ylabel('ERA5 kernel RS  (W m⁻²)', color=C_RE)
ax2.set_ylabel('CERES T07 RS  (W m⁻²)', color=C_RC)
ax.tick_params(axis='y', colors=C_RE)
ax2.tick_params(axis='y', colors=C_RC)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper right')
ax.set_title('(b) Domain-mean RS (ERA5 kernel 1982–2024 | CERES 2000–2024)',
             fontweight='bold')
ax.set_xlim(1982, 2025)
for sp in ['top']: ax.spines[sp].set_visible(False)

# ── (c,d) Index time series ───────────────────────────────────────────────────
for col, (name, clr) in enumerate([('AO', C_AO), ('PDO', C_PDO)]):
    if name not in indices: continue
    ax = fig.add_subplot(gs[1, col*2:(col+1)*2])
    idx = indices[name]
    lp  = idx_lp.get(name, np.full_like(idx, np.nan))
    ax.fill_between(yr_sie, idx, lp, color=clr, alpha=0.20)
    ax.plot(yr_sie, idx, color='#aaa', lw=0.9, alpha=0.7)
    ax.plot(yr_sie, lp,  color=clr, lw=2.2, label=f'{LP_WIN}-yr low-pass')
    ax.axhline(0, color='#ccc', lw=0.7, ls=':')
    ax.set_title(f'({chr(99+col)}) {name} index (AMJJAS)', fontweight='bold')
    ax.set_ylabel(f'{name} anomaly')
    ax.legend(fontsize=8); ax.set_xlim(yr_sie[0]-1, yr_sie[-1]+1)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

# ── (e-h) Scatter: SIE vs each index, ERA5 RS vs each index ──────────────────
panel = 0
rs_sources = [('SIE', yr_sie, sie_arr, C_SIE, 'SIE (Mkm²)')]
if len(rs_era5) > 0:
    rs_sources.append(('ERA5 RS', yr_era5, rs_era5, C_RE, 'ERA5 RS (W m⁻²)'))
elif len(rs_ceres) > 0:
    rs_sources.append(('CERES RS', yr_ceres, rs_ceres, C_RC, 'CERES RS (W m⁻²)'))

for var_name, var_yr, var_arr, var_clr, var_lbl in rs_sources:
    for idx_name, idx_clr in [('AO', C_AO), ('PDO', C_PDO)]:
        if idx_name not in indices or panel >= 4:
            panel += 1; continue
        ax = fig.add_subplot(gs[2, panel])
        idx_aligned = np.array([indices[idx_name][np.where(yr_sie==y)[0][0]]
                                  for y in var_yr if y in yr_sie])
        var_aligned = np.array([var_arr[i]
                                  for i, y in enumerate(var_yr) if y in yr_sie])
        yr_aligned  = np.array([y for y in var_yr if y in yr_sie])
        mask = np.isfinite(idx_aligned) & np.isfinite(var_aligned)
        sc = ax.scatter(idx_aligned[mask], var_aligned[mask],
                        c=yr_aligned[mask], cmap='plasma',
                        s=38, edgecolors='white', lw=0.4, zorder=3)
        res_s = ols(idx_aligned, var_aligned)
        if res_s and abs(res_s.rvalue) > 0.1:
            xr_ = np.array([np.nanmin(idx_aligned), np.nanmax(idx_aligned)])
            ax.plot(xr_, res_s.slope*xr_+res_s.intercept, color=idx_clr, lw=2.0)
        if res_s:
            ax.text(0.05, 0.95,
                    f'r = {res_s.rvalue:+.3f}  {pstar(res_s.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(fc='white', ec=idx_clr, alpha=0.2, pad=3))
        ax.set_xlabel(f'{idx_name} (AMJJAS anomaly)')
        ax.set_ylabel(var_lbl)
        ax.set_title(f'({chr(101+panel)}) {var_name} vs {idx_name}',
                     fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Year', shrink=0.85)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        panel += 1

# ── (i-l) CERES RS scatter panels (bottom row) ───────────────────────────────
if len(rs_ceres) > 0:
    panel2 = 0
    for idx_name, idx_clr in [('AO', C_AO), ('PDO', C_PDO)]:
        if idx_name not in indices or panel2 >= 4:
            panel2 += 1; continue
        ax = fig.add_subplot(gs[3, panel2])
        idx_aligned = np.array([indices[idx_name][np.where(yr_sie==y)[0][0]]
                                  for y in yr_ceres if y in yr_sie])
        var_aligned = rs_ceres.copy()
        yr_aligned  = yr_ceres.copy()
        mask = np.isfinite(idx_aligned) & np.isfinite(var_aligned)
        sc = ax.scatter(idx_aligned[mask], var_aligned[mask],
                        c=yr_aligned[mask], cmap='plasma',
                        s=38, edgecolors='white', lw=0.4, zorder=3)
        res_s = ols(idx_aligned, var_aligned)
        if res_s and abs(res_s.rvalue) > 0.1:
            xr_ = np.array([np.nanmin(idx_aligned), np.nanmax(idx_aligned)])
            ax.plot(xr_, res_s.slope*xr_+res_s.intercept, color=idx_clr, lw=2.0)
        if res_s:
            ax.text(0.05, 0.95,
                    f'r = {res_s.rvalue:+.3f}  {pstar(res_s.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(fc='white', ec=idx_clr, alpha=0.2, pad=3))
        ax.set_xlabel(f'{idx_name} (AMJJAS anomaly)')
        ax.set_ylabel('CERES RS (W m⁻²)')
        ax.set_title(f'({chr(105+panel2)}) CERES RS vs {idx_name}',
                     fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Year', shrink=0.85)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        panel2 += 1

fig.suptitle(
    'Arctic SIE & RS vs AO/PDO  (AMJJAS)  |  '
    'SIE: NSIDC CDR PSN25 1982–2024  |  '
    'RS: ERA5 Huang kernel×α 1982–2024  +  CERES EBAF T07 2000–2024',
    fontsize=11, fontweight='bold')

out_path = os.path.join(OUT_DIR, 'sie_rs_teleconnection.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved -> {out_path}')
print('Done.')'''

#!/usr/bin/env python
"""
diagnose_rs_monthly_sic.py
──────────────────────────────────────────────────────────────────────────────
Investigates the CERES-RS-down / ERA5-kernel-RS-up discrepancy.

Produces three figure sets:
  Fig 1  – AMJJAS per-month time series: CERES vs ERA5-kernel, full domain
  Fig 2  – Per-month time series split by SIC regime
              (a) SIC >= 0.15  (sea-ice zone)
              (b) SIC <  0.15  (open-ocean / marginal-ice zone)
  Fig 3  – Scatter: CERES vs ERA5-kernel in overlap (2000-2024), per month,
            coloured by regime, to see if the sign-flip is ice vs ocean

All RS diagnostics use the same functions / paths / kernels as
compute_sie_rs_teleconnection.py so results are directly comparable.

Run on Derecho:
  module load python/3.10; python diagnose_rs_monthly_sic.py
"""

#!/usr/bin/env python
"""
diagnose_rs_monthly_sic_v2.py
──────────────────────────────────────────────────────────────────────────────
Updated version of diagnose_rs_monthly_sic.py with:
  • Figs 1 & 2: ERA5 and CERES on the SAME y-axis (shown as anomalies
    relative to 2000-2010 mean so the ~30 W/m² absolute offset doesn't
    dominate; the raw absolute means are printed in the trend table)
  • Fig 3: square axes, same x/y limits, dashed 1:1 line
  • Fig 5 (new): RS METHOD COMPARISON — tests whether the discrepancy
    between ERA5-kernel and CERES comes from using k_psn×(1-α) vs k_psn
    directly (the approach siaf_analysis.py uses and trusts).
    Three curves per month:
      (A) k_psn × (1-α)   [current teleconnection approach]
      (B) k_psn alone      [siaf_analysis.py approach — ≈ rsds]
      (C) CERES T07        [reference]

Rationale for the RS discrepancy
─────────────────────────────────
siaf_analysis.py uses k_psn directly as RS.  The ERA5 Huang kernel
  K = ∂(absorbed SW) / ∂(α × 100)  [W m⁻² per % albedo]
so
  |K| × 100 = |∂RS/∂α| ≈ rsds   (incident surface SW)

For the SIAF weight, siaf_analysis.py correctly uses rsds (incident SW),
because SIAF = -(dα/dT) × rsds = how much more SW would be absorbed if
albedo changed by dα/dT per K.

For a time-varying RS metric (trend analysis), the absorbed SW is
  RS_abs = rsds × (1-α) = k_psn × (1-α)
but this diverges from CERES because the kernel approach misses
cloud–albedo covariance: when sea-ice retreats, clouds tend to increase,
partially cancelling the albedo-driven increase in absorbed SW.
CERES captures this full effect; the kernel approach does not.

The Fig 5 comparison will quantify how much of the ERA5/CERES discrepancy
is method-driven vs physics-driven.
"""

#!/usr/bin/env python
"""
rs_cfc_analysis.py
──────────────────
Analyse Arctic shortwave RS (CERES T07) trends since 2000 and correlate
with ERA5 total cloud cover (TCC) back to 1979, stratified by SIC regime.

Cloud data: ERA5 monthly TCC (total_cloud_cover, 0-1 fraction)
            downloaded via CDS API → era5_tcc_arctic_amjjas.nc
"""

#!/usr/bin/env python
"""
rs_cfc_analysis.py
──────────────────
Analyse Arctic shortwave RS (CERES T07) trends since 2000 and correlate
with ERA5 total cloud cover (TCC) back to 1979, stratified by SIC regime.

Cloud data: ERA5 monthly TCC (total_cloud_cover, 0-1 fraction)
            downloaded via CDS API → era5_tcc_arctic_amjjas.nc
"""

#!/usr/bin/env python
"""
rs_cfc_analysis.py
──────────────────
Analyse Arctic shortwave RS (CERES T07) trends since 2000 and correlate
with ERA5 total cloud cover (TCC) back to 1979, stratified by SIC regime.

Cloud data: ERA5 monthly TCC (total_cloud_cover, 0-1 fraction)
            downloaded via CDS API → era5_tcc_arctic_amjjas.nc
"""

import gc, glob, os, warnings
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────
DIR_SIC       = '/glade/derecho/scratch/jrotondo/SIC_data/'
CERES_FILE    = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
                 'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
ERA5_TCC_FILE    = '/glade/work/jrotondo/era5_tcc_arctic_amjjas.nc'
MODIS_CWP_FILE   = '/glade/work/jrotondo/modis_cod_arctic_amjjas.nc'
CALIPSO_CF_FILE  = '/glade/work/jrotondo/calipso_cf_arctic_amjjas.nc'

# Set True to skip Figs 1-11 and A-E and only produce Fig F
ONLY_FINAL_FIG = True

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/rs_tcc_analysis/'
os.makedirs(OUT_DIR, exist_ok=True)

YEARS_CERES = list(range(2000, 2026))
YEARS_TCC   = list(range(1979, 2026))
YEARS_ALL   = list(range(1979, 2026))

REF_YRS = list(range(2000, 2011))

MONTH_NUMS        = [4, 5, 6, 7, 8, 9]
MONTH_NAMES_SHORT = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}

ARCTIC_LAT_MIN = 60.
SIC_ICE_THRESH = 0.15

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')


# ── PSN25 GRID ────────────────────────────────────────────────────────────────
print('Setting up PSN25 grid ...')
_sic_sample = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(_sic_sample) as _ds:
    _px = _ds['x'].values.astype(np.float64)
    _py = _ds['y'].values.astype(np.float64)
_pxx, _pyy = np.meshgrid(_px, _py)
PSN_SHAPE  = _pxx.shape
_tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON, PSN_LAT = _tfm.transform(_pxx, _pyy)
del _pxx, _pyy

_phi_ts = np.deg2rad(70.)
_k_ps   = (1. + np.sin(_phi_ts)) / np.maximum(
    1. + np.sin(np.deg2rad(PSN_LAT)), 1e-9)
AREA_W     = (1. / _k_ps**2).astype(np.float32)
DOMAIN_PSN = (PSN_LAT >= ARCTIC_LAT_MIN)
print(f'  PSN_SHAPE={PSN_SHAPE}  domain pixels: {DOMAIN_PSN.sum():,}')


# ── XYZ HELPER ────────────────────────────────────────────────────────────────
def _xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat).ravel().astype(np.float64))
    lo = np.deg2rad(np.asarray(lon).ravel().astype(np.float64))
    return np.column_stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo),
                             np.sin(la)])


# ── GRID UTILITIES ────────────────────────────────────────────────────────────
def scatter_avg(flat, idx, shape):
    n   = int(np.prod(shape))
    acc = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    ok  = np.isfinite(flat)
    np.add.at(acc, idx[ok], flat[ok].astype(np.float32))
    np.add.at(cnt, idx[ok], 1.)
    out = np.full(n, np.nan, np.float32)
    m   = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out.reshape(shape)


def domain_mean_masked(arr, mask):
    combined = DOMAIN_PSN & mask
    v = arr[combined]; w = AREA_W[combined]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok]*w[ok]) / np.sum(w[ok])) if ok.sum() else np.nan


def domain_mean(arr):
    return domain_mean_masked(arr, np.ones(PSN_SHAPE, bool))


# ── CERES → PSN25 INDEX ───────────────────────────────────────────────────────
print('Building CERES → PSN25 index ...')
with xr.open_dataset(CERES_FILE) as _ds:
    _cla = _ds['lat'].values; _clo = _ds['lon'].values
_clon2d, _clat2d = np.meshgrid(_clo, _cla)
_, IDX_CERES_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
    _xyz(_clat2d, _clon2d), workers=-1)
IDX_CERES_TO_PSN = IDX_CERES_TO_PSN.astype(np.int32)
del _clat2d, _clon2d


# ── ERA5 TCC → PSN25 ─────────────────────────────────────────────────────────
print('Loading ERA5 TCC and building regrid index ...')
_tcc_2d = {}
with xr.open_dataset(ERA5_TCC_FILE) as _tds:
    print(f'  Vars: {list(_tds.data_vars)}  Coords: {list(_tds.coords)}')
    _lat_n  = next(n for n in ['latitude', 'lat']          if n in _tds.coords)
    _lon_n  = next(n for n in ['longitude', 'lon']         if n in _tds.coords)
    _time_n = next(n for n in ['valid_time', 'time']       if n in _tds.coords)
    _var_n  = next(n for n in ['tcc', 'total_cloud_cover'] if n in _tds)

    _t_lat = _tds[_lat_n].values
    _t_lon = _tds[_lon_n].values
    _t_lon2d, _t_lat2d = np.meshgrid(_t_lon, _t_lat)

    _, IDX_ERA5_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
        _xyz(_t_lat2d, _t_lon2d), workers=-1)
    IDX_ERA5_TO_PSN = IDX_ERA5_TO_PSN.astype(np.int32)
    del _t_lat2d, _t_lon2d

    times = pd.DatetimeIndex(_tds[_time_n].values)
    for _i, _t in enumerate(times):
        if _t.month not in MONTH_NUMS or _t.year not in YEARS_TCC:
            continue
        _arr = _tds[_var_n].isel({_time_n: _i}).values.astype(np.float32)
        _arr = np.where(np.isfinite(_arr) & (_arr >= 0.) & (_arr <= 1.),
                        _arr, np.nan)
        _psn = scatter_avg(_arr.ravel(), IDX_ERA5_TO_PSN, PSN_SHAPE)
        _tcc_2d[(_t.year, _t.month)] = _psn
        if _t.year % 10 == 0 and _t.month == 6:
            print(f'  ERA5 TCC {_t.year}-{_t.month:02d}  '
                  f'domain_mean={domain_mean(_psn):.3f}')

print(f'  Loaded {len(_tcc_2d)} ERA5 TCC month-fields '
      f'({min(k[0] for k in _tcc_2d)}–{max(k[0] for k in _tcc_2d)})')


# ── MODIS MOD08_M3 CWP and Cloud Fraction → PSN25 ────────────────────────────
print('\nLoading MODIS MOD08_M3 cloud properties ...')
_modis_cf_2d  = {}   # Cloud_Retrieval_Fraction_Combined_FMean  (0-1)
_modis_cwp_2d = {}   # Cloud_Water_Path_Liquid_Mean_Mean         (g/m²)
YEARS_MODIS   = list(range(2000, 2026))

with xr.open_dataset(MODIS_CWP_FILE) as _mds:
    print(f'  Vars: {list(_mds.data_vars)}')
    _m_lat  = _mds['lat'].values
    _m_lon  = _mds['lon'].values
    _m_lon2d, _m_lat2d = np.meshgrid(_m_lon, _m_lat)

    _, IDX_MODIS_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
        _xyz(_m_lat2d, _m_lon2d), workers=-1)
    IDX_MODIS_TO_PSN = IDX_MODIS_TO_PSN.astype(np.int32)
    del _m_lat2d, _m_lon2d

    _CF_VAR  = 'Cloud_Retrieval_Fraction_Combined_FMean'
    _CWP_VAR = 'Cloud_Water_Path_Liquid_Mean_Mean'
    _m_times = pd.DatetimeIndex(_mds['time'].values)

    for _i, _t in enumerate(_m_times):
        if _t.month not in MONTH_NUMS or _t.year not in YEARS_MODIS:
            continue
        _cf_arr = _mds[_CF_VAR].isel(time=_i).values.astype(np.float32)
        _cf_arr = np.where(np.isfinite(_cf_arr) & (_cf_arr >= 0.) & (_cf_arr <= 1.),
                           _cf_arr, np.nan)
        _modis_cf_2d[(_t.year, _t.month)] = scatter_avg(
            _cf_arr.ravel(), IDX_MODIS_TO_PSN, PSN_SHAPE)

        _cwp_arr = _mds[_CWP_VAR].isel(time=_i).values.astype(np.float32)
        _cwp_arr = np.where(np.isfinite(_cwp_arr) & (_cwp_arr >= 0.), _cwp_arr, np.nan)
        _modis_cwp_2d[(_t.year, _t.month)] = scatter_avg(
            _cwp_arr.ravel(), IDX_MODIS_TO_PSN, PSN_SHAPE)

        if _t.year % 5 == 0 and _t.month == 7:
            print(f'  MODIS {_t.year}-{_t.month:02d}  '
                  f'CF={domain_mean(_modis_cf_2d[(_t.year,_t.month)]):.3f}  '
                  f'CWP_liq={domain_mean(_modis_cwp_2d[(_t.year,_t.month)]):.1f} g/m²')

print(f'  Loaded {len(_modis_cf_2d)} MODIS month-fields')


# ── CALIPSO L3 → PSN25 ───────────────────────────────────────────────────────
# Grid: (lat=85 S→N at 2°, lon=144 at 2.5°), 2006–2023 AMJJAS
# CF computed from sample counts: nanmax over altitude levels (column occurrence)
print('\nLoading CALIPSO L3 cloud fraction ...')
_calipso_cf_2d     = {}   # (yr, mo) → PSN25 total CF
_calipso_water_2d  = {}   # (yr, mo) → PSN25 liquid water CF
_calipso_opq_2d    = {}   # (yr, mo) → PSN25 opaque CF (thick clouds only)
YEARS_CALIPSO      = list(range(2006, 2024))

try:
    with xr.open_dataset(CALIPSO_CF_FILE) as _cds:
        print(f'  Vars: {list(_cds.data_vars)}')
        _c_lat   = _cds['lat'].values
        _c_lon   = _cds['lon'].values
        _c_lon2d, _c_lat2d = np.meshgrid(_c_lon, _c_lat)

        _, IDX_CALIPSO_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
            _xyz(_c_lat2d, _c_lon2d), workers=-1)
        IDX_CALIPSO_TO_PSN = IDX_CALIPSO_TO_PSN.astype(np.int32)
        del _c_lat2d, _c_lon2d

        _c_times = pd.DatetimeIndex(_cds['time'].values)
        for _i, _t in enumerate(_c_times):
            if _t.month not in MONTH_NUMS or _t.year not in YEARS_CALIPSO:
                continue
            def _clip01(arr):
                a = arr.astype(np.float32)
                return np.where(np.isfinite(a) & (a >= 0.) & (a <= 1.), a, np.nan)
            _cf_tot = _clip01(_cds['cf_total'].isel(time=_i).values)
            _cf_wat = _clip01(_cds['cf_water'].isel(time=_i).values)
            _cf_opq = _clip01(_cds['cf_opaque'].isel(time=_i).values)
            _calipso_cf_2d[(_t.year, _t.month)]  = scatter_avg(
                _cf_tot.ravel(), IDX_CALIPSO_TO_PSN, PSN_SHAPE)
            _calipso_water_2d[(_t.year, _t.month)] = scatter_avg(
                _cf_wat.ravel(), IDX_CALIPSO_TO_PSN, PSN_SHAPE)
            _calipso_opq_2d[(_t.year, _t.month)] = scatter_avg(
                _cf_opq.ravel(), IDX_CALIPSO_TO_PSN, PSN_SHAPE)

    print(f'  Loaded {len(_calipso_cf_2d)} CALIPSO month-fields '
          f'({min(k[0] for k in _calipso_cf_2d)}–{max(k[0] for k in _calipso_cf_2d)})')
except FileNotFoundError:
    print(f'  WARNING: {CALIPSO_CF_FILE} not found — CALIPSO skipped')
except Exception as _ce:
    print(f'  WARNING: CALIPSO load failed: {_ce}')


# ── SIC LOADING ───────────────────────────────────────────────────────────────
def _find_sic_var(ds):
    preferred = ['cdr_seaice_conc', 'seaice_conc', 'siconc',
                 'sea_ice_concentration', 'sic', 'ice_conc']
    for name in preferred:
        if name in ds.data_vars: return name
    for v in ds.data_vars:
        vl = v.lower()
        if ('conc' in vl or 'sic' in vl) and not any(
                x in vl for x in ('flag','source','stdev','spatial',
                                   'interp','bt_','nt_')):
            return v
    return None


_sic_cache = {}

def load_psn_sic(year, month):
    key = (year, month)
    if key in _sic_cache: return _sic_cache[key]
    files = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}{month:02d}*.nc'))
    if not files:
        _sic_cache[key] = None; return None
    try:
        with xr.open_dataset(files[0]) as ds:
            vname = _find_sic_var(ds)
            if vname is None: _sic_cache[key] = None; return None
            arr = ds[vname].values.squeeze().astype(np.float32)
        if float(np.nanmax(arr)) > 1.5: arr /= 100.
        arr = np.where((arr >= 0.) & (arr <= 1.), arr, np.nan)
        _sic_cache[key] = arr; return arr
    except Exception as e:
        print(f'  WARNING load_psn_sic({year},{month}): {e}')
        _sic_cache[key] = None; return None


def get_sic_masks(year, month):
    sic = load_psn_sic(year, month)
    if sic is None or sic.shape != PSN_SHAPE:
        for dy in [1, -1, 2, -2]:
            sic = load_psn_sic(year + dy, month)
            if sic is not None and sic.shape == PSN_SHAPE:
                print(f'    SIC fallback: {year+dy} used for {year}-{month:02d}')
                break
        else:
            return None, None
    return (np.isfinite(sic) & (sic >= SIC_ICE_THRESH),
            np.isfinite(sic) & (sic <  SIC_ICE_THRESH))


# ── CERES RS (T07 INVERSION) ──────────────────────────────────────────────────
_CERES_VARS = ['solar_mon', 'toa_sw_all_mon', 'sfc_sw_down_all_mon',
               'sfc_sw_up_all_mon', 'toa_sw_clr_c_mon',
               'sfc_sw_down_clr_c_mon', 'sfc_sw_up_clr_c_mon']

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt > 1., rsdt, np.nan)
        Sd = np.where(rsds > 1., rsds, np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds/S;  q  = rsut/S
        d  = q + p - 2.*p*a
        R  = np.clip((q-p*a) / np.where(np.abs(d)>1e-6, d, np.nan), 0., .99)
        A  = np.clip(1.-p*(1.-a*R) / np.where((1.-R)>.01, 1.-R, np.nan), 0., .99)
        Sc = np.where(rsdscs > 1., rsdscs, np.nan)
        ac = np.clip(rsuscs / Sc, 0., 1.)
        pc = rsdscs/S;  qc = rsutcs/S
        dc = qc + pc - 2.*pc*ac
        Rc = np.clip((qc-pc*ac) / np.where(np.abs(dc)>1e-6, dc, np.nan), 0., .99)
        Rl = np.clip(R - Rc, 0., .99)
        RS = (S*(1.-A)*(1.-Rl)**2
              / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS.astype(np.float32)


print('\nPre-loading CERES RS, surface SWCRE, and TOA SWCRE fields ...')
_CERES_DS    = xr.open_dataset(CERES_FILE)
_CERES_TIMES = pd.DatetimeIndex(_CERES_DS.time.values)
_ceres_rs_2d       = {}
_ceres_swcre_2d    = {}   # surface SW CRE = net_SW_all - net_SW_clr  (W/m²)
_ceres_toaswcre_2d = {}   # TOA SW CRE = rsut_clr - rsut_all  (W/m²)
_ceres_rs_clr_2d   = {}   # clear-sky absorbed SW at surface = rsdscs - rsuscs
                           # negative = clouds cool (reflect more SW to space)

for _t_idx, _t in enumerate(_CERES_TIMES):
    if _t.month not in MONTH_NUMS or _t.year not in YEARS_CERES: continue
    _sl = _CERES_DS.isel(time=_t_idx).load()
    _vs = {k: _sl[k].values.astype(np.float32) for k in _CERES_VARS}
    _rs = t07_invert(*[_vs[k] for k in _CERES_VARS])
    _ceres_rs_2d[(_t.year, _t.month)] = scatter_avg(
        _rs.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)

    # Surface SW CRE: (net SW all-sky) - (net SW clear-sky) at surface
    _rsds   = _vs['sfc_sw_down_all_mon'];  _rsus   = _vs['sfc_sw_up_all_mon']
    _rsdscs = _vs['sfc_sw_down_clr_c_mon']; _rsuscs = _vs['sfc_sw_up_clr_c_mon']
    with np.errstate(invalid='ignore'):
        _swcre = (_rsds - _rsus) - (_rsdscs - _rsuscs)
        _swcre = np.where(np.isfinite(_swcre), _swcre, np.nan)
    _ceres_swcre_2d[(_t.year, _t.month)] = scatter_avg(
        _swcre.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)

    # TOA SW CRE: rsut_clr - rsut_all  (negative = clouds reflect more SW → cool)
    with np.errstate(invalid='ignore'):
        _toaswcre = _vs['toa_sw_clr_c_mon'] - _vs['toa_sw_all_mon']
        _toaswcre = np.where(np.isfinite(_toaswcre), _toaswcre, np.nan)
    _ceres_toaswcre_2d[(_t.year, _t.month)] = scatter_avg(
        _toaswcre.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)

    # Clear-sky absorbed SW at surface = rsdscs - rsuscs
    # This is what SIAF predicts: as ice melts → darker surface → more clear-sky SW absorbed
    with np.errstate(invalid='ignore'):
        _rs_clr = np.where(np.isfinite(_rsdscs) & np.isfinite(_rsuscs),
                           _rsdscs - _rsuscs, np.nan)
    _ceres_rs_clr_2d[(_t.year, _t.month)] = scatter_avg(
        _rs_clr.ravel(), IDX_CERES_TO_PSN, PSN_SHAPE)

    if _t.month == 7:
        print(f'  CERES {_t.year}-{_t.month:02d}  '
              f'RS={domain_mean(_ceres_rs_2d[(_t.year, _t.month)]):.2f} W/m²  '
              f'sfcSWCRE={domain_mean(_ceres_swcre_2d[(_t.year, _t.month)]):+.2f}  '
              f'toaSWCRE={domain_mean(_ceres_toaswcre_2d[(_t.year, _t.month)]):+.2f} W/m²')
_CERES_DS.close()


# ── MAIN COMPUTATION ──────────────────────────────────────────────────────────
regimes      = ('all', 'ice', 'ocean')
rs_data      = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
tcc_data     = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
swcre_data   = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
toaswcre_data  = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
modis_cf_data  = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
modis_cwp_data = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}
ocean_frac_data = {mn: {} for mn in MONTH_NUMS}   # fraction of domain that is ocean/MIZ
rs_clr_data     = {r: {mn: {} for mn in MONTH_NUMS} for r in regimes}  # clear-sky surface absorbed SW

print('\nComputing per-year per-month RS and TCC by regime ...')
for year in YEARS_ALL:
    for month in MONTH_NUMS:
        ice_mask, ocean_mask = get_sic_masks(year, month)
        if ice_mask is None:
            masks = {'all': np.ones(PSN_SHAPE, bool)}
        else:
            masks = {'all':   np.ones(PSN_SHAPE, bool),
                     'ice':   ice_mask,
                     'ocean': ocean_mask}
            # Area-weighted ocean fraction of the domain (SIC < 0.15)
            w_ice   = float(np.sum(AREA_W[DOMAIN_PSN & ice_mask]))
            w_ocean = float(np.sum(AREA_W[DOMAIN_PSN & ocean_mask]))
            w_tot   = w_ice + w_ocean
            if w_tot > 0:
                ocean_frac_data[month][year] = w_ocean / w_tot

        key = (year, month)

        if key in _ceres_rs_2d:
            rs_2d = _ceres_rs_2d[key]
            for reg, msk in masks.items():
                rs_data[reg][month][year] = domain_mean_masked(rs_2d, msk)

        if key in _ceres_swcre_2d:
            swcre_2d = _ceres_swcre_2d[key]
            for reg, msk in masks.items():
                swcre_data[reg][month][year] = domain_mean_masked(swcre_2d, msk)

        if key in _ceres_toaswcre_2d:
            toaswcre_2d = _ceres_toaswcre_2d[key]
            for reg, msk in masks.items():
                toaswcre_data[reg][month][year] = domain_mean_masked(toaswcre_2d, msk)

        if key in _ceres_rs_clr_2d:
            rs_clr_2d = _ceres_rs_clr_2d[key]
            for reg, msk in masks.items():
                rs_clr_data[reg][month][year] = domain_mean_masked(rs_clr_2d, msk)

        if key in _tcc_2d:
            tcc_2d = _tcc_2d[key]
            for reg, msk in masks.items():
                tcc_data[reg][month][year] = domain_mean_masked(tcc_2d, msk)

        if key in _modis_cf_2d:
            for reg, msk in masks.items():
                modis_cf_data[reg][month][year]  = domain_mean_masked(_modis_cf_2d[key],  msk)
                modis_cwp_data[reg][month][year] = domain_mean_masked(_modis_cwp_2d[key], msk)

    if year % 5 == 0:
        print(f'  {year}: done')

gc.collect()


# ── SIAF FOOTPRINT MASKS (extent = retreating-ice zone, darkening = stable-ice) ──
# Loads per-month SIC trend arrays saved by siaf_analysis.py and projects
# them from the SIAL grid to PSN25 so they can mask CERES/ERA5/MODIS data.
CACHE_OBS_SIAF = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
DIR_SIAL_CLARA = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
SIAF_PVAL      = 0.05
_MONTH_FULL    = {4:'April', 5:'May', 6:'June',
                  7:'July',  8:'August', 9:'September'}

print('\nBuilding SIAF extent/darkening masks from CACHE_OBS ...')
_SIAL_OK = False
IDX_SIAL_TO_PSN = None
try:
    _sial_sample = sorted(glob.glob(f'{DIR_SIAL_CLARA}/*.nc'))[0]
    with xr.open_dataset(_sial_sample) as _ds:
        _sla = _ds['lat'].values; _slo = _ds['lon'].values
    if _sla.ndim == 1:
        _SIAL_LON, _SIAL_LAT = np.meshgrid(_slo, _sla)
    else:
        _SIAL_LAT, _SIAL_LON = _sla, _slo
    _, IDX_SIAL_TO_PSN = cKDTree(_xyz(PSN_LAT, PSN_LON)).query(
        _xyz(_SIAL_LAT, _SIAL_LON), workers=-1)
    IDX_SIAL_TO_PSN = IDX_SIAL_TO_PSN.astype(np.int32)
    print(f'  SIAL grid shape: {_SIAL_LAT.shape}')
    _SIAL_OK = True
except Exception as _e:
    print(f'  WARNING: SIAL grid load failed: {_e}')

EXT_MASK_PSN = {}   # month → bool PSN_SHAPE — retreating-ice pixels
DRK_MASK_PSN = {}   # month → bool PSN_SHAPE — stable-ice pixels

if _SIAL_OK:
    for _mn in MONTH_NUMS:
        _mname = _MONTH_FULL[_mn]
        try:
            _slope = np.load(f'{CACHE_OBS_SIAF}/sic_trend_slope_sial_{_mname}.npy').astype(np.float32)
            _pval  = np.load(f'{CACHE_OBS_SIAF}/sic_trend_pval_sial_{_mname}.npy').astype(np.float32)
            _sic_e = np.load(f'{CACHE_OBS_SIAF}/sic_early_sial_{_mname}.npy').astype(np.float32)
            _sic_l = np.load(f'{CACHE_OBS_SIAF}/sic_late_sial_{_mname}.npy').astype(np.float32)

            _ext_sial = (np.isfinite(_slope) & np.isfinite(_pval) & np.isfinite(_sic_e) &
                         (_slope < 0) & (_pval < SIAF_PVAL) & (_sic_e >= SIC_ICE_THRESH))
            _sic_max  = np.fmax(np.where(np.isfinite(_sic_e), _sic_e, 0.),
                                np.where(np.isfinite(_sic_l), _sic_l, 0.))
            _drk_sial = (np.isfinite(_slope) & np.isfinite(_pval) &
                         ~_ext_sial & (_sic_max >= SIC_ICE_THRESH))

            _ext_frac = scatter_avg(_ext_sial.astype(np.float32).ravel(),
                                    IDX_SIAL_TO_PSN, PSN_SHAPE)
            _drk_frac = scatter_avg(_drk_sial.astype(np.float32).ravel(),
                                    IDX_SIAL_TO_PSN, PSN_SHAPE)
            EXT_MASK_PSN[_mn] = np.isfinite(_ext_frac) & (_ext_frac > 0.3)
            DRK_MASK_PSN[_mn] = np.isfinite(_drk_frac) & (_drk_frac > 0.3)
            print(f'  {_mname}: ext={EXT_MASK_PSN[_mn].sum():,}  '
                  f'drk={DRK_MASK_PSN[_mn].sum():,} PSN25 pixels')
        except Exception as _e:
            print(f'  WARNING: {_mname} mask load: {_e}')
            EXT_MASK_PSN[_mn] = np.zeros(PSN_SHAPE, bool)
            DRK_MASK_PSN[_mn] = np.zeros(PSN_SHAPE, bool)
else:
    for _mn in MONTH_NUMS:
        EXT_MASK_PSN[_mn] = np.zeros(PSN_SHAPE, bool)
        DRK_MASK_PSN[_mn] = np.zeros(PSN_SHAPE, bool)


# ── SIAF FOOTPRINT TIME SERIES ────────────────────────────────────────────────
rs_ext_data       = {mn: {} for mn in MONTH_NUMS}
rs_drk_data       = {mn: {} for mn in MONTH_NUMS}
tcc_ext_data      = {mn: {} for mn in MONTH_NUMS}
tcc_drk_data      = {mn: {} for mn in MONTH_NUMS}
cf_ext_data       = {mn: {} for mn in MONTH_NUMS}
cf_drk_data       = {mn: {} for mn in MONTH_NUMS}
cal_cf_ext_data   = {mn: {} for mn in MONTH_NUMS}   # CALIPSO total CF
cal_cf_drk_data   = {mn: {} for mn in MONTH_NUMS}
cal_wat_ext_data  = {mn: {} for mn in MONTH_NUMS}   # CALIPSO liquid water CF
cal_wat_drk_data  = {mn: {} for mn in MONTH_NUMS}
cal_opq_ext_data  = {mn: {} for mn in MONTH_NUMS}   # CALIPSO opaque (thick) CF
cal_opq_drk_data  = {mn: {} for mn in MONTH_NUMS}

print('\nComputing RS / TCC / CF over SIAF footprints ...')
for _year in YEARS_ALL:
    for _month in MONTH_NUMS:
        _key    = (_year, _month)
        _ext_m  = EXT_MASK_PSN.get(_month)
        _drk_m  = DRK_MASK_PSN.get(_month)
        if _ext_m is None or _ext_m.sum() == 0: continue

        if _key in _ceres_rs_2d:
            rs_ext_data[_month][_year] = domain_mean_masked(_ceres_rs_2d[_key], _ext_m)
            rs_drk_data[_month][_year] = domain_mean_masked(_ceres_rs_2d[_key], _drk_m)
        if _key in _tcc_2d:
            tcc_ext_data[_month][_year] = domain_mean_masked(_tcc_2d[_key], _ext_m)
            tcc_drk_data[_month][_year] = domain_mean_masked(_tcc_2d[_key], _drk_m)
        if _key in _modis_cf_2d:
            cf_ext_data[_month][_year] = domain_mean_masked(_modis_cf_2d[_key], _ext_m)
            cf_drk_data[_month][_year] = domain_mean_masked(_modis_cf_2d[_key], _drk_m)
        if _key in _calipso_cf_2d:
            cal_cf_ext_data[_month][_year]  = domain_mean_masked(_calipso_cf_2d[_key],    _ext_m)
            cal_cf_drk_data[_month][_year]  = domain_mean_masked(_calipso_cf_2d[_key],    _drk_m)
            cal_wat_ext_data[_month][_year] = domain_mean_masked(_calipso_water_2d[_key], _ext_m)
            cal_wat_drk_data[_month][_year] = domain_mean_masked(_calipso_water_2d[_key], _drk_m)
            cal_opq_ext_data[_month][_year] = domain_mean_masked(_calipso_opq_2d[_key],   _ext_m)
            cal_opq_drk_data[_month][_year] = domain_mean_masked(_calipso_opq_2d[_key],   _drk_m)

gc.collect()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _savefig(fig, path, label, dpi=150):
    """Save and close fig unless ONLY_FINAL_FIG is set."""
    if ONLY_FINAL_FIG:
        plt.close(fig)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {label} → {path}')

def to_ts(d_mn, years):
    return np.array([d_mn.get(y, np.nan) for y in years])

def ols_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return None
    return stats.linregress(x[mask], y[mask])

def pstar(p):
    if p is None: return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def sig_alpha(p):
    """Line/text alpha keyed to significance: n.s. → nearly invisible."""
    if p is None: return 0.10
    try: p = float(p)
    except Exception: return 0.10
    if np.isnan(p) or p >= 0.05:  return 0.10
    if p >= 0.01:  return 0.55
    if p >= 0.001: return 0.82
    return 1.00

def sig_lw(p):
    """Line width keyed to significance."""
    if p is None: return 0.7
    try: p = float(p)
    except Exception: return 0.7
    if np.isnan(p) or p >= 0.05: return 0.7
    if p >= 0.01:  return 1.6
    if p >= 0.001: return 2.2
    return 2.8

def ref_anom(ts, yr_vec, ref_yrs=REF_YRS):
    ref_mask = np.array([y in ref_yrs for y in yr_vec])
    ref = np.nanmean(ts[ref_mask]) if ref_mask.any() else np.nan
    return ts - ref, float(ref)

def trend_label(yr, ts, unit=''):
    res = ols_safe(yr.astype(float), ts)
    if res is None: return 'n/a'
    u = f' {unit}/yr' if unit else '/yr'
    return f'{res.slope:+.4f}{u} {pstar(res.pvalue)}'

yr_ceres   = np.array(YEARS_CERES)
yr_tcc     = np.array(YEARS_TCC)
yr_all     = np.array(YEARS_ALL)
yr_overlap = np.array([y for y in YEARS_CERES if y in YEARS_TCC])


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print('\n\n══════════════════════════════════════════════════════════════════════')
print('CERES RS TRENDS 2000–2025  (W/m²/yr)')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = [trend_label(yr_ceres, to_ts(rs_data[reg][mn], yr_ceres), 'W/m²')
           for reg in regimes]
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')

print('\nERA5 TCC TRENDS 1979–2025  (fraction/yr × 10⁻³)')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = []
    for reg in regimes:
        ts  = to_ts(tcc_data[reg][mn], yr_tcc)
        res = ols_safe(yr_tcc.astype(float), ts)
        row.append(f'{res.slope*1e3:+.3f}×10⁻³ {pstar(res.pvalue)}' if res else 'n/a')
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')

print('\nRS–TCC PEARSON r  (overlap 2000–2025, anomalies)')
print(f'{"Month":<5}  {"All domain":>16}  {"Ice zone":>16}  {"Ocean/MIZ":>16}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = []
    for reg in regimes:
        rs_a,  _ = ref_anom(to_ts(rs_data[reg][mn],  yr_overlap), yr_overlap)
        tcc_a, _ = ref_anom(to_ts(tcc_data[reg][mn], yr_overlap), yr_overlap)
        m = np.isfinite(rs_a) & np.isfinite(tcc_a)
        if m.sum() >= 5:
            r, p = stats.pearsonr(rs_a[m], tcc_a[m])
            row.append(f'{r:+.3f}{pstar(p)}')
        else:
            row.append('n/a')
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>16}  {row[1]:>16}  {row[2]:>16}')
print('══════════════════════════════════════════════════════════════════════\n')


# ── COLOURS ───────────────────────────────────────────────────────────────────
C_RS    = '#b5000a'
C_TCC   = '#2077b4'
C_ICE   = '#1d5fa8'
C_OCEAN = '#d67a00'
C_ALL   = '#444444'
C_SWCRE = '#5a2d82'
C_TOA   = '#2c7b45'
C_CF    = '#1d7a3a'
C_CWP   = '#8b2be2'

REG_COLORS = {'all': C_ALL, 'ice': C_ICE, 'ocean': C_OCEAN}
REG_LABELS = {'all': 'Full domain',
              'ice': 'SIC ≥ 0.15 (Ice)',
              'ocean': 'SIC < 0.15 (Ocean/MIZ)'}


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — CERES RS time series (2000–2025) by month and SIC regime
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 3, figsize=(17, 10), facecolor='white')
fig1.suptitle(
    'CERES T07 surface RS (2000–2025) — anomaly relative to 2000–2010\n'
    'Black = full domain  |  Blue = ice zone (SIC ≥ 0.15)  '
    '|  Orange = ocean/MIZ (SIC < 0.15)',
    fontsize=12, fontweight='bold')

for ax, mn in zip(axes1.ravel(), MONTH_NUMS):
    for reg in regimes:
        ts_raw = to_ts(rs_data[reg][mn], yr_ceres)
        ts_a, ref = ref_anom(ts_raw, yr_ceres)
        mask = np.isfinite(ts_a)
        clr = REG_COLORS[reg]
        lw  = 2.0 if reg == 'all' else 1.5
        ls  = '-' if reg == 'all' else ('--' if reg == 'ice' else ':')
        ax.plot(yr_ceres[mask], ts_a[mask],
                color=clr, lw=lw, ls=ls, alpha=0.9,
                label=f'{REG_LABELS[reg]} (ref={ref:.1f})')
        res = ols_safe(yr_ceres.astype(float), ts_a)
        if res:
            yt = res.slope*yr_ceres + (np.nanmean(ts_a) - res.slope*np.mean(yr_ceres))
            ax.plot(yr_ceres[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            row_i = ['all','ice','ocean'].index(reg)
            ax.text(0.02, 0.99 - row_i*0.12,
                    f'{REG_LABELS[reg]}: {res.slope:+.3f} W/m²/yr {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
    ax.axhline(0, color='#bbb', lw=0.8, ls=':')
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.set_ylabel('RS anomaly (W m⁻²)', fontsize=9)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_xlim(1999, 2026)
    ax.legend(fontsize=6.5, loc='lower left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'fig1_ceres_rs_by_regime.png')
_savefig(fig1, out1, 'Fig 1')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ERA5 TCC time series (1979–2025) by month and SIC regime
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(17, 10), facecolor='white')
fig2.suptitle(
    'ERA5 Total Cloud Cover (TCC) 1979–2025 — anomaly relative to 2000–2010\n'
    'Black = full domain  |  Blue = ice zone  |  Orange = ocean/MIZ',
    fontsize=12, fontweight='bold')

for ax, mn in zip(axes2.ravel(), MONTH_NUMS):
    for reg in regimes:
        ts_raw = to_ts(tcc_data[reg][mn], yr_tcc)
        ts_a, ref = ref_anom(ts_raw, yr_tcc)
        mask = np.isfinite(ts_a)
        clr = REG_COLORS[reg]
        lw  = 2.0 if reg == 'all' else 1.5
        ls  = '-' if reg == 'all' else ('--' if reg == 'ice' else ':')
        ax.plot(yr_tcc[mask], ts_a[mask],
                color=clr, lw=lw, ls=ls, alpha=0.9,
                label=f'{REG_LABELS[reg]} (ref={ref:.3f})')
        res = ols_safe(yr_tcc.astype(float), ts_a)
        if res:
            yt = res.slope*yr_tcc + (np.nanmean(ts_a) - res.slope*np.mean(yr_tcc))
            ax.plot(yr_tcc[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            row_i = ['all','ice','ocean'].index(reg)
            ax.text(0.02, 0.99 - row_i*0.12,
                    f'{REG_LABELS[reg]}: {res.slope*1e3:+.3f}×10⁻³/yr {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
    ax.axhline(0, color='#bbb', lw=0.8, ls=':')
    ax.axvspan(2000, 2026, color='#f0f0f0', alpha=0.5, zorder=0,
               label='CERES overlap')
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.set_ylabel('TCC anomaly (fraction)', fontsize=9)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_xlim(1978, 2026)
    ax.legend(fontsize=6.5, loc='lower left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'fig2_tcc_by_regime.png')
_savefig(fig2, out2, 'Fig 2')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — RS and TCC dual y-axis (2000–2025 overlap), 6 months × 3 regimes
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(6, 3, figsize=(18, 28), facecolor='white')
fig3.suptitle(
    'CERES RS (red, left axis) vs ERA5 TCC (blue, right axis)\n'
    'Anomalies relative to 2000–2010  |  Overlap period 2000–2025',
    fontsize=13, fontweight='bold')

for row, mn in enumerate(MONTH_NUMS):
    for col, reg in enumerate(regimes):
        ax_rs  = axes3[row, col]
        ax_tcc = ax_rs.twinx()

        rs_raw  = to_ts(rs_data[reg][mn],  yr_overlap)
        tcc_raw = to_ts(tcc_data[reg][mn], yr_overlap)
        rs_a,  rs_ref  = ref_anom(rs_raw,  yr_overlap)
        tcc_a, tcc_ref = ref_anom(tcc_raw, yr_overlap)

        m_rs = np.isfinite(rs_a)
        ax_rs.plot(yr_overlap[m_rs], rs_a[m_rs],
                   color=C_RS, lw=2.0, alpha=0.9, label='RS anom')
        ax_rs.fill_between(yr_overlap[m_rs], 0, rs_a[m_rs],
                           color=C_RS, alpha=0.12)
        res_rs = ols_safe(yr_overlap.astype(float), rs_a)
        if res_rs:
            yt = (res_rs.slope*yr_overlap
                  + np.nanmean(rs_a) - res_rs.slope*np.mean(yr_overlap))
            ax_rs.plot(yr_overlap[m_rs], yt[m_rs],
                       color=C_RS, lw=sig_lw(res_rs.pvalue), ls='--',
                       alpha=sig_alpha(res_rs.pvalue))

        m_tcc = np.isfinite(tcc_a)
        ax_tcc.plot(yr_overlap[m_tcc], tcc_a[m_tcc],
                    color=C_TCC, lw=2.0, alpha=0.9, label='TCC anom')
        ax_tcc.fill_between(yr_overlap[m_tcc], 0, tcc_a[m_tcc],
                             color=C_TCC, alpha=0.10)
        res_tcc = ols_safe(yr_overlap.astype(float), tcc_a)
        if res_tcc:
            yt = (res_tcc.slope*yr_overlap
                  + np.nanmean(tcc_a) - res_tcc.slope*np.mean(yr_overlap))
            ax_tcc.plot(yr_overlap[m_tcc], yt[m_tcc],
                        color=C_TCC, lw=sig_lw(res_tcc.pvalue), ls='--',
                        alpha=sig_alpha(res_tcc.pvalue))

        m_both = m_rs & m_tcc
        if m_both.sum() >= 5:
            r, p = stats.pearsonr(rs_a[m_both], tcc_a[m_both])
            ax_rs.text(0.03, 0.97, f'r = {r:+.3f}{pstar(p)}',
                       transform=ax_rs.transAxes, va='top', fontsize=9,
                       fontweight='bold', color='black',
                       bbox=dict(fc='white', ec='gray', alpha=0.7, pad=2))

        if res_rs:
            ax_rs.text(0.03, 0.84,
                       f'RS: {res_rs.slope:+.3f} W/m²/yr {pstar(res_rs.pvalue)}',
                       transform=ax_rs.transAxes, va='top', fontsize=7.5,
                       color=C_RS, fontweight='bold',
                       alpha=max(sig_alpha(res_rs.pvalue), 0.35))
        if res_tcc:
            ax_rs.text(0.03, 0.74,
                       f'TCC: {res_tcc.slope*1e3:+.3f}×10⁻³/yr {pstar(res_tcc.pvalue)}',
                       transform=ax_rs.transAxes, va='top', fontsize=7.5,
                       color=C_TCC, fontweight='bold',
                       alpha=max(sig_alpha(res_tcc.pvalue), 0.35))

        ax_rs.axhline(0, color='#bbb', lw=0.7, ls=':')
        ax_tcc.axhline(0, color='#bbb', lw=0.7, ls=':')
        if row == 0:
            ax_rs.set_title(REG_LABELS[reg], fontsize=10, fontweight='bold')
        ax_rs.set_ylabel(f'{MONTH_NAMES_SHORT[mn]} RS anom (W m⁻²)',
                         color=C_RS, fontsize=8)
        ax_tcc.set_ylabel('TCC anom (fraction)', color=C_TCC, fontsize=8)
        ax_rs.tick_params(axis='y', labelcolor=C_RS, labelsize=7)
        ax_tcc.tick_params(axis='y', labelcolor=C_TCC, labelsize=7)
        ax_rs.set_xlim(1999, 2026)
        ax_rs.set_xlabel('Year', fontsize=8)
        for sp in ['top']: ax_rs.spines[sp].set_visible(False)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'fig3_rs_vs_tcc_dual_axis.png')
_savefig(fig3, out3, 'Fig 3')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Scatter: RS anomaly vs TCC anomaly per month, coloured by year
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 3, figsize=(17, 11), facecolor='white')
fig4.suptitle(
    'RS anomaly vs ERA5 TCC anomaly (2000–2025 overlap, ref 2000–2010)\n'
    '■ = full domain   ● = ice zone   ▲ = ocean/MIZ   Colour = year',
    fontsize=12, fontweight='bold')

markers = {'all': 's', 'ice': 'o', 'ocean': '^'}
sizes   = {'all': 40,  'ice': 35,  'ocean': 35}

for ax, mn in zip(axes4.ravel(), MONTH_NUMS):
    any_plotted = False
    for reg in regimes:
        rs_raw  = to_ts(rs_data[reg][mn],  yr_overlap)
        tcc_raw = to_ts(tcc_data[reg][mn], yr_overlap)
        rs_a,  _ = ref_anom(rs_raw,  yr_overlap)
        tcc_a, _ = ref_anom(tcc_raw, yr_overlap)
        mask = np.isfinite(rs_a) & np.isfinite(tcc_a)
        if mask.sum() < 5: continue

        sc = ax.scatter(tcc_a[mask], rs_a[mask],
                        c=yr_overlap[mask], cmap='plasma',
                        s=sizes[reg], marker=markers[reg],
                        edgecolors=REG_COLORS[reg], lw=1.1,
                        alpha=0.75, label=REG_LABELS[reg], zorder=3)
        res = ols_safe(tcc_a, rs_a)
        if res:
            xi = np.linspace(np.nanmin(tcc_a[mask]), np.nanmax(tcc_a[mask]), 50)
            ax.plot(xi, res.slope*xi + res.intercept,
                    color=REG_COLORS[reg], lw=sig_lw(res.pvalue), ls='--',
                    alpha=sig_alpha(res.pvalue))
            row_i = list(regimes).index(reg)
            ax.text(0.97, 0.99 - row_i*0.13,
                    f'{REG_LABELS[reg]}: r={res.rvalue:+.3f}{pstar(res.pvalue)}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color=REG_COLORS[reg], fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
        any_plotted = True

    if any_plotted:
        plt.colorbar(sc, ax=ax, label='Year', shrink=0.85)
    ax.axhline(0, color='#ccc', lw=0.6, ls=':')
    ax.axvline(0, color='#ccc', lw=0.6, ls=':')
    ax.set_xlabel('TCC anomaly (fraction)', fontsize=8)
    ax.set_ylabel('RS anomaly (W m⁻²)', fontsize=8)
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.legend(fontsize=7, loc='lower right')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, 'fig4_rs_vs_tcc_scatter.png')
_savefig(fig4, out4, 'Fig 4')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — ERA5 TCC long record + CERES RS overlay, full domain
# ═══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(2, 3, figsize=(17, 10), facecolor='white')
fig5.suptitle(
    'ERA5 TCC long record (1979–2025, blue) with CERES RS overlay (2000–2025, red)\n'
    'Anomalies relative to 2000–2010  |  Full domain',
    fontsize=11, fontweight='bold')

for ax, mn in zip(axes5.ravel(), MONTH_NUMS):
    tcc_ts  = to_ts(tcc_data['all'][mn], yr_tcc)
    tcc_a, tcc_ref = ref_anom(tcc_ts, yr_tcc)
    m_tcc = np.isfinite(tcc_a)

    ax2 = ax.twinx()
    ax2.plot(yr_tcc[m_tcc], tcc_a[m_tcc],
             color=C_TCC, lw=1.6, alpha=0.75, label='TCC anom')
    ax2.fill_between(yr_tcc[m_tcc], 0, tcc_a[m_tcc],
                     color=C_TCC, alpha=0.10)
    res_tcc = ols_safe(yr_tcc.astype(float), tcc_a)
    if res_tcc:
        yt = (res_tcc.slope*yr_tcc
              + np.nanmean(tcc_a) - res_tcc.slope*np.mean(yr_tcc))
        ax2.plot(yr_tcc[m_tcc], yt[m_tcc],
                 color=C_TCC, lw=sig_lw(res_tcc.pvalue), ls='--',
                 alpha=sig_alpha(res_tcc.pvalue))
        ax2.text(0.02, 0.99,
                 f'TCC 1979–: {res_tcc.slope*1e3:+.3f}×10⁻³/yr {pstar(res_tcc.pvalue)}',
                 transform=ax2.transAxes, va='top', fontsize=8,
                 color=C_TCC, fontweight='bold',
                 alpha=max(sig_alpha(res_tcc.pvalue), 0.35))

    rs_ts = to_ts(rs_data['all'][mn], yr_ceres)
    rs_a, rs_ref = ref_anom(rs_ts, yr_ceres)
    m_rs = np.isfinite(rs_a)
    ax.plot(yr_ceres[m_rs], rs_a[m_rs],
            color=C_RS, lw=2.0, alpha=0.9, label='RS anom')
    ax.fill_between(yr_ceres[m_rs], 0, rs_a[m_rs],
                    color=C_RS, alpha=0.15)
    res_rs = ols_safe(yr_ceres.astype(float), rs_a)
    if res_rs:
        yt = (res_rs.slope*yr_ceres
              + np.nanmean(rs_a) - res_rs.slope*np.mean(yr_ceres))
        ax.plot(yr_ceres[m_rs], yt[m_rs],
                color=C_RS, lw=sig_lw(res_rs.pvalue), ls='--',
                alpha=sig_alpha(res_rs.pvalue))
        ax.text(0.02, 0.87,
                f'RS 2000–: {res_rs.slope:+.3f} W/m²/yr {pstar(res_rs.pvalue)}',
                transform=ax.transAxes, va='top', fontsize=8,
                color=C_RS, fontweight='bold',
                alpha=max(sig_alpha(res_rs.pvalue), 0.35))

    rs_ov  = to_ts(rs_data['all'][mn],  yr_overlap)
    tcc_ov = to_ts(tcc_data['all'][mn], yr_overlap)
    rs_ov_a,  _ = ref_anom(rs_ov,  yr_overlap)
    tcc_ov_a, _ = ref_anom(tcc_ov, yr_overlap)
    m_both = np.isfinite(rs_ov_a) & np.isfinite(tcc_ov_a)
    if m_both.sum() >= 5:
        r, p = stats.pearsonr(rs_ov_a[m_both], tcc_ov_a[m_both])
        ax.text(0.02, 0.75,
                f'r(RS,TCC) overlap = {r:+.3f}{pstar(p)}',
                transform=ax.transAxes, va='top', fontsize=8,
                color='black', fontweight='bold',
                bbox=dict(fc='white', ec='gray', alpha=0.6, pad=2))

    ax.axhline(0, color='#bbb', lw=0.7, ls=':')
    ax2.axhline(0, color='#bbb', lw=0.7, ls=':')
    ax.axvspan(2000, 2026, color='#fee8e8', alpha=0.25, zorder=0)

    ax.set_ylabel('RS anomaly (W m⁻²)', color=C_RS, fontsize=9)
    ax2.set_ylabel('TCC anomaly (fraction)', color=C_TCC, fontsize=9)
    ax.tick_params(axis='y', labelcolor=C_RS)
    ax2.tick_params(axis='y', labelcolor=C_TCC)
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_xlim(1978, 2026)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='lower left')
    for sp in ['top']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out5 = os.path.join(OUT_DIR, 'fig5_tcc_long_rs_overlay.png')
_savefig(fig5, out5, 'Fig 5')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Surface SW Cloud Radiative Effect (SWCRE) by month and SIC regime
#          SWCRE = net_SW_all - net_SW_clr  (W/m²)
#          Negative = clouds reduce absorbed SW (expected sign in Arctic)
#          A less-negative SWCRE over time = clouds becoming less effective
# ═══════════════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(2, 3, figsize=(17, 10), facecolor='white')
fig6.suptitle(
    'CERES surface SW Cloud Radiative Effect (2000–2025)\n'
    'SWCRE = net SW all-sky − net SW clear-sky  |  Anomaly relative to 2000–2010\n'
    'Negative = clouds cool surface  |  Trend toward zero = clouds losing cooling power',
    fontsize=11, fontweight='bold')

for ax, mn in zip(axes6.ravel(), MONTH_NUMS):
    for reg in regimes:
        ts_raw = to_ts(swcre_data[reg][mn], yr_ceres)
        ts_a, ref = ref_anom(ts_raw, yr_ceres)
        mask = np.isfinite(ts_a)
        clr = REG_COLORS[reg]
        lw  = 2.0 if reg == 'all' else 1.5
        ls  = '-' if reg == 'all' else ('--' if reg == 'ice' else ':')
        ax.plot(yr_ceres[mask], ts_a[mask],
                color=clr, lw=lw, ls=ls, alpha=0.9,
                label=f'{REG_LABELS[reg]} (ref={ref:.1f})')
        res = ols_safe(yr_ceres.astype(float), ts_a)
        if res:
            yt = res.slope*yr_ceres + (np.nanmean(ts_a) - res.slope*np.mean(yr_ceres))
            ax.plot(yr_ceres[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            row_i = ['all','ice','ocean'].index(reg)
            ax.text(0.02, 0.99 - row_i*0.12,
                    f'{REG_LABELS[reg]}: {res.slope:+.3f} W/m²/yr {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
    ax.axhline(0, color='#bbb', lw=0.8, ls=':')
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.set_ylabel('SWCRE anomaly (W m⁻²)', fontsize=9)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_xlim(1999, 2026)
    ax.legend(fontsize=6.5, loc='lower left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out6 = os.path.join(OUT_DIR, 'fig6_swcre_by_regime.png')
_savefig(fig6, out6, 'Fig 6')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — RS vs SWCRE dual-axis per regime (same layout as Fig 3)
#          Asks: does more cloud cooling (SWCRE) suppress RS, and is that
#          relationship strengthening or weakening over time?
# ═══════════════════════════════════════════════════════════════════════════════
fig7, axes7 = plt.subplots(6, 3, figsize=(18, 28), facecolor='white')
fig7.suptitle(
    'CERES RS (red, left) vs CERES surface SWCRE (purple, right)\n'
    'Anomalies relative to 2000–2010  |  2000–2025',
    fontsize=13, fontweight='bold')

for row, mn in enumerate(MONTH_NUMS):
    for col, reg in enumerate(regimes):
        ax_rs   = axes7[row, col]
        ax_swcre = ax_rs.twinx()

        rs_raw    = to_ts(rs_data[reg][mn],    yr_ceres)
        swcre_raw = to_ts(swcre_data[reg][mn], yr_ceres)
        rs_a,    rs_ref    = ref_anom(rs_raw,    yr_ceres)
        swcre_a, swcre_ref = ref_anom(swcre_raw, yr_ceres)

        m_rs = np.isfinite(rs_a)
        ax_rs.plot(yr_ceres[m_rs], rs_a[m_rs],
                   color=C_RS, lw=2.0, alpha=0.9, label='RS anom')
        ax_rs.fill_between(yr_ceres[m_rs], 0, rs_a[m_rs],
                           color=C_RS, alpha=0.12)
        res_rs = ols_safe(yr_ceres.astype(float), rs_a)
        if res_rs:
            yt = (res_rs.slope*yr_ceres
                  + np.nanmean(rs_a) - res_rs.slope*np.mean(yr_ceres))
            ax_rs.plot(yr_ceres[m_rs], yt[m_rs], color=C_RS,
                       lw=sig_lw(res_rs.pvalue), ls='--',
                       alpha=sig_alpha(res_rs.pvalue))

        m_sw = np.isfinite(swcre_a)
        ax_swcre.plot(yr_ceres[m_sw], swcre_a[m_sw],
                      color=C_SWCRE, lw=2.0, alpha=0.9, label='SWCRE anom')
        ax_swcre.fill_between(yr_ceres[m_sw], 0, swcre_a[m_sw],
                               color=C_SWCRE, alpha=0.10)
        res_sw = ols_safe(yr_ceres.astype(float), swcre_a)
        if res_sw:
            yt = (res_sw.slope*yr_ceres
                  + np.nanmean(swcre_a) - res_sw.slope*np.mean(yr_ceres))
            ax_swcre.plot(yr_ceres[m_sw], yt[m_sw], color=C_SWCRE,
                          lw=sig_lw(res_sw.pvalue), ls='--',
                          alpha=sig_alpha(res_sw.pvalue))

        m_both = m_rs & m_sw
        if m_both.sum() >= 5:
            r, p = stats.pearsonr(rs_a[m_both], swcre_a[m_both])
            ax_rs.text(0.03, 0.97, f'r(RS,SWCRE) = {r:+.3f}{pstar(p)}',
                       transform=ax_rs.transAxes, va='top', fontsize=9,
                       fontweight='bold', color='black',
                       bbox=dict(fc='white', ec='gray', alpha=0.7, pad=2))

        if res_rs:
            ax_rs.text(0.03, 0.84,
                       f'RS: {res_rs.slope:+.3f} W/m²/yr {pstar(res_rs.pvalue)}',
                       transform=ax_rs.transAxes, va='top', fontsize=7.5,
                       color=C_RS, fontweight='bold',
                       alpha=max(sig_alpha(res_rs.pvalue), 0.35))
        if res_sw:
            ax_rs.text(0.03, 0.74,
                       f'SWCRE: {res_sw.slope:+.3f} W/m²/yr {pstar(res_sw.pvalue)}',
                       transform=ax_rs.transAxes, va='top', fontsize=7.5,
                       color=C_SWCRE, fontweight='bold',
                       alpha=max(sig_alpha(res_sw.pvalue), 0.35))

        ax_rs.axhline(0, color='#bbb', lw=0.7, ls=':')
        ax_swcre.axhline(0, color='#bbb', lw=0.7, ls=':')
        if row == 0:
            ax_rs.set_title(REG_LABELS[reg], fontsize=10, fontweight='bold')
        ax_rs.set_ylabel(f'{MONTH_NAMES_SHORT[mn]} RS anom (W m⁻²)',
                         color=C_RS, fontsize=8)
        ax_swcre.set_ylabel('SWCRE anom (W m⁻²)', color=C_SWCRE, fontsize=8)
        ax_rs.tick_params(axis='y', labelcolor=C_RS, labelsize=7)
        ax_swcre.tick_params(axis='y', labelcolor=C_SWCRE, labelsize=7)
        ax_rs.set_xlim(1999, 2026)
        ax_rs.set_xlabel('Year', fontsize=8)
        for sp in ['top']: ax_rs.spines[sp].set_visible(False)

plt.tight_layout()
out7 = os.path.join(OUT_DIR, 'fig7_rs_vs_swcre_dual_axis.png')
_savefig(fig7, out7, 'Fig 7')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — TOA SW CRE by month and SIC regime
#          TOA SWCRE = rsut_clr - rsut_all
#          Negative = clouds are reflecting more SW to space (cooling)
#          Trend toward zero = clouds losing TOA cooling power
# ═══════════════════════════════════════════════════════════════════════════════
fig8, axes8 = plt.subplots(2, 3, figsize=(17, 10), facecolor='white')
fig8.suptitle(
    'CERES TOA SW Cloud Radiative Effect (2000–2025)\n'
    'TOA SWCRE = reflected SW clear-sky − reflected SW all-sky\n'
    'Negative = clouds cool TOA  |  Trend toward zero = clouds losing TOA cooling power',
    fontsize=11, fontweight='bold')

for ax, mn in zip(axes8.ravel(), MONTH_NUMS):
    for reg in regimes:
        ts_raw = to_ts(toaswcre_data[reg][mn], yr_ceres)
        ts_a, ref = ref_anom(ts_raw, yr_ceres)
        mask = np.isfinite(ts_a)
        clr = REG_COLORS[reg]
        lw  = 2.0 if reg == 'all' else 1.5
        ls  = '-' if reg == 'all' else ('--' if reg == 'ice' else ':')
        ax.plot(yr_ceres[mask], ts_a[mask],
                color=clr, lw=lw, ls=ls, alpha=0.9,
                label=f'{REG_LABELS[reg]} (ref={ref:.1f})')
        res = ols_safe(yr_ceres.astype(float), ts_a)
        if res:
            yt = res.slope*yr_ceres + (np.nanmean(ts_a) - res.slope*np.mean(yr_ceres))
            ax.plot(yr_ceres[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            row_i = ['all','ice','ocean'].index(reg)
            ax.text(0.02, 0.99 - row_i*0.12,
                    f'{REG_LABELS[reg]}: {res.slope:+.3f} W/m²/yr {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
    ax.axhline(0, color='#bbb', lw=0.8, ls=':')
    ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=12)
    ax.set_ylabel('TOA SWCRE anomaly (W m⁻²)', fontsize=9)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_xlim(1999, 2026)
    ax.legend(fontsize=6.5, loc='lower left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out8 = os.path.join(OUT_DIR, 'fig8_toa_swcre_by_regime.png')
_savefig(fig8, out8, 'Fig 8')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Attribution: RS, surface SWCRE, TOA SWCRE on three stacked panels
#          per month (columns) × regime (rows), anomalies 2000–2025.
#
#          Reading key:
#          • If TOA SWCRE and sfc SWCRE trend together → cloud effect is real
#          • If RS trends opposite to SWCRE → cloud-driven
#          • If RS trends same direction as SWCRE → non-cloud driver (ice albedo?)
#          • TOA−sfc SWCRE gap = atmospheric SW absorption by clouds
# ═══════════════════════════════════════════════════════════════════════════════
fig9, axes9 = plt.subplots(3, 6, figsize=(26, 12), facecolor='white',
                            sharex='col')
fig9.suptitle(
    'Attribution: RS (top), surface SWCRE (mid), TOA SWCRE (bottom) — anomalies 2000–2025\n'
    'If cloud CREs trend together with RS: cloud-driven  |  '
    'If RS trends alone: ice-albedo-driven\n'
    'Columns = months (Apr–Sep)  |  Regime = full domain only',
    fontsize=11, fontweight='bold')

_panel_vars = [
    ('rs',        rs_data,        yr_ceres, C_RS,    'RS anom (W m⁻²)',         '-'),
    ('swcre',     swcre_data,     yr_ceres, C_SWCRE, 'Sfc SWCRE anom (W m⁻²)',  '-'),
    ('toaswcre',  toaswcre_data,  yr_ceres, C_TOA,   'TOA SWCRE anom (W m⁻²)',  '-'),
]

for panel_row, (vname, vdict, yr_vec, clr, ylabel, ls) in enumerate(_panel_vars):
    for col, mn in enumerate(MONTH_NUMS):
        ax = axes9[panel_row, col]
        ts_raw = to_ts(vdict['all'][mn], yr_vec)
        ts_a, ref = ref_anom(ts_raw, yr_vec)
        mask = np.isfinite(ts_a)
        ax.plot(yr_vec[mask], ts_a[mask], color=clr, lw=1.8, alpha=0.85)
        ax.fill_between(yr_vec[mask], 0, ts_a[mask], color=clr, alpha=0.12)
        res = ols_safe(yr_vec.astype(float), ts_a)
        if res:
            yt = res.slope*yr_vec + (np.nanmean(ts_a) - res.slope*np.mean(yr_vec))
            ax.plot(yr_vec[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            ax.text(0.03, 0.97,
                    f'{res.slope:+.3f} W/m²/yr {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
        ax.axhline(0, color='#bbb', lw=0.7, ls=':')
        ax.set_xlim(1999, 2026)
        if panel_row == 0:
            ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=clr)
        ax.tick_params(labelsize=7)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        if panel_row == 2:
            ax.set_xlabel('Year', fontsize=8)

plt.tight_layout()
out9 = os.path.join(OUT_DIR, 'fig9_attribution_rs_swcre_toa.png')
_savefig(fig9, out9, 'Fig 9')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — "More clouds or thicker clouds?"
#           MODIS cloud retrieval fraction (CF) vs MODIS liquid CWP by regime
#           Both trend together → both mechanisms active
#           CF trends alone    → more clouds driving SWCRE increase
#           CWP trends alone   → optical thickening driving SWCRE increase
# ═══════════════════════════════════════════════════════════════════════════════
yr_modis = np.array(YEARS_MODIS)

fig10, axes10 = plt.subplots(4, 6, figsize=(26, 16), facecolor='white',
                              sharex='col')
fig10.suptitle(
    'MODIS MOD08_M3: Cloud Fraction vs Liquid Cloud Water Path (2000–2025)\n'
    'Rows: CF all-domain / CF ice-zone / CWP all-domain / CWP ice-zone\n'
    'Columns: Apr–Sep  |  Transparency = significance (opaque = p<0.001)',
    fontsize=11, fontweight='bold')

_panel_cfg = [
    (modis_cf_data,  'all',   C_CF,  'CF (all dom.)'),
    (modis_cf_data,  'ice',   C_CF,  'CF (ice zone)'),
    (modis_cwp_data, 'all',   C_CWP, 'CWP_liq g/m² (all)'),
    (modis_cwp_data, 'ice',   C_CWP, 'CWP_liq g/m² (ice)'),
]

for panel_row, (vdict, reg, clr, ylabel) in enumerate(_panel_cfg):
    for col, mn in enumerate(MONTH_NUMS):
        ax = axes10[panel_row, col]
        ts_raw = to_ts(vdict[reg][mn], yr_modis)
        ts_a, ref = ref_anom(ts_raw, yr_modis)
        mask = np.isfinite(ts_a)
        ax.plot(yr_modis[mask], ts_a[mask], color=clr, lw=1.6, alpha=0.85)
        ax.fill_between(yr_modis[mask], 0, ts_a[mask], color=clr, alpha=0.12)
        res = ols_safe(yr_modis.astype(float), ts_a)
        if res:
            yt = res.slope*yr_modis + (np.nanmean(ts_a) - res.slope*np.mean(yr_modis))
            ax.plot(yr_modis[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue), ls='--', alpha=sig_alpha(res.pvalue))
            unit = '/yr' if panel_row < 2 else ' g/m²/yr'
            ax.text(0.03, 0.97,
                    f'{res.slope:+.4f}{unit} {pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.35))
        ax.axhline(0, color='#bbb', lw=0.6, ls=':')
        ax.set_xlim(1999, 2026)
        if panel_row == 0:
            ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=clr)
        ax.tick_params(labelsize=7)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        if panel_row == 3:
            ax.set_xlabel('Year', fontsize=8)

plt.tight_layout()
out10 = os.path.join(OUT_DIR, 'fig10_modis_cf_cwp_trends.png')
_savefig(fig10, out10, 'Fig 10')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — SWCRE vs CF and SWCRE vs CWP scatter, per month
#           If SWCRE is better predicted by CF than CWP → cloud fraction drives it
#           If SWCRE is better predicted by CWP → cloud thickening drives it
#           2000-2025, full domain
# ═══════════════════════════════════════════════════════════════════════════════
fig11, axes11 = plt.subplots(2, 6, figsize=(26, 9), facecolor='white')
fig11.suptitle(
    'Surface SWCRE vs MODIS cloud fraction (top) and liquid CWP (bottom)\n'
    '2000–2025, full domain anomalies relative to 2000–2010\n'
    'r value shows which predictor better explains SWCRE variance',
    fontsize=11, fontweight='bold')

for col, mn in enumerate(MONTH_NUMS):
    swcre_ts = to_ts(swcre_data['all'][mn], yr_modis)
    swcre_a, _ = ref_anom(swcre_ts, yr_modis)

    for row, (vdict, clr, xlabel) in enumerate([
            (modis_cf_data,  C_CF,  'MODIS CF anomaly'),
            (modis_cwp_data, C_CWP, 'MODIS CWP_liq anomaly (g/m²)'),
    ]):
        ax = axes11[row, col]
        x_ts = to_ts(vdict['all'][mn], yr_modis)
        x_a, _ = ref_anom(x_ts, yr_modis)
        mask = np.isfinite(swcre_a) & np.isfinite(x_a)
        if mask.sum() >= 5:
            sc = ax.scatter(x_a[mask], swcre_a[mask],
                            c=yr_modis[mask], cmap='plasma',
                            s=40, edgecolors=clr, lw=0.8, alpha=0.75, zorder=3)
            res = ols_safe(x_a, swcre_a)
            if res:
                xi = np.linspace(np.nanmin(x_a[mask]), np.nanmax(x_a[mask]), 50)
                ax.plot(xi, res.slope*xi + res.intercept,
                        color=clr, lw=sig_lw(res.pvalue), ls='--',
                        alpha=sig_alpha(res.pvalue))
                r, p = stats.pearsonr(x_a[mask], swcre_a[mask])
                ax.text(0.97, 0.97,
                        f'r = {r:+.3f}{pstar(p)}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=9, color=clr, fontweight='bold',
                        alpha=max(sig_alpha(p), 0.35))
            plt.colorbar(sc, ax=ax, label='Year', shrink=0.8)
        ax.axhline(0, color='#ccc', lw=0.5, ls=':')
        ax.axvline(0, color='#ccc', lw=0.5, ls=':')
        ax.set_xlabel(xlabel, fontsize=8)
        if col == 0:
            ax.set_ylabel('Sfc SWCRE anomaly (W m⁻²)', fontsize=8)
        if row == 0:
            ax.set_title(MONTH_NAMES_SHORT[mn], fontweight='bold', fontsize=11)
        ax.tick_params(labelsize=7)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
out11 = os.path.join(OUT_DIR, 'fig11_swcre_vs_cf_cwp_scatter.png')
_savefig(fig11, out11, 'Fig 11')


# ─── PRINT MODIS TREND TABLE ──────────────────────────────────────────────────
print('\n\n══════════════════════════════════════════════════════════════════════')
print('MODIS CLOUD FRACTION (retrieval) TRENDS 2000–2025')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = [trend_label(yr_modis, to_ts(modis_cf_data[reg][mn], yr_modis))
           for reg in regimes]
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')

print('\nMODIS LIQUID CWP TRENDS 2000–2025  (g/m²/yr)')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = [trend_label(yr_modis, to_ts(modis_cwp_data[reg][mn], yr_modis), 'g/m²')
           for reg in regimes]
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')
print('══════════════════════════════════════════════════════════════════════\n')


# ─── PRINT SWCRE / TOA SWCRE TREND TABLE ──────────────────────────────────────
print('\n\n══════════════════════════════════════════════════════════════════════')
print('SURFACE SWCRE TRENDS 2000–2025  (W/m²/yr)')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = [trend_label(yr_ceres, to_ts(swcre_data[reg][mn], yr_ceres), 'W/m²')
           for reg in regimes]
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')

print('\nTOA SWCRE TRENDS 2000–2025  (W/m²/yr)')
print(f'{"Month":<5}  {"All domain":>22}  {"Ice zone":>22}  {"Ocean/MIZ":>22}')
print('──────────────────────────────────────────────────────────────────────')
for mn in MONTH_NUMS:
    row = [trend_label(yr_ceres, to_ts(toaswcre_data[reg][mn], yr_ceres), 'W/m²')
           for reg in regimes]
    print(f'{MONTH_NAMES_SHORT[mn]:<5}  {row[0]:>22}  {row[1]:>22}  {row[2]:>22}')
print('══════════════════════════════════════════════════════════════════════\n')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG A — Co-evolution: RS / Surface SW CRE / MODIS Cloud Fraction
#          Ice zone only, 2000–2025, 3 rows × 6 columns (months Apr–Sep)
#          Row 1: CERES T07 surface RS anomaly            (W m⁻², red)
#          Row 2: CERES EBAF surface SW CRE anomaly       (W m⁻², purple)
#          Row 3: MODIS MOD08_M3 cloud fraction anomaly   (unitless, green)
# ═══════════════════════════════════════════════════════════════════════════════
_A_cfg = [
    (rs_data,       'ice', yr_ceres, C_RS,    'RS anomaly\n(W m⁻²)',         None),
    (swcre_data,    'ice', yr_ceres, C_SWCRE, 'Sfc SW CRE anomaly\n(W m⁻²)', None),
    (modis_cf_data, 'ice', yr_modis, C_CF,    'Cloud fraction\nanomaly',      1e3),
]
_A_src = ['CERES EBAF', 'CERES EBAF', 'MODIS MOD08_M3']

fig_A, axes_A = plt.subplots(3, 6, figsize=(20, 9), facecolor='white',
                              sharex='col', sharey='row')
fig_A.suptitle(
    'Arctic ice-zone shortwave decline: co-evolution of surface RS, '
    'cloud radiative effect, and cloud fraction (2000–2025, SIC ≥ 0.15)',
    fontsize=12, fontweight='bold', y=1.02)

_ltrs = [chr(ord('a') + i) for i in range(18)]
_pidx = 0

for r, (vdict, reg, yr_vec, clr, ylabel, scale) in enumerate(_A_cfg):
    for c, mn in enumerate(MONTH_NUMS):
        ax = axes_A[r, c]

        ts_raw = to_ts(vdict[reg][mn], yr_vec)
        ts_a, _ref = ref_anom(ts_raw, yr_vec)
        mask = np.isfinite(ts_a)

        ax.fill_between(yr_vec[mask], 0, ts_a[mask],
                        color=clr, alpha=0.16, linewidth=0)
        ax.plot(yr_vec[mask], ts_a[mask],
                color=clr, lw=1.3, alpha=0.75, zorder=3)
        ax.axhline(0, color='#ccc', lw=0.65, ls=':')

        res = ols_safe(yr_vec.astype(float), ts_a)
        if res:
            yt = (res.slope * yr_vec
                  + np.nanmean(ts_a) - res.slope * np.mean(yr_vec))
            ax.plot(yr_vec[mask], yt[mask], color=clr,
                    lw=sig_lw(res.pvalue) * 1.2, ls='--',
                    alpha=sig_alpha(res.pvalue), zorder=4)
            slope_txt = (f'{res.slope * scale:+.2f}×10⁻³ yr⁻¹'
                         if scale is not None
                         else f'{res.slope:+.3f} W m⁻² yr⁻¹')
            ax.text(0.04, 0.97,
                    slope_txt + f'\n{pstar(res.pvalue)}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=clr, fontweight='bold',
                    alpha=max(sig_alpha(res.pvalue), 0.40),
                    linespacing=1.3)

        ax.text(0.97, 0.04, f'({_ltrs[_pidx]})',
                transform=ax.transAxes, va='bottom', ha='right',
                fontsize=8, color='#777', fontstyle='italic')
        _pidx += 1

        if r == 0:
            ax.set_title(MONTH_NAMES_SHORT[mn], fontsize=12,
                         fontweight='bold', pad=6)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=8.5, color=clr, labelpad=3)
        if r == 2:
            ax.set_xlabel('Year', fontsize=8)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

        ax.tick_params(labelsize=7)
        ax.set_xlim(1999, 2026)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

for r, lbl in enumerate(_A_src):
    axes_A[r, -1].annotate(
        lbl, xy=(1.03, 0.5), xycoords='axes fraction',
        ha='left', va='center', fontsize=7.5, color='#888',
        rotation=-90)

plt.tight_layout(rect=[0, 0, 0.985, 1])
outA = os.path.join(OUT_DIR, 'figA_coevolution_ice_zone.png')
_savefig(fig_A, outA, 'Fig A', dpi=180)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG B — Trend attribution bar chart (2000–2025, ice zone)
#          Left panel:  RS, sfc SWCRE, TOA SWCRE trends (W m⁻² yr⁻¹)
#          Right panel: MODIS CF trend (×10⁻³ yr⁻¹) — ice vs ocean/MIZ
#          Bars = ±1 SE; significance stars above/below
# ═══════════════════════════════════════════════════════════════════════════════
def _get_trend(vdict, reg, yr_vec, mn):
    ts = to_ts(vdict[reg][mn], yr_vec)
    res = ols_safe(yr_vec.astype(float), ts)
    if res:
        return res.slope, res.stderr, res.pvalue
    return np.nan, np.nan, np.nan

def _sig_bar_labels(ax, bars, pvals, se_arr, offset):
    for bar, p, se in zip(bars, pvals, se_arr):
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if np.isnan(pf) or pf >= 0.05:
            continue
        lbl = pstar(pf)
        h   = bar.get_height()
        sv  = float(se) if np.isfinite(float(se)) else 0.
        xc  = bar.get_x() + bar.get_width() / 2.
        if h >= 0:
            ax.text(xc, h + sv + offset, lbl,
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color='#333')
        else:
            ax.text(xc, h - sv - offset, lbl,
                    ha='center', va='top', fontsize=7.5,
                    fontweight='bold', color='#333')

_mn_labels = [MONTH_NAMES_SHORT[mn] for mn in MONTH_NUMS]
_xb = np.arange(len(MONTH_NUMS))
_wb = 0.22

_rs_sl,  _rs_se,  _rs_p   = zip(*[_get_trend(rs_data,        'ice',   yr_ceres, mn) for mn in MONTH_NUMS])
_sw_sl,  _sw_se,  _sw_p   = zip(*[_get_trend(swcre_data,     'ice',   yr_ceres, mn) for mn in MONTH_NUMS])
_toa_sl, _toa_se, _toa_p  = zip(*[_get_trend(toaswcre_data,  'ice',   yr_ceres, mn) for mn in MONTH_NUMS])
_cfi_sl, _cfi_se, _cfi_p  = zip(*[_get_trend(modis_cf_data,  'ice',   yr_modis, mn) for mn in MONTH_NUMS])
_cfo_sl, _cfo_se, _cfo_p  = zip(*[_get_trend(modis_cf_data,  'ocean', yr_modis, mn) for mn in MONTH_NUMS])

_rs_sl  = np.asarray(_rs_sl,  float); _rs_se  = np.asarray(_rs_se,  float)
_sw_sl  = np.asarray(_sw_sl,  float); _sw_se  = np.asarray(_sw_se,  float)
_toa_sl = np.asarray(_toa_sl, float); _toa_se = np.asarray(_toa_se, float)
_cfi_sl = np.asarray(_cfi_sl, float); _cfi_se = np.asarray(_cfi_se, float)
_cfo_sl = np.asarray(_cfo_sl, float); _cfo_se = np.asarray(_cfo_se, float)

fig_B, (ax_B1, ax_B2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
fig_B.suptitle(
    'Arctic linear trends 2000–2025  (bars = ±1 SE)'
    '         * p<0.05   ** p<0.01   *** p<0.001',
    fontsize=12, fontweight='bold')

# --- Left panel: energy budget ---
ax_B1.axhline(0, color='#bbb', lw=0.9, ls=':')
_ekw = dict(elinewidth=1.2, capsize=3, ecolor='#444')
brs  = ax_B1.bar(_xb - _wb, _rs_sl,  _wb, yerr=_rs_se,
                 color=C_RS,    alpha=0.88,
                 label='Surface RS (CERES T07)', error_kw=_ekw)
bsw  = ax_B1.bar(_xb,        _sw_sl,  _wb, yerr=_sw_se,
                 color=C_SWCRE, alpha=0.88,
                 label='Sfc SW CRE (CERES EBAF)', error_kw=_ekw)
btoa = ax_B1.bar(_xb + _wb,  _toa_sl, _wb, yerr=_toa_se,
                 color=C_TOA,   alpha=0.88,
                 label='TOA SW CRE (CERES EBAF)', error_kw=_ekw)

_sig_bar_labels(ax_B1, brs,  _rs_p,  _rs_se,  0.03)
_sig_bar_labels(ax_B1, bsw,  _sw_p,  _sw_se,  0.03)
_sig_bar_labels(ax_B1, btoa, _toa_p, _toa_se, 0.03)

ax_B1.set_xticks(_xb)
ax_B1.set_xticklabels(_mn_labels, fontsize=10)
ax_B1.set_ylabel('Trend (W m⁻² yr⁻¹)', fontsize=10)
ax_B1.set_title('Ice-zone energy budget trends (SIC ≥ 0.15)',
                fontsize=11, fontweight='bold')
ax_B1.legend(fontsize=8.5, loc='lower left')
for sp in ['top', 'right']:
    ax_B1.spines[sp].set_visible(False)

# --- Right panel: MODIS CF by regime ---
ax_B2.axhline(0, color='#bbb', lw=0.9, ls=':')
bci = ax_B2.bar(_xb - _wb / 2, _cfi_sl * 1e3, _wb, yerr=_cfi_se * 1e3,
                color=C_ICE,   alpha=0.88,
                label='Ice zone (SIC ≥ 0.15)', error_kw=_ekw)
bco = ax_B2.bar(_xb + _wb / 2, _cfo_sl * 1e3, _wb, yerr=_cfo_se * 1e3,
                color=C_OCEAN, alpha=0.88,
                label='Ocean/MIZ (SIC < 0.15)', error_kw=_ekw)

_sig_bar_labels(ax_B2, bci, _cfi_p, _cfi_se * 1e3, 0.02)
_sig_bar_labels(ax_B2, bco, _cfo_p, _cfo_se * 1e3, 0.02)

ax_B2.set_xticks(_xb)
ax_B2.set_xticklabels(_mn_labels, fontsize=10)
ax_B2.set_ylabel('Cloud fraction trend (×10⁻³ yr⁻¹)', fontsize=10)
ax_B2.set_title('MODIS cloud fraction trends — SIC regime',
                fontsize=11, fontweight='bold')
ax_B2.legend(fontsize=8.5)
for sp in ['top', 'right']:
    ax_B2.spines[sp].set_visible(False)

plt.tight_layout()
outB = os.path.join(OUT_DIR, 'figB_trend_attribution.png')
_savefig(fig_B, outB, 'Fig B', dpi=180)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG C — Cloud masking of SIAF: three lines of evidence
#
#  (a) Jun/Jul ice zone: clear-sky RS (SIAF predicted, should rise as ice melts)
#       vs all-sky RS (observed, declining).  The divergence IS the cloud masking.
#       Clouds block SW before it reaches the darkening surface → RS falls even
#       though surface albedo is falling (which in clear sky would raise RS).
#
#  (b) ERA5 TCC trends by month, ice zone vs ocean/MIZ (1979–2025).
#       Ice zone cloudiness rising systematically; cloud increase concentrated
#       over remaining sea ice, not over the newly exposed open ocean.
#
#  (c) Ocean/MIZ area fraction 1979–2025 (May–Aug).
#       The zone with the highest clear-sky RS is expanding every decade, but
#       clouds suppress its RS too → growing open water ≠ growing domain-mean RS.
# ═══════════════════════════════════════════════════════════════════════════════

# ── pre-compute additional trends ────────────────────────────────────────────
_rs_ocn_sl, _rs_ocn_se, _rs_ocn_p = zip(*[_get_trend(rs_data, 'ocean', yr_ceres, mn) for mn in MONTH_NUMS])
_rs_all_sl, _rs_all_se, _rs_all_p = zip(*[_get_trend(rs_data, 'all',   yr_ceres, mn) for mn in MONTH_NUMS])

_tcc_ice_sl, _tcc_ice_se, _tcc_ice_p = zip(*[_get_trend(tcc_data, 'ice',   yr_tcc, mn) for mn in MONTH_NUMS])
_tcc_ocn_sl, _tcc_ocn_se, _tcc_ocn_p = zip(*[_get_trend(tcc_data, 'ocean', yr_tcc, mn) for mn in MONTH_NUMS])

_tcc_ice_sl = np.asarray(_tcc_ice_sl, float) * 1e3   # → ×10⁻³ yr⁻¹
_tcc_ice_se = np.asarray(_tcc_ice_se, float) * 1e3
_tcc_ocn_sl = np.asarray(_tcc_ocn_sl, float) * 1e3
_tcc_ocn_se = np.asarray(_tcc_ocn_se, float) * 1e3

yr_sic = np.array(sorted(set(YEARS_ALL)))

fig_C = plt.figure(figsize=(19, 6), facecolor='white')
fig_C.suptitle(
    'Cloud masking of Arctic sea-ice albedo feedback (SIAF) — why RS is declining\n'
    'Clouds intercept SW before it reaches the darkening surface, suppressing SIAF '
    'and driving RS decline across both remaining-ice and newly ice-free zones',
    fontsize=11, fontweight='bold', y=1.03)

gs_C = gridspec.GridSpec(1, 3, figure=fig_C, wspace=0.36)
ax_C1 = fig_C.add_subplot(gs_C[0])
ax_C2 = fig_C.add_subplot(gs_C[1])
ax_C3 = fig_C.add_subplot(gs_C[2])

_ekw3 = dict(elinewidth=1.1, capsize=3, ecolor='#444')
_wb3  = 0.22
C_RS_CLR = '#e8956d'   # clear-sky colour (lighter warm red)

# ── Panel (a): clear-sky vs all-sky RS, ice zone, Jun + Jul ──────────────────
ax_C1.axhline(0, color='#ddd', lw=0.7, ls=':')
_ann_row = 0

for mn, lw_base in [(6, 1.7), (7, 1.2)]:
    mn_lbl = MONTH_NAMES_SHORT[mn]
    rs_a,    _ = ref_anom(to_ts(rs_data['ice'][mn],     yr_ceres), yr_ceres)
    rsclr_a, _ = ref_anom(to_ts(rs_clr_data['ice'][mn], yr_ceres), yr_ceres)
    ls = '-' if mn == 6 else '--'
    m_rs, m_rc = np.isfinite(rs_a), np.isfinite(rsclr_a)

    ax_C1.fill_between(yr_ceres[m_rs], 0, rs_a[m_rs],
                       color=C_RS, alpha=0.15 if mn == 6 else 0.07, linewidth=0)
    ax_C1.plot(yr_ceres[m_rs], rs_a[m_rs],
               color=C_RS, lw=lw_base, alpha=0.85, ls=ls,
               label=f'All-sky RS {mn_lbl}')

    ax_C1.fill_between(yr_ceres[m_rc], 0, rsclr_a[m_rc],
                       color=C_RS_CLR, alpha=0.13 if mn == 6 else 0.06, linewidth=0)
    ax_C1.plot(yr_ceres[m_rc], rsclr_a[m_rc],
               color=C_RS_CLR, lw=lw_base, alpha=0.85, ls=ls,
               label=f'Clear-sky RS {mn_lbl}')

    for ts_a, mask, clr, lbl in [
        (rs_a,    m_rs, C_RS,     f'{mn_lbl} all-sky'),
        (rsclr_a, m_rc, C_RS_CLR, f'{mn_lbl} clr-sky'),
    ]:
        res = ols_safe(yr_ceres.astype(float), ts_a)
        if res:
            yt = res.slope * yr_ceres + np.nanmean(ts_a) - res.slope * np.mean(yr_ceres)
            ax_C1.plot(yr_ceres[mask], yt[mask], color=clr,
                       lw=sig_lw(res.pvalue) * 1.2, ls='--',
                       alpha=sig_alpha(res.pvalue))
            ax_C1.text(0.97, 0.97 - _ann_row * 0.10,
                       f'{lbl}: {res.slope:+.3f} W m⁻² yr⁻¹ {pstar(res.pvalue)}',
                       transform=ax_C1.transAxes, ha='right', va='top',
                       fontsize=7.5, color=clr, fontweight='bold',
                       alpha=max(sig_alpha(res.pvalue), 0.45))
            _ann_row += 1

ax_C1.set_xlabel('Year', fontsize=9)
ax_C1.set_ylabel('SW anomaly (W m⁻²)', fontsize=9)
ax_C1.set_title('(a) Clear-sky vs all-sky surface RS — ice zone\n'
                'Clear-sky ↑ = SIAF working  |  All-sky ↓ = clouds suppress it',
                fontsize=9.5, fontweight='bold')
ax_C1.legend(fontsize=7.5, ncol=2, loc='lower left')
ax_C1.set_xlim(1999, 2026)
ax_C1.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax_C1.tick_params(labelsize=7)
for sp in ['top', 'right']:
    ax_C1.spines[sp].set_visible(False)

# ── Panel (b): ERA5 TCC trends, ice zone vs ocean/MIZ ────────────────────────
ax_C2.axhline(0, color='#bbb', lw=0.8, ls=':')
b_tcc_i = ax_C2.bar(_xb - _wb3 / 2, _tcc_ice_sl, _wb3, yerr=_tcc_ice_se,
                    color=C_ICE,   alpha=0.88,
                    label='Ice zone (SIC ≥ 0.15)', error_kw=_ekw3)
b_tcc_o = ax_C2.bar(_xb + _wb3 / 2, _tcc_ocn_sl, _wb3, yerr=_tcc_ocn_se,
                    color=C_OCEAN, alpha=0.88,
                    label='Ocean/MIZ (SIC < 0.15)', error_kw=_ekw3)
_sig_bar_labels(ax_C2, b_tcc_i, _tcc_ice_p, _tcc_ice_se, 0.03)
_sig_bar_labels(ax_C2, b_tcc_o, _tcc_ocn_p, _tcc_ocn_se, 0.03)
ax_C2.set_xticks(_xb)
ax_C2.set_xticklabels(_mn_labels, fontsize=9)
ax_C2.set_ylabel('TCC trend (×10⁻³ yr⁻¹)  [ERA5, 1979–2025]', fontsize=9)
ax_C2.set_title('(b) Cloud cover trends — ice zone vs open ocean\n'
                'Ice zone getting cloudier; cloud increase is regime-specific',
                fontsize=9.5, fontweight='bold')
ax_C2.legend(fontsize=8.5, loc='upper left')
ax_C2.tick_params(labelsize=8)
for sp in ['top', 'right']:
    ax_C2.spines[sp].set_visible(False)

# ── Panel (c): Ocean/MIZ area fraction growing (1979–2025) ───────────────────
_frac_styles = {5: (':',  '#f0aa44'), 6: ('-',  '#d67a00'),
                7: ('--', '#b04000'), 8: ('-.', '#883300')}
_ann_row3 = 0
for mn in [5, 6, 7, 8]:
    frac_ts = np.array([ocean_frac_data[mn].get(y, np.nan) for y in yr_sic])
    mask_f  = np.isfinite(frac_ts)
    if mask_f.sum() < 5:
        continue
    ls_f, clr_f = _frac_styles[mn]
    ax_C3.plot(yr_sic[mask_f], frac_ts[mask_f] * 100,
               color=clr_f, lw=1.6, ls=ls_f, alpha=0.8,
               label=MONTH_NAMES_SHORT[mn])
    res_f = ols_safe(yr_sic.astype(float), frac_ts)
    if res_f:
        yt_f = (res_f.slope * yr_sic
                + np.nanmean(frac_ts) - res_f.slope * np.mean(yr_sic))
        ax_C3.plot(yr_sic[mask_f], yt_f[mask_f] * 100,
                   color=clr_f, lw=sig_lw(res_f.pvalue) * 1.1,
                   ls='--', alpha=sig_alpha(res_f.pvalue))
        ax_C3.text(0.97, 0.97 - _ann_row3 * 0.10,
                   f'{MONTH_NAMES_SHORT[mn]}: {res_f.slope * 100 * 10:+.1f} %/decade {pstar(res_f.pvalue)}',
                   transform=ax_C3.transAxes, ha='right', va='top',
                   fontsize=7.5, color=clr_f, fontweight='bold',
                   alpha=max(sig_alpha(res_f.pvalue), 0.45))
        _ann_row3 += 1
ax_C3.set_xlabel('Year', fontsize=9)
ax_C3.set_ylabel('Ocean/MIZ fraction of domain (%)', fontsize=9)
ax_C3.set_title('(c) Expanding ocean/MIZ area (% of Arctic domain)\n'
                'More open water each decade — high RS potential, but cloud-cooled',
                fontsize=9.5, fontweight='bold')
ax_C3.set_xlim(1978, 2026)
ax_C3.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax_C3.tick_params(labelsize=8)
ax_C3.legend(fontsize=8.5, loc='upper left')
for sp in ['top', 'right']:
    ax_C3.spines[sp].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 1])
outC = os.path.join(OUT_DIR, 'figC_cloud_masking_siaf.png')
_savefig(fig_C, outC, 'Fig C', dpi=180)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG D — Area-weighted decomposition of Arctic domain-mean RS trend
#
#  RS_all(t) = f_ice(t) × RS_ice(t) + f_ocean(t) × RS_ocean(t)
#
#  First-order Taylor decomposition of dRS_all/dt:
#    (1) Within-ice   = f̄_ice   × slope(RS_ice)       cloud-driven decline
#    (2) Within-ocean = f̄_ocean × slope(RS_ocean)      cloud suppression of SIAF
#    (3) Area-shift   = (R̄_ocean − R̄_ice) × slope(f_ocean)   SIAF mechanism (positive)
#
#  Grouped bars show all three per month.
#  Black dots = directly observed RS_all trend.  Dashed line = sum of components.
#  If area-shift (green) < |within-ice + within-ocean| → clouds overwhelm SIAF.
# ═══════════════════════════════════════════════════════════════════════════════

_D_wi, _D_wo, _D_as, _D_tot, _D_obs = [], [], [], [], []

for mn in MONTH_NUMS:
    rs_i = to_ts(rs_data['ice'][mn],   yr_ceres)
    rs_o = to_ts(rs_data['ocean'][mn], yr_ceres)
    rs_a = to_ts(rs_data['all'][mn],   yr_ceres)
    f_o  = np.array([ocean_frac_data[mn].get(y, np.nan) for y in yr_ceres])

    valid = np.isfinite(rs_i) & np.isfinite(rs_o) & np.isfinite(f_o)
    yr_v  = yr_ceres[valid].astype(float)

    if valid.sum() < 5:
        for lst in (_D_wi, _D_wo, _D_as, _D_tot, _D_obs):
            lst.append(np.nan)
        continue

    fi_mean = float(np.nanmean((1 - f_o)[valid]))
    fo_mean = float(np.nanmean(f_o[valid]))
    ri_mean = float(np.nanmean(rs_i[valid]))
    ro_mean = float(np.nanmean(rs_o[valid]))

    def _sl(x, y):
        res = ols_safe(x, y)
        return res.slope if res else np.nan

    s_ri = _sl(yr_v, rs_i[valid])
    s_ro = _sl(yr_v, rs_o[valid])
    s_fo = _sl(yr_v, f_o[valid])
    s_ra = _sl(yr_ceres.astype(float), rs_a)

    wi  = fi_mean * s_ri
    wo  = fo_mean * s_ro
    ash = (ro_mean - ri_mean) * s_fo

    _D_wi.append(wi)
    _D_wo.append(wo)
    _D_as.append(ash)
    _D_tot.append(wi + wo + ash if all(np.isfinite([wi, wo, ash])) else np.nan)
    _D_obs.append(s_ra)

_D_wi  = np.array(_D_wi,  float)
_D_wo  = np.array(_D_wo,  float)
_D_as  = np.array(_D_as,  float)
_D_tot = np.array(_D_tot, float)
_D_obs = np.array(_D_obs, float)

_C_SHIFT = '#2ca02c'   # green = SIAF / area-shift (positive = ice-albedo feedback working)

fig_D, ax_D = plt.subplots(figsize=(10, 5.5), facecolor='white')
fig_D.suptitle(
    'Area-weighted decomposition of Arctic RS trend (2000–2025)\n'
    r'$\frac{d\overline{RS}}{dt} \approx \bar{f}_{ice}\cdot\frac{dRS_{ice}}{dt}$'
    r' $+\ \bar{f}_{ocean}\cdot\frac{dRS_{ocean}}{dt}$'
    r' $+\ (\overline{RS}_{ocean}-\overline{RS}_{ice})\cdot\frac{df_{ocean}}{dt}$',
    fontsize=11, fontweight='bold')

_xd = np.arange(len(MONTH_NUMS))
_wd_d = 0.22

b_wi = ax_D.bar(_xd - _wd_d, _D_wi, _wd_d, color=C_ICE,    alpha=0.88, zorder=3,
                label=r'Within-ice zone:  $\bar{f}_{ice}\cdot\Delta RS_{ice}$  (cloud-driven decline)')
b_wo = ax_D.bar(_xd,          _D_wo, _wd_d, color=C_OCEAN,  alpha=0.88, zorder=3,
                label=r'Within-ocean/MIZ: $\bar{f}_{ocean}\cdot\Delta RS_{ocean}$  (cloud suppression of SIAF)')
b_as = ax_D.bar(_xd + _wd_d, _D_as, _wd_d, color=_C_SHIFT,  alpha=0.88, zorder=3,
                label=r'Area-shift (SIAF): $(\overline{RS}_{ocean}-\overline{RS}_{ice})\cdot\Delta f_{ocean}$')

# Observed RS_all trend (black diamonds)
ax_D.scatter(_xd, _D_obs, color='black', s=75, zorder=6, marker='D',
             label='Observed RS_all trend (direct OLS)')

# Sum of components — tests decomposition accuracy
ax_D.plot(_xd, _D_tot, color='#555', lw=1.8, ls='--', zorder=5,
          label='Sum of components (should ≈ observed)')

# Significance stars on within-ice bars (re-use ice-zone RS pvalues)
for i, (bar, p) in enumerate(zip(b_wi, _rs_p)):
    try:
        pf = float(p)
    except Exception:
        continue
    if np.isnan(pf) or pf >= 0.05:
        continue
    h = bar.get_height()
    ax_D.text(bar.get_x() + bar.get_width() / 2,
              h - 0.02, pstar(pf),
              ha='center', va='top', fontsize=8, fontweight='bold', color='#222')

ax_D.axhline(0, color='#aaa', lw=0.8, ls=':')
ax_D.set_xticks(_xd)
ax_D.set_xticklabels(_mn_labels, fontsize=10)
ax_D.set_ylabel('Contribution to dRS_all/dt  (W m⁻² yr⁻¹)', fontsize=10)
ax_D.set_xlabel('Month', fontsize=10)
ax_D.legend(fontsize=8.5, loc='lower left')
ax_D.tick_params(labelsize=9)

# Annotate the key insight
ax_D.text(0.97, 0.97,
          'Green bars = SIAF operating (ice loss → more open water → +RS)\n'
          'Blue + orange = cloud cooling overcoming SIAF\n'
          'Net (black ◆) = observed domain-mean RS trend',
          transform=ax_D.transAxes, ha='right', va='top',
          fontsize=8, color='#444', fontstyle='italic',
          bbox=dict(fc='#f8f8f8', ec='#ccc', pad=4, alpha=0.85))

for sp in ['top', 'right']:
    ax_D.spines[sp].set_visible(False)

plt.tight_layout()
outD = os.path.join(OUT_DIR, 'figD_rs_trend_decomposition.png')
_savefig(fig_D, outD, 'Fig D', dpi=180)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG E — RS / TCC / CF in the SIAF extent footprint vs darkening footprint
#
# Top row:  AMJJAS-mean time series (2000-2024) for each variable
#           ext = retreating-ice pixels (solid), drk = stable-ice (dashed)
# Bottom row: per-month OLS trends — grouped bars, ext vs drk
# ═══════════════════════════════════════════════════════════════════════════════
print('\nBuilding Fig E — SIAF footprint RS/TCC/CF trends ...')

_C_EXT = '#c0392b'   # extent zone — deep red
_C_DRK = '#2471a3'   # darkening zone — steel blue
_MN_LBL = [MONTH_NAMES_SHORT[mn] for mn in MONTH_NUMS]

fig_E, axes_E = plt.subplots(2, 3, figsize=(16, 9), facecolor='white')

# ── Row 0: AMJJAS-mean time series ─────────────────────────────────────────────
_panel_vars = [
    ('RS',     rs_ext_data,  rs_drk_data,  yr_ceres, 'W m⁻²',      C_RS),
    ('ERA5 TCC', tcc_ext_data, tcc_drk_data, yr_tcc,  'cloud fraction', C_TCC),
    ('MODIS CF', cf_ext_data,  cf_drk_data,  yr_ceres, 'cloud fraction', C_CF),
]

for _ci, (_label, _ext_d, _drk_d, _yr_vec, _unit, _c_base) in enumerate(_panel_vars):
    ax = axes_E[0, _ci]

    # Build AMJJAS-mean time series (average over all 6 months per year)
    _ext_ts = np.array([
        np.nanmean([_ext_d[mn].get(y, np.nan) for mn in MONTH_NUMS])
        for y in _yr_vec])
    _drk_ts = np.array([
        np.nanmean([_drk_d[mn].get(y, np.nan) for mn in MONTH_NUMS])
        for y in _yr_vec])

    # Filter to years with ≥3 valid months
    _ext_ok = np.array([
        np.sum(np.isfinite([_ext_d[mn].get(y, np.nan) for mn in MONTH_NUMS])) >= 3
        for y in _yr_vec])
    _drk_ok = np.array([
        np.sum(np.isfinite([_drk_d[mn].get(y, np.nan) for mn in MONTH_NUMS])) >= 3
        for y in _yr_vec])
    _ext_ts[~_ext_ok] = np.nan
    _drk_ts[~_drk_ok] = np.nan

    ax.plot(_yr_vec, _ext_ts, color=_C_EXT, lw=1.5, alpha=0.85,
            label='Extent (retreating ice)')
    ax.plot(_yr_vec, _drk_ts, color=_C_DRK, lw=1.5, alpha=0.85,
            ls='--', label='Darkening (stable ice)')

    # Trend lines
    for _ts, _col in [(_ext_ts, _C_EXT), (_drk_ts, _C_DRK)]:
        _res = ols_safe(_yr_vec.astype(float), _ts)
        if _res:
            _fit = _res.intercept + _res.slope * _yr_vec.astype(float)
            _ok_mask = np.isfinite(_ts)
            ax.plot(_yr_vec[_ok_mask], _fit[_ok_mask],
                    color=_col, lw=sig_lw(_res.pvalue) + 0.5,
                    alpha=sig_alpha(_res.pvalue), ls='-')
            ax.text(0.02 if _col == _C_EXT else 0.98,
                    0.98 if _col == _C_EXT else 0.88,
                    f'{_res.slope*10:+.3f}/decade {pstar(_res.pvalue)}',
                    transform=ax.transAxes,
                    ha='left' if _col == _C_EXT else 'right', va='top',
                    fontsize=7.5, color=_col)

    ax.set_title(_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(_unit, fontsize=9)
    ax.set_xlabel('Year', fontsize=9)
    ax.legend(fontsize=8, loc='lower left')
    ax.tick_params(labelsize=8)
    for _sp in ['top', 'right']:
        ax.spines[_sp].set_visible(False)

    # Panel letter
    ax.text(-0.08, 1.04, chr(ord('a') + _ci), transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')

# ── Row 1: per-month trend bars ─────────────────────────────────────────────────
_panel_trends = [
    ('RS trend (W m⁻² yr⁻¹)',         rs_ext_data,  rs_drk_data,  yr_ceres),
    ('ERA5 TCC trend (×10⁻³ yr⁻¹)',    tcc_ext_data, tcc_drk_data, yr_tcc),
    ('MODIS CF trend (×10⁻³ yr⁻¹)',    cf_ext_data,  cf_drk_data,  yr_ceres),
]

_xbar = np.arange(len(MONTH_NUMS))
_wd   = 0.35

for _ci, (_ylabel, _ext_d, _drk_d, _yr_vec) in enumerate(_panel_trends):
    ax = axes_E[1, _ci]
    _scale = 1e3 if 'TCC' in _ylabel or 'CF' in _ylabel else 1.0

    _ext_sl, _ext_se, _ext_p = [], [], []
    _drk_sl, _drk_se, _drk_p = [], [], []

    for _mn in MONTH_NUMS:
        _ts_e = to_ts(_ext_d[_mn], _yr_vec)
        _ts_d = to_ts(_drk_d[_mn], _yr_vec)
        _re   = ols_safe(_yr_vec.astype(float), _ts_e)
        _rd   = ols_safe(_yr_vec.astype(float), _ts_d)
        _ext_sl.append(_re.slope * _scale if _re else 0.)
        _ext_se.append(_re.stderr * _scale if _re else 0.)
        _ext_p.append(_re.pvalue if _re else 1.)
        _drk_sl.append(_rd.slope * _scale if _rd else 0.)
        _drk_se.append(_rd.stderr * _scale if _rd else 0.)
        _drk_p.append(_rd.pvalue if _rd else 1.)

    _be = ax.bar(_xbar - _wd/2, _ext_sl, _wd, yerr=_ext_se,
                 color=_C_EXT, alpha=0.85, label='Extent', capsize=3,
                 error_kw=dict(lw=1, capthick=1))
    _bd = ax.bar(_xbar + _wd/2, _drk_sl, _wd, yerr=_drk_se,
                 color=_C_DRK, alpha=0.85, label='Darkening', capsize=3,
                 error_kw=dict(lw=1, capthick=1))

    # Significance stars on bars
    for _i, (_be_bar, _bd_bar, _pe, _pd) in enumerate(
            zip(_be, _bd, _ext_p, _drk_p)):
        for _bar, _p in [(_be_bar, _pe), (_bd_bar, _pd)]:
            _s = pstar(_p)
            if _s and _s != 'n.s.':
                _h = _bar.get_height()
                _sign = 1 if _h >= 0 else -1
                ax.text(_bar.get_x() + _bar.get_width()/2,
                        _h + _sign * max(abs(_h)*0.05, 0.001),
                        _s, ha='center',
                        va='bottom' if _h >= 0 else 'top',
                        fontsize=8, color='#222')

    ax.axhline(0, color='#999', lw=0.7, ls=':')
    ax.set_xticks(_xbar)
    ax.set_xticklabels(_MN_LBL, fontsize=9)
    ax.set_ylabel(_ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    for _sp in ['top', 'right']:
        ax.spines[_sp].set_visible(False)
    ax.text(-0.08, 1.04, chr(ord('d') + _ci), transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')

fig_E.suptitle(
    'RS, cloud cover, and cloud fraction over SIAF footprints (2000–2024)\n'
    'Extent = retreating-ice pixels (SIC trend < 0, p < 0.05);  '
    'Darkening = stable ice-covered pixels',
    fontsize=11, y=1.01)

plt.tight_layout()
outE = os.path.join(OUT_DIR, 'figE_siaf_footprint_trends.png')
_savefig(fig_E, outE, 'Fig E', dpi=180)


print('\nBuilding Fig F — publication figure ...')
from matplotlib.patches import Patch as _Patch

# ── Palette ───────────────────────────────────────────────────────────────────
_C_EXT  = '#b5000a'   # extent  — dark red
_C_DRK  = '#1a5fa8'   # darken  — steel blue
_C_OPQ  = '#5a189a'   # opaque  — deep purple
_C_EXT2 = '#e8a091'   # extent CALIPSO (lighter, distinct from MODIS)
_C_DRK2 = '#90b8d8'   # darken CALIPSO

_MN_X     = np.arange(len(MONTH_NUMS))
_MN_LBL_F = [MONTH_NAMES_SHORT[mn] for mn in MONTH_NUMS]
_yr_cal    = np.array(YEARS_CALIPSO)
_has_cal   = any(bool(cal_cf_ext_data[mn]) for mn in MONTH_NUMS)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _means(d, yrs):
    mu, sd = [], []
    for mn in MONTH_NUMS:
        v = np.array([d[mn].get(y, np.nan) for y in yrs])
        mu.append(np.nanmean(v)); sd.append(np.nanstd(v))
    return np.array(mu), np.array(sd)

def _trends(d, yrs):
    sl, se, pv = [], [], []
    for mn in MONTH_NUMS:
        v = np.array([d[mn].get(y, np.nan) for y in yrs])
        r = ols_safe(yrs.astype(float), v)
        sl.append(r.slope  if r else np.nan)
        se.append(r.stderr if r else np.nan)
        pv.append(r.pvalue if r else 1.)
    return np.array(sl), np.array(se), np.array(pv)

def _polish(ax, ylabel, title, letter, zero=False, legend_handles=None,
            note=None):
    ax.set_xticks(_MN_X)
    ax.set_xticklabels(_MN_LBL_F, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if zero:
        ax.axhline(0, color='#aaa', lw=0.9, ls='--', zorder=0)
    if legend_handles is not None:
        ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85,
                  edgecolor='#ccc')
    ax.tick_params(labelsize=10)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='bottom')
    if note:
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, color='#555',
                fontstyle='italic',
                bbox=dict(fc='#f7f7f7', ec='#ddd', pad=3, alpha=0.9))

def _trend_bars(ax, datasets, scale=1., w=0.38):
    """
    datasets: list of (sl, se, pv, color, label)
    Draws bars, error bars, and returns Patch handles for legend.
    Solid = p<0.05; hatched = n.s.
    """
    n = len(datasets)
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    handles = []
    for (_sl, _se, _pv, _col, _lbl), _off in zip(datasets, offsets):
        for i in range(len(MONTH_NUMS)):
            _sig = _pv[i] < 0.05
            ax.bar(_MN_X[i] + _off, _sl[i] * scale, w,
                   color=_col, alpha=0.85 if _sig else 0.35,
                   hatch=None if _sig else '//',
                   edgecolor=_col, linewidth=0.5)
        ax.errorbar(_MN_X + _off, _sl * scale, yerr=_se * scale,
                    fmt='none', color=_col, capsize=3.5,
                    lw=1.2, capthick=1.2, zorder=5)
        handles.append(_Patch(facecolor=_col, alpha=0.85, label=_lbl,
                               edgecolor=_col))
    return handles

# ── Figure setup ──────────────────────────────────────────────────────────────
fig_F, axes_F = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
fig_F.subplots_adjust(hspace=0.42, wspace=0.32)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0 — Climatological means
# ══════════════════════════════════════════════════════════════════════════════

# ── (a) RS mean: extent vs darkening ─────────────────────────────────────────
ax = axes_F[0, 0]
mu_re, sd_re = _means(rs_ext_data, yr_ceres)
mu_rd, sd_rd = _means(rs_drk_data, yr_ceres)
ax.bar(_MN_X - 0.2, mu_re, 0.38, yerr=sd_re, capsize=4,
       color=_C_EXT, alpha=0.88, error_kw=dict(lw=1.3, capthick=1.3))
ax.bar(_MN_X + 0.2, mu_rd, 0.38, yerr=sd_rd, capsize=4,
       color=_C_DRK, alpha=0.88, error_kw=dict(lw=1.3, capthick=1.3))
for i in range(len(MONTH_NUMS)):
    diff = mu_re[i] - mu_rd[i]
    if np.isfinite(diff):
        top = max(mu_re[i] + sd_re[i], mu_rd[i] + sd_rd[i])
        ax.text(_MN_X[i], top * 1.012, f'{diff:+.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#111')
_polish(ax, 'W m⁻²', 'Surface absorbed SW  (RS, 2000–2024)', 'a',
        legend_handles=[_Patch(facecolor=_C_EXT, label='Extent (retreating ice)'),
                        _Patch(facecolor=_C_DRK, label='Darkening (stable ice)')],
        note='Extent zone absorbs less SW\nacross all months')

# ── (b) CF mean: MODIS (solid) + CALIPSO (hatched) both footprints ───────────
ax = axes_F[0, 1]
mu_me, sd_me = _means(cf_ext_data, yr_ceres)
mu_md, sd_md = _means(cf_drk_data, yr_ceres)
w4 = 0.19
ax.bar(_MN_X - 1.5*w4, mu_me, w4, yerr=sd_me, capsize=2.5,
       color=_C_EXT, alpha=0.88, error_kw=dict(lw=1))
ax.bar(_MN_X - 0.5*w4, mu_md, w4, yerr=sd_md, capsize=2.5,
       color=_C_DRK, alpha=0.88, error_kw=dict(lw=1))
_leg_b = [_Patch(facecolor=_C_EXT, label='MODIS — Extent'),
          _Patch(facecolor=_C_DRK, label='MODIS — Darkening')]
if _has_cal:
    mu_ce, sd_ce = _means(cal_cf_ext_data, _yr_cal)
    mu_cd, sd_cd = _means(cal_cf_drk_data, _yr_cal)
    ax.bar(_MN_X + 0.5*w4, mu_ce, w4, yerr=sd_ce, capsize=2.5,
           color=_C_EXT2, alpha=0.95, edgecolor=_C_EXT, lw=1.2,
           error_kw=dict(lw=1))
    ax.bar(_MN_X + 1.5*w4, mu_cd, w4, yerr=sd_cd, capsize=2.5,
           color=_C_DRK2, alpha=0.95, edgecolor=_C_DRK, lw=1.2,
           error_kw=dict(lw=1))
    _leg_b += [_Patch(facecolor=_C_EXT2, edgecolor=_C_EXT, lw=1.2,
                       label='CALIPSO — Extent'),
               _Patch(facecolor=_C_DRK2, edgecolor=_C_DRK, lw=1.2,
                       label='CALIPSO — Darkening')]
_polish(ax, 'Cloud fraction (0–1)',
        'Cloud fraction — MODIS & CALIPSO agree\n(extent zone consistently cloudier)',
        'b', legend_handles=_leg_b,
        note='Active lidar (CALIPSO) confirms\npassive sensor (MODIS)')

# ── (c) CALIPSO: total vs opaque CF — are extent clouds optically thick? ──────
ax = axes_F[0, 2]
if _has_cal:
    mu_tot_e, _ = _means(cal_cf_ext_data,  _yr_cal)
    mu_opq_e, _ = _means(cal_opq_ext_data, _yr_cal)
    mu_tot_d, _ = _means(cal_cf_drk_data,  _yr_cal)
    mu_opq_d, _ = _means(cal_opq_drk_data, _yr_cal)
    ax.bar(_MN_X - 1.5*w4, mu_tot_e, w4, color=_C_EXT, alpha=0.85)
    ax.bar(_MN_X - 0.5*w4, mu_opq_e, w4, color=_C_OPQ, alpha=0.85)
    ax.bar(_MN_X + 0.5*w4, mu_tot_d, w4, color=_C_EXT2,
           edgecolor=_C_EXT, lw=1.2, alpha=0.9)
    ax.bar(_MN_X + 1.5*w4, mu_opq_d, w4, color=_C_OPQ,
           alpha=0.38, edgecolor=_C_OPQ, lw=1.2)
    # Annotate opaque/total ratio for extent zone
    for i in range(len(MONTH_NUMS)):
        if mu_tot_e[i] > 0 and np.isfinite(mu_opq_e[i]):
            frac = mu_opq_e[i] / mu_tot_e[i]
            ax.text(_MN_X[i] - 0.5*w4, mu_opq_e[i] + 0.008,
                    f'{frac:.0%}', ha='center', va='bottom',
                    fontsize=8.5, color=_C_OPQ, fontweight='bold')
    _polish(ax, 'Cloud fraction (0–1)',
            'CALIPSO: cloud optical thickness\n(% = opaque fraction of extent clouds)',
            'c',
            legend_handles=[
                _Patch(facecolor=_C_EXT,  label='Extent — total CF'),
                _Patch(facecolor=_C_OPQ,  label='Extent — opaque CF  ← %'),
                _Patch(facecolor=_C_EXT2, edgecolor=_C_EXT, lw=1.2,
                        label='Darkening — total CF'),
                _Patch(facecolor=_C_OPQ,  alpha=0.38, edgecolor=_C_OPQ,
                        lw=1.2, label='Darkening — opaque CF'),
            ],
            note='Extent clouds are inherently\nmore optically thick')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — OLS trends  (solid bar = p<0.05; hatched = n.s.)
# ══════════════════════════════════════════════════════════════════════════════

# ── (d) RS trend ─────────────────────────────────────────────────────────────
ax = axes_F[1, 0]
sl_re, se_re, pv_re = _trends(rs_ext_data, yr_ceres)
sl_rd, se_rd, pv_rd = _trends(rs_drk_data, yr_ceres)
h = _trend_bars(ax, [
    (sl_re, se_re, pv_re, _C_EXT, 'Extent'),
    (sl_rd, se_rd, pv_rd, _C_DRK, 'Darkening'),
])
_polish(ax, 'W m⁻² yr⁻¹', 'RS trend (2000–2024)', 'd', zero=True,
        legend_handles=h + [_Patch(facecolor='#aaa', alpha=0.35, hatch='//',
                                    label='n.s. (p ≥ 0.05)')],
        note='RS declining in BOTH zones\n→ cloud forcing at work everywhere')

# ── (e) CF trends: MODIS + CALIPSO total, extent vs darkening ────────────────
# Shows: only darkening has significant CF increase (explains its RS decline)
ax = axes_F[1, 1]
sl_me, se_me, pv_me = _trends(cf_ext_data,    yr_ceres)
sl_md, se_md, pv_md = _trends(cf_drk_data,    yr_ceres)
if _has_cal:
    sl_ce, se_ce, pv_ce = _trends(cal_cf_ext_data, _yr_cal)
    sl_cd, se_cd, pv_cd = _trends(cal_cf_drk_data, _yr_cal)
    datasets_e = [
        (sl_me, se_me, pv_me, _C_EXT,  'MODIS — Extent'),
        (sl_md, se_md, pv_md, _C_DRK,  'MODIS — Darkening'),
        (sl_ce, se_ce, pv_ce, _C_EXT2, 'CALIPSO — Extent'),
        (sl_cd, se_cd, pv_cd, _C_DRK2, 'CALIPSO — Darkening'),
    ]
else:
    datasets_e = [
        (sl_me, se_me, pv_me, _C_EXT, 'MODIS — Extent'),
        (sl_md, se_md, pv_md, _C_DRK, 'MODIS — Darkening'),
    ]
h = _trend_bars(ax, datasets_e, scale=1e3, w=0.19)
_polish(ax, '×10⁻³ yr⁻¹',
        'Cloud fraction trend — MODIS & CALIPSO\n(significant ↑ over darkening zone only)',
        'e', zero=True,
        legend_handles=h + [_Patch(facecolor='#aaa', alpha=0.35, hatch='//',
                                    label='n.s. (p ≥ 0.05)')],
        note='Darkening zone: significant CF ↑\n→ cloud-driven RS decline over stable ice')

# ── (f) CALIPSO total vs opaque CF trends — frequency ↑ but not thickness ────
ax = axes_F[1, 2]
if _has_cal:
    sl_ct_e, se_ct_e, pv_ct_e = _trends(cal_cf_ext_data,  _yr_cal)
    sl_co_e, se_co_e, pv_co_e = _trends(cal_opq_ext_data, _yr_cal)
    sl_ct_d, se_ct_d, pv_ct_d = _trends(cal_cf_drk_data,  _yr_cal)
    sl_co_d, se_co_d, pv_co_d = _trends(cal_opq_drk_data, _yr_cal)
    h = _trend_bars(ax, [
        (sl_ct_e, se_ct_e, pv_ct_e, _C_EXT,  'Extent — total CF'),
        (sl_co_e, se_co_e, pv_co_e, _C_OPQ,  'Extent — opaque CF'),
        (sl_ct_d, se_ct_d, pv_ct_d, _C_EXT2, 'Darkening — total CF'),
        (sl_co_d, se_co_d, pv_co_d, _C_DRK2, 'Darkening — opaque CF'),
    ], scale=1e3, w=0.19)
    _polish(ax, '×10⁻³ yr⁻¹',
            'CALIPSO total vs opaque CF trend\n(cloud frequency ↑, thickness unchanged)',
            'f', zero=True,
            legend_handles=h + [_Patch(facecolor='#aaa', alpha=0.35, hatch='//',
                                        label='n.s. (p ≥ 0.05)')],
            note='New clouds are not getting thicker;\nfrequency drives the signal')

fig_F.suptitle(
    'Arctic cloud–albedo interactions over SIAF footprints  '
    '(extent = retreating-ice pixels;  darkening = stable-ice pixels)',
    fontsize=13, fontweight='bold', y=1.005)

plt.tight_layout()
outF = os.path.join(OUT_DIR, 'figF_siaf_footprint_mean_cf_rs.png')
fig_F.savefig(outF, dpi=200, bbox_inches='tight')
plt.close(fig_F)
print(f'Saved Fig F → {outF}')


# ── DONE ──────────────────────────────────────────────────────────────────────
print('\nAll done.')
print(f'Output directory: {OUT_DIR}')
for f in [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
          outA, outB, outC, outD, outE, outF]:
    print(f'  {f}')

print("""
Figure guide:
  Fig 1  CERES RS trends (2000–2025) per month, split by SIC regime
  Fig 2  ERA5 TCC trends (1979–2025) per month, split by SIC regime
  Fig 3  RS vs TCC dual-axis overlays by month × regime (6×3 grid)
  Fig 4  RS vs TCC scatter per month, coloured by year (2000–2025)
  Fig 5  ERA5 TCC full record + CERES RS overlay, full domain only
  Fig 6  CERES surface SWCRE trends per month, split by SIC regime
  Fig 7  RS vs surface SWCRE dual-axis overlays by month × regime (6×3 grid)
  Fig 8  CERES TOA SWCRE trends per month, split by SIC regime
  Fig 9  Attribution panel: RS / sfc SWCRE / TOA SWCRE stacked, full domain
  Fig 10 MODIS CF and CWP_liq trends — more clouds vs thicker clouds (4×6 grid)
  Fig 11 SWCRE vs MODIS CF and CWP scatter — which predictor explains SWCRE?
  Fig E  SIAF footprint RS/TCC/CF trends — extent (retreating) vs darkening zones
  Fig F  Mean MODIS CF and RS in extent vs darkening zones — is extent cloudier?

Significance transparency key:
  Opaque trend line = *** p<0.001 | solid = ** p<0.01 | faded = * p<0.05
  Nearly invisible = n.s. (p≥0.05)

Attribution reading guide (Fig 9):
  Cloud-driven RS:    SWCRE and RS trend in OPPOSITE directions
                      (less negative SWCRE → less cloud cooling → more RS)
  Ice-albedo-driven:  RS trends while SWCRE is flat/n.s.
  Anti-SIAF signal:   RS DECLINES while SWCRE becomes MORE negative
                      (more cloud cooling offsetting ice-albedo forcing)
  TOA−sfc gap trend:  Changing atmospheric absorption of SW by clouds
""")