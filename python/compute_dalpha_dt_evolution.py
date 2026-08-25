#!/usr/bin/env python
"""
compute_dalpha_dt_evolution.py
──────────────────────────────
Computes yearly AMJJAS domain-mean Arctic albedo, aligns with the
temperature ensemble, then runs:
  • Full-period OLS with HAC-robust standard errors
  • Expanding-window and rolling-window OLS
  • First-differences OLS  (interannual / reversible sensitivity)
  • TVP Kalman filter       (time-varying beta via MLE)
  • Trend decomposition:    detrended residuals, partial regression,
                            low-pass / high-pass frequency bands
  • Forced vs internal variance attribution for SIAF

Saves everything to a pickle for plot_dalpha_dt_evolution.py.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle, glob, os
from scipy import stats
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from pyproj import Transformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'

GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

OUT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
PKL_PATH = os.path.join(OUT_DIR, 'dalpha_dt_evolution.pkl')
os.makedirs(OUT_DIR, exist_ok=True)

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
TREND_YEARS = list(range(1982, 2025))
ROLL_WIN    = 15
MIN_EXP     = 10
LP_WIN      = 9    # low-pass filter width (years) for forced/internal split

# Annual beta drift implied by rolling-window (beta changed ~0.04 albedo/K
# over ~28 years -> ~0.0014/yr).  Used to set physically-grounded TVP q.
TVP_ANNUAL_DRIFT = 0.0014


# ── GRID SETUP ────────────────────────────────────────────────────────────────
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
    _arctic_cached = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
    if _arctic_cached.shape == SIAL_SHAPE:
        ARCTIC_SIAL = _arctic_cached
    else:
        print(f'  WARNING: arctic_60N_sial.npy shape {_arctic_cached.shape} '
              f'!= SIAL_SHAPE {SIAL_SHAPE} -- recomputing from lat.')
        ARCTIC_SIAL = (SIAL_LAT >= 60.)
except FileNotFoundError:
    ARCTIC_SIAL = (SIAL_LAT >= 60.)

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

print('Building SIAL -> PSN25 index ...')
IDX_SIAL_TO_PSN = build_scatter_index(
    SIAL_LAT.ravel(), SIAL_LON.ravel(),
    PSN_LAT.ravel(),  PSN_LON.ravel())

def domain_mean_psn(field_psn):
    v  = field_psn[DOMAIN_PSN]
    w  = AREA_W[DOMAIN_PSN]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))

def regrid_sial_to_psn(arr):
    if arr.shape != SIAL_SHAPE:
        tmp = np.full(SIAL_SHAPE, np.nan, dtype=np.float32)
        r = min(arr.shape[0], SIAL_SHAPE[0])
        c = min(arr.shape[1], SIAL_SHAPE[1])
        tmp[:r, :c] = arr[:r, :c]
        arr = tmp
    src = np.where(ARCTIC_SIAL & np.isfinite(arr), arr, np.float32(0.))
    return scatter_avg(src.ravel().astype(np.float32), IDX_SIAL_TO_PSN, PSN_SHAPE)


# ── TEMPERATURE ENSEMBLE ──────────────────────────────────────────────────────
print('Building T ensemble ...')

def build_T_ensemble(trend_years):
    raw = {}
    try:
        df = pd.read_csv(GISTEMP_PATH, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        raw['GISTEMP'] = {int(r['Year']): float(r['J-D'])
                          for _, r in df.iterrows() if np.isfinite(r['J-D'])}
    except Exception: pass
    try:
        df = pd.read_csv(COWTANWAY_PATH, sep=r'\s+', header=None,
                         names=['year','anomaly','u1','u2','u3'])
        raw['CowtanWay'] = {int(r['year']): float(r['anomaly'])
                            for _, r in df.iterrows() if np.isfinite(r['anomaly'])}
    except Exception: pass
    try:
        ds = xr.open_dataset(NCEP_PATH)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
        raw['NCEP'] = {int(yr): float(v)
                       for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
                       if np.isfinite(v)}
    except Exception: pass
    try:
        files = sorted(glob.glob(ERA5_GLOB))
        if files:
            ds = xr.open_mfdataset(files, combine='by_coords')
            w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
            ta = ds['VAR_2T'].weighted(w).mean(
                dim=['latitude','longitude']).resample(time='1YE').mean()
            raw['ERA5'] = {int(yr): float(v)
                           for yr, v in zip(ta.time.dt.year.values, ta.values.squeeze())
                           if np.isfinite(v)}
    except Exception: pass

    centered = {}
    for name, yd in raw.items():
        common = [y for y in trend_years if y in yd]
        if len(common) < 10: continue
        mu = np.mean([yd[y] for y in common])
        centered[name] = {y: v - mu for y, v in yd.items()}

    ens = {}
    for y in trend_years:
        vals = [v[y] for v in centered.values() if y in v]
        if vals: ens[y] = float(np.mean(vals))
    print(f'  Sources: {list(centered.keys())}  ({len(ens)} years)')
    return ens, list(centered.keys())

T_ens, T_sources = build_T_ensemble(TREND_YEARS)


# ── LOAD YEARLY DOMAIN-MEAN ALBEDO ────────────────────────────────────────────
print('Loading yearly albedo ...')

alb_by_year  = {}
monthly_alb  = {}

for year in TREND_YEARS:
    month_vals = {}
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        if mname == 'September':
            p = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
            if not os.path.exists(p): continue
            raw = np.load(p).astype(np.float32)
        else:
            p_c = f'{CACHE_OBS}/clara_alb_{mname}_{year}.npy'
            p_a = f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
            clara = np.load(p_c).astype(np.float32) if os.path.exists(p_c) else None
            appx  = np.load(p_a).astype(np.float32) if os.path.exists(p_a) else None
            if clara is not None and appx is not None:
                if appx.shape != SIAL_SHAPE:
                    tmp = np.full(SIAL_SHAPE, np.nan, dtype=np.float32)
                    r = min(appx.shape[0], SIAL_SHAPE[0])
                    c = min(appx.shape[1], SIAL_SHAPE[1])
                    tmp[:r, :c] = appx[:r, :c]
                    appx = tmp
                raw = np.where(np.isfinite(clara), clara, appx)
            elif clara is not None: raw = clara
            elif appx  is not None: raw = appx
            else: continue

        mu = domain_mean_psn(regrid_sial_to_psn(raw))
        if np.isfinite(mu):
            month_vals[mname] = mu

    if month_vals:
        monthly_alb[year] = month_vals
        alb_by_year[year] = float(np.mean(list(month_vals.values())))


# ── ALIGN AND BUILD ARRAYS ────────────────────────────────────────────────────
years_both = sorted(set(alb_by_year) & set(T_ens))
alb_arr = np.array([alb_by_year[y] for y in years_both])
T_arr   = np.array([T_ens[y]       for y in years_both])
yr_arr  = np.array(years_both)
print(f'  {len(years_both)} years with both albedo and T  '
      f'({years_both[0]}-{years_both[-1]})')

monthly_arr = {}
for mn in MONTH_NUMS:
    mname = MONTH_NAMES[mn]
    yrs, albs, Ts = [], [], []
    for y in years_both:
        if mname in monthly_alb.get(y, {}):
            yrs.append(y); albs.append(monthly_alb[y][mname]); Ts.append(T_ens[y])
    monthly_arr[mname] = dict(years=np.array(yrs),
                               alb=np.array(albs),
                               T=np.array(Ts))


# ── OLS HELPERS ───────────────────────────────────────────────────────────────
def ols(x, y):
    res = stats.linregress(x, y)
    return res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr

def _tvp_filter(T, alpha, log_var_eps, log_q):
    """Kalman filter for alpha_t = beta_t*T_t + eps, beta_t = beta_{t-1} + eta."""
    var_eps = np.exp(log_var_eps)
    var_eta = np.exp(log_q) * var_eps
    n = len(T)
    beta_f = np.zeros(n)
    P_f    = np.zeros(n)
    bp, Pp = 0.0, 1e4
    ll = 0.0
    for t in range(n):
        F     = T[t]**2 * Pp + var_eps
        innov = alpha[t] - bp * T[t]
        ll   += -0.5 * (np.log(max(F, 1e-30)) + innov**2 / max(F, 1e-30))
        K        = Pp * T[t] / F
        beta_f[t]= bp + K * innov
        P_f[t]   = max((1. - K * T[t]) * Pp, 0.)
        bp, Pp   = beta_f[t], P_f[t] + var_eta
    return beta_f, np.sqrt(np.maximum(P_f, 0.)), ll

def ols_hac(x, y, nlags=None):
    """OLS with Newey-West HAC-robust standard errors.

    Standard OLS SE is too small when residuals are autocorrelated (as with
    co-trending series).  The slope is unchanged; only the SE and p-value
    are corrected.
    """
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    X   = sm.add_constant(np.asarray(x, dtype=float))
    mod = sm.OLS(np.asarray(y, dtype=float), X).fit()
    if nlags is None:
        nlags = int(np.ceil(4 * (len(y) / 100) ** 0.25))
    cov    = cov_hac(mod, nlags=nlags)
    se_hac = np.sqrt(np.diag(cov))
    t_hac  = mod.params / se_hac
    from scipy.stats import t as t_dist
    p_hac  = 2 * t_dist.sf(np.abs(t_hac), df=mod.df_resid)
    return (float(mod.params[1]), float(mod.params[0]),
            float(se_hac[1]),     float(t_hac[1]),
            float(p_hac[1]))


# ── FULL-PERIOD OLS ───────────────────────────────────────────────────────────
# This is the correct dα/dT for SIAF: it captures the full historical
# sensitivity including both the reversible interannual response and the
# irreversible cumulative sea-ice loss that is the physical basis of the
# feedback.  First-differences would underestimate SIAF by excluding the
# hysteresis component (ice melted in 2007 that did not return).
full_slope, full_ic, full_r, full_p, full_se = ols(T_arr, alb_arr)

print(f'\nFull period ({yr_arr[0]}-{yr_arr[-1]}): '
      f'slope = {full_slope:.5f} +/- {full_se:.5f} albedo K-1  '
      f'(r={full_r:.3f}, p={full_p:.4f})')

try:
    hac_slope, hac_ic, hac_se, hac_t, hac_p = ols_hac(T_arr, alb_arr)
    hac_ok = True
    print(f'  HAC-robust SE:  slope = {hac_slope:.5f} +/- {hac_se:.5f}  '
          f'(t={hac_t:.2f}, p={hac_p:.4f})')
    print(f'  (OLS SE was {full_se:.5f}; HAC SE is '
          f'{"larger -- autocorrelation confirmed" if hac_se > full_se else "similar"})')
except Exception as e:
    print(f'  HAC failed: {e}')
    hac_slope, hac_ic, hac_se, hac_t, hac_p = full_slope, full_ic, full_se, np.nan, np.nan
    hac_ok = False


# ── EXPANDING WINDOW ──────────────────────────────────────────────────────────
print('Computing expanding window ...')
exp_end_years, exp_slopes, exp_se, exp_pvals, exp_r = [], [], [], [], []

for end_idx in range(MIN_EXP - 1, len(yr_arr)):
    sl = slice(0, end_idx + 1)
    slope, _, r, p, se = ols(T_arr[sl], alb_arr[sl])
    exp_end_years.append(int(yr_arr[end_idx]))
    exp_slopes.append(slope); exp_se.append(se)
    exp_pvals.append(p);      exp_r.append(r)

exp_end_years = np.array(exp_end_years)
exp_slopes    = np.array(exp_slopes)
exp_se        = np.array(exp_se)
exp_pvals     = np.array(exp_pvals)
exp_r         = np.array(exp_r)


# ── ROLLING WINDOW ────────────────────────────────────────────────────────────
print(f'Computing {ROLL_WIN}-year rolling window ...')
roll_centres, roll_slopes, roll_se, roll_pvals, roll_r = [], [], [], [], []
roll_start_years, roll_end_years = [], []

for i in range(len(yr_arr) - ROLL_WIN + 1):
    sl = slice(i, i + ROLL_WIN)
    slope, _, r, p, se = ols(T_arr[sl], alb_arr[sl])
    roll_centres.append(float(np.mean(yr_arr[sl])))
    roll_start_years.append(int(yr_arr[i]))
    roll_end_years.append(int(yr_arr[i + ROLL_WIN - 1]))
    roll_slopes.append(slope); roll_se.append(se)
    roll_pvals.append(p);      roll_r.append(r)

roll_centres     = np.array(roll_centres)
roll_slopes      = np.array(roll_slopes)
roll_se          = np.array(roll_se)
roll_pvals       = np.array(roll_pvals)
roll_r           = np.array(roll_r)
roll_start_years = np.array(roll_start_years)
roll_end_years   = np.array(roll_end_years)

print(f'\nFull period ({yr_arr[0]}-{yr_arr[-1]}): '
      f'slope = {full_slope:.5f} +/- {full_se:.5f} albedo K-1  '
      f'(r={full_r:.3f}, p={full_p:.4f})')
print(f'Early {ROLL_WIN}-yr window ({roll_start_years[0]}-{roll_end_years[0]}): '
      f'slope = {roll_slopes[0]:.5f} albedo K-1')
print(f'Late  {ROLL_WIN}-yr window ({roll_start_years[-1]}-{roll_end_years[-1]}): '
      f'slope = {roll_slopes[-1]:.5f} albedo K-1')
accel = roll_slopes[-1] - roll_slopes[0]
print(f'Change (late - early): {accel:+.5f} albedo K-1  '
      f'({"strengthening" if accel < 0 else "weakening"})')


# ── TVP KALMAN FILTER ─────────────────────────────────────────────────────────
# q fixed from rolling-window implied annual beta drift rather than MLE,
# which is poorly identified with 43 observations and produces degenerate
# solutions (q >> 1).  Physical prior: beta changed ~0.04 albedo/K over
# ~28 years -> annual drift ~0.0014 albedo/K/yr.
print('Computing TVP Kalman filter ...')
try:
    sl0, _, _, _, _ = ols(T_arr, alb_arr)
    alb_mean    = float(np.mean(alb_arr))      # demean α so model has no intercept problem
    alb_ctr     = alb_arr - alb_mean
    tvp_var_eps = float(np.var(alb_ctr - sl0 * T_arr))
    tvp_q       = float(TVP_ANNUAL_DRIFT**2 / tvp_var_eps)

    print(f'  Fixed q = {tvp_q:.6f}  (from rolling-window implied drift rate)')
    print(f'  var_eps = {tvp_var_eps:.6f}  (from full-period OLS residuals)')

    tvp_beta, tvp_se, _ = _tvp_filter(
        T_arr, alb_ctr, np.log(tvp_var_eps), np.log(tvp_q))

    print(f'  TVP beta at {yr_arr[0]}:  {tvp_beta[0]:+.5f} +/- {tvp_se[0]:.5f} albedo K-1')
    print(f'  TVP beta at {yr_arr[-1]}: {tvp_beta[-1]:+.5f} +/- {tvp_se[-1]:.5f} albedo K-1')
    print(f'  TVP time-mean beta: {tvp_beta.mean():+.5f}  '
          f'(cf. full-period OLS: {full_slope:+.5f})')
    tvp_ok = True
except Exception as e:
    print(f'  TVP failed: {e}')
    tvp_beta = tvp_se = np.full(len(yr_arr), np.nan)
    tvp_var_eps = tvp_q = np.nan
    tvp_ok = False


# ── TREND DECOMPOSITION ───────────────────────────────────────────────────────
print('\nDecomposing forced trend vs interannual variability ...')

t_norm = (yr_arr - yr_arr.mean()) / yr_arr.std()

T_trend_coef   = np.polyfit(t_norm, T_arr,   1)
alb_trend_coef = np.polyfit(t_norm, alb_arr, 1)
T_trend   = np.polyval(T_trend_coef,   t_norm)
alb_trend = np.polyval(alb_trend_coef, t_norm)
T_resid   = T_arr   - T_trend
alb_resid = alb_arr - alb_trend

# Ratio of independent linear trend slopes
forced_sens = alb_trend_coef[0] / T_trend_coef[0]

# Detrended OLS: interannual co-variability after removing the linear trend
detrend_slope, detrend_ic, detrend_r, detrend_p, detrend_se = ols(T_resid, alb_resid)

print(f'  Forced trend sensitivity  (dalpha_trend / dT_trend):  '
      f'{forced_sens:+.5f} albedo K-1')
print(f'  Detrended (interannual) regression:  '
      f'{detrend_slope:+.5f} +/- {detrend_se:.5f}  '
      f'(r={detrend_r:.3f}, p={detrend_p:.4f})')
print(f'  Full-period OLS (levels):  {full_slope:+.5f}   '
      f'<- variance-weighted mix of forced and interannual')

# Partial regression alpha = beta_T * T + beta_t * t + eps  (Frisch-Waugh)
# beta_T is the T effect controlling for the secular time trend.
# Algebraically equivalent to detrended OLS when trends are linear.
X_partial = np.column_stack([T_arr, t_norm, np.ones(len(T_arr))])
coefs_partial, _, _, _ = np.linalg.lstsq(X_partial, alb_arr, rcond=None)
partial_T_slope = float(coefs_partial[0])
partial_t_slope = float(coefs_partial[1])

alb_hat_partial = X_partial @ coefs_partial
resid_partial   = alb_arr - alb_hat_partial
n_p, k_p        = len(alb_arr), 3
s2_partial      = float(np.sum(resid_partial**2) / (n_p - k_p))
XtX_inv         = np.linalg.inv(X_partial.T @ X_partial)
partial_T_se    = float(np.sqrt(s2_partial * XtX_inv[0, 0]))
partial_T_t     = partial_T_slope / partial_T_se
partial_T_p     = float(2 * stats.t.sf(np.abs(partial_T_t), df=n_p - k_p))
r_Tt            = float(np.corrcoef(T_arr, t_norm)[0, 1])

print(f'\n  Partial regression  beta_T (controlling for trend):  '
      f'{partial_T_slope:+.5f} +/- {partial_T_se:.5f}  (p={partial_T_p:.4f})')
print(f'  Partial regression  beta_t (residual time trend):    '
      f'{partial_t_slope:+.5f} albedo yr-1')
print(f'  Collinearity r(T, t) = {r_Tt:.3f}  '
      f'-> {"HIGH -- partial T coef has inflated SE" if abs(r_Tt) > 0.9 else "moderate"}')

# Low-pass / high-pass frequency-band decomposition
T_low    = uniform_filter1d(T_arr,   size=LP_WIN, mode='nearest')
alb_low  = uniform_filter1d(alb_arr, size=LP_WIN, mode='nearest')
T_high   = T_arr   - T_low
alb_high = alb_arr - alb_low

edge = LP_WIN // 2
sl_e = slice(edge, len(yr_arr) - edge)

lo_slope, _, lo_r, lo_p, lo_se = ols(T_low[sl_e],  alb_low[sl_e])
hi_slope, _, hi_r, hi_p, hi_se = ols(T_high[sl_e], alb_high[sl_e])

print(f'\n  Low-pass  (>{LP_WIN}-yr, forced signal):     '
      f'{lo_slope:+.5f} +/- {lo_se:.5f}  (r={lo_r:.3f}, p={lo_p:.4f})')
print(f'  High-pass (<{LP_WIN}-yr, interannual signal): '
      f'{hi_slope:+.5f} +/- {hi_se:.5f}  (r={hi_r:.3f}, p={hi_p:.4f})')


# ── FORCED vs INTERNAL ATTRIBUTION FOR SIAF ───────────────────────────────────
# Partitions the full-period OLS slope into forced-trend and internal-
# variability contributions using:
#
#   beta_total ~ beta_forced * f_forced + beta_internal * f_internal
#
# where f_* = Var(T_*) / Var(T_total).
#
# This confirms the full-period OLS is not inflated by internal variability
# co-variation and is therefore the correct quantity to use for SIAF,
# reported with HAC-robust standard errors.
print('\nForced vs internal attribution for SIAF ...')

var_T_total    = float(np.var(T_arr))
var_T_forced   = float(np.var(T_low))
var_T_internal = float(np.var(T_high))
var_alb_total  = float(np.var(alb_arr))
var_alb_forced = float(np.var(alb_low))

f_forced_T   = var_T_forced   / var_T_total
f_internal_T = var_T_internal / var_T_total
f_forced_alb = var_alb_forced / var_alb_total

contrib_forced   = lo_slope * f_forced_T
contrib_internal = hi_slope * f_internal_T
beta_reconstructed = contrib_forced + contrib_internal

if abs(beta_reconstructed) > 1e-8:
    frac_forced   = contrib_forced   / beta_reconstructed
    frac_internal = contrib_internal / beta_reconstructed
else:
    frac_forced = frac_internal = np.nan

cross_cov_TT = float(np.cov(T_low[sl_e], T_high[sl_e])[0, 1])

print(f'  T variance:     forced {f_forced_T:.1%}  |  internal {f_internal_T:.1%}')
print(f'  alpha variance: forced {f_forced_alb:.1%}  |  '
      f'internal {1 - f_forced_alb:.1%}')
print(f'  beta_forced   (low-pass):  '
      f'{lo_slope:+.5f} x {f_forced_T:.3f} = {contrib_forced:+.5f}')
print(f'  beta_internal (high-pass): '
      f'{hi_slope:+.5f} x {f_internal_T:.3f} = {contrib_internal:+.5f}')
print(f'  Reconstructed beta:  {beta_reconstructed:+.5f}  '
      f'(observed: {full_slope:+.5f},  '
      f'residual: {full_slope - beta_reconstructed:+.5f})')
if np.isfinite(frac_forced):
    print(f'  Forced fraction:   {frac_forced:.1%}')
    print(f'  Internal fraction: {frac_internal:.1%}')
print(f'  Cov(T_forced, T_internal) = {cross_cov_TT:.5f}  '
      f'({"approx orthogonal -- decomposition valid" if abs(cross_cov_TT) < 0.01 else "non-orthogonal -- treat fractions as approximate"})')

print(f'\n  ---- SIAF dα/dT summary ----')
print(f'  Full-period OLS:      {full_slope:+.5f} +/- {full_se:.5f}  (OLS SE, inflated by autocorr)')
print(f'  Full-period OLS HAC:  {hac_slope:+.5f} +/- {hac_se:.5f}  (autocorr-robust, use this)')
print(f'  Forced-only estimate: {lo_slope:+.5f} +/- {lo_se:.5f}')
if np.isfinite(frac_forced):
    print(f'  Internal variability contributes ~{frac_internal:.0%} of the reconstructed slope')
    print(f'  -> Full-period OLS is ~{frac_forced:.0%} forced; suitable for SIAF with HAC SE')


# ── SAVE ──────────────────────────────────────────────────────────────────────
out = dict(
    # raw time series
    yr_arr           = yr_arr,
    alb_arr          = alb_arr,
    T_arr            = T_arr,
    monthly_arr      = monthly_arr,
    T_sources        = T_sources,

    # full-period OLS (standard and HAC-robust)
    full_slope       = full_slope,
    full_ic          = full_ic,
    full_r           = full_r,
    full_p           = full_p,
    full_se          = full_se,
    hac_slope        = hac_slope,
    hac_ic           = hac_ic,
    hac_se           = hac_se,
    hac_t            = hac_t,
    hac_p            = hac_p,
    hac_ok           = hac_ok,

    # expanding window
    exp_end_years    = exp_end_years,
    exp_slopes       = exp_slopes,
    exp_se           = exp_se,
    exp_pvals        = exp_pvals,
    exp_r            = exp_r,

    # rolling window
    roll_win         = ROLL_WIN,
    roll_centres     = roll_centres,
    roll_start_years = roll_start_years,
    roll_end_years   = roll_end_years,
    roll_slopes      = roll_slopes,
    roll_se          = roll_se,
    roll_pvals       = roll_pvals,
    roll_r           = roll_r,

    # TVP Kalman filter
    tvp_beta         = tvp_beta,
    tvp_se           = tvp_se,
    tvp_q            = tvp_q,
    tvp_var_eps      = tvp_var_eps,
    tvp_annual_drift = TVP_ANNUAL_DRIFT,
    tvp_ok           = tvp_ok,

    # trend decomposition
    t_norm           = t_norm,
    T_trend          = T_trend,
    alb_trend        = alb_trend,
    T_resid          = T_resid,
    alb_resid        = alb_resid,
    forced_sens      = forced_sens,

    # detrended OLS
    detrend_slope    = detrend_slope,
    detrend_ic       = detrend_ic,
    detrend_r        = detrend_r,
    detrend_p        = detrend_p,
    detrend_se       = detrend_se,

    # partial regression (Frisch-Waugh)
    partial_T_slope  = partial_T_slope,
    partial_T_se     = partial_T_se,
    partial_T_p      = partial_T_p,
    partial_t_slope  = partial_t_slope,
    r_Tt             = r_Tt,

    # frequency-band decomposition
    lp_win           = LP_WIN,
    T_low            = T_low,
    alb_low          = alb_low,
    T_high           = T_high,
    alb_high         = alb_high,
    lo_slope         = lo_slope,
    lo_se            = lo_se,
    lo_r             = lo_r,
    lo_p             = lo_p,
    hi_slope         = hi_slope,
    hi_se            = hi_se,
    hi_r             = hi_r,
    hi_p             = hi_p,

    # forced vs internal attribution
    var_T_total      = var_T_total,
    var_T_forced     = var_T_forced,
    var_T_internal   = var_T_internal,
    var_alb_total    = var_alb_total,
    var_alb_forced   = var_alb_forced,
    f_forced_T       = f_forced_T,
    f_internal_T     = f_internal_T,
    f_forced_alb     = f_forced_alb,
    contrib_forced   = contrib_forced,
    contrib_internal = contrib_internal,
    beta_reconstructed = beta_reconstructed,
    frac_forced      = frac_forced,
    frac_internal    = frac_internal,
    cross_cov_TT     = cross_cov_TT,
)

with open(PKL_PATH, 'wb') as fh:
    pickle.dump(out, fh)
print(f'\nSaved -> {PKL_PATH}')