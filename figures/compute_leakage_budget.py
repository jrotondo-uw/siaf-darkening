#!/usr/bin/env python
"""
compute_leakage_budget.py
=========================
Rebuilds the data behind Supplementary Fig. 3, the ABSOLUTE two-sided
classification leakage budget, which previously had no generator.

It reuses compute_siaf_covariance.py's loaders, masks and epochs verbatim so
that the two supplementary figures rest on exactly the same decomposition:

    dalb      = late-epoch minus early-epoch SIAL
    dsic      = late-epoch minus early-epoch SIC
    dalb_sic  = (alpha_ice - alpha_ocn) * dsic      SIC-driven part
    dalb_res  = dalb - dalb_sic                     residual part

The two leakage directions are
    blue   residual darkening inside the declining-ice region   mean(dalb_res | Omega_area)
    orange concentration change inside the darkening region     mean(dalb_sic | Omega_drk)

computed twice: with the month's own ice-ocean contrast, and with the fixed
season-mean contrast 0.54 used as a benchmark elsewhere in the paper.

Writes leakage_budget.csv; makes no figure. Diagnostic only, no core hours.
"""
import os
import gc
import json
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

CACHE_OBS = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_CSV = ('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/'
           'revision_20261028/leakage_budget.csv')
MONTHLY_CSV = ('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/'
               'revision_20261028/siaf_pixelwise_summary_monthly.csv')

EARLY_YEARS = list(range(1982, 1992))
LATE_YEARS = list(range(2015, 2025))
MONTHS = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September'}
PVAL_THRESH = 0.05
SIC_THRESHOLD = 0.15
FIXED_CONTRAST = 0.54
PRODUCTS = {'CLARA-A3': 'clara_alb_{mname}_{year}.npy',
            'APP-x': 'appx_alb_{mname}_{year}.npy'}

# published values from the existing fs03, used only as a cross-check
PUB = {'a': dict(res=0.03181, sic=0.00701, net=0.02479),
       'b': dict(res=0.00927, sic=0.00842, net=0.00085)}
PUB_BLUE_A = {4: 0.00085, 5: 0.00365, 6: 0.00660, 7: 0.01047, 8: 0.00808, 9: 0.00207}


def load_alb_mean_by_period(cache_dir, mname, years):
    """Verbatim from compute_siaf_covariance.py so the two supplementary
    figures rest on identical albedo fields:
      September        -> APP-x only
      All other months -> CLARA-A3 primary; NaN pixels filled from APP-x
    """
    def _stack_mean(pattern):
        s, c = None, None
        for y in years:
            p = os.path.join(cache_dir, pattern.format(mname=mname, year=y))
            if not os.path.exists(p):
                continue
            a = np.load(p).astype(np.float32)
            fin = np.isfinite(a)
            a = np.where(fin, a, 0.0)
            s = a.copy() if s is None else s + a
            c = fin.astype(np.int32) if c is None else c + fin.astype(np.int32)
        if s is None:
            return None
        return np.where(c > 0, s / c, np.nan).astype(np.float32)

    if mname == 'September':
        return _stack_mean('appx_alb_{mname}_{year}.npy')

    clara = _stack_mean('clara_alb_{mname}_{year}.npy')
    appx = _stack_mean('appx_alb_{mname}_{year}.npy')
    if clara is None:
        return appx
    if appx is None:
        return clara
    if appx.shape != clara.shape:
        appx = regrid_to(appx, clara.shape)
    return np.where(np.isfinite(clara), clara, appx)


def load_sic(cache_dir, mname):
    out = {}
    for k in ('trend_slope', 'trend_pval', 'early', 'late'):
        p = os.path.join(cache_dir, f'sic_{k}_sial_{mname}.npy')
        if not os.path.exists(p):
            return None
        out[k] = np.load(p).astype(np.float32)
    return out


def regrid_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    f = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr.astype(np.float64), f, order=0, mode='nearest').astype(np.float32)


def classify_pixels(sic):
    slope, pval = sic['trend_slope'], sic['trend_pval']
    valid = np.isfinite(slope) & np.isfinite(pval)
    extent = valid & (slope < 0) & (pval < PVAL_THRESH)
    stable = valid & ~extent & (np.fmax(sic['early'], sic['late']) >= SIC_THRESHOLD)
    return extent, stable


def alpha_stats(cache_dir, mname, years, sic2d):
    alb = load_alb_mean_by_period(cache_dir, mname, years)
    if alb is None:
        return np.nan, np.nan
    alb = regrid_to(alb, sic2d.shape)
    fin = np.isfinite(alb) & np.isfinite(sic2d)
    ice, ocn = fin & (sic2d >= SIC_THRESHOLD), fin & (sic2d < SIC_THRESHOLD)
    return (float(np.nanmean(alb[ice])) if ice.sum() > 10 else np.nan,
            float(np.nanmean(alb[ocn])) if ocn.sum() > 10 else np.nan)


rows = []
for mn, mname in MONTHS.items():
    ae = load_alb_mean_by_period(CACHE_OBS, mname, EARLY_YEARS)
    al = load_alb_mean_by_period(CACHE_OBS, mname, LATE_YEARS)
    sic = load_sic(CACHE_OBS, mname)
    if ae is None or al is None or sic is None:
        print(f'{mname}: missing data, skipped')
        continue

    dalb = (al - ae).astype(np.float32)
    for k in sic:
        sic[k] = regrid_to(sic[k], dalb.shape)
    dsic = (sic['late'] - sic['early']).astype(np.float32)

    mask_ext, mask_stb = classify_pixels(sic)
    valid = np.isfinite(dalb) & np.isfinite(dsic)
    m_ext, m_stb = mask_ext & valid, mask_stb & valid

    a_ice_e, a_ocn_e = alpha_stats(CACHE_OBS, mname, EARLY_YEARS, sic['early'])
    a_ice_l, a_ocn_l = alpha_stats(CACHE_OBS, mname, LATE_YEARS, sic['late'])
    alpha_ice = np.nanmean([a_ice_e, a_ice_l]) if np.isfinite(a_ice_e) else 0.60
    alpha_ocn = np.nanmean([a_ocn_e, a_ocn_l]) if np.isfinite(a_ocn_e) else 0.06
    contrast = alpha_ice - alpha_ocn

    rec = dict(month=mname, mn=mn, alpha_ice=alpha_ice, alpha_ocn=alpha_ocn,
               contrast=contrast, n_ext=int(m_ext.sum()), n_drk=int(m_stb.sum()),
               dalb_ext=float(np.nanmean(dalb[m_ext])),
               dalb_drk=float(np.nanmean(dalb[m_stb])),
               dsic_ext=float(np.nanmean(dsic[m_ext])),
               dsic_drk=float(np.nanmean(dsic[m_stb])))

    for tag, c in (('mon', contrast), ('fix', FIXED_CONTRAST)):
        d_sic = c * dsic
        d_res = dalb - d_sic
        rec[f'res_ext_{tag}'] = float(np.nanmean(d_res[m_ext]))   # blue
        rec[f'sic_drk_{tag}'] = float(np.nanmean(d_sic[m_stb]))   # orange
    rows.append(rec)
    print(f"{mname:10s} contrast={contrast:.3f}  "
          f"res_ext_mon={rec['res_ext_mon']:+.5f}  sic_drk_mon={rec['sic_drk_mon']:+.5f}  "
          f"res_ext_fix={rec['res_ext_fix']:+.5f}  sic_drk_fix={rec['sic_drk_fix']:+.5f}")

    del ae, al, sic, dalb, dsic
    gc.collect()

df = pd.DataFrame(rows)

# geometric (area x annual) weight per month, from the paper's own tables:
# SIAF_global / SIAF_per_area = IS_global / IS_per_area, the RS terms cancel.
m = pd.read_csv(MONTHLY_CSV)
m = m[(m.rs_method == 'CERES') & (m.period == 'CMIP6') & (np.isclose(m.pval, 0.05))]
m = m.set_index('month')
df['G_ext'] = [float(m.loc[r.month, 'SIAF_ext_global'] /
                     m.loc[r.month, 'SIAF_ext_per_area']) for r in df.itertuples()]
df['G_drk'] = [float(m.loc[r.month, 'SIAF_drk_global'] /
                     m.loc[r.month, 'SIAF_drk_per_area']) for r in df.itertuples()]

# implied per-month conversion from the published figure, for diagnosis
df['C_implied'] = [PUB_BLUE_A[r.mn] / abs(r.res_ext_mon) if r.res_ext_mon else np.nan
                   for r in df.itertuples()]
df['C_over_Gext'] = df.C_implied / df.G_ext

df.to_csv(OUT_CSV, index=False)
print('\nwrote', OUT_CSV)
print(df[['month', 'contrast', 'res_ext_mon', 'sic_drk_mon',
          'G_ext', 'C_implied', 'C_over_Gext']].to_string(
              index=False, float_format=lambda x: f'{x:0.5f}'))
print('\nIf C_over_Gext is ~constant across months it is RS, and the published '
      'figure is flux = dalbedo * G * RS.')
print('published annual sums for cross-check:', json.dumps(PUB))
