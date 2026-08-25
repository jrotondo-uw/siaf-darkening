#!/usr/bin/env python
"""
compute_siaf_covariance.py
==========================
Loads pixel-level albedo and SIC caches from siaf_obs_analysis,
computes the SIC-driven vs. residual darkening decomposition for
extent pixels, and saves plot_data_covariance.pkl for the notebook.

Inputs  (sensitivity_cache_obs):
  clara_alb_{Month}_{year}.npy   APP-x: appx_alb_{Month}_{year}.npy
  sic_{trend_slope,trend_pval,early,late}_sial_{Month}.npy

Output:
  {OUT_DIR}/plot_data_covariance.pkl
"""

import os, gc, pickle
import numpy as np
from scipy.ndimage import zoom

# ── paths ──────────────────────────────────────────────────────────────
CACHE_OBS = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_DIR   = '/glade/work/jrotondo/ncc_paper/plots/extended_data_figs/'
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────
EARLY_YEARS   = list(range(1982, 1992))
LATE_YEARS    = list(range(2015, 2025))
MONTHS        = {4:'April', 5:'May', 6:'June',
                 7:'July',  8:'August', 9:'September'}
PVAL_THRESH   = 0.05
SIC_THRESHOLD = 0.15

# Products to average for albedo mean (both used for all months)
PRODUCTS = {
    'CLARA-A3': 'clara_alb_{mname}_{year}.npy',
    'APP-x':    'appx_alb_{mname}_{year}.npy',
}

# ── helpers ─────────────────────────────────────────────────────────────
def load_alb_mean(cache_dir, mname, years):
    """Average across years and products; returns NaN where no data."""
    stack_sum = None
    stack_cnt = None
    for product, pattern in PRODUCTS.items():
        for y in years:
            p = os.path.join(cache_dir, pattern.format(mname=mname, year=y))
            if not os.path.exists(p):
                continue
            a   = np.load(p).astype(np.float32)
            fin = np.isfinite(a)
            a   = np.where(fin, a, 0.0)
            stack_sum = a.copy()              if stack_sum is None else stack_sum + a
            stack_cnt = fin.astype(np.int32)  if stack_cnt is None else stack_cnt + fin.astype(np.int32)
    if stack_sum is None:
        return None, None, None
    with np.errstate(invalid='ignore'):
        mean = np.where(stack_cnt > 0, stack_sum / stack_cnt, np.nan).astype(np.float32)

    # per-product early/late means for Δα_ice, Δα_ocn estimation
    return mean, stack_sum, stack_cnt


def load_alb_mean_by_period(cache_dir, mname, years):
    """
    Temporal mean albedo for a given month + year set.
      September        → APP-x only
      All other months → CLARA-A3 primary; NaN pixels filled from APP-x
                         (APP-x regridded to CLARA-A3 shape if needed)
    """
    def _stack_mean(pattern):
        s, c = None, None
        for y in years:
            p = os.path.join(cache_dir, pattern.format(mname=mname, year=y))
            if not os.path.exists(p):
                continue
            a   = np.load(p).astype(np.float32)
            fin = np.isfinite(a)
            a   = np.where(fin, a, 0.0)
            s   = a.copy()             if s is None else s + a
            c   = fin.astype(np.int32) if c is None else c + fin.astype(np.int32)
        if s is None:
            return None
        return np.where(c > 0, s / c, np.nan).astype(np.float32)

    if mname == 'September':
        return _stack_mean('appx_alb_{mname}_{year}.npy')

    clara = _stack_mean('clara_alb_{mname}_{year}.npy')
    appx  = _stack_mean('appx_alb_{mname}_{year}.npy')

    if clara is None:
        return appx
    if appx is None:
        return clara

    # Regrid APP-x to CLARA-A3 shape if they differ before pixel-level fill
    if appx.shape != clara.shape:
        appx = regrid_to(appx, clara.shape)

    return np.where(np.isfinite(clara), clara, appx)

def load_sic(cache_dir, mname):
    keys = ('trend_slope', 'trend_pval', 'early', 'late')
    out  = {}
    for k in keys:
        p = os.path.join(cache_dir, f'sic_{k}_sial_{mname}.npy')
        if not os.path.exists(p):
            print(f'  missing: {p}')
            return None
        out[k] = np.load(p).astype(np.float32)
    return out


def regrid_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr.astype(np.float64), factors, order=0,
                mode='nearest').astype(np.float32)


def classify_pixels(sic):
    slope = sic['trend_slope']; pval = sic['trend_pval']
    sic_e = sic['early'];       sic_l = sic['late']
    valid  = np.isfinite(slope) & np.isfinite(pval)
    extent = valid & (slope < 0) & (pval < PVAL_THRESH)
    stable = valid & ~extent & (np.fmax(sic_e, sic_l) >= SIC_THRESHOLD)
    return extent, stable


def alpha_stats_for_period(cache_dir, mname, years, sic2d, mask):
    """
    Pixel-level ice/ocean classification using SIC for a given period;
    returns mean albedo over ice and ocean pixels within mask.
    """
    alb = load_alb_mean_by_period(cache_dir, mname, years)
    if alb is None:
        return np.nan, np.nan
    alb = regrid_to(alb, sic2d.shape)
    fin = np.isfinite(alb) & np.isfinite(sic2d) & mask
    ice = fin & (sic2d >= SIC_THRESHOLD)
    ocn = fin & (sic2d <  SIC_THRESHOLD)
    a_ice = float(np.nanmean(alb[ice])) if ice.sum() > 10 else np.nan
    a_ocn = float(np.nanmean(alb[ocn])) if ocn.sum() > 10 else np.nan
    return a_ice, a_ocn


# ── main computation ─────────────────────────────────────────────────────
monthly_res_ext  = {}   # residual Δα values in extent pixels (for KDE)
monthly_dalb_drk = {}   # ΔSIAL values in stable-darkening pixels (for dotted KDE)
monthly_dalb_ext = {}   # mean total ΔSIAL in extent pixels
monthly_dalb_sic = {}   # mean SIC-driven component
monthly_dalb_res = {}   # mean residual component
monthly_meta     = {}   # alpha_ice/ocn used for each month

for mn, mname in MONTHS.items():
    print(f'Processing {mname} ...', flush=True)
    ae  = load_alb_mean_by_period(CACHE_OBS, mname, EARLY_YEARS)
    al  = load_alb_mean_by_period(CACHE_OBS, mname, LATE_YEARS)
    sic = load_sic(CACHE_OBS, mname)
    if ae is None or al is None or sic is None:
        print(f'  skipping {mname} (missing data)')
        continue

    dalb = (al - ae).astype(np.float32)
    for key in sic:
        sic[key] = regrid_to(sic[key], dalb.shape)
    dsic = (sic['late'] - sic['early']).astype(np.float32)

    mask_ext, mask_stb = classify_pixels(sic)
    valid = np.isfinite(dalb) & np.isfinite(dsic)

    # Derive α_ice / α_ocn from the data itself rather than hard-coding
    arctic_mask = np.ones(dalb.shape, dtype=bool)   # use full domain
    a_ice_e, a_ocn_e = alpha_stats_for_period(CACHE_OBS, mname, EARLY_YEARS,
                                               sic['early'], arctic_mask)
    a_ice_l, a_ocn_l = alpha_stats_for_period(CACHE_OBS, mname, LATE_YEARS,
                                               sic['late'],  arctic_mask)
    # Use average of early and late for the decomposition coefficient
    alpha_ice = np.nanmean([a_ice_e, a_ice_l]) if np.isfinite(a_ice_e) else 0.60
    alpha_ocn = np.nanmean([a_ocn_e, a_ocn_l]) if np.isfinite(a_ocn_e) else 0.06
    print(f'  α_ice={alpha_ice:.3f}  α_ocn={alpha_ocn:.3f}', flush=True)

    dalb_sic = (alpha_ice - alpha_ocn) * dsic
    dalb_res = dalb - dalb_sic

    m_ext = mask_ext & valid
    m_stb = mask_stb & valid

    if m_ext.sum() > 50:
        monthly_res_ext[mn]  = dalb_res[m_ext].ravel()
        monthly_dalb_ext[mn] = float(np.nanmean(dalb[m_ext]))
        monthly_dalb_sic[mn] = float(np.nanmean(dalb_sic[m_ext]))
        monthly_dalb_res[mn] = float(np.nanmean(dalb_res[m_ext]))
        monthly_meta[mn]     = dict(alpha_ice=alpha_ice, alpha_ocn=alpha_ocn,
                                    n_ext_px=int(m_ext.sum()))
        print(f'  extent px: {m_ext.sum():,}  '
              f'Δα_tot={monthly_dalb_ext[mn]:+.4f}  '
              f'Δα_sic={monthly_dalb_sic[mn]:+.4f}  '
              f'Δα_res={monthly_dalb_res[mn]:+.4f}')
    else:
        print(f'  too few extent pixels ({m_ext.sum()}) — skipping')

    if m_stb.sum() > 50:
        monthly_dalb_drk[mn] = dalb[m_stb].ravel()
        print(f'  stable-darkening px: {m_stb.sum():,}')

    del ae, al, sic, dalb, dsic, dalb_sic, dalb_res
    gc.collect()

# ── save ─────────────────────────────────────────────────────────────────
PKL_OUT = os.path.join(OUT_DIR, 'plot_data_covariance.pkl')
with open(PKL_OUT, 'wb') as fh:
    pickle.dump({
        'monthly_res_ext':  monthly_res_ext,
        'monthly_dalb_drk': monthly_dalb_drk,
        'monthly_dalb_ext': monthly_dalb_ext,
        'monthly_dalb_sic': monthly_dalb_sic,
        'monthly_dalb_res': monthly_dalb_res,
        'monthly_meta':     monthly_meta,
        'MONTHS':           MONTHS,
        'PVAL_THRESH':      PVAL_THRESH,
        'SIC_THRESHOLD':    SIC_THRESHOLD,
        'EARLY_YEARS':      EARLY_YEARS,
        'LATE_YEARS':       LATE_YEARS,
    }, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f'\nSaved → {PKL_OUT}')