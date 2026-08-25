#!/usr/bin/env python
"""
compute_sic_trend_maps.py
=========================
Loads SIC trend + albedo cache arrays for obs and CMIP6 experiments,
classifies pixels as extent-loss vs surface-darkening at each p-value
threshold, and saves plot_data_sic_trend_maps.pkl for the notebook.

Inputs  (sensitivity_cache_{obs,cmip6}/):
  arctic_60N_sial.npy
  sic_{trend_slope,trend_pval,early,late}_sial_{Month}.npy

Output:
  {PLOT_DIR}/plot_data_sic_trend_maps.pkl
"""

import os, pickle
import numpy as np
import xarray as xr
import glob

# ── paths ──────────────────────────────────────────────────────────────────
DIR_CLARA = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)

EXPERIMENTS = [
    ('obs',   'Observational Era (1982–2024)',
     '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'),
    ('cmip6', 'CMIP6 Historical (1982–2014)',
     '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'),
]

MONTHS          = ['April', 'May', 'June', 'July', 'August', 'September']
PVAL_THRESHOLDS = [0.01, 0.05, 0.10]
SIC_ICE_THRESHOLD = 0.15

# ── SIAL grid (lat/lon) ─────────────────────────────────────────────────────
sample_nc = sorted(glob.glob(f'{DIR_CLARA}/*.nc'))[0]
with xr.open_dataset(sample_nc) as ds:
    s_lat = ds['lat'].values
    s_lon = ds['lon'].values
if s_lat.ndim == 1:
    lons, lats = np.meshgrid(s_lon, s_lat)
else:
    lons, lats = s_lon, s_lat

print(f'SIAL grid shape: {lats.shape}')

# ── compute sig_maps ────────────────────────────────────────────────────────
# sig_maps[(exp_name, mname, pthresh)] = 2-D float array
#   1.0 = significant negative SIC trend (extent loss)
#   0.0 = non-significant or positive trend (surface darkening / other)
#   NaN = no ice / outside domain

sig_maps = {}
meta     = {}  # pixel counts per (exp, month, pthresh)

for exp_name, exp_label, cache_dir in EXPERIMENTS:
    print(f'\n{exp_name}  ({exp_label})')
    arctic_60N = np.load(f'{cache_dir}/arctic_60N_sial.npy').astype(bool)

    for mname in MONTHS:
        slope   = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32)
        pval    = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32)
        early   = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32)
        late    = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32)
        sic_max = np.fmax(early, late)

        has_ice = arctic_60N & np.isfinite(slope) & (sic_max >= SIC_ICE_THRESHOLD)
        print(f'  {mname}: {int(has_ice.sum()):,} ice pixels')

        for pthresh in PVAL_THRESHOLDS:
            sig = np.full(slope.shape, np.nan, dtype=np.float32)
            sig_mask = has_ice & (pval < pthresh) & (slope < 0)
            drk_mask = has_ice & ~((pval < pthresh) & (slope < 0))
            sig[sig_mask] = 1.0
            sig[drk_mask] = 0.0

            sig_maps[(exp_name, mname, pthresh)] = sig

            n_all = int(np.isfinite(sig).sum())
            n_ext = int((sig == 1.0).sum())
            n_drk = int((sig == 0.0).sum())
            meta[(exp_name, mname, pthresh)] = dict(
                n_all=n_all, n_ext=n_ext, n_drk=n_drk,
                pct_ext=100 * n_ext / n_all if n_all > 0 else 0,
                pct_drk=100 * n_drk / n_all if n_all > 0 else 0,
            )
            print(f'    p<{pthresh:.2f}: ext={n_ext:,} ({100*n_ext/n_all if n_all else 0:.0f}%)  '
                  f'drk={n_drk:,} ({100*n_drk/n_all if n_all else 0:.0f}%)')

# ── save ────────────────────────────────────────────────────────────────────
PKL_OUT = os.path.join(PLOT_DIR, 'plot_data_sic_trend_maps.pkl')
with open(PKL_OUT, 'wb') as fh:
    pickle.dump({
        'sig_maps':        sig_maps,
        'meta':            meta,
        'lons':            lons,
        'lats':            lats,
        'EXPERIMENTS':     EXPERIMENTS,
        'MONTHS':          MONTHS,
        'PVAL_THRESHOLDS': PVAL_THRESHOLDS,
        'SIC_ICE_THRESHOLD': SIC_ICE_THRESHOLD,
    }, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f'\nSaved → {PKL_OUT}')