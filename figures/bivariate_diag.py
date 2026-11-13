#!/usr/bin/env python
"""
Bivariate albedo diagnostic: separate d(alpha)/d(SIC) from d(alpha)/dT.

Replaces the sequential two-regression factorisation (which is circular,
because the reported "albedo response per unit concentration" is the ratio
IS / L rather than a fitted slope) with a single per-pixel bivariate fit:

    alpha_t = a + b * c_t + d * T_t + e_t          c = SIC, T = global mean T

With the paper's sign convention  I = -dalpha/dT,  L = -dc/dT, this identity
is EXACT (Frisch-Waugh-Lovell), not approximate:

    I = b * L + D,        D = -d

    b : albedo change per unit concentration, holding temperature fixed
        (benchmark against the ice-ocean contrast, ~0.54)
    D : albedo change per kelvin, holding concentration fixed
        -> darkening that is NOT concentration-mediated

The residual term vanishes identically because OLS forces e _|_ T, so there
is no closure ratio to explain. Crucially I, L, b and D are all computed on
the SAME valid subset, which the saved sequential slopes were not.

Stages
  1  rebuild monthly SIC on the SIAL grid from the daily PSN25 source and
     verify it reproduces the cached sic_trend_slope/pval  (provenance)
  2  audit each monthly mask: no significant-decline pixel in the darkening
     region, using the rebuilt fields
  3  bivariate fit; verify the identity; region means of I, b*L, D
  4  high-SIC test: regress alpha on T where c > 0.9 (no mixing model, no
     decomposition -- a direct falsification test of the darkening claim)
  5  linearity: slopes binned by SIC decile, plus an optional c^2 term
  6  detrend alpha, c and T in time and refit, to split L and b into
     secular and interannual parts

Writes a CSV per stage plus a single summary.txt. Changes nothing else.

Usage
  python bivariate_diag.py                 # full run (rebuilds SIC, caches it)
  python bivariate_diag.py --months June   # one month, for a fast check
  python bivariate_diag.py --skip-sic      # reuse the cached SIC stack
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sp_stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------- config ---
DIR_SIAL = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_OBS = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/bivariate_diag/'

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

TREND_YEARS = list(range(1982, 2025))
EARLY_YEARS = set(range(1982, 1992))
LATE_YEARS = set(range(2015, 2025))

MONTHS = {4: 'April', 5: 'May', 6: 'June',
          7: 'July', 8: 'August', 9: 'September'}

SIC_THRESH = 0.15          # active-ice threshold, matches the pipeline
SIC_ICE_THRESHOLD = 0.15   # matches build_albedo_cache.py
PVAL = 0.05                # reference classification threshold
MIN_ARCTIC_PX = 1000
MIN_PAIRS = 10             # matches compute_pixel_ols_slopes
HIGH_SIC = [0.80, 0.90]    # stage 4 thresholds
ICE_OCEAN_CONTRAST = 0.54  # benchmark for b, from the leakage budget


# --------------------------------------------------------------- helpers ---
def latlon_to_xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    cl = np.cos(la)
    return np.column_stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)])


def build_nn_index(src_lat, src_lon, tgt_lat, tgt_lon):
    """Nearest source cell for each target cell (same convention as the
    pipeline's build_nn_index)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(latlon_to_xyz(src_lat, src_lon))
    _, idx = tree.query(latlon_to_xyz(tgt_lat, tgt_lon), k=1)
    return idx


MIN_SIC_SD = 0.005   # see note below


def ols_bivariate(y, x1, x2, min_n=MIN_PAIRS, min_sd_x1=MIN_SIC_SD):
    """Per-column OLS of y on [x1, x2] with intercept.

    y, x1 : (T, N) with NaNs.  x2 : (T,) common to all columns.
    Returns b, d, n, r_x1x2, and the univariate slopes of y and x1 on x2
    computed on the IDENTICAL valid subset.

    `min_sd_x1` drops pixels where x1 (concentration) is effectively constant
    over the record. There the partial slope b is not identified: S11 -> 0, so
    b = (S22*S1y - S12*S2y)/det is numerically unbounded even though b*L stays
    finite. Requiring S11 > 0 alone is not enough -- a pixel with sd(SIC) ~1e-6
    passes it and returns |b| ~ 1e3. In practice only BCC-CSM2-MR has such
    pixels (2.8% of its area-loss region, 1.6% of its darkening region; every
    other model has none), and they were enough to drive its area-weighted
    mean b to -898. Excluded pixels contribute to no term, so the identity
    I = b*L + D still closes exactly on the retained set.
    """
    T, N = y.shape
    x2b = np.broadcast_to(np.asarray(x2, float).reshape(-1, 1), (T, N))

    v = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2b)
    n = v.sum(0).astype(float)

    yz = np.where(v, y, 0.0)
    x1z = np.where(v, x1, 0.0)
    x2z = np.where(v, x2b, 0.0)

    nz = np.maximum(n, 1.0)
    ym, x1m, x2m = yz.sum(0) / nz, x1z.sum(0) / nz, x2z.sum(0) / nz

    yc = np.where(v, y - ym, 0.0)
    x1c = np.where(v, x1 - x1m, 0.0)
    x2c = np.where(v, x2b - x2m, 0.0)

    S11 = (x1c * x1c).sum(0)
    S22 = (x2c * x2c).sum(0)
    S12 = (x1c * x2c).sum(0)
    S1y = (x1c * yc).sum(0)
    S2y = (x2c * yc).sum(0)

    det = S11 * S22 - S12 ** 2
    # sd of x1 on each column's own valid subset
    sd_x1 = np.sqrt(np.maximum(S11, 0.0) / nz)
    ok = ((n >= min_n) & (det > 0) & (S22 > 0) & (S11 > 0)
          & (sd_x1 >= min_sd_x1))

    b = np.full(N, np.nan)
    d = np.full(N, np.nan)
    b[ok] = (S22[ok] * S1y[ok] - S12[ok] * S2y[ok]) / det[ok]
    d[ok] = (S11[ok] * S2y[ok] - S12[ok] * S1y[ok]) / det[ok]

    # univariate slopes on the same subset -> the identity closes exactly
    uni_y = np.full(N, np.nan)
    uni_x1 = np.full(N, np.nan)
    good22 = (n >= min_n) & (S22 > 0)
    uni_y[good22] = S2y[good22] / S22[good22]
    uni_x1[good22] = S12[good22] / S22[good22]

    r = np.full(N, np.nan)
    gr = ok & (S11 > 0) & (S22 > 0)
    r[gr] = S12[gr] / np.sqrt(S11[gr] * S22[gr])

    return b, d, n, r, uni_y, uni_x1


def ols_univariate(y, x, min_n=MIN_PAIRS):
    """Per-column OLS slope of y (T,N) on x (T,)."""
    T, N = y.shape
    xb = np.broadcast_to(np.asarray(x, float).reshape(-1, 1), (T, N))
    v = np.isfinite(y) & np.isfinite(xb)
    n = v.sum(0).astype(float)
    nz = np.maximum(n, 1.0)
    ym = np.where(v, y, 0.0).sum(0) / nz
    xm = np.where(v, xb, 0.0).sum(0) / nz
    yc = np.where(v, y - ym, 0.0)
    xc = np.where(v, xb - xm, 0.0)
    num = (xc * yc).sum(0)
    den = (xc * xc).sum(0)
    out = np.full(N, np.nan)
    ok = (n >= min_n) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out, n


def detrend_time(a):
    """Remove a linear-in-index trend from each column of (T,N), NaN-safe."""
    T = a.shape[0]
    t = np.arange(T, dtype=float)
    sl, _ = ols_univariate(a, t, min_n=3)
    v = np.isfinite(a)
    tm = np.where(v, t.reshape(-1, 1), 0.0).sum(0) / np.maximum(v.sum(0), 1)
    am = np.where(v, a, 0.0).sum(0) / np.maximum(v.sum(0), 1)
    fit = am + sl * (t.reshape(-1, 1) - tm)
    return np.where(v, a - fit, np.nan)


def wmean(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


# ------------------------------------------------------------- stage one ---
def build_sic_stacks(months, cache, idx_psn_to_sial, sial_shape,
                     arctic_psn, psn_shape):
    """Monthly-mean SIC per year, PSN grid -> SIAL grid, cached to disk."""
    out = {}
    for mn in months:
        mname = MONTHS[mn]
        f_stack = os.path.join(cache, f'bivdiag_sic_sial_{mname}.npy')
        f_years = os.path.join(cache, f'bivdiag_sic_years_{mname}.npy')
        f_psn = os.path.join(cache, f'bivdiag_sic_psn_{mname}.npy')
        if os.path.exists(f_stack) and os.path.exists(f_years):
            out[mn] = (np.load(f_stack), np.load(f_years), np.load(f_psn))
            print(f'  {mname}: cached stack '
                  f'{out[mn][0].shape[0]} yr')
            continue

        print(f'  {mname}: rebuilding from daily PSN25 ...', flush=True)
        psn_years, yrs = [], []
        for year in TREND_YEARS:
            daily = []
            pat = os.path.join(DIR_SIC, f'sic_psn25_{year}*.nc')
            for f in sorted(glob.glob(pat)):
                # month from the filename; avoids opening every daily file
                stem = os.path.basename(f).split('_')[2]
                if len(stem) < 6 or int(stem[4:6]) != mn:
                    continue
                try:
                    with xr.open_dataset(f) as ds:
                        raw = ds['cdr_seaice_conc'].squeeze().values.astype(float)
                    if raw.ndim > 2:
                        raw = raw[0]
                    if not np.any(np.isfinite(raw)):
                        continue
                    if np.nanmax(raw) > 1.5:
                        raw /= 100.0
                    raw = np.where((raw >= 0.0) & (raw <= 1.0), raw, np.nan)
                    if np.sum(np.isfinite(raw[arctic_psn])) < MIN_ARCTIC_PX:
                        continue
                    daily.append(raw)
                except Exception:
                    continue
            if daily:
                psn_years.append(np.nanmean(daily, axis=0))
                yrs.append(year)

        if len(yrs) < MIN_PAIRS:
            print(f'    only {len(yrs)} yr -> skipped')
            continue

        psn_stk = np.asarray(psn_years, dtype=np.float32)
        sial_stk = np.stack(
            [p.ravel()[idx_psn_to_sial].reshape(sial_shape) for p in psn_stk]
        ).astype(np.float32)
        yrs = np.asarray(yrs)

        np.save(f_psn, psn_stk)
        np.save(f_stack, sial_stk)
        np.save(f_years, yrs)
        out[mn] = (sial_stk, yrs, psn_stk)
        print(f'    {len(yrs)} yr built and cached')
    return out


def verify_provenance(sic_stacks, idx_psn_to_sial, sial_shape, rows):
    """Recompute SIC-vs-YEAR trend/pval from the rebuild and compare with the
    cached fields the masks were built from."""
    for mn, (sial_stk, yrs, psn_stk) in sic_stacks.items():
        mname = MONTHS[mn]
        try:
            cslope = np.load(
                f'{CACHE_OBS}/sic_trend_slope_sial_{mname}.npy').astype(float)
            cpval = np.load(
                f'{CACHE_OBS}/sic_trend_pval_sial_{mname}.npy').astype(float)
        except Exception as e:
            print(f'  {mname}: cached trend unreadable ({e})')
            continue

        flat = psn_stk.reshape(psn_stk.shape[0], -1).astype(float)
        early = np.nanmean(
            flat[[i for i, y in enumerate(yrs) if y in EARLY_YEARS], :], 0)
        late = np.nanmean(
            flat[[i for i, y in enumerate(yrs) if y in LATE_YEARS], :], 0)
        good = np.where((np.sum(np.isfinite(flat), 0) >= MIN_PAIRS) &
                        ((early >= SIC_ICE_THRESHOLD) |
                         (late >= SIC_ICE_THRESHOLD)))[0]

        sl = np.full(flat.shape[1], np.nan)
        pv = np.full(flat.shape[1], np.nan)
        yf = np.asarray(yrs, float)
        for px in good:
            col = flat[:, px]
            m = np.isfinite(col)
            if m.sum() < MIN_PAIRS:
                continue
            r = sp_stats.linregress(yf[m], col[m])
            sl[px], pv[px] = r.slope, r.pvalue

        rs = sl.ravel()[idx_psn_to_sial].reshape(sial_shape)
        rp = pv.ravel()[idx_psn_to_sial].reshape(sial_shape)

        m = np.isfinite(rs) & np.isfinite(cslope)
        dslope = np.abs(rs - cslope)[m]
        sig_new = (rs < 0) & (rp < PVAL)
        sig_old = (cslope < 0) & (cpval < PVAL)
        both = np.isfinite(rs) & np.isfinite(cslope)
        disagree = int((sig_new != sig_old)[both].sum())

        rows.append(dict(
            month=mname, n_overlap=int(m.sum()),
            slope_max_abs_diff=float(dslope.max()) if dslope.size else np.nan,
            slope_corr=float(np.corrcoef(rs[m], cslope[m])[0, 1])
            if m.sum() > 2 else np.nan,
            sig_pixels_rebuilt=int(sig_new[both].sum()),
            sig_pixels_cached=int(sig_old[both].sum()),
            classification_disagreements=disagree,
        ))
        flag = 'OK' if disagree == 0 else f'*** {disagree} DISAGREE ***'
        print(f'  {mname}: slope r={rows[-1]["slope_corr"]:.5f}  '
              f'max|dslope|={rows[-1]["slope_max_abs_diff"]:.2e}  {flag}')


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='*', default=None,
                    help='subset of month names, e.g. June September')
    ap.add_argument('--skip-sic', action='store_true',
                    help='require the cached SIC stack; do not rebuild')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    months = [m for m, n in MONTHS.items()
              if args.months is None or n in args.months]

    print('=' * 72)
    print('BIVARIATE ALBEDO DIAGNOSTIC')
    print('=' * 72)

    # --- grids ------------------------------------------------------------
    print('\n[grid] building SIAL and PSN25 geometry ...')
    sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
    with xr.open_dataset(sample_sial) as ds:
        sla, slo = ds['lat'].values, ds['lon'].values
    if sla.ndim == 1:
        SIAL_LON, SIAL_LAT = np.meshgrid(slo, sla)
    else:
        SIAL_LAT, SIAL_LON = sla, slo
    sial_shape = SIAL_LAT.shape

    try:
        arctic_sial = np.load(
            f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
        print(f'  Arctic mask: saved arctic_60N_sial.npy '
              f'({int(arctic_sial.sum()):,} px)')
    except Exception as e:
        arctic_sial = (SIAL_LAT >= 60.)
        print(f'  Arctic mask: *** FALLBACK lat>=60 *** ({e}) '
              f'({int(arctic_sial.sum()):,} px) - land not excluded')

    from pyproj import Transformer
    sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
    with xr.open_dataset(sample_sic) as ds:
        px, py = ds['x'].values.astype(float), ds['y'].values.astype(float)
    pxx, pyy = np.meshgrid(px, py)
    psn_shape = pxx.shape
    tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
    PSN_LON, PSN_LAT = tfm.transform(pxx, pyy)
    arctic_psn = (PSN_LAT >= 60.)

    cos_sial = np.cos(np.deg2rad(SIAL_LAT)).astype(np.float32)

    print('  PSN25 -> SIAL nearest-neighbour index ...')
    idx_psn_to_sial = build_nn_index(
        PSN_LAT.ravel(), PSN_LON.ravel(),
        SIAL_LAT.ravel(), SIAL_LON.ravel())

    # APP-x -> SIAL, same construction as the pipeline. September is APP-x
    # only, and APP-x is on its own grid, so without this every September
    # year fails the shape check and the month is silently dropped.
    print('  APP-x -> SIAL nearest-neighbour index ...')
    idx_appx_to_sial = None
    try:
        from netCDF4 import Dataset as NC4Dataset
        sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc',
                                       recursive=True))[0]
        with NC4Dataset(sample_appx, 'r') as ds:
            al = np.ma.filled(ds.variables['latitude'][:],
                              np.nan).astype(float)
            ao = np.ma.filled(ds.variables['longitude'][:],
                              np.nan).astype(float)
        if al.ndim == 1:
            ao2, al2 = np.meshgrid(ao, al)
        else:
            al2, ao2 = al, ao
        idx_appx_to_sial = build_nn_index(
            al2.ravel(), ao2.ravel(), SIAL_LAT.ravel(), SIAL_LON.ravel())
        print(f'    built from {os.path.basename(sample_appx)}')
    except Exception as e:
        print(f'    WARNING: APP-x -> SIAL failed ({e}); '
              f'September will be unavailable')

    def to_sial(arr):
        """Regrid an APP-x field onto SIAL if it is not already there."""
        if arr.shape == sial_shape:
            return arr
        if idx_appx_to_sial is None:
            return None
        return arr.ravel()[idx_appx_to_sial].reshape(sial_shape)

    # --- temperature ------------------------------------------------------
    T_all = np.load(f'{CACHE_OBS}/T_annual.npy')
    T_yrs = np.load(f'{CACHE_OBS}/T_annual_years.npy')
    if T_all.ndim > 1:            # ensemble -> take the member mean
        T_all = np.nanmean(T_all, axis=tuple(range(T_all.ndim - 1)))
    T_map = {int(y): float(t) for y, t in zip(T_yrs, T_all)}
    print(f'[T] {len(T_map)} years, {min(T_map)}-{max(T_map)}')

    # --- SIC --------------------------------------------------------------
    print('\n[stage 1] SIC rebuild + provenance')
    if args.skip_sic:
        sic_stacks = {}
        for mn in months:
            mname = MONTHS[mn]
            f = os.path.join(CACHE_OBS, f'bivdiag_sic_sial_{mname}.npy')
            if not os.path.exists(f):
                sys.exit(f'--skip-sic but {f} is missing; run without it once.')
            sic_stacks[mn] = (
                np.load(f),
                np.load(os.path.join(CACHE_OBS,
                                     f'bivdiag_sic_years_{mname}.npy')),
                np.load(os.path.join(CACHE_OBS,
                                     f'bivdiag_sic_psn_{mname}.npy')))
    else:
        sic_stacks = build_sic_stacks(months, CACHE_OBS, idx_psn_to_sial,
                                      sial_shape, arctic_psn, psn_shape)

    prov_rows = []
    verify_provenance(sic_stacks, idx_psn_to_sial, sial_shape, prov_rows)
    pd.DataFrame(prov_rows).to_csv(
        os.path.join(args.out, 'stage1_provenance.csv'), index=False)

    # --- per-month analysis ----------------------------------------------
    mask_rows, fit_rows, high_rows, bin_rows, detr_rows = [], [], [], [], []

    for mn in months:
        mname = MONTHS[mn]
        if mn not in sic_stacks:
            continue
        print(f'\n[{mname}]')
        sic_stk, sic_years, _ = sic_stacks[mn]

        # albedo stack on the same years
        alb_maps, alb_years = [], []
        for y in sorted(T_map):
            pc = f'{CACHE_OBS}/clara_alb_{mname}_{y}.npy'
            pa = f'{CACHE_OBS}/appx_alb_{mname}_{y}.npy'
            if mname == 'September':
                if not os.path.exists(pa):
                    continue
                a = to_sial(np.load(pa).astype(np.float32))
                if a is None:
                    continue
            else:
                c = np.load(pc).astype(np.float32) if os.path.exists(pc) else None
                aa = np.load(pa).astype(np.float32) if os.path.exists(pa) else None
                if aa is not None:
                    aa = to_sial(aa)
                if c is not None and aa is not None:
                    a = np.where(np.isfinite(c), c, aa)
                elif c is not None:
                    a = c
                elif aa is not None:
                    a = aa
                else:
                    continue
            alb_maps.append(a)
            alb_years.append(y)

        common = sorted(set(alb_years) & set(int(y) for y in sic_years))
        if len(common) < MIN_PAIRS:
            print(f'  only {len(common)} common years -> skipped')
            continue
        ai = {y: i for i, y in enumerate(alb_years)}
        si = {int(y): i for i, y in enumerate(sic_years)}

        A = np.stack([alb_maps[ai[y]] for y in common]).reshape(
            len(common), -1).astype(float)
        C = np.stack([sic_stk[si[y]] for y in common]).reshape(
            len(common), -1).astype(float)
        T = np.array([T_map[y] for y in common], float)
        print(f'  {len(common)} common years ({common[0]}-{common[-1]})')

        # ---- masks (stage 2) --------------------------------------------
        cslope = np.load(
            f'{CACHE_OBS}/sic_trend_slope_sial_{mname}.npy').astype(float)
        cpval = np.load(
            f'{CACHE_OBS}/sic_trend_pval_sial_{mname}.npy').astype(float)
        sic_e = np.load(f'{CACHE_OBS}/sic_early_sial_{mname}.npy').astype(float)
        sic_l = np.load(f'{CACHE_OBS}/sic_late_sial_{mname}.npy').astype(float)
        IS_cached = -np.load(
            f'{CACHE_OBS}/merged_alb_dalphadT_{mname}.npy').astype(float)

        valid = arctic_sial & np.isfinite(IS_cached)
        chang = (valid & np.isfinite(cslope) & np.isfinite(cpval)
                 & (cslope < 0) & (cpval < PVAL))
        stable = (valid & ~chang
                  & (np.fmax(sic_e, sic_l) >= SIC_THRESH))

        # does a significant-decline pixel sit inside the darkening region,
        # judged by the REBUILT trend rather than the cached one?
        yf = np.array(common, float)
        sl_rb, _ = ols_univariate(C, yf)
        sl_rb = sl_rb.reshape(sial_shape)
        leak = stable & np.isfinite(sl_rb) & (sl_rb < 0) & (cpval < PVAL)

        mask_rows.append(dict(
            month=mname,
            overlap_chang_stable=int((chang & stable).sum()),
            n_chang=int(chang.sum()), n_stable=int(stable.sum()),
            area_frac_chang=wmean(chang[arctic_sial].astype(float),
                                  cos_sial[arctic_sial]),
            area_frac_stable=wmean(stable[arctic_sial].astype(float),
                                   cos_sial[arctic_sial]),
            rebuilt_sigdecline_in_stable=int(leak.sum()),
        ))
        print(f'  mask: chang={chang.sum():,} stable={stable.sum():,} '
              f'overlap={int((chang & stable).sum())} '
              f'rebuilt-sig-in-stable={int(leak.sum())}')

        # ---- stage 3: bivariate -----------------------------------------
        b, d, n, r_ct, I_uni, mL_uni = ols_bivariate(A, C, T)
        I = -I_uni          # -dalpha/dT, same subset
        L = -mL_uni         # -dSIC/dT,   same subset
        D = -d              # -dalpha/dT at fixed SIC
        ident = I - (b * L + D)

        fin = np.isfinite(ident)
        print(f'  identity I = b*L + D : max|resid| = '
              f'{np.nanmax(np.abs(ident[fin])) if fin.any() else np.nan:.3e}')

        for reg, m2 in (('area_loss', chang), ('darkening', stable)):
            sel = m2.ravel() & np.isfinite(b) & np.isfinite(D) & np.isfinite(I)
            w = cos_sial.ravel()[sel]
            if sel.sum() < 20:
                continue
            fit_rows.append(dict(
                month=mname, region=reg, n_pixels=int(sel.sum()),
                I=wmean(I[sel], w),
                b=wmean(b[sel], w),
                L=wmean(L[sel], w),
                bL=wmean((b * L)[sel], w),
                D=wmean(D[sel], w),
                D_share_of_I=wmean(D[sel], w) / wmean(I[sel], w)
                if wmean(I[sel], w) else np.nan,
                b_minus_contrast=wmean(b[sel], w) - ICE_OCEAN_CONTRAST,
                mean_abs_r_SIC_T=wmean(np.abs(r_ct[sel]), w),
                frac_collinear_r_gt_0p9=float(
                    np.mean(np.abs(r_ct[sel]) > 0.9)),
                I_cached=wmean(IS_cached.ravel()[sel], w),
                mean_n_years=wmean(n[sel], w),
            ))
            q = fit_rows[-1]
            print(f'    {reg:10s} I={q["I"]:.5f} b*L={q["bL"]:.5f} '
                  f'D={q["D"]:.5f} ({100*q["D_share_of_I"]:.1f}% of I)  '
                  f'b={q["b"]:.3f}  |r(c,T)|={q["mean_abs_r_SIC_T"]:.2f}')

        # ---- stage 4: high-SIC ------------------------------------------
        cbar = np.nanmean(C, axis=0)
        slope_aT, n_aT = ols_univariate(A, T)
        for thr in HIGH_SIC:
            sel = (cbar > thr) & np.isfinite(slope_aT) & arctic_sial.ravel()
            if sel.sum() < 20:
                continue
            w = cos_sial.ravel()[sel]
            high_rows.append(dict(
                month=mname, sic_threshold=thr, n_pixels=int(sel.sum()),
                dalpha_dT=wmean(slope_aT[sel], w),
                darkening_rate=-wmean(slope_aT[sel], w),
                mean_sic=wmean(cbar[sel], w),
                frac_of_arctic=float(sel.sum()) / float(arctic_sial.sum()),
            ))
            print(f'    c>{thr}: darkening = '
                  f'{high_rows[-1]["darkening_rate"]:.5f} /K  '
                  f'({sel.sum():,} px)')

        # ---- stage 5: linearity ------------------------------------------
        edges = np.arange(0.0, 1.0001, 0.1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ((cbar >= lo) & (cbar < hi) & arctic_sial.ravel()
                   & np.isfinite(b) & np.isfinite(D))
            if sel.sum() < 20:
                continue
            w = cos_sial.ravel()[sel]
            bin_rows.append(dict(
                month=mname, sic_bin_lo=round(lo, 1), sic_bin_hi=round(hi, 1),
                n_pixels=int(sel.sum()), I=wmean(I[sel], w),
                b=wmean(b[sel], w), D=wmean(D[sel], w),
                L=wmean(L[sel], w)))

        # ---- stage 6: detrended ------------------------------------------
        Ad, Cd = detrend_time(A), detrend_time(C)
        Td = T - np.polyval(np.polyfit(np.arange(len(T)), T, 1),
                            np.arange(len(T)))
        bD, dD, _, rD, IuD, LuD = ols_bivariate(Ad, Cd, Td)
        for reg, m2 in (('area_loss', chang), ('darkening', stable)):
            sel = m2.ravel() & np.isfinite(bD) & np.isfinite(b)
            if sel.sum() < 20:
                continue
            w = cos_sial.ravel()[sel]
            detr_rows.append(dict(
                month=mname, region=reg, n_pixels=int(sel.sum()),
                L_raw=wmean(L[sel], w), L_detrended=wmean(-LuD[sel], w),
                L_interannual_share=(wmean(-LuD[sel], w) / wmean(L[sel], w)
                                     if wmean(L[sel], w) else np.nan),
                b_raw=wmean(b[sel], w), b_detrended=wmean(bD[sel], w),
                D_raw=wmean(D[sel], w), D_detrended=wmean(-dD[sel], w)))
            q = detr_rows[-1]
            print(f'    {reg:10s} L raw={q["L_raw"]:.4f} '
                  f'detrended={q["L_detrended"]:.4f} '
                  f'(interannual share {100*q["L_interannual_share"]:.0f}%)')

    # --- write -------------------------------------------------------------
    for name, rows in (('stage2_masks', mask_rows),
                       ('stage3_bivariate', fit_rows),
                       ('stage4_high_sic', high_rows),
                       ('stage5_sic_bins', bin_rows),
                       ('stage6_detrended', detr_rows)):
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out, f'{name}.csv'), index=False)

    with open(os.path.join(args.out, 'summary.txt'), 'w') as fh:
        fh.write('BIVARIATE ALBEDO DIAGNOSTIC\n')
        fh.write(f'alpha = a + b*SIC + d*T ; I = b*L + D (exact), D = -d\n')
        fh.write(f'ice-ocean contrast benchmark for b: {ICE_OCEAN_CONTRAST}\n\n')
        for name, rows in (('stage1_provenance', prov_rows),
                           ('stage2_masks', mask_rows),
                           ('stage3_bivariate', fit_rows),
                           ('stage4_high_sic', high_rows),
                           ('stage6_detrended', detr_rows)):
            fh.write(f'--- {name} ---\n')
            fh.write(pd.DataFrame(rows).to_string(index=False) + '\n\n')

    print(f'\nWrote CSVs + summary.txt to {args.out}')


if __name__ == '__main__':
    main()
