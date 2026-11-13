#!/usr/bin/env python
"""
Bivariate albedo diagnostic applied to the CMIP6 ensemble.

Observational counterpart: bivariate_diag.py. Same decomposition, same
conventions, so the two are directly comparable.

Per pixel and month, one regression with both predictors:

    alpha_t = a + b * SIC_t + d * T_t

With I = -dalpha/dT and L = -dSIC/dT this gives the exact identity

    I = b * L + D ,      D = -d

which answers the reviewer's three possibilities term by term:

    L too small  -> insufficient ice loss per kelvin
    b too small  -> insufficient albedo response to a concentration change
    D too small  -> insufficient intrinsic darkening at fixed concentration

Everything upstream of the fit -- file discovery, SIC regridding onto the
albedo grid, the SIC-trend classification, the Arctic domain -- is imported
from cmip6_siaf_calculation.py so that the masks and samples are identical
to the published pipeline rather than a reimplementation of them.

One member per model by default, matching the saved factor diagnostic.

Output
    model_bivariate_by_month.csv   per model / month / region
    model_bivariate_summary.csv    ensemble means per month / region
    model_bivariate_models.csv     per model, season-aggregated
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '/glade/work/jrotondo/siaf_drk_paper_final/python')
sys.path.insert(0,
                '/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028')

import cmip6_siaf_calculation as C          # noqa: E402
from bivariate_diag import ols_bivariate    # noqa: E402  (self-tested)

OUT_DIR = '/glade/work/jrotondo/ncc_paper/plots/bivariate_diag/'
MIN_PAIRS = 10


def wmean(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


def member_bivariate(model, member, T_annual_dict):
    """Mirror of C.compute_member_IS_native up to the regression, then fit
    the bivariate model instead of the univariate one.

    Returns a list of dict rows (one per month x region).
    """
    inst = C.MODEL_INFO[model]['institution']
    rsus_f = C.find_var_files(inst, model, member, 'rsus')
    rsds_f = C.find_var_files(inst, model, member, 'rsds')
    sic_f = C.find_var_files(inst, model, member, 'siconc')
    if not all([rsus_f, rsds_f, sic_f]):
        return []

    all_yrs = list(range(1980, 2016))
    ru = C.load_filtered(rsus_f, 'rsus', all_yrs, C.MONTH_NUMS)
    rd = C.load_filtered(rsds_f, 'rsds', all_yrs, C.MONTH_NUMS)
    si = C.load_filtered(sic_f, 'siconc', all_yrs, C.MONTH_NUMS)
    if ru is None or rd is None or si is None:
        return []

    _, alb_lat, alb_lon, alb_is2d = C._std_latlon(ru)
    _, sic_lat, sic_lon, sic_is2d = C._std_latlon(si)
    if alb_lat is None or sic_lat is None:
        return []

    def _ym(da):
        return set(zip([int(t.dt.year) for t in da.time],
                       [int(t.dt.month) for t in da.time]))

    comm = _ym(ru) & _ym(rd) & _ym(si)
    if not comm:
        return []

    def _tmask(da):
        return np.array([(int(t.dt.year), int(t.dt.month)) in comm
                         for t in da.time], dtype=bool)

    ru, rd, si = (ru.isel(time=_tmask(ru)), rd.isel(time=_tmask(rd)),
                  si.isel(time=_tmask(si)))

    sv = si.values.astype(float)
    if np.nanmax(sv) > 1.5:
        sv /= 100.

    alb_sh = alb_lat.shape if alb_is2d else (len(alb_lat), len(alb_lon))

    # --- SIC onto the albedo grid, exactly as the pipeline does -----------
    if sv.shape[1:] != alb_sh:
        if sic_is2d:
            sla_raw = C._mask_fills(sic_lat).ravel()
            slo_raw = C._mask_fills(sic_lon).ravel()
        else:
            slo2d, sla2d = np.meshgrid(sic_lon, sic_lat)
            sla_raw = C._mask_fills(sla2d).ravel()
            slo_raw = C._mask_fills(slo2d).ravel()
        if alb_is2d:
            ala = C._mask_fills(alb_lat).ravel()
            alo = C._mask_fills(alb_lon).ravel()
        else:
            alo2d, ala2d = np.meshgrid(alb_lon, alb_lat)
            ala, alo = ala2d.ravel(), alo2d.ravel()

        sic_valid = np.isfinite(sla_raw) & np.isfinite(slo_raw)
        sv = sv.reshape(sv.shape[0], -1)[:, sic_valid]
        _, idx_s2a = cKDTree(
            C.latlon_to_xyz(sla_raw[sic_valid], slo_raw[sic_valid])).query(
            C.latlon_to_xyz(ala, alo).astype(np.float32), workers=-1)
        sv = sv[:, idx_s2a].reshape((sv.shape[0],) + alb_sh)

    ru_yrs = np.array([int(t.dt.year) for t in ru.time])
    ru_mon = np.array([int(t.dt.month) for t in ru.time])
    si_yrs = np.array([int(t.dt.year) for t in si.time])
    si_mon = np.array([int(t.dt.month) for t in si.time])
    del si
    gc.collect()

    native_ctx = C.build_native_ctx(alb_lat, alb_lon, alb_is2d)
    lat2d = native_ctx['lat2d']
    coslat = np.cos(np.deg2rad(lat2d)).ravel()

    allT = np.array([T_annual_dict[y] for y in C.TREND_YEARS
                     if y in T_annual_dict])
    T_center = float(allT.mean()) if allT.size else 0.

    rows = []
    for mn in C.MONTH_NUMS:
        yearly_alb = {}
        for year in C.TREND_YEARS:
            if year not in T_annual_dict:
                continue
            m = (ru_yrs == year) & (ru_mon == mn)
            if m.sum() == 0:
                continue
            u, d = ru.isel(time=m), rd.isel(time=m)
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(d.mean('time').values > 1.,
                               u.mean('time').values / d.mean('time').values,
                               np.nan)
            yearly_alb[year] = np.clip(alb, 0., 1.)

        yrs = sorted(yearly_alb)
        if len(yrs) < MIN_PAIRS:
            continue

        # SIC stack on exactly those years
        sic_by_year = {}
        for y in yrs:
            m = (si_yrs == y) & (si_mon == mn)
            if m.sum():
                sic_by_year[y] = np.nanmean(sv[m], axis=0)
        yrs = [y for y in yrs if y in sic_by_year]
        if len(yrs) < MIN_PAIRS:
            continue

        A = np.stack([yearly_alb[y] for y in yrs]).reshape(len(yrs), -1)
        Cc = np.stack([sic_by_year[y] for y in yrs]).reshape(len(yrs), -1)
        T = np.array([T_annual_dict[y] - T_center for y in yrs], float)

        # --- classification, identical to the pipeline --------------------
        tm = (si_mon == mn) & np.isin(si_yrs, C.TREND_YEARS)
        em = (si_mon == mn) & np.isin(si_yrs, C.EARLY_YEARS)
        lm = (si_mon == mn) & np.isin(si_yrs, C.LATE_YEARS)
        if tm.sum() < MIN_PAIRS or em.sum() == 0 or lm.sum() == 0:
            continue
        sic_e = np.nanmean(sv[em], axis=0)
        sic_l = np.nanmean(sv[lm], axis=0)
        slope_sic, pval_sic = C.pixel_ols_trend(si_yrs[tm].astype(float),
                                                sv[tm])
        all_ice = (lat2d >= 60.) & (sic_e >= C.SIC_ICE_THRESHOLD)
        changing = (all_ice & np.isfinite(slope_sic) & np.isfinite(pval_sic)
                    & (slope_sic < 0) & (pval_sic < C.PVAL))
        stable = (all_ice & ~changing
                  & (np.fmax(sic_e, sic_l) >= C.SIC_ICE_THRESHOLD))

        # --- the fit ------------------------------------------------------
        b, dd, n, r_ct, I_uni, L_uni = ols_bivariate(A, Cc, T)
        I, L, D = -I_uni, -L_uni, -dd
        resid = np.nanmax(np.abs(I - (b * L + D)))

        for reg, msk in (('area_loss', changing), ('darkening', stable)):
            sel = (msk.ravel() & np.isfinite(b) & np.isfinite(D)
                   & np.isfinite(I))
            if sel.sum() < 20:
                continue
            w = coslat[sel]
            rows.append(dict(
                model=model, member=member, month=C.MONTH_NAMES[mn],
                region=reg, n_pixels=int(sel.sum()), n_years=len(yrs),
                I=wmean(I[sel], w), b=wmean(b[sel], w), L=wmean(L[sel], w),
                bL=wmean((b * L)[sel], w), D=wmean(D[sel], w),
                D_share_of_I=(wmean(D[sel], w) / wmean(I[sel], w)
                              if wmean(I[sel], w) else np.nan),
                mean_abs_r_SIC_T=wmean(np.abs(r_ct[sel]), w),
                identity_max_resid=float(resid),
            ))

        del A, Cc
        gc.collect()

    del ru, rd, sv
    gc.collect()
    return rows


def _one_model(model):
    """Worker: returns (model, rows, error_or_None). Per-model cost varies by
    two orders of magnitude (MIROC6 ~15 s, ACCESS-CM2 >13 min, mostly cold
    reads), so models are processed in parallel rather than in sequence."""
    import time
    t0 = time.time()
    try:
        inst = C.MODEL_INFO[model]['institution']
        members = C.find_all_members(inst, model)
        if not members:
            return model, [], 'no usable member', time.time() - t0
        member = members[0]
        T_annual = C.compute_T_annual_global(model, member, C.TREND_YEARS)
        if not T_annual:
            return model, [], 'no tas', time.time() - t0
        rows = member_bivariate(model, member, T_annual)
        gc.collect()
        if not rows:
            return model, [], 'no rows', time.time() - t0
        return model, rows, None, time.time() - t0
    except Exception as e:
        return model, [], f'{type(e).__name__}: {e}'[:160], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--nproc', type=int, default=1,
                    help='parallel workers across models')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = args.models or sorted(C.MODEL_INFO)
    print(f'{len(models)} models requested, nproc={args.nproc}', flush=True)

    all_rows, failures = [], []
    if args.nproc > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        done = 0
        with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
            for model, rows, err, dt in pool.imap_unordered(_one_model,
                                                            models):
                done += 1
                if err:
                    print(f'[{done}/{len(models)}] {model}: FAILED {err} '
                          f'({dt:.0f}s)', flush=True)
                    failures.append((model, err))
                else:
                    all_rows.extend(rows)
                    print(f'[{done}/{len(models)}] {model}: OK {len(rows)} '
                          f'rows ({dt:.0f}s)', flush=True)
                    # checkpoint: one slow model must not hold the whole
                    # result hostage, and a crash must not lose the rest
                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(args.out,
                                     'model_bivariate_by_month.partial.csv'),
                        index=False)
    else:
        for i, model in enumerate(models, 1):
            model, rows, err, dt = _one_model(model)
            if err:
                print(f'[{i}/{len(models)}] {model}: FAILED {err} ({dt:.0f}s)',
                      flush=True)
                failures.append((model, err))
            else:
                all_rows.extend(rows)
                print(f'[{i}/{len(models)}] {model}: OK {len(rows)} rows '
                      f'({dt:.0f}s)', flush=True)
            gc.collect()

    if not all_rows:
        sys.exit('no results')

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, 'model_bivariate_by_month.csv'),
              index=False)

    summ = (df.groupby(['month', 'region'])
              [['I', 'b', 'L', 'bL', 'D', 'D_share_of_I',
                'mean_abs_r_SIC_T']]
              .agg(['mean', 'std', 'count']))
    summ.to_csv(os.path.join(args.out, 'model_bivariate_summary.csv'))

    per_model = (df.groupby(['model', 'region'])
                   [['I', 'b', 'L', 'bL', 'D']].mean().reset_index())
    per_model.to_csv(os.path.join(args.out, 'model_bivariate_models.csv'),
                     index=False)

    print('\n' + '=' * 72)
    print('ENSEMBLE MEAN (one member per model)')
    print('=' * 72)
    s = (df.groupby(['month', 'region'])
           [['I', 'bL', 'D', 'D_share_of_I', 'b', 'L']].mean())
    print(s.round(5).to_string())
    print(f'\nmodels contributing: {df.model.nunique()}')
    print(f'max identity residual: {df.identity_max_resid.max():.2e}')
    if failures:
        print('\nfailures:')
        for m, why in failures:
            print(f'  {m}: {why}')
    print(f'\nwrote CSVs to {args.out}')


if __name__ == '__main__':
    main()
