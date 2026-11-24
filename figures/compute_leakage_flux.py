#!/usr/bin/env python
"""
compute_leakage_flux.py
=======================
Supplementary Figure 3: the ABSOLUTE two-sided classification-leakage budget,
in W m-2. Reimplements the flux conversion that `compute_leakage_budget.py`
left out, so the figure has an end-to-end generator for the first time.

`compute_leakage_budget.py` gives the leakage in albedo units: the mean
residual darkening inside the declining-ice region, and the mean
concentration-driven albedo change inside the darkening region. Turning those
into a radiative flux needs the same three ingredients the published SIAF uses,
and this script takes all three from `siaf_obs_core` (the verbatim head of
python/siaf_obs_analysis.py) rather than reimplementing any of them:

  RS            per-pixel radiative sensitivity, T07-inverted from CERES EBAF
                over 2000-2024 (CERES_OBS), monthly mean, scattered to PSN25
  the masks     compute_IS(), which is what partitions the domain into
                Omega_area and Omega_drk and sets the NaN structure that
                domain_mean() then averages over
  the weights   domain_mean() with AREA_W over DOMAIN_PSN

The published pipeline forms each month's component as

    SIAF_month = domain_mean(IS * RS) * AREA_RATIO * SEASON_FRAC

and reports the seasonal value as the mean over the six months, so

    SIAF_season = sum_months domain_mean(IS * RS) * AREA_RATIO / 12

which is exactly the "annual sum" of the supplementary figure: each month's
point is that month's term, and the six sum to a number directly comparable
with SIAF_area and SIAF_drk. The leakage terms are built by passing the two
decomposition fields through compute_IS() in place of dalpha/dT, so they
inherit the published partition unchanged:

    blue    compute_IS(dalb_res)[ext]  ->  -dalb_res on Omega_area, 0 on Omega_drk
    orange  compute_IS(dalb_sic)[drk]  ->  -dalb_sic on Omega_drk,  0 on Omega_area

Both are epoch differences, not per-K sensitivities, so the result is in W m-2
rather than W m-2 K-1, matching the published axis.

Albedo fields follow the pipeline's sensor rule verbatim: September is APP-x
only; April-August take CLARA-A3 with APP-x filling its NaN pixels. Unlike
compute_leakage_budget.py, APP-x is mapped onto the SIAL grid with the
pipeline's own nearest-neighbour index rather than zoomed, because compute_IS,
ARCTIC_SIAL and regrid_sial_to_psn all require the SIAL grid. For April-August
that changes nothing; for September it is the pipeline-faithful choice. The
run prints the albedo-unit means next to compute_leakage_budget.py's validated
values so the difference is visible rather than assumed.

Diagnostic only: reads cached fields, runs on a login node, no core hours.
Writes leakage_flux.csv; makes no figure (see s03.py).
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import siaf_obs_core as C   # noqa: E402  (verbatim head of siaf_obs_analysis.py)

PVAL = 0.05
EARLY_YEARS = list(range(1982, 1992))
LATE_YEARS = list(range(2015, 2025))
FIXED_CONTRAST = 0.54
BUDGET_CSV = os.path.join(HERE, 'leakage_budget.csv')
OUT_CSV = os.path.join(HERE, 'leakage_flux.csv')

# published fs03 annual sums, cross-check only
PUB = {'mon': dict(res=0.03181, sic=0.00701, net=0.02479),
       'fix': dict(res=0.00927, sic=0.00842, net=0.00085)}


def load_alb_epoch_mean_sial(mname, years):
    """Epoch-mean albedo on the SIAL grid.

    The sensor logic is lifted from siaf_obs_core.load_alb_yearly_with_T:
    September is APP-x only, other months are CLARA-A3 with APP-x filling NaN
    pixels, and APP-x is remapped with IDX_APPX_TO_SIAL. The only change is
    that the yearly maps are averaged over the epoch instead of regressed.
    """
    maps = []
    for year in years:
        if mname == 'September':
            p = f'{C.CACHE_OBS}/appx_alb_{mname}_{year}.npy'
            if not os.path.exists(p):
                continue
            raw = np.load(p).astype(np.float32)
            if C.IDX_APPX_TO_SIAL is not None and raw.shape != C.SIAL_SHAPE:
                raw = raw.ravel()[C.IDX_APPX_TO_SIAL].reshape(C.SIAL_SHAPE)
            alb = raw
        else:
            p_c = f'{C.CACHE_OBS}/clara_alb_{mname}_{year}.npy'
            p_a = f'{C.CACHE_OBS}/appx_alb_{mname}_{year}.npy'
            clara, appx = None, None
            if os.path.exists(p_c):
                clara = np.load(p_c).astype(np.float32)
            if os.path.exists(p_a):
                raw_a = np.load(p_a).astype(np.float32)
                if C.IDX_APPX_TO_SIAL is not None and raw_a.shape != C.SIAL_SHAPE:
                    raw_a = raw_a.ravel()[C.IDX_APPX_TO_SIAL].reshape(C.SIAL_SHAPE)
                appx = raw_a

            if clara is not None and appx is not None:
                alb = np.where(np.isfinite(clara), clara, appx)
            elif clara is not None:
                alb = clara
            elif appx is not None:
                alb = appx
            else:
                continue
        maps.append(alb.astype(np.float32))

    if not maps:
        return None
    stk = np.stack(maps, axis=0)
    with np.errstate(invalid='ignore'):
        return np.nanmean(stk, axis=0).astype(np.float32)


budget = pd.read_csv(BUDGET_CSV).set_index('mn')

print('\n' + '=' * 72)
print('ABSOLUTE LEAKAGE BUDGET  —  W m-2')
print('=' * 72)
print(f'  RS: CERES EBAF {C.CERES_OBS[0]}-{C.CERES_OBS[-1]}, T07-inverted, PSN25')
print(f'  AREA_RATIO={C.AREA_RATIO}  SEASON_FRAC={C.SEASON_FRAC}  '
      f'-> per-month factor AREA_RATIO/12 = {C.AREA_RATIO / 12:.6g}')

rows = []
for mn in C.MONTH_NUMS:
    mname = C.MONTH_NAMES[mn]
    print(f'\n  {mname} ...', flush=True)

    sic = C.load_sic_data(C.CACHE_OBS, mname)
    ae = load_alb_epoch_mean_sial(mname, EARLY_YEARS)
    al = load_alb_epoch_mean_sial(mname, LATE_YEARS)
    if sic is None or ae is None or al is None:
        print('    missing data, skipped')
        continue

    dalb = (al - ae).astype(np.float32)
    dsic = (sic['sic_l'] - sic['sic_e']).astype(np.float32)

    RS_mean = C.load_RS_ceres_mean(C.CERES_OBS, mn)
    if RS_mean is None:
        print('    no CERES RS, skipped')
        continue
    RS_psn = C.scatter_avg(RS_mean.ravel(), C.IDX_CERES_TO_PSN, C.PSN_SHAPE)
    rs_domain = C.domain_mean(RS_psn)

    rec = dict(month=mname, mn=mn, RS_domain_mean=rs_domain)
    contrast_mon = float(budget.loc[mn, 'contrast'])
    rec['contrast_mon'] = contrast_mon

    for tag, c in (('mon', contrast_mon), ('fix', FIXED_CONTRAST)):
        d_sic = (c * dsic).astype(np.float32)
        d_res = (dalb - d_sic).astype(np.float32)

        # compute_IS negates its argument and applies the published partition:
        # [1] is the Omega_area field, [2] the Omega_drk field, both NaN
        # outside Omega_area | Omega_drk so domain_mean averages over the union.
        res_ext = C.compute_IS(d_res, sic, PVAL)[1]      # blue
        sic_drk = C.compute_IS(d_sic, sic, PVAL)[2]      # orange

        blue = C.domain_mean(C.regrid_sial_to_psn(res_ext) * RS_psn) \
            * C.AREA_RATIO / 12.
        orange = C.domain_mean(C.regrid_sial_to_psn(sic_drk) * RS_psn) \
            * C.AREA_RATIO / 12.
        rec[f'blue_{tag}'] = float(blue)
        rec[f'orange_{tag}'] = float(orange)

        # albedo-unit means over each region, to sit beside the validated
        # compute_leakage_budget.py values (sign restored to that convention)
        rec[f'res_ext_{tag}'] = -float(np.nanmean(
            np.where(res_ext != 0, res_ext, np.nan)))
        rec[f'sic_drk_{tag}'] = -float(np.nanmean(
            np.where(sic_drk != 0, sic_drk, np.nan)))

    rows.append(rec)
    print(f"    RS domain mean {rs_domain:7.2f} W m-2   "
          f"blue {rec['blue_mon']:+.5f}  orange {rec['orange_mon']:+.5f}  "
          f"(fixed: {rec['blue_fix']:+.5f} / {rec['orange_fix']:+.5f})")

    del sic, ae, al, dalb, dsic, RS_mean, RS_psn
    gc.collect()

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print('\n' + '-' * 72)
print('Albedo-unit cross-check against compute_leakage_budget.py')
print('-' * 72)
print(f"{'month':11s} {'res_ext here':>13s} {'budget':>10s} "
      f"{'sic_drk here':>13s} {'budget':>10s}")
for r in df.itertuples():
    b = budget.loc[r.mn]
    print(f'{r.month:11s} {r.res_ext_mon:13.5f} {b.res_ext_mon:10.5f} '
          f'{r.sic_drk_mon:13.5f} {b.sic_drk_mon:10.5f}')

print('\n' + '-' * 72)
print('Annual sums (W m-2), against the published figure')
print('-' * 72)
for tag, label in (('mon', 'a  monthly ice albedo - 0.06'),
                   ('fix', 'b  fixed contrast 0.54')):
    res = float(df[f'blue_{tag}'].sum())
    sic_ = float(df[f'orange_{tag}'].sum())
    print(f'\n{label}')
    print(f'  Residual        {res:9.5f}   published {PUB[tag]["res"]:9.5f}   '
          f'diff {res - PUB[tag]["res"]:+9.5f}')
    print(f'  Concentration   {sic_:9.5f}   published {PUB[tag]["sic"]:9.5f}   '
          f'diff {sic_ - PUB[tag]["sic"]:+9.5f}')
    print(f'  Net to darkening{res - sic_:9.5f}   published {PUB[tag]["net"]:9.5f}   '
          f'diff {res - sic_ - PUB[tag]["net"]:+9.5f}')

print('\nwrote', OUT_CSV)
