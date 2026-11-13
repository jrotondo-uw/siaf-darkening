#!/usr/bin/env python
"""Why does BCC-CSM2-MR give |b| up to 898? Grid artefact or degenerate SIC?

Compares BCC-CSM2-MR against a control model on:
  - native albedo vs siconc grid shapes (was a regrid needed, and how coarse?)
  - duplication factor from the nearest-neighbour SIC->albedo mapping
  - per-pixel SIC standard deviation inside the masks
  - the conditioning term that sets the standard error of b
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/glade/work/jrotondo/siaf_drk_paper_final/python')
import cmip6_siaf_calculation as C  # noqa: E402

MONTH = 6  # June


def probe(model):
    inst = C.MODEL_INFO[model]['institution']
    member = C.find_all_members(inst, model)[0]
    ru = C.load_filtered(C.find_var_files(inst, model, member, 'rsus'),
                         'rsus', list(range(1980, 2016)), [MONTH])
    rd = C.load_filtered(C.find_var_files(inst, model, member, 'rsds'),
                         'rsds', list(range(1980, 2016)), [MONTH])
    si = C.load_filtered(C.find_var_files(inst, model, member, 'siconc'),
                         'siconc', list(range(1980, 2016)), [MONTH])

    _, alb_lat, alb_lon, alb_2d = C._std_latlon(ru)
    _, sic_lat, sic_lon, sic_2d = C._std_latlon(si)
    alb_sh = alb_lat.shape if alb_2d else (len(alb_lat), len(alb_lon))
    sv = si.values.astype(float)
    if np.nanmax(sv) > 1.5:
        sv /= 100.
    sic_sh = sv.shape[1:]

    print(f'\n=== {model}  (member {member}) ===')
    print(f'  albedo grid : {alb_sh}   ({np.prod(alb_sh):,} cells)')
    print(f'  siconc grid : {sic_sh}   ({np.prod(sic_sh):,} cells)')
    print(f'  regrid needed: {sic_sh != alb_sh}')

    dup = None
    if sic_sh != alb_sh:
        if sic_2d:
            sla = C._mask_fills(sic_lat).ravel()
            slo = C._mask_fills(sic_lon).ravel()
        else:
            slo2, sla2 = np.meshgrid(sic_lon, sic_lat)
            sla, slo = C._mask_fills(sla2).ravel(), C._mask_fills(slo2).ravel()
        if alb_2d:
            ala = C._mask_fills(alb_lat).ravel()
            alo = C._mask_fills(alb_lon).ravel()
        else:
            alo2, ala2 = np.meshgrid(alb_lon, alb_lat)
            ala, alo = ala2.ravel(), alo2.ravel()
        ok = np.isfinite(sla) & np.isfinite(slo)
        _, idx = cKDTree(C.latlon_to_xyz(sla[ok], slo[ok])).query(
            C.latlon_to_xyz(ala, alo).astype(np.float32), workers=-1)
        uniq = len(np.unique(idx))
        dup = len(idx) / uniq
        print(f'  NN mapping  : {len(idx):,} targets <- {uniq:,} distinct '
              f'sources  (mean duplication {dup:.2f}x)')
        sv = sv.reshape(sv.shape[0], -1)[:, ok][:, idx].reshape(
            (sv.shape[0],) + alb_sh)

    ctx = C.build_native_ctx(alb_lat, alb_lon, alb_2d)
    lat2d = ctx['lat2d']

    yrs = np.array([int(t.dt.year) for t in si.time])
    mon = np.array([int(t.dt.month) for t in si.time])
    tm = (mon == MONTH) & np.isin(yrs, C.TREND_YEARS)
    em = (mon == MONTH) & np.isin(yrs, C.EARLY_YEARS)
    lm = (mon == MONTH) & np.isin(yrs, C.LATE_YEARS)

    sic_e, sic_l = np.nanmean(sv[em], 0), np.nanmean(sv[lm], 0)
    slope, pval = C.pixel_ols_trend(yrs[tm].astype(float), sv[tm])
    all_ice = (lat2d >= 60.) & (sic_e >= C.SIC_ICE_THRESHOLD)
    chang = (all_ice & np.isfinite(slope) & np.isfinite(pval)
             & (slope < 0) & (pval < C.PVAL))
    stable = all_ice & ~chang & (np.fmax(sic_e, sic_l) >= C.SIC_ICE_THRESHOLD)

    sd = np.nanstd(sv[tm], axis=0)
    for name, msk in (('area_loss', chang), ('darkening', stable)):
        s = sd[msk]
        s = s[np.isfinite(s)]
        if not s.size:
            print(f'  {name:10s}: empty'); continue
        print(f'  {name:10s}: n={s.size:6,}  SIC sd  median={np.median(s):.4f}'
              f'  p10={np.percentile(s,10):.4f}'
              f'  frac(sd<0.01)={np.mean(s<0.01):.3f}'
              f'  frac(sd<0.001)={np.mean(s<0.001):.4f}')
    return dup


if __name__ == '__main__':
    for m in ['BCC-CSM2-MR', 'MIROC6', 'CESM2']:
        try:
            probe(m)
        except Exception as e:
            print(f'{m}: FAILED {e}')
