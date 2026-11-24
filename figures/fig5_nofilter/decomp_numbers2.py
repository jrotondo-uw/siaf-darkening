#!/usr/bin/env python
"""Every number the paper quotes from the direct-slope decomposition (Fig. 5c-d,
Results, Methods, Supp. Table 8, response R2 C5), from one model CSV.

usage: python decomp_numbers2.py MODEL_CSV [TAG] [OBS_STAGE3_CSV]
Observations: the historical-only (1982-2014) rerun of 12 Nov 2026. Never the
ncc_paper/plots/bivariate_diag/stage3 file (mixed period).
"""
import sys
import numpy as np
import pandas as pd

MODEL_CSV = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
NB = '/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028'
OBS = '/glade/derecho/scratch/jrotondo/fig5_review_20261112/observations/stage3_bivariate.csv'
if len(sys.argv) > 3:
    OBS = sys.argv[3]   # alternative observational stage3 (e.g. the no-filter sensitivity run)
SIG = NB + '/f05_cd_obs_sigma.csv'
M = ['April', 'May', 'June', 'July', 'August', 'September']

d = pd.read_csv(MODEL_CSV)
o = pd.read_csv(OBS).set_index(['region', 'month'])
sg = pd.read_csv(SIG)
sg['month_name'] = sg.month.map(dict(zip(range(4, 10), M)))
sg = sg.set_index(['region', 'month_name'])
print(f'===== {TAG}  {MODEL_CSV}\n      obs: {OBS}')
print('models:', d.model.nunique(), ' rows:', len(d), ' n_years:',
      d.n_years.min(), '-', d.n_years.max())


def monthly(region, var):
    rows = []
    for m in M:
        v = d[(d.region == region) & (d.month == m)][var].astype(float).dropna()
        rows.append(dict(month=m, n=len(v), mean=v.mean(), sd1=v.std(ddof=1),
                         obs=float(o.loc[(region, m), var])))
    t = pd.DataFrame(rows).set_index('month')
    t['ratio'] = t['mean'] / t['obs']
    return t


def season_models(region, var):
    """Per-model season mean, only for models present in all six months."""
    s = d[d.region == region].pivot_table(index='model', columns='month', values=var)
    full = s.dropna()
    return s, full[M].mean(axis=1)


for region, var, lab in (('area_loss', 'L', 'L on area-loss'),
                         ('darkening', 'I', 'IS on darkening'),
                         ('darkening', 'L', 'residual L on darkening'),
                         ('area_loss', 'I', 'total I on area-loss')):
    t = monthly(region, var)
    s_mod, s_obs = t['mean'].mean(), t['obs'].mean()
    print(f'\n--- {lab}')
    print(t.round(4).to_string())
    print(f'season: model {s_mod:.4f}  obs {s_obs:.4f}  ratio-of-means {s_mod/s_obs:.4f}'
          f'  mean-of-ratios {t.ratio.mean():.4f}  monthly ratio range '
          f'{t.ratio.min():.2f}-{t.ratio.max():.2f}')
    allm, full = season_models(region, var)
    below = int((full < s_obs).sum())
    print(f'models in all 6 months: {len(full)} of {len(allm)};  below obs season mean: '
          f'{below};  above: {sorted(full[full >= s_obs].index)}')
    if (region, var) in (('area_loss', 'L'), ('darkening', 'I')):
        # overlap of model mean +/- 2 sd (ddof=1) with obs +/- model-based 2 sigma
        hw = sg.loc[region]['halfwidth_2sigma'].reindex(M).values
        lo_m, hi_m = t['mean'] - 2 * t['sd1'], t['mean'] + 2 * t['sd1']
        lo_o, hi_o = t['obs'] - hw, t['obs'] + hw
        ov = (lo_m <= hi_o) & (lo_o <= hi_m)
        print('obs 2sigma half-width:', dict(zip(M, np.round(hw, 4))))
        print('intervals overlap:', dict(zip(M, ov.values)))
        inside = (t['obs'] >= lo_m) & (t['obs'] <= hi_m)
        print('obs value inside model mean +/- 2sd:', dict(zip(M, inside.values)))
        # per-month counts of models below the observed monthly value
        cnt = {m: int((d[(d.region == region) & (d.month == m)][var] < t.loc[m, 'obs']).sum())
               for m in M}
        print('models below obs, by month:', cnt)

# r across models between total albedo sensitivity I and L, each region
print('\n--- r(I, L) across models')
for region in ('area_loss', 'darkening'):
    piv = d[d.region == region].pivot_table(index='model', columns='month', values=['I', 'L'])
    full = piv.dropna()
    sI, sL = full['I'][M].mean(axis=1), full['L'][M].mean(axis=1)
    r_full = np.corrcoef(sI, sL)[0, 1]
    # alternative: season mean over available months
    aI = piv['I'].mean(axis=1)
    aL = piv['L'].mean(axis=1)
    r_avail = np.corrcoef(aI, aL)[0, 1]
    print(f'{region}: r(all-6-month models, n={len(full)}) = {r_full:+.3f};  '
          f'r(available months, n={len(aI)}) = {r_avail:+.3f}')

# optical shortfall after removing the residual concentration term
tI = monthly('darkening', 'I')
tL = monthly('darkening', 'L')
IS_m, IS_o = tI['mean'].mean(), tI['obs'].mean()
Lr_m, Lr_o = tL['mean'].mean(), tL['obs'].mean()
print('\n--- optical shortfall: 1 - (IS_m - c Lres_m)/(IS_o - c Lres_o)')
for c in (0.54, 0.70):
    print(f'c={c}: {1 - (IS_m - c * Lr_m) / (IS_o - c * Lr_o):.4f}')
print(f'deficits: L {1 - monthly("area_loss", "L")["mean"].mean() / monthly("area_loss", "L")["obs"].mean():.4f}'
      f'  IS {1 - IS_m / IS_o:.4f}')
may = tI.loc['May']
print(f'May darkening: model {may["mean"]:.4f} obs {may["obs"]:.4f} deficit {1 - may["ratio"]:.3f}')
