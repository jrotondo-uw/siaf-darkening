#!/usr/bin/env python
"""Source data for Supplementary Fig. 5, the likelihood-selected subset (24 Nov 2026).

s05.py draws the CMIP6 models whose total-SIAF BMA weight in the 21-model pool
(full_T07, at least five members) is at or above the pool's 90th percentile:
one Gaussian per model (member mean and member spread), the unweighted,
BMA-weighted and variance-constrained mixtures of those Gaussians, and the
observed center with its model-based uncertainty. This script writes those
plotted quantities, one row per component and model, with s05.py's own
selection and weighting.

Columns
  component               ext (area loss), drk (darkening) or tot (total), as
                          in SourceData_Figure8.csv
  model, n_members        subset model and its retained members
  model_mean, model_sigma member mean and 1-sigma member spread (ddof=1),
                          W m-2 K-1: the model's plotted Gaussian
  pool_weight_tot         the model's total-SIAF BMA weight in the 21-model
                          pool, which is the selection criterion
  pool_percentile         percent of pool models whose weight is at or below it
  subset_weight           BMA weight within the subset for this component, as
                          used by the BMA-weighted curve
  obs_center, obs_2sigma  observed SIAF (CERES-APRP, 1982-2014, p < 0.05) and
                          its model-based 2-sigma; the plotted observed
                          Gaussian has scale obs_2sigma / 2

Inputs are byte-identical copies of the files s05.py reads:
SourceData_Figure6.csv (siaf_cmip6_summary_final.csv), SourceData_Figure4.csv
(siaf_pixelwise_summary.csv) and uncertainty.csv. Output goes to
../source_data/ when run from the repository, or to the path given as the
first argument.
usage: python s05_source_data.py [output.csv]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SD = HERE.parent / 'source_data'
MIN_MEMBERS, TOP_PCT = 5, 90
COLS = {'ext': 'siaf_ext_global', 'drk': 'siaf_drk_global', 'tot': 'siaf_global'}
OBSCOL = {'ext': 'SIAF_ext_global', 'drk': 'SIAF_drk_global', 'tot': 'SIAF_tot_global'}
UNC = {'ext': 'area_loss', 'drk': 'darkening', 'tot': 'total'}

d = pd.read_csv(SD / 'SourceData_Figure6.csv')
d = d[d.rs_method == 'full_T07']
cnt = d.groupby('model').size()
d = d[d.model.isin(cnt[cnt >= MIN_MEMBERS].index)]

o = pd.read_csv(SD / 'SourceData_Figure4.csv')
o = o[(o.rs_method == 'CERES') & (o.period == 'CMIP6') & np.isclose(o.pval, 0.05)].iloc[0]
u = (pd.read_csv(HERE / 'uncertainty.csv')
     .query("method == 'Common MMM trend' and block_years == 3").set_index('component'))
OBS = {k: (float(o[OBSCOL[k]]), float(u.loc[UNC[k], 'joint_2sigma'])) for k in COLS}


def norm(x, mu, sd):
    sd = max(float(sd), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def bma_w(g, mu_o, sd_o):
    w = np.array([norm(mu_o, m, np.sqrt(s ** 2 + sd_o ** 2))
                  for m, s in zip(g['mean'], g['std'])])
    return w / w.sum() if w.sum() > 0 else np.ones(len(g)) / len(g)


# Selection on the total SIAF, exactly as in s05.py.
mu_t, two_t = OBS['tot']
g_sel = d.groupby('model')[COLS['tot']].agg(['mean', 'std']).dropna()
w_sel = bma_w(g_sel, mu_t, two_t / 2)
wmap = dict(zip(g_sel.index, w_sel))
keep = g_sel.index[w_sel >= np.percentile(w_sel, TOP_PCT)].tolist()
keep = sorted(keep, key=lambda m: -wmap[m])
sub = d[d.model.isin(keep)]

rows = []
for k, col in COLS.items():
    mu_o, two_o = OBS[k]
    g = sub.groupby('model')[col].agg(['mean', 'std', 'count']).dropna()
    w = dict(zip(g.index, bma_w(g, mu_o, two_o / 2)))
    for m in keep:
        rows.append(dict(component=k, model=m, n_members=int(g.loc[m, 'count']),
                         model_mean=g.loc[m, 'mean'], model_sigma=g.loc[m, 'std'],
                         pool_weight_tot=wmap[m],
                         pool_percentile=100 * (w_sel <= wmap[m]).mean(),
                         subset_weight=w[m], obs_center=mu_o, obs_2sigma=two_o))
res = pd.DataFrame(rows)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else SD / 'SourceData_Supplementary_Figure5.csv'
res.to_csv(out, index=False)
print(f'pool: {g_sel.shape[0]} models, {len(d)} members; '
      f'subset: {len(keep)} models, {len(sub)} members')
for m in keep:
    print(f'   {m:18s} w={wmap[m]:.3f}  {100 * (w_sel <= wmap[m]).mean():.0f}th pct')
for k in COLS:
    r = res[res.component == k]
    print(f'  {k}: obs={OBS[k][0]:.4f}+/-{OBS[k][1] / 2:.4f} (1 sigma)  '
          f'subset mean={r.model_mean.mean():.4f}')
print('->', out)
