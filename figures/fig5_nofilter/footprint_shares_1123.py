#!/usr/bin/env python
"""Darkening share of each model's active-ice footprint, by month (31 models).

share = drk / (drk + loss) per model and month, from sd_filter_check.csv
(one member per model, the Fig. 5c-d member). Two measures:
  count  - native-grid pixel counts (what the 24-model text quoted)
  area   - cos-latitude-weighted area, the weighting of the model region means
Both are the full footprints (pixels the SIC-sigma filter refuses are counted).
A region-month absent for a model (fewer than 20 valid pixels) counts as zero.
Observed shares are true area fractions from stage2_masks.csv.
"""
import pandas as pd

D = '/glade/derecho/scratch/jrotondo'
M = ['April', 'May', 'June', 'July', 'August', 'September']
c = pd.read_csv(f'{D}/fig5_rerun31_patched_20261123/sd_filter_check.csv')
c = c[c._merge != 'right_only']
o = pd.read_csv(f'{D}/fig5_review_20261112/observations/stage2_masks.csv').set_index('month')
obs = (o.area_frac_stable / (o.area_frac_chang + o.area_frac_stable)).reindex(M)
print('observed darkening share of active-ice area:', (100 * obs).round(1).to_dict())
for lab, col in (('count', 'n_pixels'), ('area', 'area')):
    p = c.pivot_table(index=['model', 'month'], columns='region', values=col, aggfunc='sum').fillna(0.0)
    p['share'] = 100 * p.darkening / (p.darkening + p.area_loss)
    s = p.share.unstack('month').reindex(columns=M)
    print(f'\n[{lab}] models per month:', s.notna().sum().to_dict())
    print(pd.DataFrame({'median': s.median(), 'min': s.min(), 'p10': s.quantile(.1),
                        'p90': s.quantile(.9), 'max': s.max(),
                        'n_above_obs': (s.gt(100 * obs, axis=1)).sum()}).round(1).to_string())
