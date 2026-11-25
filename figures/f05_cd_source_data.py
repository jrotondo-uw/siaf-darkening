#!/usr/bin/env python
"""Source data for Figure 5c-d: the individual model points drawn in grey.

Reads figures/f05_cd_models.csv, the table f05.py scatters behind the model mean,
and writes source_data/SourceData_Figure5_cd_models.csv with the plotted value only:
L = -dSIC/dT on the area-loss region (panel c) and IS = -dalpha/dT on the darkening
region (panel d), both per kelvin of global mean temperature, one retained member
per model. Values that are missing for a month are dropped, as in the figure, and
the horizontal jitter in the figure is for display only. The model means, the
model 2-sigma spread and the observations are in SourceData_Figure5_cd.csv.
"""
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parent
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']
m = pd.read_csv(R / 'f05_cd_models.csv')
parts = []
for panel, region, col, name in (('c', 'area_loss', 'L', 'L'), ('d', 'darkening', 'I', 'IS')):
    s = m[(m.region == region) & m[col].notna()]
    parts.append(pd.DataFrame({'panel': panel, 'region': region, 'month': s.month,
                               'model': s.model, 'member': s.member, 'variable': name,
                               'value_per_K': s[col].astype(float),
                               'n_pixels': s.n_pixels, 'n_years': s.n_years}))
out = pd.concat(parts)
out['month'] = pd.Categorical(out.month, MONTHS, ordered=True)
out = out.sort_values(['panel', 'month', 'model']).reset_index(drop=True)
dest = R.parent / 'source_data' / 'SourceData_Figure5_cd_models.csv'
out.to_csv(dest, index=False, float_format='%.6g')
print(f'wrote {dest.name}: {len(out)} rows')
# cross-check against the published panel table (not written anywhere)
cd = pd.read_csv(R.parent / 'source_data' / 'SourceData_Figure5_cd.csv')
g = out.groupby(['region', 'month'], observed=True).value_per_K.agg(['mean', 'std', 'count']).reset_index()
chk = cd.merge(g, on=['region', 'month'])
chk['d_mean'] = chk['mean'] - chk.model_mean
chk['d_2sig'] = 2 * chk['std'] - chk.model_2sigma
print(chk[['panel', 'region', 'month', 'n_models', 'count', 'model_mean', 'd_mean', 'd_2sig']].to_string(index=False))
