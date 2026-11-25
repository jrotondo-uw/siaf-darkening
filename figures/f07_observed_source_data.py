#!/usr/bin/env python
"""Source data for the observational elements of main Fig. 7 (24 Nov 2026).

SourceData_Figure7.csv holds every plotted ensemble member. This script writes
SourceData_Figure7_Observed.csv: the elements drawn on top of the members, with
the intervals exactly as f07.py draws them.

  Observed, CERES-APRP / ERA5 kernel  the two observational points. RS interval:
                                      pairs-bootstrap 2-sigma for CERES-APRP; the
                                      ERA5-kernel RS is a fixed climatological
                                      field with no bootstrap uncertainty (0).
                                      IS interval: the observed residual-based 2-sigma
                                      (observed_minus_MMM_consistent_widths.csv,
                                      3-year blocks), which is also
                                      the gray band.
  Family-weighted multi-model mean    18 family means (members -> models ->
                                      families); interval is 2-sigma across the
                                      family means (ddof=1), as in f07.py.

Inputs are the files f07.py reads. Output goes to ../source_data/ when run from
the repository, or to the path given as the first argument.
usage: python f07_observed_source_data.py [output.csv]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CMIP6_CSV = HERE/'siaf_cmip6_summary_final.csv'
OBS_CSV = HERE/'siaf_pixelwise_summary.csv'
MIN_MEMBERS = 3

FAMILY = {**{m: 'CESM2' for m in ['CESM2', 'CESM2-WACCM', 'CESM2-FV2', 'CESM2-WACCM-FV2']},
          **{m: 'EC-Earth3' for m in ['EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR',
                                      'EC-Earth3-CC', 'EC-Earth3-AerChem', 'EC-Earth3-ESM-1']},
          **{m: 'CMCC' for m in ['CMCC-CM2-SR5', 'CMCC-ESM2', 'CMCC-CM2-HR4']},
          **{m: 'E3SM' for m in ['E3SM-1-0', 'E3SM-1-1', 'E3SM-1-1-ECA']},
          **{m: 'HadGEM3/UKESM' for m in ['HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
                                          'UKESM1-0-LL', 'UKESM1-1-LL']},
          **{m: 'MPI/ICON' for m in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
                                     'MPI-ESM-1-2-HAM', 'ICON-ESM-LR']},
          **{m: 'NorESM/NorCPM' for m in ['NorESM2-LM', 'NorESM2-MM', 'NorCPM1']},
          **{m: 'CNRM' for m in ['CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1']},
          **{m: 'CanESM5' for m in ['CanESM5', 'CanESM5-CanOE']},
          **{m: 'MIROC' for m in ['MIROC6', 'MIROC-ES2L']},
          **{m: 'IPSL' for m in ['IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'IPSL-CM5A2-INCA']},
          **{m: 'INM' for m in ['INM-CM4-8', 'INM-CM5-0']},
          **{m: 'FGOALS' for m in ['FGOALS-g3', 'FGOALS-f3-L']},
          **{m: 'BCC' for m in ['BCC-CSM2-MR', 'BCC-ESM1']},
          **{m: 'ACCESS' for m in ['ACCESS-CM2', 'ACCESS-ESM1-5']},
          **{m: 'GISS' for m in ['GISS-E2-1-H', 'GISS-E2-1-G']}}

U = pd.read_csv(HERE.parent/'analysis_data/observed_uncertainty_20261125/observed_minus_MMM_consistent_widths.csv')
IS_2S_MODEL = float(U.query("metric=='full_IS' and component=='total'").block3_2sigma.iloc[0])

O = pd.read_csv(OBS_CSV)
rows = []
RS_INTERVAL = {'CERES': 'pairs-bootstrap 2 sigma',
               'Kernel': 'none: fixed climatological field, no bootstrap uncertainty'}
for method, label in (('CERES', 'CERES-APRP'), ('Kernel', 'ERA5 kernel')):
    r = O[(O.rs_method == method) & (O.period == 'CMIP6') & np.isclose(O.pval, 0.05)].iloc[0]
    if method == 'Kernel':
        assert float(r.sigma2_RS_arctic) == 0.0, r.sigma2_RS_arctic
    rows.append(dict(element='Observed', rs_method=label,
                     rs_arctic_Wm2=float(r.RS_arctic), rs_arctic_2sigma_Wm2=float(r.sigma2_RS_arctic),
                     rs_interval=RS_INTERVAL[method],
                     is_tot_arctic_perK=float(r.IS_tot_arctic), is_tot_arctic_2sigma_perK=IS_2S_MODEL,
                     is_interval='observed residual-based 2 sigma (gray band)'))

df = pd.read_csv(CMIP6_CSV)
df = df[df.rs_method == 'full_T07'].copy()
cnt = df.groupby('model').size()
df = df[df.model.isin(cnt[cnt >= MIN_MEMBERS].index)].dropna(subset=['rs_arctic', 'is_tot_arctic'])
df['family'] = df.model.map(lambda m: FAMILY.get(m, m))
fam = df.groupby(['family', 'model'])[['rs_arctic', 'is_tot_arctic']].mean().groupby('family').mean()
assert (len(df), df.model.nunique(), len(fam)) == (378, 31, 18), (len(df), df.model.nunique(), len(fam))
rows.append(dict(element='Family-weighted multi-model mean', rs_method='CMIP6 (378 members, 31 models, 18 families)',
                 rs_arctic_Wm2=fam.rs_arctic.mean(), rs_arctic_2sigma_Wm2=2 * fam.rs_arctic.std(ddof=1),
                 rs_interval='2 sigma across the 18 family means',
                 is_tot_arctic_perK=fam.is_tot_arctic.mean(), is_tot_arctic_2sigma_perK=2 * fam.is_tot_arctic.std(ddof=1),
                 is_interval='2 sigma across the 18 family means'))

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'source_data' / 'SourceData_Figure7_Observed.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string())
print('wrote', out)
