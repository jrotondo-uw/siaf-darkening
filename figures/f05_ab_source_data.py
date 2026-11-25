#!/usr/bin/env python
"""Source data for Figure 5a-b: the CMIP6 family means and the observed point.

Runs the same aggregation as f05.py (the saved sie_aggregation_source.py without its
plotting section) and writes source_data/SourceData_Figure5.csv with one row per
CMIP6 family (kind = family) and one row for the observations (kind = observed):
  n_models, n_members        models and members behind each family mean
  sie_trend_1e6km2_per_yr    sea ice extent trend, panel a x axis (10^6 km2 yr-1)
  sie_1982_1991_1e6km2       1982-1991 sea ice extent, panel b x axis (10^6 km2)
  siaf_total                 total SIAF, y axis of both panels (W m-2 K-1)
  *_2sigma                   the error bars drawn: 2 sigma across a family's members,
                             and for the observed total the model-based 2-sigma interval
Family values are family means of model means, as plotted.
The inputs (sie_aggregation_source.py, sie_df_common.pkl, obs_sie_by_month.pkl,
siaf_pixelwise_summary.csv, siaf_cmip6_summary_final.csv) are read from the folder
given as the first argument, default this script's folder, as f05.py reads them.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
OBS_2SIGMA = 0.04384      # the observed interval f05.py draws (model-based 2 sigma)
assert 'yerr=.04384' in (HERE / 'f05.py').read_text(), 'f05.py no longer draws 0.04384'

src = (R / 'sie_aggregation_source.py').read_text()
src = src[:src.index('lbls = list(fam.index)')]
src = src.replace("os.makedirs(OUT_DIR, exist_ok=True)", '')
src = (src.replace("pd.read_pickle(SIE_PKL)", "pd.read_pickle(R/'sie_df_common.pkl')")
          .replace("load_cache('obs_sie_by_month')", "pd.read_pickle(R/'obs_sie_by_month.pkl')")
          .replace('pd.read_csv(OBS_SIAF_CSV)', "pd.read_csv(R/'siaf_pixelwise_summary.csv')")
          .replace('pd.read_csv(CMIP6_SIAF_CSV)', "pd.read_csv(R/'siaf_cmip6_summary_final.csv')"))
exec(compile(src, 'saved_sie_aggregation', 'exec'))

fams = pd.DataFrame({'kind': 'family', 'family': fam.index,
                     'n_models': fam.n_mod.astype(int).values, 'n_members': fam.n_mem.astype(int).values,
                     'sie_trend_1e6km2_per_yr': fam.trend.values, 'sie_trend_2sigma': fam_sd.trend.values,
                     'sie_1982_1991_1e6km2': fam.early.values, 'sie_1982_1991_2sigma': fam_sd.early.values,
                     'siaf_total': fam.siaf.values, 'siaf_total_2sigma': fam_sd.siaf.values})
obs = pd.DataFrame([{'kind': 'observed', 'family': 'observations',
                     'sie_trend_1e6km2_per_yr': obs_slope, 'sie_1982_1991_1e6km2': obs_early,
                     'siaf_total': OBS_TOT, 'siaf_total_2sigma': OBS_2SIGMA}])
out = pd.concat([fams, obs], ignore_index=True)
out['n_models'] = out.n_models.astype('Int64'); out['n_members'] = out.n_members.astype('Int64')
dest = HERE.parent / 'source_data' / 'SourceData_Figure5.csv'
out.to_csv(dest, index=False, float_format='%.6g')
print(f'wrote {dest.name}: {len(fams)} families + observed')
