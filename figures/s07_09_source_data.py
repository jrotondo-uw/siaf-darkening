#!/usr/bin/env python
"""Source data for Supplementary Figures 7-9: the SIC-trend classification maps.

Reads the same pickle as s07.py, s08.py and s09.py and writes, for each figure, one
row per grid cell that is classified in at least one of its twelve panels:
  lat, lon                    cell centre (degrees north, degrees east)
  <period>_<month>_p<p>       1 = area loss (significant negative SIC trend at p),
                              0 = darkening (active ice outside significant retreat),
                              blank = not active ice (open ocean with mean SIC below
                              15% in both the first and the last decade) or land
Supplementary Fig. 7 is April-May, Fig. 8 June-July and Fig. 9 August-September.
The per-panel shares printed on the maps are in SourceData_Supplementary_Figure<N>.csv.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PKL = '/glade/work/jrotondo/ncc_paper/plots/plot_data_sic_trend_maps.pkl'
SD = Path(__file__).resolve().parent.parent / 'source_data'
PERIOD = {'obs': '1982-2024', 'cmip6': '1982-2014'}
FIGS = {7: ['April', 'May'], 8: ['June', 'July'], 9: ['August', 'September']}

with open(PKL, 'rb') as fh:
    d = pickle.load(fh)
lats, lons = d['lats'], d['lons']
for n, months in FIGS.items():
    cols, any_ok = {}, np.zeros(lats.shape, bool)
    for exp, _, _ in d['EXPERIMENTS']:
        for mo in months:
            for p in d['PVAL_THRESHOLDS']:
                a = np.asarray(d['sig_maps'][(exp, mo, p)], dtype=float)
                ok = np.isfinite(a)
                assert set(np.unique(a[ok])) <= {0.0, 1.0}
                m = d['meta'][(exp, mo, p)]
                assert int(ok.sum()) == m['n_all'] and int((a == 1).sum()) == m['n_ext'], (exp, mo, p)
                cols[f'{PERIOD[exp]}_{mo}_p{p:.2f}'] = a
                any_ok |= ok
    out = pd.DataFrame({'lat': lats[any_ok], 'lon': lons[any_ok]})
    for k, a in cols.items():
        out[k] = pd.array(np.where(np.isfinite(a[any_ok]), a[any_ok], np.nan), dtype='Int8')
    dest = SD / f'SourceData_Supplementary_Figure{n}_maps.csv'
    out.to_csv(dest, index=False, float_format='%.4f')
    print(f'wrote {dest.name}: {len(out)} cells x {len(cols)} panels')
