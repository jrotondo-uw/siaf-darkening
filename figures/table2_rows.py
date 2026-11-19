"""Rows for the new summary table (Table 2).

Sources are the LIVE files, not the Oct 27 copies that sit in revision_20261028:
  obs    /glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv
  models /glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv
  unc    revision_20261028/uncertainty.csv   (common-MMM, 3-yr blocks)

The local response is SIAF per unit active area divided by SCALE = AREA_RATIO *
SEASON_FRAC, the same normalization f06.py uses for its panel b, which is what
reproduces the 32.8 and 13.9 W m-2 K-1 quoted in the Results.
"""
import numpy as np
import pandas as pd

NB = '/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028'
SCALE = 0.0347 * 0.5

F6 = open(NB + '/f06.py').read()
_hdr = F6[F6.index('MIN_MEMBERS'):F6.index('df = pd.read_csv')]
_ns = {}
exec(compile(_hdr[:_hdr.index('matplotlib.rcParams')] + _hdr[_hdr.index('FAMILY = {'):],
             'f06_header', 'exec'), _ns)
FAM, MIN_MEMBERS = _ns['FAMILY'], _ns['MIN_MEMBERS']
SCALE = _ns['SCALE']

df = pd.read_csv('/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv')
df = df[df.rs_method == 'full_T07'].copy()
cnt = df.groupby('model').size()
df = df[df.model.isin(cnt[cnt >= MIN_MEMBERS].index)].copy()
df['family'] = df.model.map(lambda m: FAM.get(m, m))
df['ext_loc'] = df.siaf_ext_per_area / SCALE
df['drk_loc'] = df.siaf_drk_per_area / SCALE

o = pd.read_csv('/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv')
o = o[(o.rs_method == 'CERES') & (o.period == 'CMIP6') & (np.isclose(o.pval, 0.05))].iloc[0]
unc = pd.read_csv(NB + '/uncertainty.csv').query(
    "method == 'Common MMM trend' and block_years == 3").set_index('component')

OBS = {'SIAF_tot':   (float(o.SIAF_tot_global), float(unc.loc['total', 'joint_2sigma'])),
       'SIAF_area':  (float(o.SIAF_ext_global), float(unc.loc['area_loss', 'joint_2sigma'])),
       'SIAF_drk':   (float(o.SIAF_drk_global), float(unc.loc['darkening', 'joint_2sigma'])),
       'local_area': (float(o.SIAF_ext_per_area) / SCALE, None),
       'local_drk':  (float(o.SIAF_drk_per_area) / SCALE, None),
       'RS':         (float(o.RS_arctic), None),
       'IS':         (float(o.IS_tot_arctic), None)}
COLS = {'SIAF_tot': 'siaf_global', 'SIAF_area': 'siaf_ext_global',
        'SIAF_drk': 'siaf_drk_global', 'local_area': 'ext_loc',
        'local_drk': 'drk_loc', 'RS': 'rs_arctic', 'IS': 'is_tot_arctic'}

hdr = '{:11s} {:>17s} {:>9s} {:>9s} {:>17s} {:>9s}'
print(hdr.format('quantity', 'observed', 'MMM-mod', 'MMM-fam', 'family range', 'deficit'))
for k, col in COLS.items():
    per_model = df.groupby('model')[col].mean()
    per_fam = df.groupby(['family', 'model'])[col].mean().groupby('family').mean()
    mmm = per_model.mean()
    fmmm = per_fam.mean()
    mu, err = OBS[k]
    obs_s = '{:.3f} +/- {:.3f}'.format(mu, err) if err else '{:.3f}'.format(mu)
    rng = '{:.3f}-{:.3f}'.format(per_fam.min(), per_fam.max())
    print(hdr.format(k, obs_s, '{:.3f}'.format(mmm), '{:.3f}'.format(fmmm), rng,
                     '{:.0f}%'.format(100 * (1 - fmmm / mu))))

print()
print('models {}  members {}  families {}'.format(
    df.model.nunique(), len(df), df.family.nunique()))
print('RS family-weighted MMM {:.0f}   model-mean range {:.0f}-{:.0f}'.format(
    df.groupby(['family', 'model']).rs_arctic.mean().groupby('family').mean().mean(),
    df.groupby('model').rs_arctic.mean().min(), df.groupby('model').rs_arctic.mean().max()))
