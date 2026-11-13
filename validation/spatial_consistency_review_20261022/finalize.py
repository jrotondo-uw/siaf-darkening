from pathlib import Path
import pandas as pd,pickle,numpy as np
R=Path(__file__).resolve().parent
for folder in ['sheba_surface_flux','tara_surface_flux','sheba_perovich_transects']:
 for p in (R/folder).glob('*matchups.csv'):
  d=pd.read_csv(p);d=d.rename(columns={c:'legacy_'+c for c in ['n_ov','sza','nobs'] if c in d});d.to_csv(p,index=False)
s=pd.read_csv(R/'spatial_consistency_stats.csv');s[~s.platform.str.startswith('Barrow')].to_csv(R/'spatial_consistency_stats.csv',index=False)
with (R/'validation_spatial_consistent.pkl').open('rb') as f:d=pickle.load(f)
with open('/glade/work/jrotondo/combined_validation/validation_data.pkl','rb') as f:old=pickle.load(f)
for kind in ['transect','heli','drone','arise','arcsix']:
 for product in ['c','a']:pd.testing.assert_frame_equal(d[f'df_{kind}_{product}'],old[f'df_{kind}_{product}'])
for kind in ['tower','summit','raven']:pd.testing.assert_frame_equal(d[f'df_{kind}_a'],old[f'df_{kind}_a'])
for kind in ['tower','summit','raven']:
 new=d[f'df_{kind}_c'];prev=old[f'df_{kind}_c'];assert new.index.isin(prev.index).all()
 cols=['tower_sza_matched','tower_scaled'] if kind=='tower' else ['tower_albedo']
 for c in cols:np.testing.assert_allclose(new[c],prev.loc[new.index,c],equal_nan=True)
print('Unchanged panels, dates and in-situ references verified. Barrow excluded from active statistics.')
