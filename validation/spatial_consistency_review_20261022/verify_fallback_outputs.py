from pathlib import Path
import pickle,pandas as pd,numpy as np
R=Path(__file__).resolve().parent;B=R.parent
with (R/'validation_IDW_fallback.pkl').open('rb') as f:D=pickle.load(f)
with open('/glade/work/jrotondo/combined_validation/validation_data.pkl','rb') as f:old=pickle.load(f)
for kind in ['transect','heli','drone','arise','arcsix']:
 a=D['df_'+kind+'_c'];b=old['df_'+kind+'_c'];pd.testing.assert_frame_equal(a[b.columns],b)
for kind in ['tower','summit','raven','transect','heli','drone','arise','arcsix']:
 for suffix in ['c','a']:
  a=D[f'df_{kind}_{suffix}'];b=old[f'df_{kind}_{suffix}']
  if not len(a):continue
  assert a.index.isin(b.index).all()
  cols=['tower_sza_matched','tower_scaled'] if kind=='tower' else [{'summit':'tower_albedo','raven':'tower_albedo','transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind]]
  for c in cols:np.testing.assert_allclose(a[c],b.loc[a.index,c],equal_nan=True)
  assert a.spatial_method.isin(['idw','nearest','idw+nearest']).all()
for folder,stem in [('sheba_surface_flux','sheba'),('tara_surface_flux','tara_broadband'),('sheba_perovich_transects','perovich')]:
 for product,col in [('clara','pentad'),('appx','date')]:
  a=pd.read_csv(R/'fallback'/folder/f'{stem}_{product}_matchups.csv').set_index(col);b=pd.read_csv(B/folder/f'{stem}_{product}_matchups.csv').set_index(col)
  assert not a.index.duplicated().any() and a.index.isin(b.index).all();np.testing.assert_allclose(a.insitu,b.loc[a.index,'insitu'])
  assert a.spatial_method.isin(['idw','nearest','idw+nearest']).all()
s=pd.read_csv(R/'IDW_fallback_stats.csv');assert not s.platform.str.contains('Barrow').any();assert (s.fallback_rows<=s.n).all()
print('Verified: original dates/reference values preserved; successful original CLARA IDW panels unchanged; fallback labels valid; Barrow excluded.')
