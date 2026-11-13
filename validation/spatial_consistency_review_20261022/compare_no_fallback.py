from pathlib import Path
import pandas as pd,numpy as np,pickle
R=Path(__file__).resolve().parent
with (R/'validation_IDW_only_controlled.pkl').open('rb') as f:S=pickle.load(f)
with (R/'validation_IDW_fallback.pkl').open('rb') as f:F=pickle.load(f)
rows=[]
def compare(name,product,s,f,x,y):
 # Validate that IDW-only data equal the fallback version wherever no fallback was used.
 common=s.index.intersection(f.index);p=f.loc[common]
 if 'spatial_method' in p:
  keep=~p.spatial_method.str.contains('nearest');idx=common[keep]
  np.testing.assert_allclose(s.loc[idx,y],f.loc[idx,y],rtol=1e-5,atol=1e-7,equal_nan=True)
 for label,d in [('IDW only',s),('IDW + nearest',f)]:
  xx=d[x].to_numpy(float);yy=d[y].to_numpy(float);ok=np.isfinite(xx)&np.isfinite(yy);e=yy[ok]-xx[ok]
  rows.append(dict(platform=name,product=product,method=label,n=len(e),bias=e.mean() if len(e) else np.nan,rmse=np.sqrt((e**2).mean()) if len(e) else np.nan,r=np.corrcoef(xx[ok],yy[ok])[0,1] if len(e)>2 else np.nan))
for kind in ['tower','summit','raven','transect','heli','drone','arise','arcsix']:
 for suffix in ['c','a']:
  s=S[f'df_{kind}_{suffix}'].copy();f=F[f'df_{kind}_{suffix}'].copy()
  if not len(s) and not len(f):continue
  if kind=='tower':
   for d in [s,f]:d['reference']=d.tower_sza_matched.fillna(d.tower_scaled)
   x,y='reference','sial_blue'
  else:x,y={'summit':'tower_albedo','raven':'tower_albedo','transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind],'sat_albedo'
  compare(kind,suffix,s,f,x,y)
for name,folder,stem in [('sheba','sheba_surface_flux','sheba'),('tara','tara_surface_flux','tara_broadband'),('perovich','sheba_perovich_transects','perovich')]:
 for product,col in [('clara','pentad'),('appx','date')]:
  s=pd.read_csv(R/'no_fallback'/folder/f'{stem}_{product}_matchups.csv').set_index(col);f=pd.read_csv(R/'fallback'/folder/f'{stem}_{product}_matchups.csv').set_index(col);compare(name,product,s,f,'insitu','sat')
pd.DataFrame(rows).to_csv(R/'IDW_only_vs_fallback.csv',index=False)
s=S['df_tower_c'];f=F['df_tower_c'];print(pd.DataFrame(rows).query('platform=="tower"').to_string(index=False))
print('Tower nearest-fallback rows:');print(f.loc[f.spatial_method.str.contains('nearest')].to_string())
print('IDW-only drops:',list(f.index.difference(s.index)))
print('All no-fallback paired values verified equal.')
