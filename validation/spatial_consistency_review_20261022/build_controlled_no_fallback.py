from pathlib import Path
import pickle,pandas as pd,numpy as np
R=Path(__file__).resolve().parent
p=R/'rebuild_fallback.py';src=p.read_text().split("with open('/glade/work/jrotondo/combined_validation/validation_data.pkl'")[0];ctx={'__file__':str(p)};exec(compile(src,str(p),'exec'),ctx)
# Reuse the exact current loader and IDW function; disable only nearest fallback.
ctx['hybrid']=ctx['base_idw'];ctx['ns']['spatial_interp_idw']=ctx['base_idw']
with (R/'validation_spatial_consistent.pkl').open('rb') as f:D=pickle.load(f)
with (R/'validation_IDW_fallback.pkl').open('rb') as f:F=pickle.load(f)
for kind in ['tower','summit','raven','transect','heli','drone','arise','arcsix']:
 for suffix in ['c','a']:
  s=D[f'df_{kind}_{suffix}'];fb=F[f'df_{kind}_{suffix}']
  if not len(s):continue
  y='sial_blue' if kind=='tower' else 'sat_albedo';common=s.index.intersection(fb.index)
  idx=common[~fb.loc[common,'spatial_method'].str.contains('nearest')]
  np.testing.assert_allclose(s.loc[idx,y],fb.loc[idx,y],rtol=1e-5,atol=1e-7,equal_nan=True)
  # Identical values in unaffected rows, avoiding even numerical-loader differences.
  s.loc[idx,y]=fb.loc[idx,y];s['spatial_method']='idw'
  x=s.tower_sza_matched.fillna(s.tower_scaled) if kind=='tower' else s[{'summit':'tower_albedo','raven':'tower_albedo','transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind]]
  z=s[y];ok=np.isfinite(x)&np.isfinite(z);err=z[ok]-x[ok]
  D[f'stats_{kind}_{suffix}']=dict(n=int(ok.sum()),bias=err.mean(),rmse=np.sqrt((err**2).mean()),r=x[ok].corr(z[ok]))
  if kind=='tower':D['x_tower_'+suffix]=x.to_numpy();D['y_tower_'+suffix]=z.to_numpy();D['doy_tower_'+suffix]=s.index.dayofyear.to_numpy()
with (R/'validation_IDW_only_controlled.pkl').open('wb') as f:pickle.dump(D,f)
for folder,stem in [('sheba_surface_flux','sheba'),('tara_surface_flux','tara_broadband'),('sheba_perovich_transects','perovich')]:
 dest=R/'no_fallback'/folder;dest.mkdir(parents=True,exist_ok=True)
 for product,col in [('clara','pentad'),('appx','date')]:
  df=pd.read_csv(R/'fallback'/folder/f'{stem}_{product}_matchups.csv',parse_dates=[col]);values=[]
  for _,r in df.iterrows():
   if r.spatial_method=='idw':a=r.sat
   elif product=='clara':a=ctx['clara'](r[col],r.lat,r.lon)
   else:a=ctx['appx'](r[col],r.lat,r.lon)
   values.append(a)
  df['sat']=values;df=df.dropna(subset=['sat']);df=df[df.sat>0];df['spatial_method']='idw';df=df.drop(columns=['n_nearest_samples','n_idw_samples'],errors='ignore');df.to_csv(dest/f'{stem}_{product}_matchups.csv',index=False)
  print(folder,product,len(df),flush=True)
print('Controlled IDW-only build complete.',flush=True)
