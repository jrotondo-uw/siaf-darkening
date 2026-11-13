from pathlib import Path
import pickle,numpy as np,pandas as pd
R=Path(__file__).resolve().parent
with (R/'validation_fallback_corrected_SIC.pkl').open('rb') as f:D=pickle.load(f)
s=pd.read_csv(R/'Figure_9_chronological_statistics.csv')
def metrics(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<2:return dict(n=len(x),bias=np.nan,rmse=np.nan,ccc=np.nan)
 bias=(y-x).mean();vx=np.mean((x-x.mean())**2);vy=np.mean((y-y.mean())**2);cov=np.mean((x-x.mean())*(y-y.mean()));den=vx+vy+bias**2
 return dict(n=len(x),bias=bias,rmse=np.sqrt(np.mean((y-x)**2)),ccc=2*cov/den if den>0 else np.nan)
assert np.isclose(metrics([0,1,2],[0,1,2])['ccc'],1)
assert np.isclose(metrics([0,1,2],[2,1,0])['ccc'],-1)
assert metrics([0,1,2],[1,2,3])['ccc']<1
out=[]
for _,r in s.iterrows():
 prod='c' if r['product']=='CLARA-A3' else 'a';name=r.dataset
 if name.startswith(('SHEBA','Tara')):
  if name=='SHEBA transects':folder,stem='sheba_perovich_transects','perovich'
  elif name.startswith('SHEBA'):folder,stem='sheba_surface_flux','sheba'
  else:folder,stem='tara_surface_flux','tara_broadband'
  p='clara' if prod=='c' else 'appx';d=pd.read_csv(R/'fallback'/folder/f'{stem}_{p}_matchups.csv');x,y=d.insitu,d.sat
 else:
  kind=next(k for prefix,k in [('MOSAiC Tower','tower'),('MOSAiC Transects','transect'),('MOSAiC Helicopter','heli'),('Summit','summit'),('HELiX','drone'),('ARISE','arise'),('ARCSIX','arcsix'),('Raven','raven')] if name.startswith(prefix))
  d=D[f'df_{kind}_{prod}']
  if kind=='tower':x,y=D['x_tower_'+prod],D['y_tower_'+prod]
  elif len(d)==0:x,y=[],[]
  else:x,y=d[{'transect':'transect_albedo','heli':'heli_albedo_corr','summit':'tower_albedo','raven':'tower_albedo','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind]],d.sat_albedo
 m=metrics(x,y);assert m['n']==r.n,(name,m['n'],r.n)
 np.testing.assert_allclose([m['bias'],m['rmse']],[r.bias,r.rmse],atol=1e-12,equal_nan=True)
 assert np.isnan(m['ccc']) or -1-1e-12<=m['ccc']<=1+1e-12
 out.append(dict(r,ccc=m['ccc']))
pd.DataFrame(out).to_csv(R/'Figure_9_statistics_CCC.csv',index=False)
print(pd.DataFrame(out)[['dataset','product','ccc']].to_string(index=False))
