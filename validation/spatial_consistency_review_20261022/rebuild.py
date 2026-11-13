"""Spatial-only consistency review on previously retained matchups.
Uses the existing Figure 9 common IDW routine for both products. Does not rerun
outlier selection, change in situ references, or add previously rejected dates.
"""
from pathlib import Path
import ast,pickle,glob,hashlib
import numpy as np,pandas as pd,xarray as xr
from scipy.spatial import cKDTree
B=Path('/glade/work/jrotondo/albedo_satellite_validation');O=Path(__file__).resolve().parent
p=B/'date_matched_review_20261022/save_validation_data.py'
nodes=[];before=True
for node in ast.parse(p.read_text()).body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 if isinstance(node,ast.FunctionDef):before=False
 if before and isinstance(node,(ast.Assign,ast.AnnAssign)):nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
# Exactly the existing shared interpolation function, accelerated by caching indices.
cache={}
def idw(lat,lon,la,lo,al,max_dist_km=25.,min_valid=2):
 la=np.asarray(la);lo=np.asarray(lo);al=np.asarray(al)
 if la.ndim==1:lo,la=np.meshgrid(lo,la)
 key=hashlib.sha256(la.tobytes()+lo.tobytes()).hexdigest()
 if key not in cache:cache[key]=cKDTree(np.column_stack([la.ravel(),lo.ravel()]))
 _,idx=cache[key].query([lat,lon],k=4);v=al.ravel()[idx];valid=np.isfinite(v)&(v>=0)&(v<=1)
 if valid.sum()<min_valid:return np.nan
 d=ns['haversine_km'](lon,lat,lo.ravel()[idx][valid],la.ravel()[idx][valid])
 if d.min()>max_dist_km:return np.nan
 w=1/(d+.01);return float(np.sum(w*v[valid])/w.sum())
def clara(date,lat,lon,limit=25):
 files=glob.glob(f'{ns["dir_sial"]}/SALpm{pd.Timestamp(date):%Y%m%d}*.nc')
 if not files:return np.nan
 with xr.open_dataset(files[0]) as ds:
  al=ds.blue_sky_albedo_all_mean.squeeze().values/100;la=ds.lat.values;lo=ds.lon.values
  a=idw(lat,lon,la,lo,al,limit)
  # verify accelerated sampler exactly reproduces source routine on every CLARA sample
  expected=ns['spatial_interp_idw'](lat,lon,la,lo,al,max_dist_km=limit)
  assert np.isclose(a,expected,equal_nan=True)
  return a
def appx(date,lat,lon):
 vals=[]
 for f in sorted(glob.glob(f'{ns["dir_appx"]}/{pd.Timestamp(date).year}/*_d{pd.Timestamp(date):%Y%m%d}_*.nc')):
  with xr.open_dataset(f) as ds:
   a=idw(lat,lon,ds.latitude.values,ds.longitude.values,ds.cdr_surface_albedo.squeeze().values)
   if np.isfinite(a) and a>0:vals.append(a)
 return float(np.mean(vals)) if vals else np.nan
stats=[]
def summary(label,product,old,new,xcol,ycol):
 x=new[xcol].to_numpy(float);y=new[ycol].to_numpy(float);valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
 e=y-x;stats.append(dict(platform=label,product=product,n=len(x),old_n=len(old),bias=e.mean() if len(e) else np.nan,rmse=np.sqrt((e**2).mean()) if len(e) else np.nan,old_rmse=np.sqrt(((old[ycol]-old[xcol])**2).mean()),r=np.corrcoef(x,y)[0,1] if len(x)>2 else np.nan))
 print(stats[-1],flush=True)
 return dict(n=len(x),bias=stats[-1]['bias'],rmse=stats[-1]['rmse'],r=stats[-1]['r'])
with open('/glade/work/jrotondo/combined_validation/validation_data.pkl','rb') as f:D=pickle.load(f)
# Most paired panels already use the same common IDW routine; preserve them exactly.
for kind in ['summit','raven']:
 old=D[f'df_{kind}_c'];new=old.copy();lat,lon=D[kind.upper()+'_LAT'],D[kind.upper()+'_LON']
 new['sat_albedo']=[clara(t,lat,lon,50) for t in new.index];new=new.dropna(subset=['sat_albedo'])
 D[f'df_{kind}_c']=new;D[f'stats_{kind}_c']=summary(kind,'CLARA-A3',old,new,'tower_albedo','sat_albedo')
 new.to_csv(O/f'{kind}_clara.csv')
# Reconstruct the original tower observation locations, preserving reference/QC values.
parts=[]
for path,pat in [('central_observatory','mosmet'),('station_30','mosmet.asfs30'),('station_50','mosmet.asfs50')]:
 d=ns['load_tower_with_sza'](str(Path(ns['base_dir'])/path),pat)
 if len(d):parts.append(d)
tower=pd.concat(parts).sort_index();old=D['df_tower_c'];new=old.copy();vals=[]
for t in new.index:
 g=tower[(tower.index>=t)&(tower.index<t+pd.Timedelta(days=5))];vals.append(clara(t,g.lat.mean(),g.lon.mean()))
new['sial_blue']=vals;new=new.dropna(subset=['sial_blue']);D['df_tower_c']=new
for df in [old,new]:df['review_reference']=df.tower_sza_matched.fillna(df.tower_scaled)
D['stats_tower_c']=summary('tower','CLARA-A3',old,new,'review_reference','sial_blue')
D['x_tower_c']=new.review_reference.to_numpy();D['y_tower_c']=new.sial_blue.to_numpy();D['doy_tower_c']=new.index.dayofyear.to_numpy();new.to_csv(O/'tower_clara.csv')
with (O/'validation_spatial_consistent.pkl').open('wb') as f:pickle.dump(D,f)
# SHEBA, Tara, and Perovich: keep original dates, reference means, and positions.
for name,folder,stem in [('sheba','sheba_surface_flux','sheba'),('tara','tara_surface_flux','tara_broadband'),('perovich','sheba_perovich_transects','perovich')]:
 (O/folder).mkdir(exist_ok=True)
 for product,col in [('clara','pentad'),('appx','date')]:
  old=pd.read_csv(B/folder/f'{stem}_{product}_matchups.csv',parse_dates=[col]);new=old.copy();fn=clara if product=='clara' else appx
  new['sat']=[fn(r[col],r.lat,r.lon) for _,r in new.iterrows()];new['spatial_method']='shared Figure9 four-neighbor IDW';new=new.dropna(subset=['sat']);new=new[new.sat>0]
  new.to_csv(O/folder/f'{stem}_{product}_matchups.csv',index=False);summary(name,product,old,new,'insitu','sat')
# Coastal Barrow excluded by user decision; no validation matchups generated here.
pd.DataFrame(stats).to_csv(O/'spatial_consistency_stats.csv',index=False)
print('FINISHED spatial review',flush=True)
