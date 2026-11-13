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
from functools import lru_cache
base_idw=idw;events=[];sphere_cache={}
def hybrid(lat,lon,la,lo,al,max_dist_km=25.,min_valid=2):
 a=base_idw(lat,lon,la,lo,al,max_dist_km,min_valid)
 if np.isfinite(a):events.append('idw');return a
 la=np.asarray(la);lo=np.asarray(lo);al=np.asarray(al)
 if la.ndim==1:lo,la=np.meshgrid(lo,la)
 key=hashlib.sha256(la.tobytes()+lo.tobytes()).hexdigest()
 def xyz(la,lo):
  la=np.radians(la);lo=np.radians(lo);return np.column_stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)])
 if key not in sphere_cache:sphere_cache[key]=cKDTree(xyz(la.ravel(),lo.ravel()))
 _,idx=sphere_cache[key].query(xyz(np.array([lat]),np.array([lon]))[0]);a=float(al.ravel()[idx])
 dist=float(ns['haversine_km'](lon,lat,lo.ravel()[idx],la.ravel()[idx]))
 if np.isfinite(a) and 0<=a<=1 and dist<=max_dist_km:events.append('nearest');return a
 events.append('unmatched');return np.nan
ns['spatial_interp_idw']=hybrid
ns['load_appx_albedo']=lru_cache(maxsize=48)(ns['load_appx_albedo'])
ns['find_appx_files_for_date']=lru_cache(maxsize=1024)(ns['find_appx_files_for_date'])
@lru_cache(maxsize=24)
def load_clara(f):
 with xr.open_dataset(f) as d:return d.lat.values,d.lon.values,d.blue_sky_albedo_all_mean.squeeze().values/100

def clara(date,lat,lon,limit=25):
 files=glob.glob(f'{ns["dir_sial"]}/SALpm{pd.Timestamp(date):%Y%m%d}*.nc')
 if not files:return np.nan
 la,lo,al=load_clara(files[0]);return hybrid(lat,lon,la,lo,al,limit)
def appx(date,lat,lon):
 vals=[]
 for f in sorted(glob.glob(f'{ns["dir_appx"]}/{pd.Timestamp(date).year}/*_d{pd.Timestamp(date):%Y%m%d}_*.nc')):
  la,lo,al=ns['load_appx_albedo'](f)
  if la is not None:
   a=hybrid(lat,lon,la,lo,al)
   if np.isfinite(a) and a>0:vals.append(a)
 return float(np.mean(vals)) if vals else np.nan

def diagnostics():
 valid=[x for x in events if x!='unmatched'];return {'spatial_method':'+'.join(sorted(set(valid))) if valid else 'unmatched','n_idw_samples':valid.count('idw'),'n_nearest_samples':valid.count('nearest')}
stats=[]
def summarize(name,product,old,new,xcol,ycol):
 x=new[xcol].to_numpy(float);y=new[ycol].to_numpy(float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];e=y-x
 st=dict(n=len(e),bias=float(e.mean()) if len(e) else np.nan,rmse=float(np.sqrt((e**2).mean())) if len(e) else np.nan,r=float(np.corrcoef(x,y)[0,1]) if len(e)>2 else np.nan)
 nfb=int(new.spatial_method.str.contains('nearest').sum()) if 'spatial_method' in new else 0
 stats.append(dict(platform=name,product=product,old_n=len(old),fallback_rows=nfb,**st));print(stats[-1],flush=True);return st
with open('/glade/work/jrotondo/combined_validation/validation_data.pkl','rb') as f:D=pickle.load(f)
# Existing CLARA common-IDW panels are already successful IDW matches; no fallback needed.
for kind in ['transect','heli','drone','arise','arcsix']:
 d=D[f'df_{kind}_c'];d['spatial_method']='idw';d['n_nearest_samples']=0
# Original tower locations and hours are required for both products.
parts=[]
for path,pat in [('central_observatory','mosmet'),('station_30','mosmet.asfs30'),('station_50','mosmet.asfs50')]:
 d=ns['load_tower_with_sza'](str(Path(ns['base_dir'])/path),pat)
 if len(d):parts.append(d)
tower=pd.concat(parts).sort_index()
# Re-evaluate all retained APP-x records: spatial fallback now precedes the inherited temporal fallback.
for kind in ['tower','summit','raven','transect','heli','drone','arise','arcsix']:
 for suffix in ['c','a']:
  if suffix=='c' and kind not in ['tower','summit','raven']:continue
  old=D[f'df_{kind}_{suffix}'];new=old.copy()
  if not len(old):continue
  values=[];diags=[]
  for idx,r in old.iterrows():
   events.clear()
   if kind=='tower':
    date=pd.Timestamp(idx);g=tower[(tower.index>=date)&(tower.index<date+pd.Timedelta(days=5 if suffix=='c' else 1))];lat,lon=g.lat.mean(),g.lon.mean();hour=float(np.mean(g.index.hour)+np.mean(g.index.minute)/60)
   elif kind in ['summit','raven']:date=pd.Timestamp(idx);lat,lon=D[kind.upper()+'_LAT'],D[kind.upper()+'_LON'];hour=14
   else:
    date=pd.to_datetime(str(r.date),format='%Y%m%d') if str(r.date).isdigit() else pd.Timestamp(r.date);lat,lon=r.lat,r.lon;hour=r.utc_hour
   limit=50 if kind in ['summit','raven'] else 25
   if suffix=='c':a=clara(date,lat,lon,limit)
   else:a,slot,offset=ns['get_appx_albedo_composite'](lat,lon,date,hour,max_dist_km=limit)
   values.append(a);diags.append(diagnostics())
  ycol='sial_blue' if kind=='tower' else 'sat_albedo';new[ycol]=values
  for k in ['spatial_method','n_idw_samples','n_nearest_samples']:new[k]=[d[k] for d in diags]
  new=new.dropna(subset=[ycol]);xcol={'tower':'tower_scaled','summit':'tower_albedo','raven':'tower_albedo','transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind]
  if kind=='tower':new['review_reference']=new.tower_sza_matched.fillna(new.tower_scaled);xcol='review_reference'
  D[f'df_{kind}_{suffix}']=new;D[f'stats_{kind}_{suffix}']=summarize(kind,suffix,old,new,xcol,ycol)
  if kind=='tower':D['x_tower_'+suffix]=new[xcol].to_numpy();D['y_tower_'+suffix]=new[ycol].to_numpy();D['doy_tower_'+suffix]=new.index.dayofyear.to_numpy()
  new.to_csv(O/f'{kind}_{suffix}_fallback.csv')
with (O/'validation_IDW_fallback.pkl').open('wb') as f:pickle.dump(D,f)
for name,folder,stem in [('sheba','sheba_surface_flux','sheba'),('tara','tara_surface_flux','tara_broadband'),('perovich','sheba_perovich_transects','perovich')]:
 dest=O/'fallback'/folder;dest.mkdir(parents=True,exist_ok=True)
 for product,col in [('clara','pentad'),('appx','date')]:
  old=pd.read_csv(B/folder/f'{stem}_{product}_matchups.csv',parse_dates=[col]);new=old.copy();fn=clara if product=='clara' else appx;values=[];diags=[]
  for _,r in old.iterrows():events.clear();values.append(fn(r[col],r.lat,r.lon));diags.append(diagnostics())
  new['sat']=values
  for k in ['spatial_method','n_idw_samples','n_nearest_samples']:new[k]=[d[k] for d in diags]
  new=new.dropna(subset=['sat']);new=new[new.sat>0];new=new.rename(columns={c:'legacy_'+c for c in ['n_ov','sza','nobs'] if c in new})
  new.to_csv(dest/f'{stem}_{product}_matchups.csv',index=False);summarize(name,product,old,new,'insitu','sat')
pd.DataFrame(stats).to_csv(O/'IDW_fallback_stats.csv',index=False)
print('FINISHED IDW + nearest fallback; Barrow excluded.',flush=True)
