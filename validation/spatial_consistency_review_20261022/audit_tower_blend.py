from pathlib import Path
import ast,pickle,glob,re
import numpy as np,pandas as pd,xarray as xr
from pyproj import CRS,Transformer,Geod
from functools import lru_cache
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent;B=R.parent
p=B/'date_matched_review_20261022/save_validation_data.py';nodes=[];before=True
for node in ast.parse(p.read_text()).body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 if isinstance(node,ast.FunctionDef):before=False
 if before and isinstance(node,(ast.Assign,ast.AnnAssign)):nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
with open('/glade/work/jrotondo/combined_validation/validation_data.pkl','rb') as f:old=pickle.load(f)
with (R/'validation_IDW_only_controlled.pkl').open('rb') as f:new=pickle.load(f)
parts=[]
for path,pat in [('central_observatory','mosmet'),('station_30','mosmet.asfs30'),('station_50','mosmet.asfs50')]:
 d=ns['load_tower_with_sza'](str(Path(ns['base_dir'])/path),pat)
 if len(d):parts.append(d)
tower=pd.concat(parts).sort_index();print('Loaded tower',len(tower),flush=True)
files=sorted(glob.glob(ns['dir_sic']+'/sic_psn25_2020*.nc'));dates=pd.DatetimeIndex([pd.to_datetime(re.search(r'sic_psn25_(\d{8})',f).group(1)) for f in files]);anchor=dates.min().normalize();starts=pd.date_range(anchor,dates.max(),freq='5D')
with xr.open_dataset(files[0]) as d:crs=CRS.from_wkt(d.crs.attrs['crs_wkt'])
tf=Transformer.from_crs(4326,crs,always_xy=True);tf_old=Transformer.from_crs(4326,3413,always_xy=True);inv=Transformer.from_crs(crs,4326,always_xy=True);geod=Geod(ellps='WGS84')
@lru_cache(maxsize=50)
def sic_grid(t):
 arrays=[]
 for f in np.array(files)[(dates>=t)&(dates<t+pd.Timedelta(days=5))]:
  with xr.open_dataset(f) as d:
   a=d.cdr_seaice_conc.squeeze().load()
   # Match original decoder/scaling to isolate coordinate lookup.
   if float(a.max())>1.5:a=a/100
   arrays.append(a)
 return xr.concat(arrays,dim='sample').mean('sample')
@lru_cache(maxsize=30)
def clara_ds(t):
 f=glob.glob(ns['dir_sial']+f'/SALpm{t:%Y%m%d}*.nc')[0]
 with xr.open_dataset(f) as d:return d[['sun_zenith_angle_mean','sun_zenith_angle_std','lat','lon','blue_sky_albedo_all_mean']].load()
def field_idw(d,var,lat,lon):
 la=d.lat.values.ravel();lo=d.lon.values.ravel();v=d[var].squeeze().values.ravel();_,ii=cKDTree(np.column_stack([la,lo])).query([lat,lon],k=4);valid=np.isfinite(v[ii])&(v[ii]>=0)&(v[ii]<=180)
 if valid.sum()<2:return np.nan
 dd=ns['haversine_km'](lon,lat,lo[ii][valid],la[ii][valid])
 if dd.min()>25:return np.nan
 w=1/(dd+.01);return float(np.sum(w*v[ii][valid])/w.sum())
def raw_mean(g,sza,std=None,product='c'):
 win=max(3.,std) if product=='c' else ns['MAX_DELTA_SZA']*2
 h=g[np.abs(g.sza-sza)<=win]
 return float(h.albedo.mean()) if len(h)>=6 else float(g.albedo.mean())
rows=[]
for product in ['c','a']:
 olddf=old['df_tower_'+product];ndf=new['df_tower_'+product]
 for t,r in olddf.iterrows():
  g=tower[(tower.index>=t)&(tower.index<t+pd.Timedelta(days=5 if product=='c' else 1))];lat,lon=float(g.lat.mean()),float(g.lon.mean());x,y=tf_old.transform(lon,lat);sx,sy=ns['_rotate_coords'](x,y,-ns['ROTATION_ANGLE']);xx,yy=tf.transform(lon,lat)
  tt=starts[np.argmin(np.abs(starts-t))];assert abs((tt-t).days)<= (2 if product=='c' else 3)
  sic=sic_grid(tt);so=float(sic.interp(x=sx,y=sy));sn=float(sic.interp(x=xx,y=yy));wronglon,wronglat=inv.transform(sx,sy);_,_,dist=geod.inv(lon,lat,wronglon,wronglat)
  if product=='c':
   d=clara_ds(t);sza_old=float(d.sun_zenith_angle_mean.squeeze().interp(x=x,y=y));std_old=float(d.sun_zenith_angle_std.squeeze().interp(x=x,y=y));sza_new=field_idw(d,'sun_zenith_angle_mean',lat,lon);std_new=field_idw(d,'sun_zenith_angle_std',lat,lon)
  else:
   hour=float(np.mean(g.index.hour)+np.mean(g.index.minute)/60);_,slot,_=ns['get_appx_albedo_composite'](lat,lon,t,hour);sza_old=ns['get_appx_sza'](lat,lon,t,slot or 14);sza_new=sza_old;std_old=std_new=6
  raw_old=raw_mean(g,sza_old,std_old,product);raw_new=raw_mean(g,sza_new,std_new,product) if np.isfinite(sza_new+std_new) else np.nan
  ocean_old=ns['ocean_albedo_payne'](sza_old);ocean_new=ns['ocean_albedo_payne'](sza_new)
  stored=r.tower_sza_matched if np.isfinite(r.tower_sza_matched) else r.tower_scaled
  reconstructed=so*raw_old+(1-so)*ocean_old;corrected=sn*raw_old+(1-sn)*ocean_old;corrected_all=sn*raw_new+(1-sn)*ocean_new
  rows.append(dict(product=product,date=t,lat=lat,lon=lon,sic_date=tt,sic_old=so,sic_corrected=sn,wrong_sic_lat=wronglat,wrong_sic_lon=wronglon,sic_location_error_km=dist/1000,sza_old=sza_old,sza_corrected=sza_new,raw_reference_old=raw_old,raw_reference_corrected=raw_new,reference_stored=stored,reference_reconstructed=reconstructed,reference_corrected_sic=corrected,reference_corrected_sic_sza=corrected_all,satellite_original=r.sial_blue,satellite_idw=ndf.loc[t,'sial_blue'] if t in ndf.index else np.nan))
 print('Completed',product,flush=True)
a=pd.DataFrame(rows);a.to_csv(R/'tower_blending_audit.csv',index=False)
print('Max reconstruction error',np.nanmax(abs(a.reference_stored-a.reference_reconstructed)))
print(a[(a.product=='c')&(a.reference_stored<.2)][['date','lat','lon','sic_old','sic_corrected','sic_location_error_km','reference_stored','reference_corrected_sic','reference_corrected_sic_sza','satellite_original','satellite_idw']].to_string(index=False))
st=[]
for prod,g in a.groupby('product'):
 for label,col in [('Original reference','reference_stored'),('Correct SIC location','reference_corrected_sic'),('Correct SIC and SZA locations','reference_corrected_sic_sza')]:
  q=g.dropna(subset=[col,'satellite_idw']);e=q.satellite_idw-q[col];st.append(dict(product=prod,reference=label,n=len(q),rmse=np.sqrt((e**2).mean()),bias=e.mean(),r=q[col].corr(q.satellite_idw)))
pd.DataFrame(st).to_csv(R/'tower_blending_audit_stats.csv',index=False);print(pd.DataFrame(st).to_string(index=False))
