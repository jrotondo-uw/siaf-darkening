from pathlib import Path
import glob,re
import pandas as pd,numpy as np,xarray as xr
from pyproj import Transformer
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'barrow_daily.csv',parse_dates=['date']);tf=Transformer.from_crs(4326,3413,always_xy=True)
s['x'],s['y']=tf.transform(s.lon.values,s.lat.values)
cl=[];ax=[];audit=[]
for f in sorted(glob.glob('/glade/derecho/scratch/jrotondo/clara_a3_obs/SALpm200[89]*.nc')+glob.glob('/glade/derecho/scratch/jrotondo/clara_a3_obs/SALpm2010*.nc')):
 t=pd.Timestamp(re.search(r'SALpm(\d{8})',f).group(1));sub=s[(s.date>=t)&(s.date<t+pd.Timedelta(days=5))]
 if sub.empty:continue
 with xr.open_dataset(f) as ds:
  for site,g in sub.groupby('site'):
   a=float(ds.blue_sky_albedo_all_mean.isel(time=0).interp(x=float(g.x.mean()),y=float(g.y.mean()),method='linear'))/100
   valid=np.isfinite(a) and 0<a<=1.05
   audit.append(dict(site=site,product='CLARA-A3',date=t,valid=valid,sat=a,n_days=len(g)))
   if valid:cl.append(dict(site=site,date=t,insitu=g.alb.mean(),sat=a,n_days=len(g)))
for date,g in s.groupby('date'):
 files=glob.glob(f'/glade/derecho/scratch/jrotondo/APP-x_observations/{date.year}/*_d{date:%Y%m%d}_*.nc');vals={site:[] for site in g.site};info={}
 for f in files:
  with xr.open_dataset(f) as ds:
   v=ds.cdr_surface_albedo
   if 'Time' in v.dims:v=v.isel(Time=0)
   la=ds.latitude.values;lo=ds.longitude.values
   for _,r in g.iterrows():
    dist=np.radians(la-r.lat)**2+(np.radians(lo-r.lon)*np.cos(np.radians(r.lat)))**2;j,k=np.unravel_index(np.nanargmin(dist),dist.shape);a=float(v.values[j,k]);info[r.site]=(float(la[j,k]),float(lo[j,k]),int(j),int(k))
    if np.isfinite(a) and 0<a<=1.05:vals[r.site].append(a)
 for _,r in g.iterrows():
  v=vals[r.site];audit.append(dict(site=r.site,product='APP-x',date=date,valid=bool(v),sat=np.mean(v) if v else np.nan,n_files=len(files)))
  if v:
   lat,lon,j,k=info[r.site];ax.append(dict(site=r.site,date=date,insitu=r.alb,sat=np.mean(v),n_ov=len(v),pixel_lat=lat,pixel_lon=lon,j=j,k=k))
 print(date.date(),'APP-x', {k:len(v) for k,v in vals.items()},flush=True)
for name,data in [('clara',cl),('appx',ax)]:pd.DataFrame(data,columns=list(data[0]) if data else ['site','date','insitu','sat']).to_csv(R/(name+'_matchups.csv'),index=False)
pd.DataFrame(audit).to_csv(R/'coverage_audit.csv',index=False)
print('CLARA',len(cl),'APP-x',len(ax),flush=True)
