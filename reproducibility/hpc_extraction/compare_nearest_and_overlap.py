"""Compare nearest-center sampling with area-overlap aggregation on a shared PSN ocean domain.
Overlap fractions use projected polygons; PSN mapscale areas are retained per fine cell.
Densified latitude edges and a convergence test limit polygon approximation error.
"""
from pathlib import Path
import numpy as np,pandas as pd,xarray as xr
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon,box
from shapely.strtree import STRtree
from pyproj import Transformer
import siaf_obs_core as C
O=Path('/glade/work/jrotondo/siaf_weight_fix_review_20261125/results')
ids=np.flatnonzero(C.DOMAIN_PSN.ravel());xx=C.psn_xx.ravel()[ids];yy=C.psn_yy.ravel()[ids];wa=C.AREA_W.ravel()[ids].astype(float);den=wa.sum();dx=abs(np.diff(C.psn_x).mean());dy=abs(np.diff(C.psn_y).mean())
fine=[box(x-dx/2,y-dy/2,x+dx/2,y+dy/2) for x,y in zip(xx,yy)]
proj=Transformer.from_crs('EPSG:4326',C.PSN25_PROJ4,always_xy=True)
with xr.open_dataset(C.KERNEL_FILE) as ds:
 latn='latitude' if 'latitude' in ds else 'lat';lonn='longitude' if 'longitude' in ds else 'lon';klat=ds[latn].values;klon=ds[lonn].values

def weights(lat,lon,nseg):
 lat=np.asarray(lat);lon=np.asarray(lon);ly=np.argsort(lat);lx=np.argsort(lon);slat=lat[ly];slon=lon[lx];lb=np.r_[-90,(slat[:-1]+slat[1:])/2,90];step=np.median(np.diff(slon));lob=np.r_[slon[0]-step/2,(slon[:-1]+slon[1:])/2,slon[-1]+step/2]
 polys=[];src=[]
 for iy in range(len(lat)):
  if lb[iy+1]<58:continue
  for ix in range(len(lon)):
   lons=np.r_[np.linspace(lob[ix],lob[ix+1],nseg+1),np.full(nseg,lob[ix+1]),np.linspace(lob[ix+1],lob[ix],nseg+1)[1:],np.full(nseg,lob[ix])]
   lats=np.r_[np.full(nseg+1,lb[iy]),np.linspace(lb[iy],lb[iy+1],nseg+1)[1:],np.full(nseg,lb[iy+1]),np.linspace(lb[iy+1],lb[iy],nseg+1)[1:]]
   x,y=proj.transform(lons,lats);p=Polygon(np.c_[x,y]);
   if not p.is_valid:p=p.buffer(0)
   polys.append(p);src.append(ly[iy]*len(lon)+lx[ix])
 tree=STRtree(polys);rows=[];cols=[];vals=[]
 for j,p in enumerate(fine):
  for ii in tree.query(p):
   a=p.intersection(polys[int(ii)]).area
   if a>0:rows.append(j);cols.append(src[int(ii)]);vals.append(a/(dx*dy))
 W=csr_matrix((vals,(rows,cols)),shape=(len(ids),len(lat)*len(lon)));s=np.asarray(W.sum(axis=1)).ravel();assert np.max(abs(s-1))<1e-5,np.max(abs(s-1))
 print('WEIGHTS',len(lat),len(lon),nseg,'max row error',np.max(abs(s-1)),flush=True);return W
mats={}
for name,lat,lon in [('CERES',C.c_la,C.c_lo),('Kernel',klat,klon)]:
 for nseg in [8,32]:mats[name,nseg]=weights(lat,lon,nseg)
rows=[]
for period,cache,years in [('1982-2014',C.CACHE_CMIP6,C.CERES_CMIP6),('1982-2024',C.CACHE_OBS,C.CERES_OBS)]:
 for mn in C.MONTH_NUMS:
  mo=C.MONTH_NAMES[mn];sic=C.load_sic_data(cache,mo);d=C.load_IS_map(cache,mo);fields=[C.regrid_sial_to_psn(z).ravel()[ids].astype(float) for z in C.compute_IS(d,sic,.05)]
  rc=C.load_RS_ceres_mean(years,mn)
  with xr.open_dataset(C.KERNEL_FILE) as ds:
   tdim=next(dim for dim,size in ds.sizes.items() if size==12);v=C.KERNEL_VAR if C.KERNEL_VAR in ds else next(x for x in ds.data_vars if x.lower()==C.KERNEL_VAR.lower());rk=np.abs(ds[v].isel({tdim:mn-1}).values.astype(np.float32))*C.KERNEL_SCALE
  for name,raw,nearest in [('CERES',rc,rc.ravel()[C.IDX_CERES_TO_PSN].reshape(C.PSN_SHAPE)),('Kernel',rk,C.load_kernel_psn(mn))]:
   assert np.isfinite(raw).all(),name
   for label,nseg in [('nearest',0),('overlap8',8),('overlap32',32)]:
    rs=nearest.ravel()[ids] if nseg==0 else mats[name,nseg]@raw.ravel()
    vv=[np.sum(wa*f*rs)/den*C.SCALE for f in fields]
    if nseg:
     W=mats[name,nseg];A=np.asarray(W.T@wa).ravel();co=[np.sum((W.T@(wa*f))*raw.ravel())/den*C.SCALE for f in fields];np.testing.assert_allclose(vv,co,rtol=1e-12,atol=1e-12)
    rows.append(dict(period=period,month=mo,method=name,remap=label,total=vv[0],area_loss=vv[1],darkening=vv[2]))
  print('DONE',period,mo,flush=True)
df=pd.DataFrame(rows);df.to_csv(O/'regrid_comparison_monthly.csv',index=False);z=df.groupby(['period','method','remap'])[['total','area_loss','darkening']].mean();z.to_csv(O/'regrid_comparison.csv');print(z.to_string())
