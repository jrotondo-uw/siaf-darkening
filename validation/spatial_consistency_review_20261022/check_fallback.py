from pathlib import Path
import numpy as np,xarray as xr
from pyproj import Transformer
p=Path(__file__).resolve().parent/'rebuild_fallback.py';s=p.read_text().split("with open('/glade/work/jrotondo/combined_validation/validation_data.pkl'")[0];ns2={'__file__':str(p)};exec(compile(s,str(p),'exec'),ns2)
h=ns2['hybrid'];ev=ns2['events'];la=np.array([[70.,70.],[70.1,70.1]]);lo=np.array([[-150.,-149.9],[-150.,-149.9]])
ev.clear();v=h(70.,-150.,la,lo,np.full((2,2),.6));assert abs(v-.6)<1e-12 and ev==['idw']
ev.clear();v=h(70.,-150.,la,lo,np.array([[.6,np.nan],[np.nan,np.nan]]));assert v==.6 and ev==['nearest']
ev.clear();v=h(70.,-150.,la,lo,np.array([[np.nan,.6],[np.nan,np.nan]]));assert np.isnan(v) and ev==['unmatched']
ev.clear();v=h(80.,0.,la,lo,np.full((2,2),.6));assert np.isnan(v) and ev==['unmatched']
print('Verified IDW precedence; valid nearest fallback; missing nearest remains unmatched; distance limit.')
f='/glade/derecho/scratch/jrotondo/clara_a3_obs/SALpm20080605000000321AVPOS01NP.nc'
with xr.open_dataset(f) as d:
 j,k=104,146;lat=float(d.lat.values[j,k]);lon=float(d.lon.values[j,k]);tx,ty=Transformer.from_crs(4326,3413,always_xy=True).transform(lon,lat)
 print('CLARA stored native xy:',float(d.x[k]),float(d.y[j]),'EPSG3413 projected actual grid coordinate:',tx,ty,'metadata:',d.x.attrs,d.y.attrs)
