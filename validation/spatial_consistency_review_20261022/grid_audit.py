from pathlib import Path
import glob,json
import xarray as xr,numpy as np
R=Path(__file__).resolve().parent
out={}
for name,pat in [('CLARA','/glade/derecho/scratch/jrotondo/clara_a3_obs/SALpm200806*.nc'),('APPx','/glade/derecho/scratch/jrotondo/APP-x_observations/2008/*_d20080605_*.nc')]:
 f=sorted(glob.glob(pat))[0]
 with xr.open_dataset(f) as d:
  out[name]={'file':f,'variables':list(d.variables),'projection':{k:str(v.attrs) for k,v in d.variables.items() if 'grid_mapping_name' in v.attrs},'sizes':dict(d.sizes),'global_resolution':{k:str(v) for k,v in d.attrs.items() if any(x in k.lower() for x in ['resol','grid','proj'])}}
  la=d['lat' if name=='CLARA' else 'latitude'].values;lo=d['lon' if name=='CLARA' else 'longitude'].values
  dist=(la-71.37)**2+((lo+156.52)*np.cos(np.radians(71.37)))**2
  j,k=np.unravel_index(np.nanargmin(dist),dist.shape)
  out[name]['nearest']={'j':int(j),'k':int(k),'lat':float(la[j,k]),'lon':float(lo[j,k])}
  np.savez(R/(name+'_local_grid.npz'),lat=la[max(0,j-3):j+4,max(0,k-3):k+4],lon=lo[max(0,j-3):j+4,max(0,k-3):k+4])
(R/'grid_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
