import glob,json
import numpy as np,xarray as xr
from pathlib import Path
out=[]
for f in sorted(glob.glob('/glade/derecho/scratch/jrotondo/SIC_data/sic_psn25_2020*.nc'))[:2]:
 with xr.open_dataset(f) as d:
  row={'file':f,'sizes':dict(d.sizes),'variables':list(d.variables),'crs':{k:v.attrs for k,v in d.variables.items() if 'grid_mapping_name' in v.attrs},'sic_attrs':d.cdr_seaice_conc.attrs,'sic_encoding':str(d.cdr_seaice_conc.encoding),'x_attrs':d.x.attrs,'y_attrs':d.y.attrs,'min':float(d.cdr_seaice_conc.min()),'max':float(d.cdr_seaice_conc.max())}
  v=d.cdr_seaice_conc.values;row['quantiles']=np.nanquantile(v,[0,.25,.5,.75,.9,1]).tolist();out.append(row)
print(json.dumps(out,indent=2,default=str));Path(__file__).with_name('sic_metadata_audit.json').write_text(json.dumps(out,indent=2,default=str))
