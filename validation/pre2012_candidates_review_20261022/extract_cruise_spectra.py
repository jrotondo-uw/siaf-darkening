from pathlib import Path
import pandas as pd,numpy as np,xlrd,re
R=Path(__file__).resolve().parent;rows=[];inventory=[]
for p in sorted((R/'raw').glob('cruise*/station*.xls')):
 b=xlrd.open_workbook(p);s=b.sheet_by_index(0);year=int(p.parent.name[-4:]);stamp=s.cell_value(0,0)
 date=xlrd.xldate_as_datetime(stamp,b.datemode) if isinstance(stamp,(int,float)) else pd.to_datetime(re.search(r'(?:June|July) \d+, \d{4}',stamp).group())
 station=p.stem.split('_')[0];n=0
 for i in range(4,s.nrows):
  try:w=float(s.cell_value(i,0));a=float(s.cell_value(i,1))
  except (ValueError,TypeError):continue
  if 350<=w<=2500:
   rows.append(dict(year=year,station=station,date=pd.Timestamp(date).date(),wavelength_nm=w,albedo=a,source=p.name));n+=1
 inventory.append(dict(year=year,station=station,date=pd.Timestamp(date).date(),file=p.name,n_wavelengths=n,workbook_station_label=s.cell_value(2,0)))
pd.DataFrame(rows).to_csv(R/'icescape_spectral_means.csv',index=False);pd.DataFrame(inventory).to_csv(R/'icescape_inventory.csv',index=False)
print(pd.DataFrame(inventory).to_string(index=False))
