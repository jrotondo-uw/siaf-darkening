"""Read original NOAA-mirrored SHEBA Perovich broadband transects.
Spatial mean is trapezoidal integration / observed transect length.
No temporal gap filling. Ambiguous April 5 and October headers are excluded.
"""
from pathlib import Path
import csv,json,hashlib
import pandas as pd,numpy as np
BASE=Path(__file__).resolve().parent;RAW=BASE/'raw';rows=[];audit=[]
for p in sorted(RAW.glob('*.CSV')):
 r=list(csv.reader(p.open(encoding='utf-8-sig')))
 if p.name in ['ALBT0405.CSV','ALBT1011.CSV']:
  audit.append(dict(file=p.name,action='excluded from dated analysis',reason='Filename/header conflict or malformed year',header=r[0]));continue
 if p.name=='ALBTMAY.CSV':
  header=r[1];start=2;dates=[]
  for s in header[1:]:
   if s in ['8-May-99','11-May-99']:
    audit.append(dict(file=p.name,action='year corrected',old=s,new=s.replace('99','98'),reason='File title and metadata explicitly May 1998; May 8 duplicated in ALBT0508.CSV'))
    s=s.replace('99','98')
   dates.append(pd.to_datetime(s,format='%d-%b-%y'))
 elif p.name=='ALBTSUM.CSV':header=r[0];start=1;dates=[pd.to_datetime(s,format='%d-%b-%y') for s in header[1:]]
 else:dates=[pd.to_datetime(r[0][0],format='%d-%b-%y')];start=1
 for row in r[start:]:
  try:x=float(row[0])
  except (ValueError,IndexError):continue
  for j,day in enumerate(dates,1):
   try:a=float(row[j])
   except (ValueError,IndexError):continue
   if not np.isfinite(a) or not 0<=a<=1:continue
   rows.append(dict(date=day,x_m=x,alb=a,file=p.name))
points=pd.DataFrame(rows)
# May 8 occurs in both the individual and monthly file: verify before dedup.
for key,g in points.groupby(['date','x_m']):
 if len(g)>1:assert np.allclose(g.alb,g.alb.iloc[0]),f'conflicting measurements {key}'
points=points.drop_duplicates(['date','x_m']).sort_values(['date','x_m'])
points.to_csv(BASE/'perovich_transect_points.csv',index=False)
means=[]
for date,g in points.groupby('date'):
 x=g.x_m.to_numpy();a=g.alb.to_numpy();assert len(x)>1 and np.all(np.diff(x)>0)
 means.append(dict(date=date,alb=np.trapezoid(a,x)/(x[-1]-x[0]),point_mean=a.mean(),spatial_sd=a.std(ddof=1),n_points=len(g),x_min_m=x[0],x_max_m=x[-1],max_gap_m=np.diff(x).max()))
s=pd.DataFrame(means)
# Existing daily camp track: positions are camp-scale proxies, not surveyed endpoints.
track=pd.read_csv('/glade/work/jrotondo/albedo_satellite_validation/sheba_surface_flux/sheba_daily.csv',parse_dates=['date'])[['date','lat','lon']]
assert track.date.is_unique
s=s.merge(track,on='date',how='left',validate='one_to_one')
missing=s.loc[s[['lat','lon']].isna().any(axis=1),'date'].astype(str).tolist()
s.to_csv(BASE/'perovich_transect_daily.csv',index=False)
manifest=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url='https://psl.noaa.gov/arctic/sheba/netcdf/datasets/Ice_snow_data/ICEDATA/OPTICS/TOTALB/'+p.name) for p in sorted(RAW.iterdir())]
(BASE/'source_and_qc.json').write_text(json.dumps(dict(dataset='UCAR 13.826',doi='10.5065/D6SB444G',files=manifest,decisions=audit,missing_position_dates=missing,spatial_average='trapezoidal integral over observed length / observed length; no extrapolation',geolocation='existing SHEBA daily camp track; proxy for 200-m transect'),indent=2)+'\n')
print('Distinct surveys:',len(s),'points:',len(points),'dates:',s.date.min(),s.date.max(),'missing positions:',missing)
print(s[['date','alb','n_points','x_min_m','x_max_m']].head().to_string(index=False))
