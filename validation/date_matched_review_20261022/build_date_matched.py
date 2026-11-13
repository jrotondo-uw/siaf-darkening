from pathlib import Path
import ast,pickle,json,hashlib
import pandas as pd,numpy as np
BASE=Path('/glade/work/jrotondo/albedo_satellite_validation')
OUT=BASE/'date_matched_review_20261022';OUT.mkdir(exist_ok=True)
source=Path('/glade/work/jrotondo/combined_validation/validation_data.pkl')
with source.open('rb') as f:D=pickle.load(f)
audit=[]
def stats(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 return dict(n=len(x),bias=float(np.mean(y-x)) if len(x) else np.nan,rmse=float(np.sqrt(np.mean((y-x)**2))) if len(x) else np.nan,r=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else np.nan)
def dates(df):
 if 'date' in df:
  v=df.date
  if len(v) and str(v.iloc[0]).isdigit():return pd.DatetimeIndex(pd.to_datetime(v.astype(str),format='%Y%m%d')).normalize()
  return pd.DatetimeIndex(pd.to_datetime(v)).normalize()
 return pd.DatetimeIndex(df.index).normalize()
def xy(df,kind):
 if kind=='tower':return df.tower_sza_matched.fillna(df.tower_scaled),df.sial_blue
 x={'transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','summit':'tower_albedo','raven':'tower_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo'}[kind]
 return (df[x],df.sat_albedo) if len(df) else ([],[])
def audit_pair(name,c0,a0,c,a,method):
 for product,old,new in [('CLARA-A3',c0,c),('APP-x',a0,a)]:
  x,y=xy(new,name) if name not in ('sheba','tara') else (new.insitu,new.sat)
  ox,oy=xy(old,name) if name not in ('sheba','tara') else (old.insitu,old.sat)
  audit.append(dict(platform=name,product=product,method=method,original_n=len(old),**stats(x,y),original_bias=stats(ox,oy)['bias'],original_rmse=stats(ox,oy)['rmse']))
def full_pentads(c,a,cd,ad):
 keep=np.array([all(s+pd.Timedelta(days=k) in set(ad) for k in range(5)) for s in cd],bool)
 starts=cd[keep];days={s+pd.Timedelta(days=k) for s in starts for k in range(5)}
 c=c.loc[keep].copy();a=a.loc[ad.isin(days)].copy()
 assert set(ad[ad.isin(days)])==days
 return c,a
for kind in ['tower','transect','heli','drone','arise','arcsix']:
 c0=D[f'df_{kind}_c'];a0=D[f'df_{kind}_a'];c=c0.copy();a=a0.copy()
 if kind=='tower':
  c,a=full_pentads(c,a,dates(c),dates(a));method='CLARA pentads with all five APP-x daily dates'
 else:
  common=set(dates(c))&set(dates(a)) if len(c) and len(a) else set()
  c=c.loc[dates(c).isin(common)].copy() if len(c) else c
  a=a.loc[dates(a).isin(common)].copy() if len(a) else a
  assert set(dates(c))==set(dates(a))
  method='Exact shared observation dates; existing pixel records retained'
 for suffix,df in [('c',c),('a',a)]:
  D[f'df_{kind}_{suffix}']=df;x,y=xy(df,kind);D[f'stats_{kind}_{suffix}']=stats(x,y)
  if kind=='tower':
   D[f'x_tower_{suffix}']=np.asarray(x);D[f'y_tower_{suffix}']=np.asarray(y);D[f'doy_tower_{suffix}']=dates(df).dayofyear
  df.to_csv(OUT/f'{kind}_{suffix}.csv',index=True)
 audit_pair(kind,c0,a0,c,a,method)
# Reuse the exact original APP-x sampler, including existing SZA and fallback logic.
p=OUT/'save_validation_data.py'
tree=ast.parse(p.read_text());nodes=[];before_functions=True
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 if isinstance(node,ast.FunctionDef):before_functions=False
 if before_functions and isinstance(node,(ast.Assign,ast.AnnAssign)):nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
for kind in ['summit','raven']:
 c0=D[f'df_{kind}_c'];a0=D[f'df_{kind}_a'];rows=[]
 lat,lon=D[f'{kind.upper()}_LAT'],D[f'{kind.upper()}_LON']
 for date,row in c0.iterrows():
  sat,slot,_=ns['get_appx_albedo_greenland'](lat,lon,date,utc_hour=14)
  sza=ns['get_appx_sza'](lat,lon,date,slot or 14)
  if not np.isfinite(sat) or not np.isfinite(sza) or sza>=D['MAX_SZA']:continue
  rows.append(dict(date=date,tower_albedo=row.tower_albedo,sat_albedo=sat,doy=pd.Timestamp(date).dayofyear))
 a=pd.DataFrame(rows,columns=['date','tower_albedo','sat_albedo','doy']).set_index('date')
 a=ns['remove_outliers'](a,'sat_albedo','tower_albedo')
 c=c0.loc[c0.index.isin(a.index)].copy();a=a.loc[c.index].copy()
 assert c.index.equals(a.index);np.testing.assert_allclose(c.tower_albedo,a.tower_albedo)
 for suffix,df in [('c',c),('a',a)]:
  D[f'df_{kind}_{suffix}']=df;D[f'stats_{kind}_{suffix}']=stats(*xy(df,kind));df.to_csv(OUT/f'{kind}_{suffix}.csv')
 audit_pair(kind,c0,a0,c,a,'APP-x sampled on CLARA pentad start dates; identical tower means; original sampler/QC')
 print(kind,len(c),'shared pentads',flush=True)
for kind in ['sheba','tara']:
 stem='tara_broadband' if kind=='tara' else 'sheba'
 c0=pd.read_csv(BASE/f'{kind}_surface_flux/{stem}_clara_matchups.csv',parse_dates=['pentad'])
 a0=pd.read_csv(BASE/f'{kind}_surface_flux/{stem}_appx_matchups.csv',parse_dates=['date'])
 c,a=full_pentads(c0,a0,pd.DatetimeIndex(c0.pentad),pd.DatetimeIndex(a0.date))
 c.to_csv(OUT/f'{stem}_clara_matchups.csv',index=False);a.to_csv(OUT/f'{stem}_appx_matchups.csv',index=False)
 audit_pair(kind,c0,a0,c,a,'CLARA pentads with all five APP-x daily dates')
with (OUT/'validation_date_matched.pkl').open('wb') as f:pickle.dump(D,f)
pd.DataFrame(audit).to_csv(OUT/'date_matched_stats.csv',index=False)
print(pd.DataFrame(audit).to_string(index=False),flush=True)
(OUT/'README.md').write_text('''# Figure 9: shared-date sensitivity check
Separate review variant; the main notebook, source pickle, and f06/f09 figures are unchanged.

MOSAiC tower, SHEBA, Tara: retain only CLARA pentads having all five APP-x daily matches; retain APP-x records on exactly those five-day windows. Native daily/pentad averaging and original spatial/SZA/QC methods remain unchanged. Existing CLARA in-situ means are not recalculated.
Flights, drone and transects: intersect observation calendar dates and retain each product's existing valid spatial records; these are not necessarily identical pixels or instruments.
Summit/Raven: existing five-day anchors differ between products. Resample APP-x on CLARA start dates with the original APP-x sampling, SZA filter, temporal fallback, and outlier filter; use the same tower means as CLARA, then intersect successful dates. APP-x remains the original representative-date estimate, not a new five-day satellite mean.
ARISE: no APP-x data, so neither product has shared dates.

This is a date-availability sensitivity check, not a controlled solar-zenith-angle experiment. Product-specific pixels, SZA corrections, temporal compositing/fallback, and averaging can still differ. Source matching code and current plotting code are copied alongside the variant for provenance.
''')
