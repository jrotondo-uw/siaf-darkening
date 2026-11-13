from pathlib import Path
import ast,pickle
import numpy as np,pandas as pd
BASE=Path('/glade/work/jrotondo/albedo_satellite_validation')
SRC=BASE/'date_matched_review_20261022';OUT=BASE/'pentad_matched_review_20261022';OUT.mkdir(exist_ok=True)
with (SRC/'validation_date_matched.pkl').open('rb') as f:D=pickle.load(f)

def stats(d):
 x=d.insitu.to_numpy(float);y=d.sat.to_numpy(float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 return dict(n=len(x),bias=float(np.mean(y-x)) if len(x) else np.nan,rmse=float(np.sqrt(np.mean((y-x)**2))) if len(x) else np.nan,r=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else np.nan)
def dates(df):
 if 'date' in df:
  v=df.date
  return pd.to_datetime(v.astype(str),format='%Y%m%d') if len(v) and str(v.iloc[0]).isdigit() else pd.to_datetime(v)
 return pd.to_datetime(df.index)
def normalize(df,kind):
 if not len(df):return pd.DataFrame(columns=['date','insitu','sat'])
 if kind=='tower':x=df.tower_sza_matched.fillna(df.tower_scaled);y=df.sial_blue
 else:
  col={'transect':'transect_albedo','heli':'heli_albedo_corr','drone':'drone_albedo','arise':'aircraft_albedo','arcsix':'aircraft_albedo','summit':'tower_albedo','raven':'tower_albedo'}[kind]
  x=df[col];y=df.sat_albedo
 return pd.DataFrame(dict(date=np.asarray(dates(df)),insitu=np.asarray(x),sat=np.asarray(y)))
def pentad(dates):
 dates=pd.DatetimeIndex(dates);jan=pd.to_datetime(dates.year.astype(str)+'-01-01');return jan+pd.to_timedelta(((dates-jan).days//5)*5,unit='D')
results=[]
def save(kind,product,before,after,method):
 after.to_csv(OUT/f'{kind}_{product}.csv',index=False)
 results.append(dict(platform=kind,product=product,method=method,**stats(after),prior_n=stats(before)['n'],prior_bias=stats(before)['bias'],prior_rmse=stats(before)['rmse'],prior_r=stats(before)['r']))
for kind in ['tower','sheba','tara']:
 if kind=='tower':c=normalize(D['df_tower_c'],kind);a=normalize(D['df_tower_a'],kind)
 else:
  stem='tara_broadband' if kind=='tara' else 'sheba'
  c=pd.read_csv(SRC/f'{stem}_clara_matchups.csv',parse_dates=['pentad']).rename(columns={'pentad':'date'})[['date','insitu','sat']]
  a=pd.read_csv(SRC/f'{stem}_appx_matchups.csv',parse_dates=['date'])[['date','insitu','sat']]
 rows=[]
 for date in c.date:
  sub=a[(a.date>=date)&(a.date<date+pd.Timedelta(days=5))]
  assert len(sub)==5 and sub.date.nunique()==5
  rows.append(dict(date=date,insitu=sub.insitu.mean(),sat=sub.sat.mean(),n_days=5))
 b=pd.DataFrame(rows)
 np.testing.assert_allclose(stats(a)['bias'],stats(b)['bias'],atol=1e-12)
 save(kind,'c',c,c,'Existing CLARA pentad mean unchanged')
 save(kind,'a',a,b,'Mean of five matched APP-x daily pairs; equal daily weights')
for kind in ['transect','heli','drone','arise','arcsix']:
 inputs={s:normalize(D[f'df_{kind}_{s}'],kind) for s in ['c','a']}
 outputs={}
 for suffix,d in inputs.items():
  if not len(d):outputs[suffix]=pd.DataFrame(columns=['date','insitu','sat','n_days']);continue
  daily=d.groupby('date',as_index=False)[['insitu','sat']].mean()
  daily['pentad']=pentad(daily.date)
  outputs[suffix]=daily.groupby('pentad').agg(insitu=('insitu','mean'),sat=('sat','mean'),n_days=('date','size')).reset_index().rename(columns={'pentad':'date'})
 assert set(outputs['c'].date)==set(outputs['a'].date)
 for s in ['c','a']:save(kind,s,inputs[s],outputs[s],'Spatial mean per sampled date, then equal-day mean within Jan-1 anchored pentads; no missing days filled')
# Greenland: original source uses a representative APP-x date, not a five-day mean.
p=SRC/'save_validation_data.py';tree=ast.parse(p.read_text());nodes=[];before=True
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 if isinstance(node,ast.FunctionDef):before=False
 if before and isinstance(node,(ast.Assign,ast.AnnAssign)):nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
for kind in ['summit','raven']:
 c=normalize(D[f'df_{kind}_c'],kind);a=normalize(D[f'df_{kind}_a'],kind)
 lat,lon=D[f'{kind.upper()}_LAT'],D[f'{kind.upper()}_LON'];rows=[];dailyrows=[]
 for _,row in c.iterrows():
  vals=[]
  for day in pd.date_range(row.date,periods=5):
   sat,slot,_=ns['get_appx_albedo_greenland'](lat,lon,day,utc_hour=14)
   sza=ns['get_appx_sza'](lat,lon,day,slot or 14)
   ok=np.isfinite(sat) and np.isfinite(sza) and sza<D['MAX_SZA']
   dailyrows.append(dict(pentad=row.date,date=day,sat=sat,sza=sza,valid=ok))
   if ok:vals.append(sat)
  if len(vals)==5:rows.append(dict(date=row.date,insitu=row.insitu,sat=np.mean(vals),n_days=5))
 b=pd.DataFrame(rows,columns=['date','insitu','sat','n_days']);keep=set(b.date)
 ckeep=c[c.date.isin(keep)].copy();akeep=a[a.date.isin(keep)].copy()
 assert ckeep.date.tolist()==b.date.tolist()
 save(kind,'c',ckeep,ckeep,'Existing CLARA pentads with all five valid APP-x days')
 save(kind,'a',akeep,b,'Original APP-x sampler evaluated daily; five-day mean; same tower reference as CLARA; no new residual trimming')
 pd.DataFrame(dailyrows).to_csv(OUT/f'{kind}_daily_audit.csv',index=False)
 print(kind,len(c),'previous shared pentads;',len(b),'complete five-day APP-x means',flush=True)
pd.DataFrame(results).to_csv(OUT/'pentad_comparison_stats.csv',index=False)
print(pd.DataFrame(results).to_string(index=False),flush=True)
(OUT/'README.md').write_text('''# Figure 9: matched-pentad sensitivity check
The main figure and previous shared-date variant are unchanged.
MOSAiC tower/SHEBA/Tara: retain exactly the previously shared pentads. Average both APP-x in-situ and satellite daily values over all five days with equal weights. CLARA values are unchanged.
Flights, drone, transects: average existing matched spatial records within each shared observation date, then average those daily means within Jan-1 anchored five-day windows. Both products have the same observed dates, but not necessarily the same pixels/instruments. Sparse campaigns are sampled-day means, not complete five-day coverage. This additionally changes spatial weighting; interpret separately from the station temporal-averaging experiment.
Summit/Raven: previous APP-x points were representative-date estimates, not pentad satellite means. Evaluate the original sampler and SZA filter on each day of each previously shared pentad; average only complete five-day sets, retain the same tower pentad reference, and intersect CLARA accordingly. Compare prior statistics on precisely these retained windows. The original sampler may use its existing +/-2 day temporal fallback. No new residual-based outlier removal is applied after averaging.
No spatial interpolation methods or existing corrections were changed. CLARA averages remain product pentad composites. Averaging may reduce scatter without reducing systematic bias; improved RMSE is not independent proof of higher accuracy.
''')
