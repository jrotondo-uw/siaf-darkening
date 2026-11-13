from pathlib import Path
import pandas as pd,numpy as np, csv,re,json
from pyproj import Transformer
R=Path(__file__).resolve().parent
tf=Transformer.from_crs(32604,4326,always_xy=True)
ln,lt=tf.transform((588422.82+588226.83)/2,(7920510.74+7920471.23)/2)
ls,ts=tf.transform((587718.95+587519.66)/2,(7920261.77+7920274.16)/2)
sites={'2008':(71.36568,-156.54417,'2008_Kipps_Avg.csv'),'2009N':(lt,ln,'Average_North.csv'),'2009S':(ts,ls,'Average_South.csv'),'2010':(71.37096,-156.50975,'Average.csv')}
rows=[]; audit=[]
for site,(lat,lon,f) in sites.items():
 df=pd.read_csv(R/'raw'/site/f).iloc[:,:2];df.columns=['d','alb'];df.alb=pd.to_numeric(df.alb,errors='coerce');df=df.dropna();df['date']=pd.to_datetime(df.d+'-'+site[:4],format='%d-%b-%Y')
 assert df.alb.between(0,1).all() and not df.date.duplicated().any()
 for _,r in df.iterrows():rows.append(dict(site=site,date=r.date,lat=lat,lon=lon,alb=r.alb,source=f,method='author-provided spatial average'))
 # Check supplied daily means against raw equally spaced point means.
 for p in (R/'raw'/site).glob('*.csv'):
  if p.name==f:continue
  rr=list(csv.reader(p.open()));a=[]
  for x in rr[2:]:
   try:
    v=float(x[1]);d=float(x[0]);
    if np.isfinite(v):a.append(v)
   except (ValueError,IndexError):pass
  if a:
   try:dt=pd.to_datetime(rr[0][0],format='%d-%b-%y') if site=='2008' else pd.to_datetime(rr[0][0]+'-'+site[:4],format='%d-%b-%Y')
   except:continue
   ref=df[df.date==dt]
   audit.append(dict(site=site,file=p.name,date=dt,raw_n=len(a),raw_mean=np.mean(a),published_mean=ref.alb.iloc[0] if len(ref) else np.nan,raw_outside_0_1=sum(not 0<=v<=1 for v in a)))
   if site=='2008' and dt==pd.Timestamp('2008-04-08'):
    rows.append(dict(site=site,date=dt,lat=lat,lon=lon,alb=np.mean(a),source=p.name,method='arithmetic mean of raw evenly spaced points'))
out=pd.DataFrame(rows).sort_values(['site','date']);out.to_csv(R/'barrow_daily.csv',index=False)
a=pd.DataFrame(audit);a['difference']=a.raw_mean-a.published_mean;a.to_csv(R/'raw_mean_audit.csv',index=False)
print(out.groupby('site').agg(n=('date','size'),start=('date','min'),end=('date','max')).to_string());print(a[a.difference.abs()>1e-6].to_string(index=False))
