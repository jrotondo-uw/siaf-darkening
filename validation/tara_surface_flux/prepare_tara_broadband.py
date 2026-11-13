"""Derive daily broadband albedo from the supplied radiometer workbook.

Requires pandas, numpy, xlrd. Timestamps are used as supplied (assumed UTC).
Equal-duration minute samples: daily albedo = sum(SW Out)/sum(SW In).
QC: finite fluxes, SW In >= 20 W/m2, 0 <= SW Out/SW In <= 1.
Positions are interpolated in polar stereographic coordinates only within
hourly track coverage, with no interpolation across gaps longer than 2 hours.
No spectral albedo, calibration adjustment, or missing-value imputation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    import xlrd
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
    import xlrd
from pyproj import Transformer

BASE = Path(__file__).resolve().parent
book = xlrd.open_workbook(str(BASE/'Tara_radiomet_Apr-Oct_clean.xls'))
frames = []
for sh in book.sheets():
    assert sh.row_values(0)[:3] == ['Date', 'SW In W/m²', 'SW Out W/m²']
    d = pd.DataFrame([sh.row_values(i)[:3] for i in range(1,sh.nrows)],
                     columns=['excel_date','sw_in','sw_out'])
    for c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    d['date'] = pd.to_datetime(d.excel_date, unit='D', origin='1899-12-30').dt.round('min')
    frames.append(d[['date','sw_in','sw_out']])
d = pd.concat(frames, ignore_index=True).sort_values('date')
source_rows = len(d)
duplicate_rows = int(d.date.duplicated(keep=False).sum())
# Ambiguous repeated timestamps are excluded, never selected arbitrarily.
d = d.loc[~d.date.duplicated(keep=False)].copy()
track = pd.read_csv(BASE/'Tara_coord_23Mar_19Sep.txt', sep=r'\s+')
track['date'] = pd.to_datetime(track[['year','month','day','hour','min']].rename(columns={'min':'minute'}))
track = track.sort_values('date').drop_duplicates('date')
fwd = Transformer.from_crs(4326,3413,always_xy=True)
inv = Transformer.from_crs(3413,4326,always_xy=True)
tx,ty = fwd.transform(track.lonE,track.latN)
xt = track.date.astype('int64').to_numpy(); dt = d.date.astype('int64').to_numpy()
k = np.searchsorted(xt,dt,side='right'); lo=np.clip(k-1,0,len(xt)-1); hi=np.clip(k,0,len(xt)-1)
position_ok = (dt>=xt[0]) & (dt<=xt[-1]) & ((xt[hi]-xt[lo])<=pd.Timedelta('2h').value)
d['x']=np.interp(dt,xt,tx); d['y']=np.interp(dt,xt,ty)
ratio=d.sw_out/d.sw_in
flux_ok=np.isfinite(d.sw_in)&np.isfinite(d.sw_out)&(d.sw_in>=20)&ratio.between(0,1)
valid=d.loc[flux_ok & position_ok].copy()
valid['day']=valid.date.dt.floor('D')
g=valid.groupby('day')
out=g.agg(sw_in_sum=('sw_in','sum'),sw_out_sum=('sw_out','sum'),n=('sw_in','size'),x=('x','mean'),y=('y','mean'))
out['alb']=out.sw_out_sum/out.sw_in_sum
out['lon'],out['lat']=inv.transform(out.x.to_numpy(),out.y.to_numpy())
out.index.name='date'
out=out.reset_index()[['date','alb','n','lat','lon','sw_in_sum','sw_out_sum']]
out['albedo_type']='broadband'
out.to_csv(BASE/'tara_broadband_daily.csv',index=False)
audit=dict(source_rows=source_rows,duplicate_timestamp_rows_excluded=duplicate_rows,valid_rows=len(valid),flux_rejected=int((~flux_ok).sum()),
           outside_or_missing_track=int((~position_ok).sum()),days=len(out),start=str(out.date.min()),end=str(out.date.max()),
           time_assumption='Source timestamps assumed UTC; timezone not stated in workbook',
           qc='SW In >=20 W/m2; finite fluxes; 0<=SW Out/SW In<=1; position within track, max 2h gap',
           averaging='Daily sum(SW Out)/sum(SW In) over accepted minute samples')
(BASE/'tara_broadband_processing.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2));print(out.groupby(out.date.dt.month).alb.agg(['count','mean','min','max']).to_string())
