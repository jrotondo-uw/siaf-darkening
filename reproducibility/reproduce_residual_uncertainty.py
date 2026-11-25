"""Recompute the primary 1982–2014 residual spreads from deposited annual series.

Usage: python reproducibility/reproduce_residual_uncertainty.py
No Derecho access or raw-data download is needed. numpy and pandas required.
This is a conditional residual spread, not a complete measurement-error model.
"""
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
D=R/'analysis_data/observed_uncertainty_20261125'
OUT=R/'output/reproduced_residual_uncertainty';OUT.mkdir(parents=True,exist_ok=True)
B=10000

def widths(y,T,year,b):
    y=np.asarray(y,dtype=float).copy();y-=y.mean()
    T=np.asarray(T,dtype=float).copy();T-=T.mean()
    t=np.asarray(year,dtype=float).copy();t-=t.mean()
    n=len(t);assert n==33
    residual=y-float(b)*t
    result={}
    for L in [1,3,5]:
        starts=np.random.default_rng(20261023+L).integers(0,n,size=(B,int(np.ceil(n/L))))
        idx=((starts[:,:,None]+np.arange(L))%n).reshape(B,-1)[:,:n]
        result[f'block{L}_2sigma']=2*(residual[idx]@T/(T@T)).std(ddof=1)
    return result

# Ensemble calendar-time trend: members within model, models within family, equal families.
cols=['total','area_loss','darkening']
m=pd.read_csv(D/'SIAF_model_member_calendar_trends.csv')
assert len(m)==378 and m.model.nunique()==31 and m.family.nunique()==18
b=m.groupby(['family','model'])[cols].mean().groupby('family').mean().mean()
rows=[]
for method in ['CERES','Kernel']:
    a=pd.read_csv(D/f'advisor_annual_{method}.csv')
    for c in cols:rows.append(dict(method=method,component=c,**widths(a[c],a.temperature,a.year,b[c])))
siaf=pd.DataFrame(rows)
ref=pd.read_csv(D/'observed_minus_MMM_uncertainty.csv')
for col in ['block1_2sigma','block3_2sigma','block5_2sigma']:
    np.testing.assert_allclose(siaf[col],ref[col],rtol=0,atol=1e-12)
siaf.to_csv(OUT/'SIAF_residual_widths.csv',index=False)

rows=[]
a=pd.read_csv(D/'observed_IS_monthly_series.csv')
bis=pd.read_csv(D/'IS_MMM_trends.csv',index_col=0).MMM_trend
for comp,z in a.groupby('component'):
    assert (z.groupby('year').month.nunique()==6).all()
    annual=z.groupby('year')[['anomaly','temperature']].mean().reset_index()
    rows.append(dict(metric='full_IS',component=comp,month=np.nan,**widths(annual.anomaly,annual.temperature,annual.year,bis[comp])))
f=pd.read_csv(D/'observed_Fig5_series.csv')
for (month,comp),z in f.groupby(['month','component']):
    z=z.sort_values('year')
    rows.append(dict(metric='Fig5',component=comp,month=month,**widths(z.anomaly,z.temperature,z.year,z.MMM_trend.iloc[0])))
other=pd.DataFrame(rows);ref=pd.read_csv(D/'observed_minus_MMM_consistent_widths.csv')
for col in ['block1_2sigma','block3_2sigma','block5_2sigma']:
    np.testing.assert_allclose(other[col],ref[col],rtol=0,atol=1e-12)
other.to_csv(OUT/'IS_and_Figure5_residual_widths.csv',index=False)
print('PASS: all deposited SIAF, IS and Figure5 widths reproduced within 1e-12.')
print(siaf[['method','component','block3_2sigma']].to_string(index=False))
