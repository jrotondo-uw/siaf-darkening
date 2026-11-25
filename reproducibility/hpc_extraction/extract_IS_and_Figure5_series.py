from pathlib import Path
import sys,json,gc,glob
import numpy as np,pandas as pd
sys.path.insert(0,'/glade/work/jrotondo/siaf_weight_fix_review_20261125/code')
import siaf_obs_core as C
ROOT=Path('/glade/work/jrotondo/siaf_weight_fix_review_20261125/observed_uncertainty')
ROOT.mkdir(exist_ok=True)
NF=Path('/glade/derecho/scratch/jrotondo/fig5_uncertainty_nofilter_20261123')
mmm=pd.read_csv(NF/'mmm_trends_nf.csv').set_index(['month','region'])
cache=C.CACHE_CMIP6;yrs=np.load(cache+'T_annual_years.npy').astype(int);T=np.load(cache+'T_annual.npy').astype(float)
assert yrs.shape==(33,) and T.shape==(33,)
tmap=dict(zip(yrs,T));allrows=[];checks=[];figrows=[]
for mn in C.MONTH_NUMS:
 mo=C.MONTH_NAMES[mn];a,t,y=C.load_alb_yearly_with_T(cache,mo,tmap);y=np.asarray(y);assert np.array_equal(y,yrs)
 s=C.load_sic_data(cache,mo);d=C.load_IS_map(cache,mo);masks=C.compute_IS(d,s,.05)
 aa=a-np.nanmean(a,axis=0)
 for j,comp in enumerate(['total','area_loss','darkening']):
  support=np.isfinite(masks[j])
  if j:support &= np.isfinite(masks[0])&(np.abs(masks[j])>0)
  vals=np.array([C.domain_mean(C.regrid_sial_to_psn(np.where(support,-np.nan_to_num(v,nan=0.),np.nan))) for v in aa])
  x=t-t.mean();rec=x@vals/(x@x);expected=C.domain_mean(C.regrid_sial_to_psn(masks[j]));checks.append(dict(metric='full_IS',month=mn,component=comp,center=expected,reconstructed=rec,difference=rec-expected))
  allrows.extend(dict(metric='full_IS',month=mn,component=comp,year=int(yr),temperature=float(temp),anomaly=float(v)) for yr,temp,v in zip(y,t,vals))
 sicdir='/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
 cy=np.load(sicdir+f'bivdiag_sic_years_{mo}.npy');raw=np.load(sicdir+f'bivdiag_sic_sial_{mo}.npy',mmap_mode='r');ci=[int(np.where(cy==yr)[0][0]) for yr in y];cc=np.array(raw[ci],dtype=np.float32);del raw
 valid=C.ARCTIC_SIAL&np.isfinite(-d)
 change=valid&np.isfinite(s['slope'])&np.isfinite(s['pval'])&(s['slope']<0)&(s['pval']<.05)
 stable=valid&~change&(np.fmax(s['sic_e'],s['sic_l'])>=.15)
 # The direct I and L slopes are evaluated on the common finite albedo/SIC pairs per native pixel.
 finite=np.isfinite(a)&np.isfinite(cc);ac=np.where(finite,a,np.nan);cs=np.where(finite,cc,np.nan)
 ua=C.compute_pixel_ols_slopes(ac,t);uc=C.compute_pixel_ols_slopes(cs,t)
 ac=ac-np.nanmean(ac,axis=0);cs=cs-np.nanmean(cs,axis=0)
 for reg,mask,var,data,sl,bname in [('area_loss',change,'L',cs,uc,'trend_c'),('darkening',stable,'I',ac,ua,'trend_a')]:
  keep=mask&np.isfinite(ua)&np.isfinite(uc)
  vals=-np.mean(np.nan_to_num(data[:,keep],nan=0.),axis=1);expect=-float(np.mean(sl[keep]));x=t-t.mean();rec=x@vals/(x@x)
  checks.append(dict(metric='Fig5',month=mn,component=reg,center=expect,reconstructed=rec,difference=rec-expect,n_pixels=int(keep.sum())))
  b=-float(mmm.loc[(mn,reg),bname]);figrows.extend(dict(metric='Fig5',month=mn,component=reg,year=int(yr),temperature=float(temp),anomaly=float(v),MMM_trend=b,variable=var) for yr,temp,v in zip(y,t,vals))
 print('MONTH',mn,'done',flush=True);del a,aa,cc,ac,cs;gc.collect()
pd.DataFrame(allrows).to_csv(ROOT/'observed_IS_monthly_series.csv',index=False)
pd.DataFrame(figrows).to_csv(ROOT/'observed_Fig5_series.csv',index=False)
pd.DataFrame(checks).to_csv(ROOT/'reconstruction_checks.csv',index=False)
# Full IS model mean calendar trend, same 378 members/31 models/18 families as saved primary.
FAM={'ACCESS':['ACCESS-ESM1-5','ACCESS-CM2'],'BCC':['BCC-CSM2-MR','BCC-ESM1'],'CESM2':['CESM2','CESM2-WACCM','CESM2-FV2'],'CMCC':['CMCC-CM2-SR5'],'CNRM':['CNRM-CM6-1','CNRM-ESM2-1'],'CanESM5':['CanESM5'],'E3SM':['E3SM-1-0'],'EC-Earth3':['EC-Earth3-Veg','EC-Earth3-Veg-LR'],'FGOALS':['FGOALS-f3-L','FGOALS-g3'],'GISS':['GISS-E2-1-H'],'HadGEM3/UKESM':['UKESM1-0-LL','HadGEM3-GC31-LL','HadGEM3-GC31-MM'],'INM':['INM-CM5-0'],'IPSL':['IPSL-CM6A-LR'],'MIROC':['MIROC-ES2L','MIROC6'],'MPI/ICON':['MPI-ESM1-2-LR','MPI-ESM1-2-HR','MPI-ESM-1-2-HAM'],'MRI':['MRI-ESM2-0'],'NESM3':['NESM3'],'NorESM':['NorCPM1','NorESM2-LM']}
m2f={m:f for f,mods in FAM.items() for m in mods};rows=[]
for p in Path('/glade/derecho/scratch/jrotondo/uncertainty_is_20261124/members').glob('*.npz'):
 z=np.load(p);x=z['years'].astype(float);x-=x.mean();q=z['q_is'].astype(float);s=x@(q-q.mean(axis=0))/(x@x)
 rows.append(dict(model=str(z['model']),member=str(z['member']),**dict(zip(['total','area_loss','darkening'],s))))
r=pd.DataFrame(rows);assert len(r)==378 and r.model.nunique()==31;r['family']=r.model.map(m2f);assert r.family.nunique()==18
r.to_csv(ROOT/'IS_model_member_trends.csv',index=False)
b=r.groupby(['family','model'])[['total','area_loss','darkening']].mean().groupby('family').mean().mean();b.rename('MMM_trend').to_csv(ROOT/'IS_MMM_trends.csv')
print('DONE',b.to_dict(),flush=True)
