from pathlib import Path
import numpy as np,pandas as pd,gc
import siaf_obs_core as C
O=Path('/glade/work/jrotondo/siaf_weight_fix_review_20261125/results')
c=C.CACHE_CMIP6;ys=np.load(c+'T_annual_years.npy').astype(int);temp=np.load(c+'T_annual.npy');td=dict(zip(ys,temp));print('T CACHE',ys.shape,temp.shape)
monthly={m:[] for m in ['CERES','Kernel']};points={m:[] for m in monthly};coarse=[]
# Grid-cell binning is a comparison on the same fine-cell area quadrature, not a new source of resolution.
for mn in C.MONTH_NUMS:
 mo=C.MONTH_NAMES[mn];stk,t,y=C.load_alb_yearly_with_T(c,mo,td);sic=C.load_sic_data(c,mo);d=C.load_IS_map(c,mo);masks=C.compute_IS(d,sic,.05)
 rs=C.load_RS_ceres_mean(C.CERES_CMIP6,mn).ravel()[C.IDX_CERES_TO_PSN].reshape(C.PSN_SHAPE)
 rp={'CERES':rs,'Kernel':C.load_kernel_psn(mn)}
 for method in monthly:points[method].append([C.domain_mean(C.regrid_sial_to_psn(v)*rp[method])*C.SCALE for v in masks])
 anomalies=stk-np.nanmean(stk,axis=0)
 for yr,arr in zip(y,anomalies):
  for method in monthly:
   vals=[]
   for j,mask in enumerate(masks):
    support=np.isfinite(mask)
    if j:support &= np.isfinite(masks[0])&(np.abs(mask)>0)
    field=np.where(support,-np.nan_to_num(arr,nan=0.),np.nan)
    vals.append(C.domain_mean(C.regrid_sial_to_psn(field)*rp[method])*C.SCALE)
   monthly[method].append([mn,yr,td[yr],*vals])
 print('MONTH',mo,flush=True);del stk,anomalies;gc.collect()
for method,data in monthly.items():
 z=pd.DataFrame(data,columns=['month','year','temperature','total','area_loss','darkening']);z.to_csv(O/f'advisor_annual_monthly_{method}.csv',index=False)
 a=z.groupby('year')[['temperature','total','area_loss','darkening']].mean();a.to_csv(O/f'advisor_annual_{method}.csv');p=np.mean(points[method],axis=0);np.save(O/f'advisor_point_{method}.npy',p)
 x=a.temperature.to_numpy();x=x-x.mean();rec=x@a[['total','area_loss','darkening']].to_numpy()/(x@x);print('RECONSTRUCTION',method,p,rec,rec-p,flush=True)
