"""Main Figure 8: one common-MMM model-based uncertainty convention."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
R=Path(__file__).resolve().parent
d=pd.read_csv('/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv')
d=d[d.rs_method.eq('full_T07')]
cnt=d.groupby('model').size();d=d[d.model.isin(cnt[cnt>=5].index)]
u=pd.read_csv(R/'uncertainty.csv').query("method == 'Common MMM trend' and block_years == 3").set_index('component')
obs=pd.read_csv('/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv')
o=obs[obs.rs_method.eq('CERES')&obs.period.eq('CMIP6')&np.isclose(obs.pval,.05)].iloc[0]
fig,axes=plt.subplots(3,1,figsize=(7.2,8.5),sharex=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
rows=[]
for j,(ax,c,modelcol,obscol,label) in enumerate(zip(axes,['area_loss','darkening','total'],['siaf_ext_global','siaf_drk_global','siaf_global'],['SIAF_ext_global','SIAF_drk_global','SIAF_tot_global'],['Area loss','Darkening','Total'])):
 g=d.groupby('model')[modelcol].agg(['mean','std','count'])
 mu=float(o[obscol]);sd=float(u.loc[c,'joint_2sigma'])/2
 x=np.linspace(-.08,.36,1800)
 w=pd.Series(norm.pdf(mu,g['mean'],np.sqrt(g['std']**2+sd**2)),index=g.index);w=w/w.sum()
 uw=np.mean(norm.pdf(x[:,None],g['mean'].to_numpy(),g['std'].to_numpy()),axis=1)
 bma=np.sum(norm.pdf(x[:,None],g['mean'].to_numpy(),g['std'].to_numpy())*w.to_numpy(),axis=1)
 vc=np.mean(norm.pdf(x[:,None],g['mean'].to_numpy(),sd),axis=1)
 ax.plot(x,uw,color='#888888',ls='--',label='Unweighted')
 ax.plot(x,bma,color='#222222',label='BMA weighted')
 ax.plot(x,vc,color='#4B3F9E',label='Common variance')
 ax.plot(x,norm.pdf(x,mu,sd),color='#b5000a',label='Observed center, model-based uncertainty')
 ax.axvline(mu,color='#b5000a',ls=':',lw=.8)
 ax.set_title(f'{chr(97+j)}  {label}',loc='left',fontsize=11,fontweight='bold')
 ax.set_ylabel('Density');ax.spines[['top','right']].set_visible(False)
 ax.set_ylim(bottom=0);ax.set_xlim(-.04,.32)
 for model,row in g.iterrows():rows.append(dict(component=c,model=model,n=int(row['count']),mean=row['mean'],sd=row['std'],weight=w.loc[model],obs_mean=mu,obs_sd=sd))
axes[0].legend(fontsize=8,frameon=False)
axes[-1].set_xlabel('SIAF (W m$^{-2}$ K$^{-1}$)')
fig.tight_layout();fig.savefig(R/'figures'/'f08.png',dpi=300)
pd.DataFrame(rows).to_csv(R/'figures'/'BMA_weights.csv',index=False)
