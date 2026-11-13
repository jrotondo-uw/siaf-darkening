from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
OUT=R/'figures'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
d=pd.read_csv(R/'siaf_pixelwise_summary_monthly.csv');d=d[np.isclose(d.pval,.05)]
months=['April','May','June','July','August','September']
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,(method,period) in zip(axs.flat,[(m,p) for m in ['CERES','Kernel'] for p in ['CMIP6','Obs-era']]):
 s=d.query('rs_method==@method and period==@period').set_index('month').loc[months]
 for k,label,col,off,mk in [('tot','Total','#222222',-.19,'o'),('ext','Area loss','#3A6BAF',0,'s'),('drk','Darkening','#D95F1A',.19,'^')]:
  y=s[f'SIAF_{k}_global']/6;x=np.arange(6)+off
  ax.plot(x,y,ls='none',marker=mk,color=col,label=label,markersize=6)
  if k=='tot':ax.errorbar(x,y,yerr=s.sigma2_tot_global/6,fmt='none',ecolor=col,capsize=3,lw=1)
 ax.set_title(('CERES APRP' if method=='CERES' else 'ERA5 kernel')+' · '+('1982–2014' if period=='CMIP6' else '1982–2024'),fontsize=12,loc='left')
 ax.axhline(0,color='.5',lw=.6);ax.grid(axis='y',color='.9');ax.set_axisbelow(True)
 ax.set_xticks(range(6),[m[:3] for m in months]);ax.tick_params(labelsize=11)
for ax in axs[:,0]:ax.set_ylabel('Contribution to annual SIAF (W m$^{-2}$ K$^{-1}$)')
for i,ax in enumerate(axs.flat):ax.text(-.1,1.12,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=13)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
fig.subplots_adjust(top=.86,bottom=.08,left=.09,right=.99,hspace=.30,wspace=.12)
for ext in ['pdf','png']:fig.savefig(OUT/f'f03.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
