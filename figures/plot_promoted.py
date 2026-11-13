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
 for k,label,col,off,mk in [('tot','Total','#222222',-.19,'o'),('ext','Area loss','#0072B2',0,'s'),('drk','Darkening','#D55E00',.19,'^')]:
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
# Reuse the saved aggregation without its plotting section; data and weighting unchanged.
src=(R/'sie_aggregation_source.py').read_text()
src=src[:src.index('lbls = list(fam.index)')]
src=src.replace("os.makedirs(OUT_DIR, exist_ok=True)",'')
src=src.replace("pd.read_pickle(SIE_PKL)","pd.read_pickle(R/'sie_df_common.pkl')").replace("load_cache('obs_sie_by_month')","pd.read_pickle(R/'obs_sie_by_month.pkl')").replace('pd.read_csv(OBS_SIAF_CSV)',"pd.read_csv(R/'siaf_pixelwise_summary.csv')").replace('pd.read_csv(CMIP6_SIAF_CSV)',"pd.read_csv(R/'siaf_cmip6_summary_final.csv')")
exec(compile(src,'saved_sie_aggregation','exec'))
fam.to_csv(OUT/'f05_family_means.csv');fam_sd.to_csv(OUT/'f05_family_spread.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
for i,(ax,k,ox,xlab,title) in enumerate(zip(axs,['trend','early'],[obs_slope,obs_early],['SIE trend (million km² yr$^{-1}$)','1982–1991 SIE (million km²)'],['Retreat rate','Initial ice extent'])):
 ax.errorbar(fam[k],fam.siaf,xerr=fam_sd[k],yerr=fam_sd.siaf,fmt='none',ecolor='#0072B2',alpha=.23,lw=.8)
 ax.scatter(fam[k],fam.siaf,c='#0072B2',s=35,label='CMIP6 family mean',zorder=3)
 r,p=stats.pearsonr(fam[k],fam.siaf);sl,b=np.polyfit(fam[k],fam.siaf,1);xr=np.linspace(fam[k].min(),fam[k].max(),100)
 ax.plot(xr,sl*xr+b,color='#0072B2',ls='--',lw=1.2)
 ax.errorbar(ox,OBS_TOT,yerr=.04384,fmt='*',color='#D55E00',markersize=13,capsize=4,label='Observed, 1982–2014',zorder=4)
 ax.text(.05,.95,f'r = {r:.2f}; p = {p:.3f}',va='top',transform=ax.transAxes)
 ax.set_title(chr(97+i)+'  '+title,loc='left',weight='bold');ax.set_xlabel(xlab);ax.grid(color='.92');ax.set_axisbelow(True)
axs[0].set_xticks([-.15,-.10,-.05,0]);axs[0].set_ylabel('Total SIAF (W m$^{-2}$ K$^{-1}$)');axs[0].set_ylim(0,.30)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.subplots_adjust(top=.82,bottom=.16,left=.09,right=.98,wspace=.12)
for ext in ['pdf','png']:fig.savefig(OUT/f'f05.{ext}',dpi=220,bbox_inches='tight')
