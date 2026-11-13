from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/glade/work/jrotondo/albedo_satellite_validation/pentad_matched_review_20261022')
st=pd.read_csv(BASE/'pentad_comparison_stats.csv')
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12})
fig,axes=plt.subplots(5,4,figsize=(24,28),layout='constrained')
layout=[('tower','c'),('transect','c'),('heli','c'),('summit','c'),('tower','a'),('transect','a'),('heli','a'),('summit','a'),('drone','c'),('arise','c'),('arcsix','c'),('raven','c'),('drone','a'),('arise','a'),('arcsix','a'),('raven','a'),('sheba','c'),('sheba','a'),('tara','c'),('tara','a')]
names={'tower':'MOSAiC Tower','transect':'MOSAiC Transects','heli':'MOSAiC Heli + HELiPOD','summit':'Summit 2012–2013','drone':'MOSAiC Drone','arise':'NASA ARISE 2014','arcsix':'NASA ARCSIX 2024','raven':'Raven / SLEIGH 2024','sheba':'SHEBA 1997–1998','tara':'Tara 2007 broadband'}
for i,(ax,(kind,suffix)) in enumerate(zip(axes.flat,layout)):
 d=pd.read_csv(BASE/f'{kind}_{suffix}.csv',parse_dates=['date']);product='CLARA-A3' if suffix=='c' else 'APP-x'
 ax.set(xlim=(0,1),ylim=(0,1),xlabel='Pentad-mean in-situ albedo',ylabel=f'{product} albedo',title=f'{chr(65+i)}) {names[kind]}\n{product}')
 ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_axisbelow(True)
 if not len(d):
  ax.text(.5,.5,'No shared dates available',transform=ax.transAxes,ha='center',color='.45',style='italic');continue
 sc=ax.scatter(d.insitu,d.sat,c=d.date.dt.dayofyear,cmap='viridis',vmin=1,vmax=366,s=48,edgecolors='.2',linewidths=.4,alpha=.85)
 ax.plot([0,1],[0,1],'--',color='.45',lw=1.2,label='1:1')
 r=st[(st.platform==kind)&(st['product']==suffix)].iloc[0]
 if np.isfinite(r.r) and r.r>=.5 and len(d)>2:
  slope,intercept=np.polyfit(d.insitu,d.sat,1);xx=np.linspace(d.insitu.min(),d.insitu.max(),100)
  ax.plot(xx,slope*xx+intercept,color='#d62728',lw=1.5,label=f'y={slope:.2f}x{intercept:+.2f}')
 ax.text(.035,.965,f'r = {r.r:.3f}\nbias = {r.bias:+.3f}\nRMSE = {r.rmse:.3f}\nn = {int(r.n)}',transform=ax.transAxes,va='top',fontsize=11,family='monospace',bbox=dict(boxstyle='round,pad=.3',fc='white',ec='.6',alpha=.95))
 ax.legend(loc='lower right',fontsize=9)
 note='Sampled-day means' if kind in ['transect','heli','drone','arise','arcsix'] else 'Five-day windows'
 ax.text(.035,.035,note,transform=ax.transAxes,fontsize=9,color='.35',style='italic')
 cb=fig.colorbar(sc,ax=ax,shrink=.76,pad=.02);cb.set_label('Pentad start: day of year',fontsize=10);cb.ax.tick_params(labelsize=9)
fig.suptitle('Figure 9 — shared dates, pentad averages',fontsize=23)
fig.supxlabel('Station APP-x daily pairs averaged over the same CLARA-A3 five-day windows.\nCampaigns: spatial mean per shared day, then mean of sampled days within each pentad; no missing days filled.\nOriginal spatial matching retained. Greenland requires all five valid APP-x days; ARISE has no shared dates.',fontsize=13)
for ext in ['png','pdf']:fig.savefig(BASE/f'Figure_9_pentad_matched.{ext}',dpi=250)
print('Saved full pentad-average Figure 9',flush=True)
