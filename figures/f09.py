from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path('/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022')
s=pd.read_csv(R/'Figure_9_statistics_CCC.csv')
rows=[('SHEBA surface flux (1997–98)','SHEBA 1997-98'),('SHEBA transects (1998)','SHEBA transects'),('Tara (2007)','Tara 2007'),('Summit (2012–13)','Summit Greenland'),('ARISE (2014)','ARISE 2014'),('MOSAiC tower (2020)','MOSAiC Tower'),('MOSAiC transects (2020)','MOSAiC Transects'),('MOSAiC helicopter (2020)','MOSAiC Helicopter'),('MOSAiC drone (2020)','HELiX Drone'),('Raven / SLEIGH (2024)','Raven Greenland'),('ARCSIX (2024)','ARCSIX 2024')]
products=[('CLARA-A3','#6A3FA5','o',-.12),('APP-x','#009E73','^',.12)]
ordered=[]
for label,prefix in rows:
 for product,_,_,_ in products:
  q=s[s.dataset.str.startswith(prefix)&s['product'].eq(product)]
  assert len(q)==1,(label,product)
  ordered.append(dict(q.iloc[0],dataset=label))
d=pd.DataFrame(ordered)
assert len(d)==22 and d.n.sum()==s.n.sum()
assert d.loc[d.n.eq(0),['bias','rmse']].isna().all().all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(7.2,6.5))
gs=fig.add_gridspec(1,4,width_ratios=[.95,.95,.95,.75],left=.30,right=.99,bottom=.17,top=.82,wspace=.22)
ax1=fig.add_subplot(gs[0]);ax2=fig.add_subplot(gs[1],sharey=ax1);ax3=fig.add_subplot(gs[2],sharey=ax1);counts=fig.add_subplot(gs[3],sharey=ax1)
for ax in [ax1,ax2,ax3,counts]:
 ax.set_ylim(10.6,-.6)
 for y in range(11):
  if y%2==0:ax.axhspan(y-.5,y+.5,color='#f2f5f7',zorder=0,lw=0)
 for y in [1.5,2.5,3.5,4.5,8.5]:ax.axhline(y,color='#d8dfe4',lw=.65,zorder=0)
for ax,metric,title in [(ax1,'bias','a  Bias'),(ax2,'rmse','b  RMSE'),(ax3,'ccc','c  CCC')]:
 for product,color,marker,offset in products:
  q=d[d['product'].eq(product)].reset_index(drop=True)
  valid=q.n.gt(0)
  ax.scatter(q.loc[valid,metric],np.arange(11)[valid]+offset,s=30,color=color,marker=marker,edgecolor='white',linewidth=.6,zorder=3)
 ax.set_title(title,loc='left',fontsize=9,fontweight='bold',pad=14)
 ax.grid(axis='x',color='#dfe4e8',linewidth=.7,zorder=0)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
ax1.set_yticks(range(11),[r[0] for r in rows]);ax1.tick_params(axis='y',pad=7,labelsize=8)
ax1.set_xlim(-.37,.075);ax1.set_xticks([-.3,0]);ax1.axvline(0,color='#66717a',lw=1.1,zorder=1);ax1.set_xlabel('Bias',labelpad=12)
ax2.set_xlim(0,.39);ax2.set_xticks([0,.1,.2,.3]);ax2.tick_params(labelleft=False);ax2.set_xlabel('RMSE',labelpad=12)
ax3.set_xlim(-1.05,1.05);ax3.set_xticks([-1,0,1]);ax3.tick_params(labelleft=False);ax3.axvline(0,color='#66717a',lw=.9);ax3.axvline(1,color='#66717a',lw=.9,ls='--');ax3.set_xlabel('CCC',labelpad=12)
counts.set_xlim(0,1);counts.axis('off');counts.text(.5,1.11,'Pairs, n',transform=counts.transAxes,ha='center',va='bottom',fontsize=9,clip_on=False)
for j,(product,color,marker,offset) in enumerate(products):
 x=.20+j*.60
 counts.text(x,1.012,('CLARA\nA3' if product=='CLARA-A3' else 'APP\nx'),transform=counts.transAxes,ha='center',va='bottom',fontsize=7.5,color=color,clip_on=False)
 q=d[d['product'].eq(product)].reset_index(drop=True)
 for i,n in enumerate(q.n):counts.text(x,i,f'{int(n):,}',ha='center',va='center',color=color,fontsize=9)
handles=[Line2D([],[],color=c,marker=m,ls='none',markersize=8,label=p) for p,c,m,_ in products]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.63,.925),ncol=2,frameon=False,columnspacing=2.8)

for ext in ['png','pdf']:fig.savefig(R/('Validation_summary_bias_RMSE_CCC.'+ext),dpi=220,facecolor='white')
print(d[['dataset','product','n','bias','rmse']].to_string(index=False))