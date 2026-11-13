from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'Figure_9_chronological_statistics.csv')
rows=[('SHEBA surface flux (1997–98)','SHEBA 1997-98'),('SHEBA transects (1998)','SHEBA transects'),('Tara (2007)','Tara 2007'),('Summit (2012–13)','Summit Greenland'),('ARISE (2014)','ARISE 2014'),('MOSAiC tower (2020)','MOSAiC Tower'),('MOSAiC transects (2020)','MOSAiC Transects'),('MOSAiC helicopter / HELiPOD (2020)','MOSAiC Helicopter'),('MOSAiC drone (2020)','HELiX Drone'),('Raven / SLEIGH (2024)','Raven Greenland'),('ARCSIX (2024)','ARCSIX 2024')]
products=[('CLARA-A3','#0072B2','o',-.12),('APP-x','#D55E00','^',.12)]
ordered=[]
for label,prefix in rows:
 for product,_,_,_ in products:
  q=s[s.dataset.str.startswith(prefix)&s['product'].eq(product)]
  assert len(q)==1,(label,product)
  ordered.append(dict(q.iloc[0],dataset=label))
d=pd.DataFrame(ordered)
assert len(d)==22 and d.n.sum()==s.n.sum()
assert d.loc[d.n.eq(0),['bias','rmse']].isna().all().all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(12.8,7.3))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,.46],left=.30,right=.985,bottom=.17,top=.82,wspace=.16)
ax1=fig.add_subplot(gs[0]);ax2=fig.add_subplot(gs[1],sharey=ax1);counts=fig.add_subplot(gs[2],sharey=ax1)
for ax in [ax1,ax2,counts]:
 ax.set_ylim(10.6,-.6)
 for y in range(11):
  if y%2==0:ax.axhspan(y-.5,y+.5,color='#f2f5f7',zorder=0,lw=0)
 for y in [1.5,2.5,3.5,4.5,8.5]:ax.axhline(y,color='#d8dfe4',lw=.65,zorder=0)
for ax,metric,title in [(ax1,'bias','a  Bias'),(ax2,'rmse','b  RMSE')]:
 for product,color,marker,offset in products:
  q=d[d['product'].eq(product)].reset_index(drop=True)
  valid=q.n.gt(0)
  ax.scatter(q.loc[valid,metric],np.arange(11)[valid]+offset,s=59,color=color,marker=marker,edgecolor='white',linewidth=.6,zorder=3)
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=14)
 ax.grid(axis='x',color='#dfe4e8',linewidth=.7,zorder=0)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
ax1.set_yticks(range(11),[r[0] for r in rows]);ax1.tick_params(axis='y',pad=12)
ax1.set_xlim(-.37,.075);ax1.set_xticks([-.3,-.2,-.1,0]);ax1.axvline(0,color='#66717a',lw=1.1,zorder=1);ax1.set_xlabel('Satellite − reference albedo',labelpad=12)
ax2.set_xlim(0,.39);ax2.set_xticks([0,.1,.2,.3]);ax2.tick_params(labelleft=False);ax2.set_xlabel('Root mean square error (albedo)',labelpad=12)
counts.set_xlim(0,1);counts.axis('off');counts.set_title('Matched pairs, n',fontsize=11,pad=14)
for j,(product,color,marker,offset) in enumerate(products):
 x=.24+j*.53
 counts.text(x,1.012,product,transform=counts.transAxes,ha='center',va='bottom',fontsize=9,color=color)
 q=d[d['product'].eq(product)].reset_index(drop=True)
 for i,n in enumerate(q.n):counts.text(x,i,f'{int(n):,}',ha='center',va='center',color=color,fontsize=10)
handles=[Line2D([],[],color=c,marker=m,ls='none',markersize=8,label=p) for p,c,m,_ in products]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.63,.925),ncol=2,frameon=False,columnspacing=2.8)
fig.suptitle('Albedo validation across field datasets',x=.52,y=.98,fontsize=17,fontweight='bold')
fig.text(.30,.025,'IDW with nearest-neighbor backup; corrected MOSAiC tower SIC blending.\nARISE has no retained APP-x matches (n = 0); no bias or RMSE is plotted.',fontsize=10,color='#47525b',linespacing=1.6)
for ext in ['png','pdf']:fig.savefig(R/('Validation_summary_bias_RMSE.'+ext),dpi=220,facecolor='white')
d.to_csv(R/'Validation_summary_statistics.csv',index=False)
(R/'Validation_summary_caption.txt').write_text('Candidate supplementary figure. Bias (satellite minus reference) and root mean square error of CLARA-A3 and APP-x albedo against each field dataset, ordered chronologically by campaign. Circles denote CLARA-A3 and triangles APP-x. Counts at right give the retained matched pairs for each product; these are not necessarily independent observations. No APP-x estimates are plotted for ARISE because no pairs were retained. The summary uses the current Figure 9 matchups: IDW with valid nearest-neighbor spatial fallback within the existing distance limits, corrected MOSAiC tower sea-ice-concentration blending, and unchanged prior observation selection and temporal matching. Products may have different dates and sample counts, so differences between scores also reflect sampling differences. Metrics are shown in albedo units.\n')
print(d[['dataset','product','n','bias','rmse']].to_string(index=False))
