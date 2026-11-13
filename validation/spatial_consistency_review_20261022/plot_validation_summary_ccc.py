from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'Figure_9_statistics_CCC.csv')
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
fig=plt.figure(figsize=(15.5,7.3))
gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.48],left=.25,right=.985,bottom=.17,top=.82,wspace=.16)
ax1=fig.add_subplot(gs[0]);ax2=fig.add_subplot(gs[1],sharey=ax1);ax3=fig.add_subplot(gs[2],sharey=ax1);counts=fig.add_subplot(gs[3],sharey=ax1)
for ax in [ax1,ax2,ax3,counts]:
 ax.set_ylim(10.6,-.6)
 for y in range(11):
  if y%2==0:ax.axhspan(y-.5,y+.5,color='#f2f5f7',zorder=0,lw=0)
 for y in [1.5,2.5,3.5,4.5,8.5]:ax.axhline(y,color='#d8dfe4',lw=.65,zorder=0)
for ax,metric,title in [(ax1,'bias','a  Bias'),(ax2,'rmse','b  RMSE'),(ax3,'ccc','c  Concordance')]:
 for product,color,marker,offset in products:
  q=d[d['product'].eq(product)].reset_index(drop=True)
  valid=q.n.gt(0)
  ax.scatter(q.loc[valid,metric],np.arange(11)[valid]+offset,s=59,color=color,marker=marker,edgecolor='white',linewidth=.6,zorder=3)
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=14)
 ax.grid(axis='x',color='#dfe4e8',linewidth=.7,zorder=0)
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
ax1.set_yticks(range(11),[r[0] for r in rows]);ax1.tick_params(axis='y',pad=12)
ax1.set_xlim(-.37,.075);ax1.set_xticks([-.3,-.2,-.1,0]);ax1.axvline(0,color='#66717a',lw=1.1,zorder=1);ax1.set_xlabel('Satellite − reference albedo',labelpad=12)
ax2.set_xlim(0,.39);ax2.set_xticks([0,.1,.2,.3]);ax2.tick_params(labelleft=False);ax2.set_xlabel('RMSE (albedo units)',labelpad=12)
ax3.set_xlim(-1.05,1.05);ax3.set_xticks([-1,-.5,0,.5,1]);ax3.tick_params(labelleft=False);ax3.axvline(0,color='#66717a',lw=.9);ax3.axvline(1,color='#66717a',lw=.9,ls='--');ax3.set_xlabel('CCC (1 = perfect agreement)',labelpad=12)
counts.set_xlim(0,1);counts.axis('off');counts.text(.5,1.075,'Matched pairs, n',transform=counts.transAxes,ha='center',va='bottom',fontsize=11,clip_on=False)
for j,(product,color,marker,offset) in enumerate(products):
 x=.24+j*.53
 counts.text(x,1.012,product,transform=counts.transAxes,ha='center',va='bottom',fontsize=9,color=color)
 q=d[d['product'].eq(product)].reset_index(drop=True)
 for i,n in enumerate(q.n):counts.text(x,i,f'{int(n):,}',ha='center',va='center',color=color,fontsize=10)
handles=[Line2D([],[],color=c,marker=m,ls='none',markersize=8,label=p) for p,c,m,_ in products]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.63,.925),ncol=2,frameon=False,columnspacing=2.8)
fig.suptitle('Figure 9 — Albedo validation across field datasets',x=.52,y=.98,fontsize=17,fontweight='bold')
fig.text(.25,.025,'IDW with nearest-neighbor backup; corrected MOSAiC tower SIC blending.\nARISE has no retained APP-x matches (n = 0); no metrics are plotted.',fontsize=10,color='#47525b',linespacing=1.6)
for ext in ['png','pdf']:fig.savefig(R/('Validation_summary_bias_RMSE_CCC.'+ext),dpi=220,facecolor='white')
d.to_csv(R/'Validation_summary_CCC_statistics.csv',index=False)
(R/'Validation_summary_CCC_caption.txt').write_text('Figure 9. Bias (satellite minus reference), root mean square error, and Lin’s concordance correlation coefficient (CCC) of CLARA-A3 and APP-x albedo against each field dataset, ordered chronologically by campaign. Circles denote CLARA-A3 and triangles APP-x. Counts at right give the retained matched pairs for each product; these are not necessarily independent observations. No APP-x estimates are plotted for ARISE because no pairs were retained. The summary uses the current Figure 9 matchups: IDW with valid nearest-neighbor spatial fallback within the existing distance limits, corrected MOSAiC tower sea-ice-concentration blending, and unchanged prior observation selection and temporal matching. Products may have different dates and sample counts, so differences between scores also reflect sampling differences. Bias and RMSE are in albedo units; CCC is dimensionless and equals 1 for perfect agreement with the 1:1 line. CCC is calculated as twice the paired covariance divided by the sum of the two variances and squared mean difference, using population moments (ddof=0). CCC is undefined for fewer than two pairs or a zero denominator. Limited reference variability can yield low CCC despite small absolute errors.\n')
print(d[['dataset','product','n','bias','rmse']].to_string(index=False))
