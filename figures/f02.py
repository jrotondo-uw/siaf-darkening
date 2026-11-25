from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R
(P/'figures').mkdir(exist_ok=True)
u=pd.read_csv(R.parent/'analysis_data/observed_uncertainty_20261125/observed_minus_MMM_uncertainty.csv')
d=pd.read_csv(R/'siaf_pixelwise_summary.csv').query('pval==.05')
print(d[['period','rs_method','SIAF_tot_global','sigma2_tot_global']].to_string(index=False))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(a,b)=plt.subplots(1,2,figsize=(8,4),gridspec_kw={'width_ratios':[2.1,1]});labels=[]
for i,(method,period) in enumerate([('CERES','CMIP6'),('CERES','Full'),('Kernel','CMIP6'),('Kernel','Full')]):
 row=d[(d.rs_method==method)&(d.period==period)]
 if row.empty:row=d[(d.rs_method==method)&(d.period!='CMIP6')]
 r=row.iloc[0];labels.append(('CERES APRP' if method=='CERES' else 'ERA5 kernel')+'\n'+('1982–2014' if period=='CMIP6' else '1982–2024'))
 for off,col,c,mark in [(-.17,'tot','black','o'),(0,'ext','#3A6BAF','s'),(.17,'drk','#D95F1A','^')]:
  a.errorbar(r[f'SIAF_{col}_global'],i+off,xerr=(float(u[(u.method==method)&(u.component=={'tot':'total','ext':'area_loss','drk':'darkening'}[col])].block3_2sigma.iloc[0]) if period=='CMIP6' else r[f'sigma2_{col}_global']),fmt=mark,color=c,ms=5,capsize=3,label={'tot':'Total','ext':'Area loss','drk':'Darkening'}[col] if i==0 else None)
 b.barh(i,r.ext_pct,color='#3A6BAF',height=.65);b.barh(i,r.drk_pct,left=r.ext_pct,color='#D95F1A',height=.65)
 b.text(r.ext_pct/2,i,f'{r.ext_pct:.0f}%',va='center',ha='center',color='white',weight='bold');b.text(r.ext_pct+r.drk_pct/2,i,f'{r.drk_pct:.0f}%',va='center',ha='center',color='white',weight='bold')
a.set_yticks(range(4),labels);b.set_yticks(range(4),['']*4)
for ax in [a,b]:ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
a.set_xlabel('Annual SIAF (W m$^{-2}$ K$^{-1}$)');a.set_title('a  Observed feedback',loc='left');a.set_xlim(0,.29)
b.set_title('b  Component shares',loc='left');b.set_xlabel('Share of total (%)');b.set_xlim(0,100)
fig.legend(*a.get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.6,0));fig.tight_layout(rect=(0,.07,1,1));fig.savefig(P/'figures/f02.png',dpi=300);fig.savefig(P/'figures/f02.pdf');plt.close(fig)
