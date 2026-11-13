"""Threshold sensitivity: dot-and-interval plots, exact saved 2SD widths."""
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=pd.read_csv('/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv')
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
fig,axs=plt.subplots(4,2,figsize=(7.2,9),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
for j,(period,method,title) in enumerate([('CMIP6','CERES','1982–2014 · CERES APRP'),('CMIP6','Kernel','1982–2014 · ERA5 kernel'),('Obs-era','CERES','1982–2024 · CERES APRP'),('Obs-era','Kernel','1982–2024 · ERA5 kernel')]):
 sub=D[D.period.eq(period)&D.rs_method.eq(method)].sort_values('pval');sub=sub[np.isclose(sub.pval,.01)|np.isclose(sub.pval,.05)|np.isclose(sub.pval,.1)]
 for k,(c,label,color,marker) in enumerate([('tot','Total','#333333','o'),('ext','Area loss','#3A6BAF','s'),('drk','Darkening','#D95F1A','^')]):
  y=np.arange(3)+(k-1)*.18
  axs[j,0].errorbar(sub[f'SIAF_{c}_global'],y,xerr=sub[f'sigma2_{c}_global'],fmt=marker,color=color,ms=4,capsize=2,label=label)
  if c!='tot':axs[j,1].scatter(100*sub[f'SIAF_{c}_global']/sub.SIAF_tot_global,y,c=color,marker=marker,s=20)
 for col in range(2):
  ax=axs[j,col];ax.set_yticks(range(3),['p < 0.01','p < 0.05','p < 0.10'] if col==0 else []);ax.set_ylim(2.5,-.5);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_title(f'{chr(97+2*j+col)}  '+(title if col==0 else 'Component share'),loc='left',fontsize=10)
 axs[j,0].set_xlim(0,.30);axs[j,1].set_xlim(0,100)
axs[0,0].legend(loc='upper right',fontsize=8,frameon=False)
axs[-1,0].set_xlabel('SIAF (W m$^{-2}$ K$^{-1}$)');axs[-1,1].set_xlabel('Percent of total')
fig.savefig('f03.pdf')
