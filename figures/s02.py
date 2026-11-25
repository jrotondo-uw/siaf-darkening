from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
d=pd.read_csv(P.parent/'analysis_data/weighting_revision_20261125/fixed_leakage_final.csv')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
x=np.arange(6);labels=d.month.str[:3]
fig,ax=plt.subplots(figsize=(7.2,3.7));ax.bar(x-.17,d.frac_conc,.34,color='#777777',label='Concentration term / darkening-region change');ax.bar(x+.17,d.frac_res,.34,color='#3A6BAF',label='Residual / area-loss-region change');ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Share of regional albedo change (%)');ax.legend(loc='upper right',fontsize=9,frameon=False);fig.tight_layout();fig.savefig(P/'figures/s02.png',dpi=300);fig.savefig(P/'figures/s02.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,3.8));ax.plot(x,d.blue_fix,'o-',color='#3A6BAF',label='Residual in area-loss region');ax.plot(x,d.orange_fix,'^-',color='#D95F1A',label='Concentration term in darkening region');ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Contribution to annual flux change\n(W m$^{-2}$)');ax.legend(loc='upper left',fontsize=10,frameon=False)
ax.text(.98,.97,f'Fixed contrast: 0.54\nAnnual sums (W m$^{{-2}}$)\nResidual: {d.blue_fix.sum():.4f}\nConcentration: {d.orange_fix.sum():.4f}\nNet to darkening: {d.blue_fix.sum()-d.orange_fix.sum():.4f}',transform=ax.transAxes,va='top',ha='right',fontsize=10);ax.set_ylim(-.0012,.0082);fig.tight_layout();fig.savefig(P/'figures/s03.png',dpi=300);fig.savefig(P/'figures/s03.pdf');plt.close(fig)
