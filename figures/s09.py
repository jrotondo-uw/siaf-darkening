"""Readable six-page validation supplement, using the unchanged 22 paired datasets."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'source_data';OUT=ROOT/'figures'
d=pd.read_csv(DATA/'validation_pairs.csv');m=pd.read_csv(DATA/'validation_metrics_ordered.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
with PdfPages(OUT/'fs09.pdf') as pdf:
 for page in range(6):
  fig,axs=plt.subplots(2,2,figsize=(7.2,7.4),layout='constrained')
  for j,ax in enumerate(axs.flat):
   i=page*4+j
   if i>=len(m):ax.axis('off');continue
   r=m.iloc[i];q=d[(d.dataset==r.dataset)&(d['product']==r['product'])]
   col='#0072B2' if r['product']=='CLARA-A3' else '#D55E00'
   ax.scatter(q.reference,q.satellite,color=col,s=12,alpha=.5,rasterized=True,linewidths=0)
   ax.plot([0,1],[0,1],color='.35',ls='--',lw=1)
   if len(q)>1 and np.corrcoef(q.reference,q.satellite)[0,1]>=.5:
    z=np.polyfit(q.reference,q.satellite,1);x=np.array([q.reference.min(),q.reference.max()]);ax.plot(x,np.polyval(z,x),color=col,lw=1)
   title=r.dataset.replace('helicopter / HELiPOD','helicopter/HELiPOD')
   ax.set_title(f'{chr(97+i)}  {title}\n{r["product"]}',loc='left',fontsize=10)
   st='No retained pairs (n = 0)' if not len(q) else f'n = {int(r.n)}\nbias = {r.bias:+.3f}\nRMSE = {r.rmse:.3f}\nCCC = {r.ccc:.3f}'
   ax.text(.04,.96,st,transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
   ax.set(xlim=(0,1),ylim=(0,1),xlabel='Reference broadband albedo',ylabel='Satellite broadband albedo',aspect='equal');ax.grid(alpha=.15)
  pdf.savefig(fig);fig.savefig(OUT/f's11_page{page+1}.pdf');fig.savefig(OUT/f's11_page{page+1}.png',dpi=170);plt.close(fig)
