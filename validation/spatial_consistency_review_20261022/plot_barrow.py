from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.dates as mdates
R=Path(__file__).resolve().parent/'barrow'; O=R/'figures';O.mkdir(exist_ok=True)
s=pd.read_csv(R/'barrow_daily.csv',parse_dates=['date']);cl=pd.read_csv(R/'clara_matchups.csv',parse_dates=['date']);ap=pd.read_csv(R/'appx_matchups.csv',parse_dates=['date']);sp=pd.read_csv(R/'icescape_spectral_means.csv')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(5,3,figsize=(17,21),gridspec_kw={'height_ratios':[1,1,1,1,1.05]})
fig.subplots_adjust(top=.938,bottom=.095,left=.065,right=.975,wspace=.29,hspace=.43)
fig.suptitle('Pre-2012 candidates — inverse-distance weighting for both products',fontsize=22,y=.984)
fig.text(.5,.96,'Broadband Barrow transects vs CLARA-A3 and APP-x; ICESCAPE spectra shown separately below',ha='center',fontsize=12)
st=[]
for i,site in enumerate(['2008','2009N','2009S','2010']):
 a=axs[i,0];obs=s[s.site==site];c=cl[cl.site==site];p=ap[ap.site==site];label={'2008':'Barrow 2008','2009N':'Barrow 2009 · North','2009S':'Barrow 2009 · South','2010':'Barrow 2010'}[site]
 a.scatter(obs.date,obs.alb,s=25,c='#202d35',label='Transect mean',zorder=4)
 a.scatter(p.date,p.sat,s=25,c='#d88935',marker='^',label='APP-x daily')
 a.errorbar(c.date+pd.Timedelta(days=2),c.sat,xerr=pd.Timedelta(days=2),fmt='s',ms=5,color='#277ca5',lw=2,label='CLARA pentad')
 a.set(title=label,ylabel='Broadband albedo',ylim=(0,1),xlabel='Observation date');a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5));a.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));a.grid(alpha=.15)
 if i==0:a.legend(loc='lower left',fontsize=9,frameon=False)
 for j,(df,product,color) in enumerate([(c,'CLARA-A3','#277ca5'),(p,'APP-x','#d88935')],1):
  a=axs[i,j];a.plot([0,1],[0,1],ls='--',c='.55',lw=1);a.scatter(df.insitu,df.sat,s=42,c=color,edgecolors='white',linewidth=.5)
  a.set(xlim=(0,1),ylim=(0,1),xlabel='Transect broadband albedo',ylabel='Satellite broadband albedo',title=product+' · '+('pentad means' if j==1 else 'survey dates'));a.set_aspect('equal');a.grid(alpha=.15)
  e=df.sat-df.insitu;n=len(df);bias=e.mean();rmse=np.sqrt((e**2).mean());r=df.insitu.corr(df.sat) if n>2 else np.nan
  a.text(.04,.96,f'n = {n}\nBias = {bias:+.3f}\nRMSE = {rmse:.3f}\nr = {r:.2f}',va='top',transform=a.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  st.append(dict(site=site,product=product,n=n,bias=bias,rmse=rmse,r=r))
for j,year in enumerate([2010,2011]):
 a=axs[4,j];g=sp[sp.year==year]
 for k,(station,d) in enumerate(g.groupby('station')):
  # Display a clean spectral window only; do not integrate or infer broadband.
  d=d[d.wavelength_nm.between(400,1300)].copy();d.loc[~d.albedo.between(0,1),'albedo']=np.nan
  a.plot(d.wavelength_nm,d.albedo,lw=1.1,alpha=.8,label=d.date.iloc[0][5:])
 a.set(xlim=(400,1300),ylim=(0,1),xlabel='Wavelength (nm) · selected spectral window',ylabel='Spectral albedo',title=f'ICESCAPE {year} · {g.station.nunique()} available files');a.grid(alpha=.15)
 a.text(.03,.05,'Source spectra — not broadband validation\nSampled surfaces; no area weighting or smoothing',transform=a.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 a.legend(fontsize=7,ncol=3,loc='upper right',frameon=False)
a=axs[4,2];a.axis('off');a.text(0,.99,'Barrow 2000–2002',fontsize=15,weight='bold',va='top');a.text(0,.84,'Published broadband transects confirmed.\n\nRaw dated observations were not located\nin the accessible sources searched.\n\nNot plotted or added to validation.\nAuthor/archive data are still needed.\n\nICESCAPE 2011: 1 of 9 linked files\nreturns 404; 8 available files shown.',va='top',fontsize=11,linespacing=1.4)
fig.text(.065,.022,'Exploratory coastal comparison: 200 m transects represent a small part of a satellite footprint. Sites are not independent satellite pixels.\nIdentical four-neighbor IDW spatial sampling for both products; original survey dates and temporal aggregation retained.\nNo date intersection, temporal gap-filling, sea-ice-concentration blending, or spectral-to-broadband conversion. Bias = satellite − in situ.',fontsize=10,linespacing=1.45)
for ext in ['png','pdf']:fig.savefig(O/('Pre2012_candidates_IDW.'+ext),dpi=150)
pd.DataFrame(st).to_csv(R/'candidate_validation_stats.csv',index=False);print(pd.DataFrame(st).to_string(index=False))
