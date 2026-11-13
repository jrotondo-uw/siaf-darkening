#!/usr/bin/env python
"""
compute_dfdt_covariance_instant.py  (v2: month-varying ice-ocean contrast)
──────────────────────────────────────────────────────────────────────────
Instantaneous residual-darkening diagnostic.

  residual_i = (dα/dT)_i − C(m) (df/dT)_i        [ = f_i dα_ice/dT ]

C(m) = ᾱ_ice(m) − ᾱ_ocn(m) is the MONTH-VARYING ice-ocean albedo contrast,
computed from the period-mean albedo over ice-covered (mean SIC≥0.8) vs
open-water (mean SIC<0.1) pixels — NOT a fixed 0.54. Late-melt-season ice is
much darker (ponds/melt), so a fixed contrast over-predicts the SIC-driven
term and spuriously reverses the residual; the monthly contrast fixes that.

Outputs: CACHE_OBS/sic_dfdt_sial_{Month}.npy, OUT/fe_covariance_instant.{png,pdf}
"""
import os, glob, re, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from netCDF4 import Dataset as NC4
from pyproj import Transformer
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROG_FILE='/glade/u/home/jrotondo/work/ncc_final/python/dfdt_progress.txt'
def prog(m):
    with open(PROG_FILE,'a') as fh: fh.write(m+'\n')
    print(m,flush=True)

DIR_SIAL='/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX='/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC ='/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6='/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS ='/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_DIR ='/glade/work/jrotondo/ncc_paper/plots/extended_data_figs/'
GISTEMP_PATH='/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH='/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH='/glade/work/jrotondo/ncep_air_sig995.nc'
PSN25_PROJ4=('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
             '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
os.makedirs(OUT_DIR,exist_ok=True)
MONTH_NUMS=[4,5,6,7,8,9]
MONTHS={4:'April',5:'May',6:'June',7:'July',8:'August',9:'September'}
TREND_OBS=list(range(1982,2025))
PVAL_THRESH=0.05
ICE_SIC=0.80; OCN_SIC=0.10
MONTH_COLORS={4:'#2166AC',5:'#4393C3',6:'#92C5DE',7:'#F4A582',8:'#D6604D',9:'#B2182B'}
COL_SIC='#3A7DC9'; COL_RES='#D95F1A'

def build_T_annual_ensemble(trend_years):
    raw={}
    try:
        df=pd.read_csv(GISTEMP_PATH,skiprows=1,na_values='***')
        df['J-D']=pd.to_numeric(df['J-D'],errors='coerce')
        raw['GISTEMP']={int(r['Year']):float(r['J-D']) for _,r in df.iterrows() if np.isfinite(r['J-D'])}
    except Exception as e: prog(f'  GISTEMP fail {e}')
    try:
        df=pd.read_csv(COWTANWAY_PATH,sep=r'\s+',header=None,names=['year','anom','u1','u2','u3'])
        raw['CowtanWay']={int(r['year']):float(r['anom']) for _,r in df.iterrows() if np.isfinite(r['anom'])}
    except Exception as e: prog(f'  Cowtan fail {e}')
    try:
        ds=xr.open_dataset(NCEP_PATH)
        w=np.cos(np.deg2rad(ds['air'].lat))
        ta=ds['air'].weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
        raw['NCEP']={int(y):float(v) for y,v in zip(ta.time.dt.year.values,ta.values.squeeze()) if np.isfinite(v)}
    except Exception as e: prog(f'  NCEP fail {e}')
    centered={}
    for name,yd in raw.items():
        common=[y for y in trend_years if y in yd]
        if len(common)<10: continue
        mv=float(np.mean([yd[y] for y in common])); centered[name]={y:v-mv for y,v in yd.items()}
    ens={}
    for y in trend_years:
        vals=[v[y] for v in centered.values() if y in v]
        if vals: ens[y]=float(np.mean(vals))
    prog(f'  T sources={list(centered.keys())} years={len(ens)}')
    return ens

def latlon_to_xyz(lat,lon):
    la=np.deg2rad(np.asarray(lat).ravel()); lo=np.deg2rad(np.asarray(lon).ravel())
    return np.column_stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)])
def build_nn_index(sl,so,tl,to):
    _,idx=cKDTree(latlon_to_xyz(sl,so)).query(latlon_to_xyz(tl,to),workers=-1); return idx.astype(np.int32)
def compute_pixel_ols_slopes(stk,x):
    T,H,W=stk.shape; flat=stk.reshape(T,-1).astype(np.float64); N=flat.shape[1]
    x=x.astype(np.float64).reshape(-1,1); valid=np.isfinite(flat); nv=valid.sum(0).astype(np.float64)
    fz=np.where(valid,flat,0.); xb=np.broadcast_to(x,(T,N)); xz=np.where(valid,xb,0.)
    ym=fz.sum(0)/np.maximum(nv,1); xm=xz.sum(0)/np.maximum(nv,1)
    fc=np.where(valid,flat-ym,0.); xc=np.where(valid,xb-xm,0.)
    num=(xc*fc).sum(0); den=(xc**2).sum(0)
    out=np.full(N,np.nan,np.float32); good=(nv>=10)&(den>0); out[good]=(num[good]/den[good]).astype(np.float32)
    return out.reshape(H,W)

prog('SETUP')
sample_sial=sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds: sla=ds['lat'].values; slo=ds['lon'].values
if sla.ndim==1: SIAL_LON,SIAL_LAT=np.meshgrid(slo,sla)
else: SIAL_LAT,SIAL_LON=sla,slo
SIAL_SHAPE=SIAL_LAT.shape
try: ARCTIC_SIAL=np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except: ARCTIC_SIAL=(SIAL_LAT>=60.)
IDX_APPX_TO_SIAL=None
try:
    sap=sorted(glob.glob(f'{DIR_APPX}/**/*.nc',recursive=True))[0]
    with NC4(sap) as ds:
        al=np.ma.filled(ds.variables['latitude'][:],np.nan).astype(float)
        ao=np.ma.filled(ds.variables['longitude'][:],np.nan).astype(float)
    if al.ndim==1: ao2,al2=np.meshgrid(ao,al)
    else: al2,ao2=al,ao
    IDX_APPX_TO_SIAL=build_nn_index(al2.ravel(),ao2.ravel(),SIAL_LAT.ravel(),SIAL_LON.ravel())
except Exception as e: prog(f'  appx idx fail {e}')
ssic=sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(ssic) as ds: sx=ds['x'].values; sy=ds['y'].values
sxx,syy=np.meshgrid(sx,sy)
tf=Transformer.from_crs(PSN25_PROJ4,'EPSG:4326',always_xy=True)
sic_lon2d,sic_lat2d=tf.transform(sxx,syy)
sic_tree=cKDTree(latlon_to_xyz(sic_lat2d.ravel(),sic_lon2d.ravel()))
_,IDX_SIC_TO_SIAL=sic_tree.query(latlon_to_xyz(SIAL_LAT.ravel(),SIAL_LON.ravel()),workers=-1)
def regrid_sic_to_sial(a): return a.ravel()[IDX_SIC_TO_SIAL].reshape(SIAL_SHAPE)

def load_alb_year(mname,year):
    if mname=='September':
        p=f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
        if not os.path.exists(p): return None
        raw=np.load(p).astype(np.float32)
        return (raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
                if IDX_APPX_TO_SIAL is not None and raw.shape!=SIAL_SHAPE else raw)
    pc=f'{CACHE_OBS}/clara_alb_{mname}_{year}.npy'; pa=f'{CACHE_OBS}/appx_alb_{mname}_{year}.npy'
    clara=np.load(pc).astype(np.float32) if os.path.exists(pc) else None
    appx=None
    if os.path.exists(pa):
        raw=np.load(pa).astype(np.float32)
        appx=(raw.ravel()[IDX_APPX_TO_SIAL].reshape(SIAL_SHAPE)
              if IDX_APPX_TO_SIAL is not None and raw.shape!=SIAL_SHAPE else raw)
    if clara is not None and appx is not None: return np.where(np.isfinite(clara),clara,appx)
    return clara if clara is not None else appx

_date_re=re.compile(r'sic_psn25_(\d{4})(\d{2})(\d{2})')
_sic_by_ym=defaultdict(list)
for f in glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'):
    m=_date_re.search(os.path.basename(f))
    if not m: continue
    y,mo=int(m.group(1)),int(m.group(2))
    if mo in MONTH_NUMS and y in TREND_OBS: _sic_by_ym[(y,mo)].append(f)
def _read_sic(f):
    try:
        with NC4(f) as ds: raw=np.ma.filled(ds.variables['cdr_seaice_conc'][:],np.nan).astype(np.float32)
    except Exception: return None
    raw=np.squeeze(raw)
    if raw.ndim>2: raw=raw[0]
    if np.isfinite(raw).any() and np.nanmax(raw)>1.5: raw=raw/100.
    return np.where((raw>=0.)&(raw<=1.),raw,np.nan)
def load_sic_year(mn,year):
    fs=_sic_by_ym.get((year,mn),[])
    if not fs: return None
    acc=cnt=None
    for f in fs:
        r=_read_sic(f)
        if r is None: continue
        ok=np.isfinite(r)
        if acc is None: acc=np.where(ok,r,0.).astype(np.float64); cnt=ok.astype(np.float32)
        else: acc+=np.where(ok,r,0.); cnt+=ok
    if acc is None: return None
    mean=np.full(acc.shape,np.nan,np.float32); v=cnt>0; mean[v]=(acc[v]/cnt[v]).astype(np.float32)
    return regrid_sic_to_sial(mean).astype(np.float32)

open(PROG_FILE,'w').close()
prog('Building T ensemble...'); T_dict=build_T_annual_ensemble(TREND_OBS)
prog('Per-month instantaneous regressions (month-varying contrast):')
R={}; CACHE_CONTRAST=[]
for mn in MONTH_NUMS:
    mname=MONTHS[mn]
    years=[]; albs=[]; sics=[]
    for y in sorted(T_dict):
        a=load_alb_year(mname,y); s=load_sic_year(mn,y)
        if a is None or s is None: continue
        years.append(y); albs.append(a); sics.append(s)
    if len(years)<10: prog(f'  {mname}: {len(years)} yr skip'); continue
    Tvec=np.array([T_dict[y] for y in years]); alb_stk=np.stack(albs,0); sic_stk=np.stack(sics,0)
    dadt=compute_pixel_ols_slopes(alb_stk,Tvec); dfdt=compute_pixel_ols_slopes(sic_stk,Tvec)
    np.save(f'{CACHE_OBS}/sic_dfdt_sial_{mname}.npy',dfdt.astype(np.float32))
    corr=np.nan
    try:
        cached=np.load(f'{CACHE_OBS}/merged_alb_dalphadT_{mname}.npy').astype(np.float32)
        mm=np.isfinite(cached)&np.isfinite(dadt)&ARCTIC_SIAL
        if mm.sum()>100: corr=float(np.corrcoef(cached[mm],dadt[mm])[0,1])
    except Exception: pass
    # extent mask
    slope=np.load(f'{CACHE_OBS}/sic_trend_slope_sial_{mname}.npy').astype(np.float32)
    pval =np.load(f'{CACHE_OBS}/sic_trend_pval_sial_{mname}.npy').astype(np.float32)
    valid=ARCTIC_SIAL&np.isfinite(dadt)&np.isfinite(dfdt)
    ext=valid&np.isfinite(slope)&np.isfinite(pval)&(slope<0)&(pval<PVAL_THRESH)
    # contrast over early / late / whole to expose the alpha_ice decline that a
    # whole-period mean conflates away (the time-covariance of loss & darkening)
    yrs=np.array(years)
    def contrast_over(mask_yr):
        ii=np.where(mask_yr)[0]
        if ii.size<3: return np.nan,np.nan
        ma=np.nanmean(alb_stk[ii],0); ms=np.nanmean(sic_stk[ii],0)
        im=(ms>=ICE_SIC)&ARCTIC_SIAL&np.isfinite(ma); om=(ms<OCN_SIC)&ARCTIC_SIAL&np.isfinite(ma)
        ai=float(np.nanmean(ma[im])) if im.sum()>30 else np.nan
        ao=float(np.nanmean(ma[om])) if om.sum()>30 else 0.06
        return ai-ao, ai
    C_all,ai_all=contrast_over(np.ones(len(years),bool))
    C_e,ai_e=contrast_over((yrs>=1982)&(yrs<=1991))
    C_l,ai_l=contrast_over((yrs>=2015)&(yrs<=2024))
    def rstats(C):
        r=(dadt-C*dfdt)[ext]; r=r[np.isfinite(r)]
        return float(np.median(r)),float(np.mean(r<0)*100),r
    med_a,neg_a,_   =rstats(C_all)
    med_e,neg_e,_   =rstats(C_e)
    med_l,neg_l,rE_l=rstats(C_l)
    # Omega_dark residual (dotted, panel a): no extent loss => residual ~ darkening
    sic_e_map=np.load(f'{CACHE_OBS}/sic_early_sial_{mname}.npy').astype(np.float32)
    sic_l_map=np.load(f'{CACHE_OBS}/sic_late_sial_{mname}.npy').astype(np.float32)
    stable=valid&(~ext)&(np.fmax(sic_e_map,sic_l_map)>=0.15)
    rD=(dadt-C_l*dfdt)[stable]; rD=rD[np.isfinite(rD)]
    # PRIMARY = late-period contrast (recent, accelerating loss => darker ice)
    sic_driven=C_l*dfdt
    sE=sic_driven[ext]; sE=sE[np.isfinite(sE)]; tE=dadt[ext]; tE=tE[np.isfinite(tE)]
    msic=float(np.mean(sE)); mtot=float(np.mean(tE)); mres=float(np.mean(rE_l))
    rpct=100.*mres/mtot if mtot!=0 else np.nan
    R[mn]=dict(rE=rE_l,rD=rD,C=C_l,C_e=C_e,ai_e=ai_e,ai_l=ai_l,fneg=neg_l,med=med_l,
               mres=mres,msic=msic,mtot=mtot,rpct=rpct)
    CACHE_CONTRAST.append(dict(month=mname,a_ice_early=ai_e,a_ice_late=ai_l,
                               C_early=C_e,C_late=C_l))
    prog(f'  {mname:9s} a_ice early={ai_e:.3f} late={ai_l:.3f} | C_late={C_l:.3f} | '
         f'med_ext={med_l:+.4f} %neg={neg_l:4.1f} resid%={rpct:+.1f} '
         f'med_dark={float(np.median(rD)):+.4f} corr={corr:.3f}')

# ── cache early/late contrast for reproducibility ───────────────────────────
pd.DataFrame(CACHE_CONTRAST).to_csv(f'{CACHE_OBS}/sic_contrast_sial.csv',index=False)

# ── 2-panel Nature-style figure ─────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,
    'ytick.labelsize':6,'legend.fontsize':6,'legend.title_fontsize':6,
    'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
    'xtick.major.size':3,'ytick.major.size':3,'lines.linewidth':1.2,
    'figure.dpi':300,'savefig.dpi':300,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig,(axa,axb)=plt.subplots(1,2,figsize=(7.2,2.8))
xg=np.linspace(-0.16,0.10,500)
for mn in MONTH_NUMS:
    d=R.get(mn)
    if d is None: continue
    if d['rE'].size>=50:
        k=gaussian_kde(d['rE']); axa.plot(xg,k(xg),color=MONTH_COLORS[mn],lw=1.5,
            label=f"{MONTHS[mn][:3]}  ({d['med']:+.3f})")
        axa.fill_between(xg,k(xg),color=MONTH_COLORS[mn],alpha=0.08)
    if d['rD'].size>=50:
        kd=gaussian_kde(d['rD']); axa.plot(xg,kd(xg),color=MONTH_COLORS[mn],lw=0.7,ls=':',alpha=0.5)
axa.axvline(0,color='k',lw=0.7,ls='--')
axa.set_ylim(0,8)
axa.set_xlabel(r'Residual darkening $\;d\alpha/dT-C(m)\,df/dT$   [K$^{-1}$]')
axa.set_ylabel('Probability density')
axa.set_title('a',loc='left',fontweight='bold',fontsize=9)
leg=axa.legend(title='Month (median, extent)',fontsize=5.6,ncol=2,handlelength=1.2,
               loc='upper left')
leg.get_title().set_fontsize(5.6)
axa.add_artist(leg)
axa.legend(handles=[Line2D([0],[0],color='0.35',lw=1.4,label='extent-loss'),
                    Line2D([0],[0],color='0.35',lw=0.9,ls=':',label='stable darkening')],
           fontsize=5.6,loc='upper right',handlelength=1.6)
mns=[m for m in MONTH_NUMS if m in R]; xpos=np.arange(len(mns))
for i,mn in enumerate(mns):
    d=R[mn]
    axb.bar(i,d['msic'],color=COL_SIC,width=0.72)
    axb.bar(i,d['mres'],bottom=d['msic'],color=COL_RES,width=0.72)
    yy=min(d['msic'],d['msic']+d['mres'])
    axb.text(i,yy-0.004,f"{100*d['mres']/abs(d['mtot']):+.0f}%",ha='center',va='top',
             color='#222',fontsize=6.5,fontweight='bold')
axb.axhline(0,color='k',lw=0.7)
axb.set_xticks(xpos); axb.set_xticklabels([MONTHS[m][:3] for m in mns])
axb.set_ylabel(r'Mean sensitivity, extent pixels  [K$^{-1}$]')
axb.set_title('b',loc='left',fontweight='bold',fontsize=9)
axb.legend(handles=[Patch(color=COL_SIC,label=r'SIC-driven $C(m)\,df/dT$'),
                    Patch(color=COL_RES,label=r'Residual $d\alpha/dT-C(m)\,df/dT$')],
           fontsize=5.6,loc='lower left')
fig.tight_layout()
for e in ('png','pdf'): fig.savefig(f'{OUT_DIR}/fe_covariance_instant.{e}',dpi=300,bbox_inches='tight')
prog(f'-> {OUT_DIR}/fe_covariance_instant.png')
prog('CAPTION NUMBERS (C_late):')
prog('  month   a_ice_e  a_ice_l   C_late   med_ext   %neg   resid%_of_total')
for mn in MONTH_NUMS:
    if mn in R:
        d=R[mn]; prog(f'  {MONTHS[mn]:9s} {d["ai_e"]:.3f}   {d["ai_l"]:.3f}   {d["C"]:.3f}   '
                      f'{d["med"]:+.4f}  {d["fneg"]:4.1f}   {d["rpct"]:+5.1f}')
prog('DONE')
