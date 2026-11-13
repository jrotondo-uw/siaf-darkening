#!/usr/bin/env python
"""
siaf_rolling_analysis.py
════════════════════════
Complete rolling SIAF pipeline in one script:

  Step 1 — Fill cache gap years 1992–2004 (CLARA-A3 + APP-x albedo)
  Step 2 — Obs rolling SIAF 1994–2024   → siaf_rolling_timeseries.csv
  Step 3 — CMIP6 rolling SIAF 1994–2014 → cmip6_rolling_siaf.csv
  Step 4 — Two figures:
              siaf_rolling_timeseries.png  (obs only, 2-panel)
              cmip6_rolling_vs_obs.png     (4-panel comparison)

Method (identical for obs and CMIP6):
  • late  = trailing WINDOW-year window [t-4, t]
  • early = 1982–1991 fixed
  • ΔT    = OLS_slope(T, 1982→t) × (t − 1982)
  • IS    = −(α_late − α_early) / ΔT
  • SIAF  = IS × RS_clim  (RS fixed to climatological mean)
  • Scale by AREA_RATIO × SEASON_FRAC → global annual W m⁻² K⁻¹
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings('ignore')
import os, gc, glob, re, sys, traceback, datetime as dt_mod
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from scipy.stats import t as t_dist
from scipy.interpolate import NearestNDInterpolator
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from global_land_mask import globe as _gm_globe; _HAS_LAND = True
except ImportError:
    _HAS_LAND = False

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
CMIP6_ROOT  = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT  = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
CMIP6_DIR   = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
OBS_CSV     = os.path.join(OUT_DIR,   'siaf_rolling_timeseries.csv')
CMIP6_CSV   = os.path.join(CMIP6_DIR, 'cmip6_rolling_siaf.csv')
OBS_PLOT    = os.path.join(OUT_DIR,   'siaf_rolling_timeseries.png')
COMP_PLOT   = os.path.join(CMIP6_DIR, 'cmip6_rolling_vs_obs.png')
GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')
for d in [OUT_DIR, CMIP6_DIR]: os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EARLY_YEARS    = list(range(1982, 1992))
WINDOW         = 5
FIRST_LATE_OBS = 1994           # obs rolling start
LAST_LATE_OBS  = 2024
TARGET_YEARS_CMIP6 = list(range(1994, 2015))
CERES_CLIM_YRS = list(range(2000, 2025))
RS_YEARS_CMIP6 = list(range(1985, 2015))
ALL_LOAD_CMIP6 = list(range(1982, 2015))
TREND_YEARS_CMIP6 = list(range(1982, 2015))
GAP_YEARS      = list(range(1992, 2005))   # fills gap between caches
CACHE_SEARCH   = [CACHE_OBS, CACHE_CMIP6]  # search order for albedo files
MONTH_NUMS     = [4, 5, 6, 7, 8, 9]
MONTH_NAMES    = {4:'April',5:'May',6:'June',7:'July',8:'August',9:'September'}
SIC_THRESH     = 0.15
PVAL_THRESH    = 0.05
AREA_RATIO     = 0.0347
SEASON_FRAC    = 6./12.
SCALE          = AREA_RATIO * SEASON_FRAC
FILL_THRESH    = 1e10
N_BOOTSTRAP    = 500
BOOT_SEED      = 42
N_WORKERS      = 4
PENTAD_DAYS    = 5
MIN_ARCTIC_PX  = 100
PSN25_PROJ4    = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
                  '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
PISTONE_REGIONS = [
    ('EU', 60, 90,    0,   60),
    ('WA', 60, 90,   60,  120),
    ('WP', 60, 90,  120,  180),
    ('EP', 60, 90, -180, -120),
    ('NA', 60, 90, -120,  -60),
    ('AO', 60, 90,  -60,    0),
]

# Module-level grid globals (populated by setup_obs_grids)
sial_shape = sial_lat2d = sial_lon2d = sial_coslat = arctic_sial = None
appx_shape = appx_lat2d = appx_lon2d = arctic_60N_appx = appx_fill = None
sic_shape  = sic_lat2d_psn = sic_lon2d_psn = arctic_60N_mask_psn = None
psn_shape  = psn_lat2d = psn_lon2d = psn_xx = psn_yy = None
AREA_WEIGHT_PSN = DOMAIN_PSN = None
REGION_MASKS_PSN = []
idx_sial_to_psn = idx_ceres_to_psn = idx_appx_to_sial = None
idx_sic_to_sial = idx_sic_to_appx = None
ceres_lat1d = ceres_lon1d = None
T_SOURCES   = []
RS_PSN_CLIM = {}   # {month: psn_array}
AE_EARLY    = {}   # {month_name: sial_array}
SIC_OBS     = {}   # {month_name: dict(slope,pval,sic_e,sic_l)}

# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt>1.,rsdt,np.nan); Sd = np.where(rsds>1.,rsds,np.nan)
        a  = np.clip(rsus/Sd,0.,1.); p=rsds/S; q=rsut/S; den=q+p-2.*p*a
        R  = np.clip((q-p*a)/np.where(np.abs(den)>1e-6,den,np.nan),0.,.99)
        A  = np.clip(1.-p*(1.-a*R)/np.where((1.-R)>.01,1.-R,np.nan),0.,.99)
        Sc = np.where(rsdscs>1.,rsdscs,np.nan); ac=np.clip(rsuscs/Sc,0.,1.)
        pc = rsdscs/S; qc=rsutcs/S; dc=qc+pc-2.*pc*ac
        Rc = np.clip((qc-pc*ac)/np.where(np.abs(dc)>1e-6,dc,np.nan),0.,.99)
        Rl = np.clip(R-Rc,0.,.99)
        RS = (S*(1.-A)*(1.-Rl)**2
              /np.where((1.-a*Rl)**2>1e-9,(1.-a*Rl)**2,np.nan))
    return RS.astype(np.float32)

def weighted_mean(field, mask, weights):
    w=weights[mask]; v=field[mask]; ok=np.isfinite(v)&np.isfinite(w)&(w>0)
    return float(np.sum(v[ok]*w[ok])/np.sum(w[ok])) if ok.sum()>0 else np.nan

def pixel_ols(years_1d, sic_3d):
    T=years_1d.shape[0]; x=years_1d-years_1d.mean(); ss=float(np.dot(x,x))
    ym=np.nanmean(sic_3d,axis=0)
    sl=np.nansum(x[:,None,None]*(sic_3d-ym[None]),axis=0)/ss
    resid=sic_3d-(ym+sl*x[:,None,None]); sse=np.nansum(resid**2,axis=0); df=T-2
    with np.errstate(invalid='ignore',divide='ignore'):
        se=np.sqrt(sse/df)/np.sqrt(ss); ts=sl/se
        pv=2.*(1.-t_dist.cdf(np.abs(ts),df))
    return sl, pv

# ══════════════════════════════════════════════════════════════════════════════
# PSN25 / OBS GRID HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    tgt_xyz=latlon_to_xyz(tgt_lat2d.ravel(),tgt_lon2d.ravel())
    src_xyz=latlon_to_xyz(src_lat2d.ravel(),src_lon2d.ravel())
    tree=cKDTree(tgt_xyz); idx=np.empty(src_xyz.shape[0],dtype=np.int32)
    for i in range(0,src_xyz.shape[0],chunk):
        e=min(i+chunk,src_xyz.shape[0]); _,idx[i:e]=tree.query(src_xyz[i:e],workers=1)
    del tree; gc.collect(); return idx, tgt_lat2d.shape

def scatter_avg(arr_flat, idx, out_shape):
    n=int(np.prod(out_shape)); accum=np.zeros(n,dtype=np.float32); count=np.zeros(n,dtype=np.float32)
    v=np.isfinite(arr_flat)
    np.add.at(accum,idx[v],arr_flat[v].astype(np.float32))
    np.add.at(count,idx[v],np.float32(1.))
    out=np.full(n,np.nan,dtype=np.float32); m=count>0; out[m]=accum[m]/count[m]
    return out.reshape(out_shape)

def regrid_sial_to_psn(arr):
    src=np.where(arctic_sial,np.where(np.isfinite(arr),arr,np.float32(0.)),np.nan).astype(np.float32)
    return scatter_avg(src.ravel(),idx_sial_to_psn,psn_shape)

def region_mean_psn(arr_psn, rmask):
    v,w=arr_psn[rmask],AREA_WEIGHT_PSN[rmask]; ok=np.isfinite(v)&np.isfinite(w)&(w>0)
    return (float(np.sum(v[ok]*w[ok])/np.sum(w[ok])),float(np.sum(w[ok]))) if ok.sum()>0 else (np.nan,0.)

def arctic_mean_psn(arr):
    num=den=0.
    for rm in REGION_MASKS_PSN:
        v,w=region_mean_psn(arr,rm)
        if np.isfinite(v) and w>0: num+=v*w; den+=w
    return (num/den) if den>0 else np.nan

# ══════════════════════════════════════════════════════════════════════════════
# OBS TEMPERATURE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_gistemp():
    if not os.path.exists(GISTEMP_PATH): return None,None
    df=pd.read_csv(GISTEMP_PATH,skiprows=1,na_values='***')
    df['J-D']=pd.to_numeric(df['J-D'],errors='coerce')
    return df['Year'].values.astype(float),df['J-D'].values.astype(float)

def _load_cowtan():
    if not os.path.exists(COWTANWAY_PATH): return None,None
    df=pd.read_csv(COWTANWAY_PATH,sep=r'\s+',header=None,names=['year','anomaly','u1','u2','u3'])
    return df['year'].values.astype(float),df['anomaly'].values.astype(float)

def _load_ncep():
    if not os.path.exists(NCEP_PATH): return None,None
    ds=xr.open_dataset(NCEP_PATH); t=ds['air']
    ta=t.weighted(np.cos(np.deg2rad(t.lat))).mean(dim=['lat','lon']).resample(time='1YE').mean()
    ds.close(); return ta.time.dt.year.values.astype(float),ta.values.squeeze().astype(float)

def _load_era5():
    files=sorted(glob.glob(ERA5_GLOB))
    if not files: return None,None
    ds=xr.open_mfdataset(files,combine='by_coords'); t=ds['VAR_2T']
    ta=t.weighted(np.cos(np.deg2rad(t.latitude))).mean(dim=['latitude','longitude']).resample(time='1YE').mean()
    ds.close(); return ta.time.dt.year.values.astype(float),ta.values.squeeze().astype(float)

def build_T_sources():
    srcs=[]
    for name,loader in [('GISTEMP',_load_gistemp),('CowtanWay',_load_cowtan),
                         ('NCEP',_load_ncep),('ERA5',_load_era5)]:
        try:
            yy,vv=loader()
            if yy is not None and np.isfinite(vv).sum()>10:
                srcs.append((name,yy,vv)); print(f'  T-source {name}: {int(yy.min())}–{int(yy.max())}',flush=True)
        except Exception as e: print(f'  T-source {name} failed: {e}',flush=True)
    return srcs

def compute_dT_ols_obs(t_end):
    """ΔT from obs T-sources: OLS(1982→t_end) × (t_end−1982)."""
    vals=[]
    for _,yy,vv in T_SOURCES:
        m=(yy>=1982)&(yy<=t_end)&np.isfinite(vv)
        if m.sum()<5: continue
        dT=stats.linregress(yy[m],vv[m]).slope*(t_end-1982.)
        if np.isfinite(dT) and dT>0: vals.append(float(dT))
    return np.array(vals,dtype=np.float32) if vals else np.array([0.7],dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# OBS DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
_CERES_VARS=['solar_mon','toa_sw_all_mon','sfc_sw_down_all_mon','sfc_sw_up_all_mon',
             'toa_sw_clr_c_mon','sfc_sw_down_clr_c_mon','sfc_sw_up_clr_c_mon']

def load_alb_yearly_stack(mname, years):
    use_appx=(mname=='September'); prefix='appx' if use_appx else 'clara'; stack=[]
    for y in years:
        raw=None
        for cache in CACHE_SEARCH:
            p=f'{cache}/{prefix}_alb_{mname}_{y}.npy'
            if os.path.exists(p): raw=np.load(p).astype(np.float32); break
        if raw is None: continue
        if use_appx and idx_appx_to_sial is not None:
            raw=raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
        stack.append(raw)
    return np.stack(stack,axis=0) if stack else None

def load_RS_ceres_clim(years, mn):
    sums={k:None for k in _CERES_VARS}; cnt=0
    with xr.open_dataset(CERES_FILE,engine='netcdf4') as ds:
        times=pd.DatetimeIndex(ds.time.values)
        for i,t in enumerate(times):
            if t.year in years and t.month==mn:
                sl=ds.isel(time=i).load()
                for k in _CERES_VARS:
                    a=sl[k].values.astype(np.float32)
                    sums[k]=a if cnt==0 else sums[k]+a
                cnt+=1; del sl
    if cnt==0: return None
    for k in sums: sums[k]/=np.float32(cnt)
    return t07_invert(*[sums[k] for k in _CERES_VARS]).astype(np.float32)

def load_sic_obs(mname):
    for cache in CACHE_SEARCH:
        try:
            return dict(
                slope=np.load(f'{cache}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
                pval =np.load(f'{cache}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
                sic_e=np.load(f'{cache}/sic_early_sial_{mname}.npy').astype(np.float32),
                sic_l=np.load(f'{cache}/sic_late_sial_{mname}.npy').astype(np.float32))
        except: continue
    return None

# ══════════════════════════════════════════════════════════════════════════════
# OBS ROLLING SIAF
# ══════════════════════════════════════════════════════════════════════════════

def compute_IS_SIAF_obs(ae, al, sic, dT, RS_psn):
    if sic is None or not (np.isfinite(dT) and dT>0): return None,None,None
    dalpha=al.astype(np.float32)-ae.astype(np.float32)
    changing=(arctic_sial & np.isfinite(dalpha) & np.isfinite(sic['slope'])
              & np.isfinite(sic['pval']) & (sic['slope']<0) & (sic['pval']<PVAL_THRESH))
    stable=(arctic_sial & ~changing & np.isfinite(dalpha)
            & (np.fmax(sic['sic_e'],sic['sic_l'])>=SIC_THRESH))
    IS_ext=np.where(changing,-dalpha/dT,np.nan).astype(np.float32)
    IS_drk=np.where(stable,  -dalpha/dT,np.nan).astype(np.float32)
    IS_tot=np.where(np.isfinite(IS_ext)|np.isfinite(IS_drk),
                    np.where(np.isfinite(IS_ext),IS_ext,0.)
                    +np.where(np.isfinite(IS_drk),IS_drk,0.),np.nan).astype(np.float32)
    IS_ext=np.where(np.isfinite(IS_tot),np.where(np.isfinite(IS_ext),IS_ext,0.),np.nan)
    IS_drk=np.where(np.isfinite(IS_tot),np.where(np.isfinite(IS_drk),IS_drk,0.),np.nan)

    IT_p=regrid_sial_to_psn(IS_tot); IX_p=regrid_sial_to_psn(IS_ext); ID_p=regrid_sial_to_psn(IS_drk)
    reg_siaf,reg_wts=[],[]
    for rm in REGION_MASKS_PSN:
        ST,w=region_mean_psn(IT_p*RS_psn,rm)
        if not(np.isfinite(ST) and w>0): continue
        SX,_=region_mean_psn(IX_p*RS_psn,rm); SD,_=region_mean_psn(ID_p*RS_psn,rm)
        reg_siaf.append([ST,SX,SD]); reg_wts.append(w)
    if not reg_wts: return None,None,None
    wts=np.array(reg_wts,dtype=np.float64); arr=np.array(reg_siaf,dtype=np.float64)
    wm=(np.sum(arr*wts[:,None],axis=0)/np.sum(wts)).tolist()
    return wm[0],wm[1],wm[2]

def run_obs_rolling():
    print('\n'+'═'*60+'\nOBS ROLLING SIAF\n'+'═'*60,flush=True)
    rng=np.random.default_rng(BOOT_SEED); rows=[]
    target_years=list(range(FIRST_LATE_OBS,LAST_LATE_OBS+1))
    for t_end in target_years:
        late=list(range(t_end-WINDOW+1,t_end+1))
        dT_samples=compute_dT_ols_obs(t_end); dT_mean=float(np.mean(dT_samples))
        if not(np.isfinite(dT_mean) and dT_mean>0):
            print(f'  {t_end}: skip (dT={dT_mean:.3f})',flush=True); continue

        month_siaf={'tot':[],'ext':[],'drk':[]}; month_boot=[]
        for mn in MONTH_NUMS:
            mname=MONTH_NAMES[mn]; ae=AE_EARLY.get(mname)
            rs_p=RS_PSN_CLIM.get(mn); sic=SIC_OBS.get(mname)
            if ae is None or rs_p is None or sic is None: continue
            al_stk=load_alb_yearly_stack(mname,late)
            if al_stk is None: continue
            al=np.nanmean(al_stk,axis=0).astype(np.float32)
            ST,SX,SD=compute_IS_SIAF_obs(ae,al,sic,dT_mean,rs_p)
            if ST is not None:
                month_siaf['tot'].append(ST); month_siaf['ext'].append(SX); month_siaf['drk'].append(SD)
            ae_stk=load_alb_yearly_stack(mname,EARLY_YEARS)
            if ae_stk is None or al_stk is None: month_boot.append(None); continue
            boot=np.full((N_BOOTSTRAP,3),np.nan)
            ne,nl,nt=ae_stk.shape[0],al_stk.shape[0],len(dT_samples)
            for b in range(N_BOOTSTRAP):
                ae_b=np.nanmean(ae_stk[rng.integers(0,ne,size=ne)],axis=0)
                al_b=np.nanmean(al_stk[rng.integers(0,nl,size=nl)],axis=0)
                dT_b=float(dT_samples[rng.integers(0,nt)])
                if dT_b<=0: continue
                bT,bX,bD=compute_IS_SIAF_obs(ae_b,al_b,sic,dT_b,rs_p)
                if bT is not None: boot[b]=[bT,bX,bD]
            month_boot.append(boot)

        if not month_siaf['tot']: continue
        seas={k:float(np.nanmean(v))*SCALE for k,v in month_siaf.items()}
        valid=[b for b in month_boot if b is not None and not np.all(np.isnan(b))]
        if valid:
            sb=np.nanmean(np.stack(valid,axis=1),axis=1)
            sig2={k:float(2*np.nanstd(sb[:,i]))*SCALE for i,k in enumerate(['tot','ext','drk'])}
        else:
            sig2={'tot':np.nan,'ext':np.nan,'drk':np.nan}
        rows.append(dict(year=t_end,late_start=t_end-WINDOW+1,dT_mean=round(dT_mean,4),
                         SIAF_tot=round(seas['tot'],6),SIAF_ext=round(seas['ext'],6),
                         SIAF_drk=round(seas['drk'],6),
                         sigma2_tot=round(sig2['tot'],6),sigma2_ext=round(sig2['ext'],6),
                         sigma2_drk=round(sig2['drk'],6)))
        print(f'  {t_end}: SIAF={seas["tot"]:+.4f} ± {sig2["tot"]:.4f}  '
              f'ext={seas["ext"]:+.4f}  drk={seas["drk"]:+.4f}',flush=True)
        gc.collect()
    df=pd.DataFrame(rows); df.to_csv(OBS_CSV,index=False)
    print(f'Obs CSV → {OBS_CSV}',flush=True); return df

# ══════════════════════════════════════════════════════════════════════════════
# CMIP6 INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

MODEL_INFO={
    'CESM2':dict(institution='NCAR',member='r1i1p1f1'),
    'E3SM-1-0':dict(institution='E3SM-Project',member='r1i1p1f1'),
    'MPI-ESM1-2-HR':dict(institution='MPI-M',member='r1i1p1f1'),
    'MPI-ESM1-2-LR':dict(institution='MPI-M',member='r1i1p1f1'),
    'IPSL-CM6A-LR':dict(institution='IPSL',member='r1i1p1f1'),
    'MIROC6':dict(institution='MIROC',member='r1i1p1f1'),
    'MIROC-ES2L':dict(institution='MIROC',member='r1i1p1f2'),
    'CanESM5':dict(institution='CCCma',member='r1i1p1f1'),
    'ACCESS-ESM1-5':dict(institution='CSIRO',member='r1i1p1f1'),
    'GFDL-ESM4':dict(institution='NOAA-GFDL',member='r1i1p1f1'),
    'UKESM1-0-LL':dict(institution='MOHC',member='r10i1p1f2'),
    'HadGEM3-GC31-LL':dict(institution='MOHC',member='r1i1p1f3'),
    'CNRM-CM6-1':dict(institution='CNRM-CERFACS',member='r10i1p1f2'),
    'BCC-CSM2-MR':dict(institution='BCC',member='r1i1p1f1'),
    'ACCESS-CM2':dict(institution='CSIRO-ARCCSS',member='r1i1p1f1'),
    'CNRM-ESM2-1':dict(institution='CNRM-CERFACS',member='r1i1p1f2'),
    'EC-Earth3':dict(institution='EC-Earth-Consortium',member='r1i1p1f1'),
    'EC-Earth3-Veg':dict(institution='EC-Earth-Consortium',member='r1i1p1f1'),
    'GFDL-CM4':dict(institution='NOAA-GFDL',member='r1i1p1f2'),
    'GISS-E2-1-H':dict(institution='NASA-GISS',member='r1i1p1f1'),
    'INM-CM4-8':dict(institution='INM',member='r1i1p1f1'),
    'INM-CM5-0':dict(institution='INM',member='r1i1p1f1'),
    'MRI-ESM2-0':dict(institution='MRI',member='r1i1p1f1'),
    'NorESM2-MM':dict(institution='NCC',member='r1i1p1f1'),
}
MODEL_COLORS={
    'CESM2':'#1f77b4','E3SM-1-0':'#ff7f0e','MPI-ESM1-2-HR':'#2ca02c',
    'MPI-ESM1-2-LR':'#98df8a','IPSL-CM6A-LR':'#d62728','MIROC6':'#9467bd',
    'MIROC-ES2L':'#c5b0d5','CanESM5':'#8c564b','ACCESS-ESM1-5':'#e377c2',
    'GFDL-ESM4':'#7f7f7f','UKESM1-0-LL':'#17becf','HadGEM3-GC31-LL':'#aec7e8',
    'CNRM-CM6-1':'#ffbb78','BCC-CSM2-MR':'#ff9896','ACCESS-CM2':'#bcbd22',
    'CNRM-ESM2-1':'#dbdb8d','EC-Earth3':'#e41a1c','EC-Earth3-Veg':'#f781bf',
    'GFDL-CM4':'#a65628','GISS-E2-1-H':'#b8e186','INM-CM4-8':'#542788',
    'INM-CM5-0':'#998ec3','MRI-ESM2-0':'#f1a340','NorESM2-MM':'#5ab4ac',
}
VAR_TABLE_MAP={
    'rsus':['Amon'],'rsds':['Amon'],'rsdt':['Amon'],'rsut':['Amon'],
    'rsutcs':['Amon','CFmon'],'rsdscs':['Amon','CFmon'],'rsuscs':['Amon','CFmon'],
    'siconc':['SImon','SIday'],'tas':['Amon'],
}
RAD_VARS=['rsdt','rsut','rsds','rsus','rsutcs','rsdscs','rsuscs']

def _mask_fills(arr):
    arr=np.asarray(arr,dtype=float)
    return np.where(np.abs(arr)<FILL_THRESH,arr,np.nan)

def _is_valid_lat(arr):
    if arr is None: return False
    flat=arr.ravel(); flat=flat[np.isfinite(flat)]
    return len(flat)>0 and float(flat.min())>=-90.1 and float(flat.max())<=90.1

def _std_latlon(da):
    aliases={'lat':['latitude','nav_lat','rlat','y'],'lon':['longitude','nav_lon','rlon','x']}
    for new,als in aliases.items():
        for c in list(da.coords):
            if c.lower() in als and c!=new:
                try: da=da.rename({c:new})
                except: pass; break
    sp=da.shape[1:] if da.ndim>=3 else da.shape
    ny=sp[0] if len(sp)>=1 else None; nx=sp[1] if len(sp)>=2 else None
    def _best(primary,aliases_):
        b1=b2=None
        for name in [primary]+aliases_:
            if name in da.coords:
                arr=_mask_fills(np.asarray(da.coords[name].values,dtype=float))
                if arr.ndim==2 and ny and nx and arr.shape==(ny,nx):
                    if b2 is None: b2=arr
                elif arr.ndim==1:
                    if b1 is None: b1=arr
        return b2 if b2 is not None else b1
    lat=_best('lat',['latitude','nav_lat','rlat','y'])
    lon=_best('lon',['longitude','nav_lon','rlon','x'])
    if not _is_valid_lat(lat):
        for name in [n for n in da.coords if 'lat' in n.lower()]:
            try:
                arr=_mask_fills(np.asarray(da[name].values,dtype=float))
                if ny and nx and arr.shape==(ny,nx) and _is_valid_lat(arr):
                    lat=arr
                    for ln in [n for n in da.coords if 'lon' in n.lower()]:
                        la=_mask_fills(np.asarray(da[ln].values,dtype=float))
                        if la.shape==(ny,nx): lon=la; break
                    break
            except: continue
    return da, lat, lon, (lat.ndim==2 if lat is not None else False)

def build_native_ctx(lat,lon,is_2d):
    if is_2d: lat2d=np.asarray(lat,dtype=np.float32); lon2d=np.asarray(lon,dtype=np.float32)
    else: lon2d,lat2d=np.meshgrid(np.asarray(lon,dtype=np.float32),np.asarray(lat,dtype=np.float32))
    lat2d=np.where(np.isfinite(lat2d),lat2d,0.).astype(np.float32)
    lon2d=np.where(np.isfinite(lon2d),lon2d,0.).astype(np.float32)
    area=np.cos(np.deg2rad(lat2d)).astype(np.float32)
    lon_pm=lon2d-360.*np.floor((lon2d+180.)/360.)
    ocean=_gm_globe.is_ocean(lat2d,lon_pm) if _HAS_LAND else np.ones(lat2d.shape,dtype=bool)
    domain=ocean&(lat2d>=60.)
    rmasks={}
    for code,lo,hi,lmn,lmx in PISTONE_REGIONS:
        lat_ok=(lat2d>=lo)&(lat2d<=hi)
        lon_ok=((lon_pm>=lmn)&(lon_pm<=lmx) if lmn<=lmx else (lon_pm>=lmn)|(lon_pm<=lmx))
        rmasks[code]=lat_ok&lon_ok&domain
    return dict(lat2d=lat2d,shape=lat2d.shape,area=area,domain=domain,rmasks=rmasks)

def arctic_mean_native(arr,ctx):
    num=den=0.
    for code in [r[0] for r in PISTONE_REGIONS]:
        rm=ctx['rmasks'][code]; v,w=arr[rm],ctx['area'][rm]
        ok=np.isfinite(v)&np.isfinite(w)&(w>0)
        if ok.sum()>0: num+=float(np.sum(v[ok]*w[ok])/np.sum(w[ok]))*float(np.sum(w[ok])); den+=float(np.sum(w[ok]))
    return (num/den) if den>0 else np.nan

def _files_cover_years(files,years):
    yr_set=set(years)
    for f in files:
        m=re.search(r'_(\d{4,6})-(\d{4,6})\.nc$',os.path.basename(f))
        if m:
            if any(int(m.group(1)[:4])<=y<=int(m.group(2)[:4]) for y in yr_set): return True
        else: return True
    return False

def _prefer_gr(files):
    if not files: return files
    bns=[os.path.basename(f) for f in files]
    hgr=any('_gr_' in b or '_gr1_' in b or '_gr2_' in b for b in bns)
    hgn=any('_gn_' in b for b in bns)
    if hgr and hgn: return [f for f,b in zip(files,bns) if '_gr_' in b or '_gr1_' in b or '_gr2_' in b]
    return files

def find_var_files(inst,model,member,variable):
    tables=VAR_TABLE_MAP.get(variable,['Amon'])
    for root in [CMIP6_ROOT,STAGE_ROOT]:
        for table in tables:
            d=os.path.join(root,inst,model,'historical',member,table,variable)
            if not os.path.isdir(d): continue
            files=sorted(glob.glob(os.path.join(d,'**',f'{variable}_*.nc'),recursive=True))
            if not files: continue
            files=_prefer_gr(files)
            rel=[f for f in files if _files_cover_years([f],ALL_LOAD_CMIP6)]
            if rel: return rel
    return []

def find_all_members(inst,model):
    members=set()
    for root in [CMIP6_ROOT,STAGE_ROOT]:
        base=os.path.join(root,inst,model,'historical')
        if not os.path.isdir(base): continue
        for mem in sorted(os.listdir(base)):
            if (find_var_files(inst,model,mem,'rsus') and
                    find_var_files(inst,model,mem,'rsds') and
                    find_var_files(inst,model,mem,'siconc')): members.add(mem)
    return sorted(members)

def _open_robust(files,variable):
    for kw in [dict(combine='by_coords',use_cftime=True,chunks={'time':12}),
               dict(combine='nested',concat_dim='time',coords='minimal',compat='override',use_cftime=True,chunks={'time':12})]:
        try: return xr.open_mfdataset(files,**kw)
        except: pass
    raise RuntimeError(f'Cannot open {variable}')

# ══════════════════════════════════════════════════════════════════════════════
# CMIP6 ROLLING SIAF  (worker function — must be at module level for multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def compute_dT_ols_cmip6(model,member,t_end):
    inst=MODEL_INFO[model]['institution']
    tf=find_var_files(inst,model,member,'tas')
    if not tf: return np.nan
    try:
        ds=_open_robust(tf,'tas'); da=ds['tas']
        da=da.isel(time=[i for i,t in enumerate(da.time) if 1982<=int(t.dt.year)<=t_end]).load()
        ds.close(); _,lat,lon,is_2d=_std_latlon(da)
        if lat is None: return np.nan
        if not is_2d and lat.ndim==1: _,lat2d=np.meshgrid(lon,lat)
        else: lat2d=lat
        w=np.cos(np.deg2rad(lat2d)); yrs=np.array([int(t.dt.year) for t in da.time])
        vals=da.values.astype(float)
        tmeans=[]
        for i in range(len(yrs)):
            ok=np.isfinite(vals[i])&(w>0)
            if ok.any(): tmeans.append(float(np.sum(vals[i][ok]*w[ok])/np.sum(w[ok])))
        if len(tmeans)<5: return np.nan
        dT=stats.linregress(yrs[:len(tmeans)],tmeans).slope*(t_end-1982.)
        return float(dT) if dT>0 else np.nan
    except: return np.nan

def process_model_rolling(model):
    """Worker: compute rolling SIAF for all members of a CMIP6 model."""
    inst=MODEL_INFO[model]['institution']; prim=MODEL_INFO[model]['member']
    members=find_all_members(inst,model)
    if not members: return []
    if prim in members: members=[prim]+[m for m in members if m!=prim]
    print(f'[{model}] {len(members)} member(s)',flush=True)
    all_rows=[]
    for member in members:
        try:
            rows=_compute_member_rolling(model,member)
            all_rows.extend(rows)
        except Exception as e:
            print(f'  [{model}/{member}] FAILED: {e}',flush=True)
            print(''.join(traceback.format_exc().splitlines(keepends=True)[-4:]),flush=True)
    return all_rows

def _compute_member_rolling(model,member):
    inst=MODEL_INFO[model]['institution']
    rsus_f=find_var_files(inst,model,member,'rsus')
    rsds_f=find_var_files(inst,model,member,'rsds')
    sic_f =find_var_files(inst,model,member,'siconc')
    if not all([rsus_f,rsds_f,sic_f]): return []

    def _load(files,var):
        ds=_open_robust(files,var); da=ds[var]
        msk=np.array([1982<=int(t.dt.year)<=2014 and int(t.dt.month) in MONTH_NUMS
                       for t in da.time],dtype=bool)
        da=da.isel(time=msk).load(); ds.close(); da,_,_,_=_std_latlon(da); return da

    try:
        ru=_load(rsus_f,'rsus'); rd=_load(rsds_f,'rsds')
        si_ds=_open_robust(sic_f,'siconc'); si_da=si_ds['siconc']
        si_msk=np.array([1982<=int(t.dt.year)<=2014 and int(t.dt.month) in MONTH_NUMS
                          for t in si_da.time],dtype=bool)
        si_da=si_da.isel(time=si_msk).load(); si_ds.close(); si_da,_,_,_=_std_latlon(si_da)
    except Exception as e:
        print(f'    [{model}/{member}] load: {e}',flush=True); return []

    _,alb_lat,alb_lon,alb_is2d=_std_latlon(ru); _,sic_lat,sic_lon,sic_is2d=_std_latlon(si_da)
    if alb_lat is None or sic_lat is None: return []
    ctx=build_native_ctx(alb_lat,alb_lon,alb_is2d); alb_shp=ctx['shape']; lat2d=ctx['lat2d']

    sv=si_da.values.astype(float)
    if np.nanmax(sv)>1.5: sv/=100.
    if sv.shape[1:]!=alb_shp:
        if sic_is2d: sla=_mask_fills(sic_lat).ravel(); slo=_mask_fills(sic_lon).ravel()
        else:
            slo2,sla2=np.meshgrid(sic_lon,sic_lat)
            sla=_mask_fills(sla2).ravel(); slo=_mask_fills(slo2).ravel()
        if alb_is2d: ala=_mask_fills(alb_lat).ravel(); alo=_mask_fills(alb_lon).ravel()
        else:
            alo2,ala2=np.meshgrid(alb_lon,alb_lat); ala=ala2.ravel(); alo=alo2.ravel()
        valid=np.isfinite(sla)&np.isfinite(slo)
        _,idx=cKDTree(latlon_to_xyz(sla[valid],slo[valid])).query(
            latlon_to_xyz(ala,alo).astype(np.float32),workers=-1)
        sv=sv.reshape(sv.shape[0],-1)[:,valid][:,idx].reshape((sv.shape[0],)+alb_shp)

    ru_yrs=np.array([int(t.dt.year) for t in ru.time]); ru_mon=np.array([int(t.dt.month) for t in ru.time])
    si_yrs=np.array([int(t.dt.year) for t in si_da.time]); si_mon=np.array([int(t.dt.month) for t in si_da.time])
    del si_da; gc.collect()

    # Fixed RS climatology
    rf={v:find_var_files(inst,model,member,v) for v in RAD_VARS}; avail={v for v,f in rf.items() if f}
    rs_clim={}
    for mn in MONTH_NUMS:
        rs_arr=None
        if set(RAD_VARS)<=avail:
            data={}; ok=True
            for v in RAD_VARS:
                try:
                    ds=_open_robust(rf[v],v); da2=ds[v]
                    msk=np.array([int(t.dt.year) in set(RS_YEARS_CMIP6) and int(t.dt.month)==mn
                                   for t in da2.time],dtype=bool)
                    if not msk.any(): ok=False; ds.close(); break
                    data[v]=da2.isel(time=msk).mean('time').load().values.astype(float); ds.close()
                except: ok=False; break
            if ok: rs_arr=t07_invert(*[data[v] for v in RAD_VARS])
        elif 'rsds' in avail:
            try:
                ds=_open_robust(rf['rsds'],'rsds'); da2=ds['rsds']
                msk=np.array([int(t.dt.year) in set(RS_YEARS_CMIP6) and int(t.dt.month)==mn
                               for t in da2.time],dtype=bool)
                if msk.any(): rs_arr=da2.isel(time=msk).mean('time').load().values.astype(np.float32)
                ds.close()
            except: pass
        rs_clim[mn]=rs_arr.astype(np.float32) if rs_arr is not None else None
    gc.collect()

    # Fixed SIC trend
    sic_trends={}
    for mn in MONTH_NUMS:
        tm=(si_mon==mn)&np.isin(si_yrs,TREND_YEARS_CMIP6); em=(si_mon==mn)&np.isin(si_yrs,EARLY_YEARS)
        if tm.sum()<10 or em.sum()==0: continue
        sl,pv=pixel_ols(si_yrs[tm].astype(float),sv[tm]); sic_e=np.nanmean(sv[em],axis=0)
        lm=(si_mon==mn)&np.isin(si_yrs,list(range(2005,2015)))
        sic_l=np.nanmean(sv[lm],axis=0) if lm.sum()>0 else sic_e
        sic_trends[mn]=dict(slope=sl,pval=pv,sic_e=sic_e,sic_l=sic_l)
    del sv; gc.collect()

    # Per-year albedo
    alb_by_year={}
    for yr in ALL_LOAD_CMIP6:
        alb_by_year[yr]={}
        for mn in MONTH_NUMS:
            msk=(ru_yrs==yr)&(ru_mon==mn)
            if not msk.any(): continue
            u=ru.isel(time=msk).mean('time').values; d=rd.isel(time=msk).mean('time').values
            with np.errstate(invalid='ignore',divide='ignore'):
                alb=np.where(d>1.,u/d,np.nan)
            alb_by_year[yr][mn]=np.clip(alb,0.,1.).astype(np.float32)
    del ru,rd; gc.collect()

    ae_by_month={mn:np.nanmean([alb_by_year[y][mn] for y in EARLY_YEARS
                                  if mn in alb_by_year.get(y,{})],axis=0).astype(np.float32)
                 for mn in MONTH_NUMS if any(mn in alb_by_year.get(y,{}) for y in EARLY_YEARS)}

    # Pre-compute ΔT
    dT_by_year={t:compute_dT_ols_cmip6(model,member,t) for t in TARGET_YEARS_CMIP6}

    rows=[]
    for t_end in TARGET_YEARS_CMIP6:
        late=list(range(t_end-WINDOW+1,t_end+1)); dT=dT_by_year[t_end]
        if not(np.isfinite(dT) and dT>0): continue
        month_siaf={'tot':[],'ext':[],'drk':[]}
        for mn in MONTH_NUMS:
            ae=ae_by_month.get(mn); rs_p=rs_clim.get(mn); sic=sic_trends.get(mn)
            if ae is None or rs_p is None or sic is None: continue
            arrs=[alb_by_year[y][mn] for y in late if mn in alb_by_year.get(y,{})]
            if not arrs: continue
            al=np.nanmean(arrs,axis=0).astype(np.float32)
            dalpha=al-ae; sl=sic['slope']; pv=sic['pval']
            ice=(lat2d>=60.)&(sic['sic_e']>=SIC_THRESH)
            changing=ice&np.isfinite(sl)&np.isfinite(pv)&(sl<0)&(pv<PVAL_THRESH)
            stable=ice&~changing&(np.fmax(sic['sic_e'],sic['sic_l'])>=SIC_THRESH)
            IS_ext=np.where(changing,-dalpha/dT,np.nan).astype(np.float32)
            IS_drk=np.where(stable,  -dalpha/dT,np.nan).astype(np.float32)
            IS_tot=np.where(np.isfinite(IS_ext)|np.isfinite(IS_drk),
                            np.where(np.isfinite(IS_ext),IS_ext,0.)
                            +np.where(np.isfinite(IS_drk),IS_drk,0.),np.nan).astype(np.float32)
            IS_ext=np.where(np.isfinite(IS_tot),np.where(np.isfinite(IS_ext),IS_ext,0.),np.nan)
            IS_drk=np.where(np.isfinite(IS_tot),np.where(np.isfinite(IS_drk),IS_drk,0.),np.nan)
            if rs_p.shape!=IS_tot.shape: continue
            for comp,IS in [('tot',IS_tot),('ext',IS_ext),('drk',IS_drk)]:
                v=arctic_mean_native((IS*rs_p).astype(np.float32),ctx)
                if np.isfinite(v): month_siaf[comp].append(v)
        if not month_siaf['tot']: continue
        rows.append(dict(model=model,member=member,year=t_end,
                         SIAF_tot=float(np.nanmean(month_siaf['tot']))*SCALE,
                         SIAF_ext=float(np.nanmean(month_siaf['ext']))*SCALE,
                         SIAF_drk=float(np.nanmean(month_siaf['drk']))*SCALE,dT=dT))
    print(f'  [{model}/{member}] {len(rows)}/{len(TARGET_YEARS_CMIP6)} windows',flush=True)
    return rows

def run_cmip6_rolling():
    print('\n'+'═'*60+'\nCMIP6 ROLLING SIAF\n'+'═'*60,flush=True)
    if os.path.exists(CMIP6_CSV):
        df=pd.read_csv(CMIP6_CSV)
        if df['year'].min()<=min(TARGET_YEARS_CMIP6):
            print(f'Loading cached CMIP6 rolling SIAF from {CMIP6_CSV}',flush=True); return df
        print(f'Cache starts at {df["year"].min()} but need {min(TARGET_YEARS_CMIP6)} — recomputing.',flush=True)
        os.remove(CMIP6_CSV)
    all_rows=[]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures={pool.submit(process_model_rolling,m):m for m in MODEL_INFO}
        for fut in as_completed(futures):
            m=futures[fut]
            try:
                rows=fut.result(); all_rows.extend(rows)
                print(f'  ✓ {m}: {len(rows)} rows',flush=True)
            except Exception as e: print(f'  ✗ {m}: {e}',flush=True)
    df=pd.DataFrame(all_rows); df.to_csv(CMIP6_CSV,index=False)
    print(f'CMIP6 CSV → {CMIP6_CSV}',flush=True); return df

# ══════════════════════════════════════════════════════════════════════════════
# CACHE GAP FILL  (1992–2004)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_date_appx(fname):
    try: return dt_mod.datetime.strptime(os.path.basename(fname).split('_d')[1][:8],"%Y%m%d")
    except: return None

def _load_sic_day(filepath):
    with xr.open_dataset(filepath) as ds:
        t=pd.Timestamp(ds['time'].values[0])
        raw=ds['cdr_seaice_conc'].squeeze().values.astype(float)
        if raw.ndim>2: raw=raw[0]
        if np.nanmax(raw)>1.5: raw/=100.
        raw=np.where((raw>=0.)&(raw<=1.),raw,np.nan)
    return t,raw

def fill_gap_years():
    print('\n'+'═'*60+f'\nCACHE GAP FILL: {GAP_YEARS[0]}–{GAP_YEARS[-1]}\n'+'═'*60,flush=True)

    # ── CLARA-A3 ──────────────────────────────────────────────────────────────
    print('Step 1: CLARA-A3...',flush=True)
    for year in GAP_YEARS:
        if all(os.path.exists(f'{CACHE_OBS}/clara_alb_{mn}_{year}.npy') for mn in MONTH_NAMES.values()):
            print(f'  {year}: all cached, skip'); continue
        sial_files=[f for f in sorted(glob.glob(f'{DIR_SIAL}/*.nc')) if str(year) in f]
        if not sial_files: print(f'  {year}: no CLARA files, skip'); continue
        print(f'  {year}...',end='',flush=True)
        try:
            ds=xr.open_mfdataset(sial_files,combine='by_coords')
            sa=ds['black_sky_albedo_snow_mean']/100.; ra=ds['black_sky_albedo_mean']/100.
            sial_pent=xr.where(~np.isnan(sa),sa,ra).resample(time=f'{PENTAD_DAYS}D').mean().load()
            ds.close()
        except Exception as e: print(f' skip ({e})'); continue
        sic_daily={}
        for f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}*.nc')):
            try:
                t,raw=_load_sic_day(f)
                sic_daily[t.date()]=raw.ravel()[idx_sic_to_sial].reshape(sial_shape)
            except: continue
        sic_pent={}
        for ptime in sial_pent.time.values:
            pd_date=pd.Timestamp(ptime).date()
            maps=[sic_daily[d] for d in [pd_date+dt_mod.timedelta(days=i) for i in range(PENTAD_DAYS)] if d in sic_daily]
            if maps: sic_pent[pd_date]=np.nanmean(maps,axis=0)
        saved=0
        for mname,mnum in MONTH_NAMES.items():
            out=f'{CACHE_OBS}/clara_alb_{mname}_{year}.npy'
            if os.path.exists(out): saved+=1; continue
            sm=sial_pent.sel(time=sial_pent['time.month']==mnum)
            if len(sm.time)==0: continue
            smean=sm.mean(dim='time').values
            sic_maps=[sic_pent[pd.Timestamp(t).date()] for t in sm.time.values if pd.Timestamp(t).date() in sic_pent]
            if not sic_maps: continue
            sicm=np.nanmean(sic_maps,axis=0)
            sout=np.where(arctic_60N_sial,smean,np.nan); scm=np.where(arctic_60N_sial,sicm,np.nan)
            vp=np.isfinite(sout)&np.isfinite(scm)&arctic_60N_sial
            if vp.sum()<MIN_ARCTIC_PX: continue
            ai=weighted_mean(sout,vp&(scm>=SIC_THRESH),sial_coslat)
            ao=weighted_mean(sout,vp&(scm<SIC_THRESH), sial_coslat)
            if np.isfinite(ai) and np.isfinite(ao):
                np.save(out,np.where(vp,sout,np.nan))
                np.save(f'{CACHE_OBS}/clara_dalpha_{mname}_{year}.npy',np.array(ai-ao)); saved+=1
        print(f' ✓ ({saved}/{len(MONTH_NAMES)})',flush=True)
        del sial_pent,sic_daily; gc.collect()

    # ── APP-x (September only) ─────────────────────────────────────────────────
    print('Step 2: APP-x (September)...',flush=True)
    for year in GAP_YEARS:
        out=f'{CACHE_OBS}/appx_alb_September_{year}.npy'
        if os.path.exists(out): print(f'  Sep {year}: cached'); continue
        print(f'  Sep {year}...',end='',flush=True)
        base=DIR_APPX.rstrip('/'); mn=9; files=[]
        for p in [f'{base}/{year}/Polar-APP-X*d{year}{mn:02d}*.nc',
                   f'{base}/Polar-APP-X*d{year}{mn:02d}*.nc']:
            files.extend(glob.glob(p))
        files=sorted(set(files))
        if not files: print(' skip (no files)'); continue
        fbd={}
        for f in files:
            od=_parse_date_appx(f)
            if od is None: continue
            os_=od.strftime('%Y%m%d')
            mt=re.search(r'_(\d{4})_d',os.path.basename(f)); op=mt.group(1) if mt else 'unk'
            mc=re.search(r'_c(\d{8})\.nc$',os.path.basename(f)); cd=mc.group(1) if mc else '0'
            if os_ not in fbd: fbd[os_]={}
            if op not in fbd[os_] or cd>fbd[os_][op][1]: fbd[os_][op]=(f,cd)
        monthly=[]
        for os_,passes in sorted(fbd.items()):
            daily=[]
            for op,(f,cd) in passes.items():
                try:
                    with NC4Dataset(f,'r') as ds:
                        vn=next((v for v in ds.variables if 'albedo' in v.lower() and ds.variables[v].ndim>=2),None)
                        if not vn: continue
                        av=ds.variables[vn]; alb=av[0] if av.ndim==3 else av[:]
                        if hasattr(alb,'filled'): alb=alb.filled(np.nan)
                        alb=alb.astype(float)
                        ff=float(getattr(av,'_FillValue',appx_fill)); alb[alb==ff]=np.nan
                        alb[(alb<0)|(alb>1)]=np.nan; daily.append(alb)
                except: continue
            if daily:
                with warnings.catch_warnings(): warnings.simplefilter('ignore')
                monthly.append(np.nanmean(daily,axis=0))
        if not monthly: print(' skip (no valid data)'); continue
        with warnings.catch_warnings(): warnings.simplefilter('ignore')
        alb_mean=np.nanmean(monthly,axis=0)
        sic_maps=[]
        for f in sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}*.nc')):
            try:
                with xr.open_dataset(f) as ds:
                    if pd.Timestamp(ds['time'].values[0]).month!=mn: continue
                    raw=ds['cdr_seaice_conc'].squeeze().values.astype(float)
                    if raw.ndim>2: raw=raw[0]
                    if np.nanmax(raw)>1.5: raw/=100.
                    raw=np.where((raw>=0.)&(raw<=1.),raw,np.nan)
                    if np.sum(np.isfinite(raw[arctic_60N_mask_psn]))<MIN_ARCTIC_PX: continue
                    sic_maps.append(raw.ravel()[idx_sic_to_appx].reshape(appx_shape))
            except: continue
        if not sic_maps: print(' skip (no SIC)'); continue
        sic_mean=np.nanmean(sic_maps,axis=0)
        aout=np.where(arctic_60N_appx,alb_mean,np.nan); sm=np.where(arctic_60N_appx,sic_mean,np.nan)
        vp=np.isfinite(aout)&np.isfinite(sm)&arctic_60N_appx
        if vp.sum()<MIN_ARCTIC_PX: print(f' skip ({vp.sum()} px)'); continue
        ai=weighted_mean(aout,vp&(sm>=SIC_THRESH),np.ones(appx_shape,dtype=float))
        ao=weighted_mean(aout,vp&(sm<SIC_THRESH), np.ones(appx_shape,dtype=float))
        if np.isfinite(ai) and np.isfinite(ao):
            np.save(out,np.where(vp,aout,np.nan))
            np.save(f'{CACHE_OBS}/appx_dalpha_September_{year}.npy',np.array(ai-ao))
            print(' ✓',flush=True)
        else: print(' skip (non-finite stats)',flush=True)
        gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_obs_rolling_plot(df):
    yr=df['year'].values; tot=df['SIAF_tot'].values; ext=df['SIAF_ext'].values
    drk=df['SIAF_drk'].values; e_t=df['sigma2_tot'].values
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True); fig.patch.set_facecolor('#FAFAFA')
    COL_TOT='#1a1a2e'; COL_EXT='#3A6BAF'; COL_DRK='#D95F1A'
    ax=axes[0]; ax.set_facecolor('#FAFAFA')
    ax.fill_between(yr,tot-e_t,tot+e_t,color=COL_TOT,alpha=0.15,label='Total SIAF ±2σ')
    ax.fill_between(yr,0,ext,color=COL_EXT,alpha=0.55,label='Extent')
    ax.fill_between(yr,ext,ext+drk,color=COL_DRK,alpha=0.55,label='Darkening')
    ax.plot(yr,tot,color=COL_TOT,lw=2.0,zorder=5,label='Total SIAF')
    ax.axhline(0,color='k',lw=0.6,ls='--',alpha=0.4)
    ax.set_ylabel('Global Annual SIAF\n(W m⁻² K⁻¹)',fontsize=11)
    ax.legend(fontsize=9,framealpha=0.9,loc='upper right',ncol=2)
    ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax2=axes[1]; ax2r=ax2.twinx(); ax2.set_facecolor('#FAFAFA')
    ext_frac=np.where(tot>0,ext/tot,np.nan)
    ax2.plot(yr,ext_frac*100,color=COL_EXT,lw=2.0,label='Extent fraction (%)')
    ax2.fill_between(yr,(ext_frac-e_t/np.where(tot>0,tot,np.nan))*100,
                        (ext_frac+e_t/np.where(tot>0,tot,np.nan))*100,
                     color=COL_EXT,alpha=0.15)
    ax2.set_ylabel('Extent fraction (%)',fontsize=11,color=COL_EXT)
    ax2.tick_params(axis='y',colors=COL_EXT); ax2.set_ylim(0,110)
    ax2r.plot(yr,df['dT_mean'].values,color='#888800',lw=1.5,ls='--',alpha=0.7,label='ΔT (K)')
    ax2r.set_ylabel('ΔT (K)',fontsize=10,color='#888800'); ax2r.tick_params(axis='y',colors='#888800')
    l1,lb1=ax2.get_legend_handles_labels(); l2,lb2=ax2r.get_legend_handles_labels()
    ax2.legend(l1+l2,lb1+lb2,fontsize=9,framealpha=0.9,loc='lower right')
    ax2.yaxis.grid(True,alpha=0.3); ax2.set_axisbelow(True)
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)
    ax2.set_xlabel(f'Year (end of trailing {WINDOW}-year late window)',fontsize=11)
    fig.suptitle(f'Sea Ice–Albedo Feedback: {WINDOW}-Year Rolling Window\n'
                 f'(late = trailing {WINDOW} yrs;  ΔT = OLS trend×yrs, 1982→t;  '
                 f'RS = CERES 2000–2024 clim;  p < {PVAL_THRESH})',
                 fontsize=12,fontweight='bold',y=1.01)
    plt.tight_layout()
    fig.savefig(OBS_PLOT,dpi=200,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig); print(f'Obs plot → {OBS_PLOT}',flush=True)


def make_comparison_plot(obs_df, cmip_df):
    OVERLAP=list(range(1994,2015))
    obs_ov=obs_df[obs_df['year'].isin(OVERLAP)].sort_values('year')
    def _trend(df_sub):
        if len(df_sub)<3: return np.nan
        return stats.linregress(df_sub['year'].values,df_sub['SIAF_tot'].values).slope
    obs_trend=_trend(obs_ov)

    fig=plt.figure(figsize=(16,14),facecolor='#FAFAFA')
    gs=fig.add_gridspec(3,2,hspace=0.42,wspace=0.35,left=0.08,right=0.97,top=0.93,bottom=0.07)
    ax1=fig.add_subplot(gs[0,:]); ax2=fig.add_subplot(gs[1,:])
    ax3=fig.add_subplot(gs[2,0]); ax4=fig.add_subplot(gs[2,1])
    for ax in [ax1,ax2,ax3,ax4]: ax.set_facecolor('#FAFAFA')

    model_means={}
    for model in cmip_df['model'].unique():
        col=MODEL_COLORS.get(model,'#888888'); msub=cmip_df[cmip_df['model']==model]; mdata=[]
        for _,grp in msub.groupby('member'):
            grp=grp.sort_values('year')
            ax1.plot(grp['year'],grp['SIAF_tot'],color=col,alpha=0.2,lw=0.8,zorder=2)
            mdata.append(grp.set_index('year')['SIAF_tot'])
        if mdata:
            mm=pd.concat(mdata,axis=1).mean(axis=1).sort_index()
            ax1.plot(mm.index,mm.values,color=col,alpha=0.85,lw=1.8,zorder=3); model_means[model]=mm

    all_m=pd.concat(list(model_means.values()),axis=1); mmm=all_m.mean(axis=1); mmm_s=all_m.std(axis=1)
    ax1.fill_between(mmm.index,mmm-mmm_s,mmm+mmm_s,color='#555555',alpha=0.18,zorder=4)
    ax1.plot(mmm.index,mmm.values,color='#333333',lw=2.5,ls='--',zorder=5,label='MMM ± 1σ')

    obs_full=obs_df.sort_values('year'); obs_ext=obs_full[obs_full['year']>2014]
    ax1.fill_between(obs_full['year'],obs_full['SIAF_tot']-obs_full['sigma2_tot'],
                     obs_full['SIAF_tot']+obs_full['sigma2_tot'],color='#b5000a',alpha=0.15,zorder=6)
    ax1.plot(obs_ov['year'],obs_ov['SIAF_tot'],color='#b5000a',lw=3.0,zorder=7,label='OBS (overlap)')
    if not obs_ext.empty:
        ax1.plot(obs_ext['year'],obs_ext['SIAF_tot'],color='#b5000a',lw=3.0,ls=':',zorder=7,label='OBS (post-historical)')
    ax1.axvline(2014,color='#444',lw=1.2,ls=':',alpha=0.7)
    ax1.set_ylabel('Global Annual SIAF\n(W m⁻² K⁻¹)',fontsize=11)
    ax1.set_title('Rolling 5-Year SIAF: CMIP6 Historical vs Observations',fontsize=12,fontweight='bold')
    ax1.legend(fontsize=9,framealpha=0.9,loc='upper right',ncol=3)
    ax1.yaxis.grid(True,alpha=0.3); ax1.set_axisbelow(True)
    for sp in ['top','right']: ax1.spines[sp].set_visible(False)

    model_trends={}
    for model in cmip_df['model'].unique():
        msub=cmip_df[cmip_df['model']==model]; trds=[]
        for _,grp in msub.groupby('member'):
            t=_trend(grp[grp['year'].isin(OVERLAP)].sort_values('year'))
            if np.isfinite(t): trds.append(t)
        if trds: model_trends[model]=np.array(trds)

    sorted_m=sorted(model_trends,key=lambda m:np.median(model_trends[m]))
    for yi,model in enumerate(sorted_m):
        col=MODEL_COLORS.get(model,'#888')
        ax2.boxplot(model_trends[model],positions=[yi],vert=False,widths=0.6,patch_artist=True,
                    boxprops=dict(facecolor=col,alpha=0.65),medianprops=dict(color='white',linewidth=2),
                    whiskerprops=dict(color=col,alpha=0.7),capprops=dict(color=col,alpha=0.7),
                    flierprops=dict(marker='o',color=col,markersize=3,alpha=0.5))
    ax2.axvline(obs_trend,color='#b5000a',lw=2.5,ls='--',zorder=5,
                label=f'OBS trend = {obs_trend*1e3:.2f}×10⁻³ W m⁻² K⁻¹ yr⁻¹')
    ax2.axvline(0,color='#888',lw=0.8,ls='-',alpha=0.5)
    ax2.set_yticks(range(len(sorted_m))); ax2.set_yticklabels(sorted_m,fontsize=8)
    ax2.set_xlabel('SIAF linear trend 1994–2014  (W m⁻² K⁻¹ yr⁻¹)',fontsize=10)
    ax2.set_title('Rate of SIAF Change (1994–2014)',fontsize=11,fontweight='bold')
    ax2.legend(fontsize=9,framealpha=0.9)
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)

    for model in model_trends:
        col=MODEL_COLORS.get(model,'#888')
        msub=cmip_df[cmip_df['model']==model]
        level=msub[msub['year'].isin(OVERLAP)]['SIAF_tot'].mean()
        ax3.scatter(level,float(np.median(model_trends[model])),color=col,s=80,zorder=4,edgecolors='white',lw=0.6)
        ax3.annotate(model,(level,float(np.median(model_trends[model]))),fontsize=6,xytext=(3,2),textcoords='offset points')
    obs_level=obs_ov['SIAF_tot'].mean()
    ax3.scatter([obs_level],[obs_trend],marker='*',s=300,color='#b5000a',edgecolors='white',zorder=6,label='OBS')
    ax3.axvline(obs_level,color='#b5000a',lw=1,ls=':',alpha=0.5); ax3.axhline(obs_trend,color='#b5000a',lw=1,ls=':',alpha=0.5)
    ax3.axhline(0,color='#888',lw=0.8,ls='-',alpha=0.4)
    ax3.set_xlabel('Mean SIAF 1994–2014  (W m⁻² K⁻¹)',fontsize=10); ax3.set_ylabel('SIAF trend  (W m⁻² K⁻¹ yr⁻¹)',fontsize=10)
    ax3.set_title('Level vs Trend',fontsize=11,fontweight='bold'); ax3.legend(fontsize=9); ax3.grid(True,alpha=0.25)
    for sp in ['top','right']: ax3.spines[sp].set_visible(False)

    obs_series=obs_ov.set_index('year')['SIAF_tot']; corrs,labels,cols_c=[],[],[]
    for model in sorted(model_trends.keys()):
        msub=cmip_df[cmip_df['model']==model]; mcorrs=[]
        for _,grp in msub.groupby('member'):
            ov=grp[grp['year'].isin(OVERLAP)].sort_values('year').set_index('year')
            common=obs_series.index.intersection(ov.index)
            if len(common)>=4:
                r,_=stats.pearsonr(obs_series[common],ov.loc[common,'SIAF_tot'])
                if np.isfinite(r): mcorrs.append(r)
        if mcorrs: corrs.append(np.mean(mcorrs)); labels.append(model); cols_c.append(MODEL_COLORS.get(model,'#888'))
    si=np.argsort(corrs); corrs=[corrs[i] for i in si]; labels=[labels[i] for i in si]; cols_c=[cols_c[i] for i in si]
    ax4.barh(range(len(corrs)),corrs,color=cols_c,alpha=0.8,edgecolor='white',linewidth=0.5)
    ax4.axvline(0,color='#888',lw=1.); ax4.axvline(0.5,color='#333',lw=0.8,ls='--',alpha=0.5,label='r = 0.5')
    ax4.set_yticks(range(len(labels))); ax4.set_yticklabels(labels,fontsize=8)
    ax4.set_xlabel('Pearson r (model vs OBS rolling SIAF, 1994–2014)',fontsize=10)
    ax4.set_title('Model–OBS Correlation',fontsize=11,fontweight='bold')
    ax4.set_xlim(-1,1); ax4.legend(fontsize=8); ax4.yaxis.grid(True,alpha=0.2); ax4.set_axisbelow(True)
    for sp in ['top','right']: ax4.spines[sp].set_visible(False)

    fig.savefig(COMP_PLOT,dpi=200,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig); print(f'Comparison plot → {COMP_PLOT}',flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# OBS GRID SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_obs_grids():
    global sial_shape,sial_lat2d,sial_lon2d,sial_coslat,arctic_sial
    global appx_shape,appx_lat2d,appx_lon2d,arctic_60N_appx,appx_fill
    global sic_shape,sic_lat2d_psn,sic_lon2d_psn,arctic_60N_mask_psn
    global psn_shape,psn_lat2d,psn_lon2d,psn_xx,psn_yy
    global AREA_WEIGHT_PSN,DOMAIN_PSN,REGION_MASKS_PSN
    global idx_sial_to_psn,idx_ceres_to_psn,idx_appx_to_sial
    global idx_sic_to_sial,idx_sic_to_appx,ceres_lat1d,ceres_lon1d
    global T_SOURCES,RS_PSN_CLIM,AE_EARLY,SIC_OBS

    print('\n'+'═'*60+'\nGRID SETUP\n'+'═'*60,flush=True)

    # SIAL
    with xr.open_dataset(sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]) as ds:
        la,lo=ds['lat'].values,ds['lon'].values
    if la.ndim==1: sial_lon2d,sial_lat2d=np.meshgrid(lo,la)
    else: sial_lat2d,sial_lon2d=la,lo
    sial_shape=sial_lat2d.shape; sial_coslat=np.cos(np.deg2rad(sial_lat2d)).astype(float)
    sial_coslat=np.where(sial_coslat>0,sial_coslat,0.)
    try: arctic_sial=np.load(f'{CACHE_OBS}/arctic_60N_sial.npy').astype(bool)
    except: arctic_sial=(sial_lat2d>=60.)
    print(f'  SIAL: {sial_shape}',flush=True)

    # APP-x
    with NC4Dataset(sorted(glob.glob(f'{DIR_APPX}/**/*.nc',recursive=True))[0],'r') as ds:
        ala=np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),np.nan).astype(float)
        alo=np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]),np.nan).astype(float)
        appx_fill=float(getattr(ds.variables['cdr_surface_albedo'],'_FillValue',9999.))
    if ala.ndim==1: appx_lon2d,appx_lat2d=np.meshgrid(alo,ala)
    else: appx_lat2d,appx_lon2d=ala,alo
    appx_shape=appx_lat2d.shape; print(f'  APP-x: {appx_shape}',flush=True)

    # PSN25 via SIC file
    with xr.open_dataset(sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]) as ds:
        px,py=ds['x'].values.astype(np.float64),ds['y'].values.astype(np.float64)
    psn_xx,psn_yy=np.meshgrid(px,py); psn_shape=psn_xx.shape
    tfm=Transformer.from_crs(PSN25_PROJ4,'EPSG:4326',always_xy=True)
    psn_lon2d,psn_lat2d=tfm.transform(psn_xx,psn_yy)
    phi_ts=np.deg2rad(70.); k_ps=(1.+np.sin(phi_ts))/np.maximum(1.+np.sin(np.deg2rad(psn_lat2d)),1e-9)
    AREA_WEIGHT_PSN=(1./k_ps**2).astype(np.float32)
    if _HAS_LAND:
        lp=np.where(psn_lon2d>180,psn_lon2d-360,psn_lon2d)
        DOMAIN_PSN=(psn_lat2d>=60.)&_gm_globe.is_ocean(psn_lat2d,lp)
    else: DOMAIN_PSN=(psn_lat2d>=60.)
    REGION_MASKS_PSN=[]
    for code,lo_b,hi_b,lmn,lmx in PISTONE_REGIONS:
        lp2=np.where(psn_lon2d>180,psn_lon2d-360.,psn_lon2d)
        lat_ok=(psn_lat2d>=lo_b)&(psn_lat2d<=hi_b)
        lon_ok=((lp2>=lmn)&(lp2<=lmx) if lmn<=lmx else (lp2>=lmn)|(lp2<=lmx))
        REGION_MASKS_PSN.append(lat_ok&lon_ok&DOMAIN_PSN)
    print(f'  PSN25: {psn_shape}  domain={DOMAIN_PSN.sum():,} cells',flush=True)

    # SIC grid for fill_gap_years regridding
    sic_tree=cKDTree(latlon_to_xyz(psn_lat2d.ravel(),psn_lon2d.ravel()))
    arctic_60N_mask_psn=(psn_lat2d>=60.); sic_shape=psn_shape
    sic_lat2d_psn=psn_lat2d; sic_lon2d_psn=psn_lon2d
    _,idx_sic_to_sial=sic_tree.query(latlon_to_xyz(sial_lat2d.ravel(),sial_lon2d.ravel()),workers=-1)
    _,idx_sic_to_appx=sic_tree.query(latlon_to_xyz(appx_lat2d.ravel(),appx_lon2d.ravel()),workers=-1)
    arctic_60N_sial_=arctic_60N_mask_psn.ravel()[idx_sic_to_sial].reshape(sial_shape)
    arctic_60N_appx= arctic_60N_mask_psn.ravel()[idx_sic_to_appx].reshape(appx_shape)
    import ctypes; globals()['arctic_60N_sial']=arctic_60N_sial_; globals()['arctic_60N_appx']=arctic_60N_appx
    del sic_tree; gc.collect()

    # SIAL → PSN25
    print('  Building SIAL→PSN25 index...',flush=True)
    idx_sial_to_psn,_=build_scatter_index(sial_lat2d,sial_lon2d,psn_lat2d,psn_lon2d)

    # APP-x → SIAL (tree from APP-x, query with SIAL)
    tree=cKDTree(latlon_to_xyz(appx_lat2d.ravel(),appx_lon2d.ravel()))
    _,idx_appx_to_sial=tree.query(latlon_to_xyz(sial_lat2d.ravel(),sial_lon2d.ravel()),workers=-1)
    del tree; gc.collect()

    # CERES → PSN25
    with xr.open_dataset(CERES_FILE,engine='netcdf4') as _ds:
        ceres_lat1d=_ds['lat'].values; ceres_lon1d=_ds['lon'].values
    clon,clat=np.meshgrid(ceres_lon1d,ceres_lat1d)
    print('  Building CERES→PSN25 index...',flush=True)
    idx_ceres_to_psn,_=build_scatter_index(clat,clon,psn_lat2d,psn_lon2d)

    # Temperature sources
    T_SOURCES=build_T_sources()

    # Fixed CERES RS climatology
    print('  Loading CERES RS climatology...',flush=True)
    for mn in MONTH_NUMS:
        RS_c=load_RS_ceres_clim(CERES_CLIM_YRS,mn)
        if RS_c is not None:
            RS_PSN_CLIM[mn]=scatter_avg(RS_c.ravel(),idx_ceres_to_psn,psn_shape)
            print(f'    Month {mn} loaded',flush=True)

    # Fixed early albedo
    for mn in MONTH_NUMS:
        stk=load_alb_yearly_stack(MONTH_NAMES[mn],EARLY_YEARS)
        if stk is not None: AE_EARLY[MONTH_NAMES[mn]]=np.nanmean(stk,axis=0).astype(np.float32)

    # SIC obs
    for mn in MONTH_NUMS:
        sic=load_sic_obs(MONTH_NAMES[mn])
        if sic is not None: SIC_OBS[MONTH_NAMES[mn]]=sic

    print('Grid setup complete.',flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import multiprocessing; multiprocessing.set_start_method('spawn', force=True)

    print('='*70+'\nSIAF ROLLING ANALYSIS PIPELINE\n'+'='*70,flush=True)

    # 1. Grid setup (needed by cache fill and obs rolling)
    setup_obs_grids()

    # 2. Fill cache gap years 1992–2004
    fill_gap_years()

    # 3. Obs rolling SIAF
    obs_df = run_obs_rolling()

    # 4. CMIP6 rolling SIAF (parallel, reads its own data)
    cmip_df = run_cmip6_rolling()

    # 5. Plots
    print('\nGenerating plots...',flush=True)
    make_obs_rolling_plot(obs_df)
    make_comparison_plot(obs_df, cmip_df)

    print('\n'+'='*70+'\nDONE\n'+'='*70,flush=True)