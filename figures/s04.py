#!/usr/bin/env python
"""Extended Data — SIAF vs Arctic Amplification, family-grouped, 2-sigma bars."""
import os, glob
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR_CMIP6_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
DIR_OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
NCEP_PATH     = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB     = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                 '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')
PLOT_OUT      = ('/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
                 'siaf_cmip6_vs_AA_family.png')
OBS_RS, OBS_PVAL = 'CERES', 0.05
EARLY_YEARS      = list(range(1982, 1992))
LATE_YEARS_CMIP6 = list(range(2005, 2015))
LATE_YEARS_OBS   = list(range(2015, 2025))
ARCTIC_LAT       = 60.

FAMILY = {**{m:'CESM2' for m in ['CESM2','CESM2-WACCM','CESM2-FV2','CESM2-WACCM-FV2']},
          **{m:'EC-Earth3' for m in ['EC-Earth3','EC-Earth3-Veg','EC-Earth3-Veg-LR',
                                     'EC-Earth3-CC','EC-Earth3-AerChem','EC-Earth3-ESM-1']},
          **{m:'CMCC' for m in ['CMCC-CM2-SR5','CMCC-ESM2','CMCC-CM2-HR4']},
          **{m:'E3SM' for m in ['E3SM-1-0','E3SM-1-1','E3SM-1-1-ECA']},
          **{m:'HadGEM3/UKESM' for m in ['HadGEM3-GC31-LL','HadGEM3-GC31-MM',
                                         'UKESM1-0-LL','UKESM1-1-LL']},
          **{m:'MPI/ICON' for m in ['MPI-ESM1-2-HR','MPI-ESM1-2-LR',
                                    'MPI-ESM-1-2-HAM','ICON-ESM-LR']},
          **{m:'NorESM/NorCPM' for m in ['NorESM2-LM','NorESM2-MM','NorCPM1']},
          **{m:'CNRM' for m in ['CNRM-CM6-1','CNRM-CM6-1-HR','CNRM-ESM2-1']},
          **{m:'CanESM5' for m in ['CanESM5','CanESM5-CanOE']},
          **{m:'MIROC' for m in ['MIROC6','MIROC-ES2L']},
          **{m:'IPSL' for m in ['IPSL-CM6A-LR','IPSL-CM6A-LR-INCA','IPSL-CM5A2-INCA']},
          **{m:'INM' for m in ['INM-CM4-8','INM-CM5-0']},
          **{m:'FGOALS' for m in ['FGOALS-g3','FGOALS-f3-L']},
          **{m:'BCC' for m in ['BCC-CSM2-MR','BCC-ESM1']},
          **{m:'ACCESS' for m in ['ACCESS-CM2','ACCESS-ESM1-5']},
          **{m:'GISS' for m in ['GISS-E2-1-H','GISS-E2-1-G']}}

def _epoch_means(da, lat_name, years):
    yrs = np.array([int(t.dt.year) for t in da.time]); mask = np.isin(yrs, years)
    if not mask.any(): return np.nan, np.nan
    arr = da.isel(time=mask).mean('time').values.astype(float)
    lat = da[lat_name].values.astype(float)
    if lat.ndim == 1:
        lon_name = 'longitude' if 'longitude' in da.coords else 'lon'
        _, lat2d = np.meshgrid(da[lon_name].values, lat)
    else:
        lat2d = lat
    w_glob = np.cos(np.deg2rad(lat2d))
    w_arct = np.where(lat2d >= ARCTIC_LAT, w_glob, 0.)
    def _wm(w):
        ok = np.isfinite(arr) & (w > 0)
        return float(np.sum(arr[ok]*w[ok])/np.sum(w[ok])) if ok.any() else np.nan
    return _wm(w_glob), _wm(w_arct)

def _aa_from_da(da, lat_name, late_years):
    g_e, a_e = _epoch_means(da, lat_name, EARLY_YEARS)
    g_l, a_l = _epoch_means(da, lat_name, late_years)
    dg, da_ = g_l - g_e, a_l - a_e
    return float(da_/dg) if (np.isfinite(dg) and dg > 0 and np.isfinite(da_)) else np.nan

def compute_obs_AA(late_years):
    aas = {}
    try:
        if os.path.exists(NCEP_PATH):
            ds = xr.open_dataset(NCEP_PATH); air = ds['air']
            ln = 'lat' if 'lat' in air.coords else 'latitude'
            aa = _aa_from_da(air, ln, late_years); ds.close()
            if np.isfinite(aa): aas['NCEP'] = aa
    except Exception as e: print(f'  NCEP AA failed: {e}')
    try:
        files = sorted(glob.glob(ERA5_GLOB))
        if files:
            ds = xr.open_mfdataset(files, combine='by_coords'); t2m = ds['VAR_2T']
            ln = 'latitude' if 'latitude' in t2m.coords else 'lat'
            aa = _aa_from_da(t2m, ln, late_years); ds.close()
            if np.isfinite(aa): aas['ERA5'] = aa
    except Exception as e: print(f'  ERA5 AA failed: {e}')
    if not aas: return np.nan, np.nan
    v = list(aas.values())
    return float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.

df = pd.read_csv(DIR_CMIP6_CSV); df = df[df.rs_method == 'full_T07'].copy()
c = df.groupby('model').size(); df = df[df.model.isin(c[c >= 3].index)].copy()
df['AA'] = df.delta_t_arctic / df.delta_t
df = df.dropna(subset=['AA','siaf_global']); df = df[df.AA > 0].copy()
df['family'] = df.model.map(lambda m: FAMILY.get(m, m))

mm  = df.groupby(['family','model'])[['AA','siaf_global']].mean()
fam = mm.groupby('family').mean()
sp  = 2 * df.groupby('family')[['AA','siaf_global']].std().fillna(0.)   # 2-sigma
fam['aa_sd'], fam['siaf_sd'] = sp.AA, sp.siaf_global
fam['n_mod'] = mm.groupby('family').size()
fam['n_mem'] = df.groupby('family').size()
fam = fam.sort_values('siaf_global'); order = list(fam.index)
print(f'{len(order)} families, {df.model.nunique()} models, {len(df)} members')

obs = {}; df_obs = pd.read_csv(DIR_OBS_CSV)
for pk, lab in [('CMIP6','CMIP6-era (1982–2014)'), ('Obs-era','Obs-era (1982–2024)')]:
    r_ = df_obs[(df_obs.period == pk) & (df_obs.rs_method == OBS_RS) &
                np.isclose(df_obs.pval, OBS_PVAL)]
    if len(r_):
        obs[pk] = dict(siaf=float(r_.iloc[0].SIAF_tot_global),
                       siaf_err=float(r_.iloc[0].sigma2_tot_global),   # 2-sigma
                       label=lab)
print('Computing observed AA...')
a1, s1 = compute_obs_AA(LATE_YEARS_CMIP6); a2, s2 = compute_obs_AA(LATE_YEARS_OBS)
if 'CMIP6'   in obs: obs['CMIP6'].update(aa_mean=a1, aa_std=2*s1)
if 'Obs-era' in obs: obs['Obs-era'].update(aa_mean=a2, aa_std=2*s2)
print(f'  CMIP6-era AA={a1:.3f}  Obs-era AA={a2:.3f}')

aa_arr, si_arr = fam.AA.values, fam.siaf_global.values
fin_ = np.isfinite(aa_arr) & np.isfinite(si_arr)
slope, intercept, r, p_ols, _ = stats.linregress(aa_arr[fin_], si_arr[fin_])
print(f'OLS (family means): r={r:+.3f} r2={r**2:.3f} p={p_ols:.4f}')

pal = list(plt.get_cmap('tab20').colors)
cols = {f: pal[i % len(pal)] for i, f in enumerate(order)}
fig, ax = plt.subplots(figsize=(9.5, 7), dpi=300)
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')

for f in order:
    s_ = df[df.family == f]
    ax.scatter(s_.AA, s_.siaf_global, color=cols[f], s=12, alpha=0.40,
               zorder=1, edgecolors='none')
for f in order:
    x_m, y_m = fam.AA[f], fam.siaf_global[f]
    ax.errorbar(x_m, y_m, xerr=fam.aa_sd[f] or None, yerr=fam.siaf_sd[f] or None,
                fmt='none', color=cols[f], alpha=0.5, lw=1.0, capsize=2.5, zorder=2)
    ax.scatter(x_m, y_m, color=cols[f], s=75, zorder=4,
               edgecolors='white', linewidths=0.6)
    ax.annotate(f'{f} (n={int(fam.n_mem[f])})', xy=(x_m, y_m), xytext=(6, 3),
                textcoords='offset points', fontsize=7, color=cols[f],
                zorder=5, fontweight='bold')

xx = np.linspace(aa_arr[fin_].min()*0.9, aa_arr[fin_].max()*1.1, 100)
ax.plot(xx, slope*xx + intercept, color='#333333', lw=1.5, ls='--', zorder=3,
        label=f'OLS (family means): r²={r**2:.2f}, p={p_ols:.3f}')
for pk, st in (('CMIP6', dict(marker='D', ms=13)), ('Obs-era', dict(marker='*', ms=18))):
    if pk not in obs: continue
    o = obs[pk]
    if not (np.isfinite(o['siaf']) and np.isfinite(o.get('aa_mean', np.nan))): continue
    ax.errorbar(o['aa_mean'], o['siaf'],
                xerr=o['aa_std'] if o['aa_std'] > 0 else None,
                yerr=o['siaf_err'] if o['siaf_err'] > 0 else None,
                fmt=st['marker'], color='#b5000a', markersize=st['ms']/1.5,
                markeredgecolor='white', markeredgewidth=0.8, capsize=4,
                elinewidth=1.8, ecolor='#b5000a', zorder=7, label=o['label'])
if 'CMIP6' in obs and 'Obs-era' in obs:
    o1, o2 = obs['CMIP6'], obs['Obs-era']
    if all(np.isfinite([o1.get('aa_mean',np.nan), o1['siaf'],
                        o2.get('aa_mean',np.nan), o2['siaf']])):
        ax.annotate('', xy=(o2['aa_mean'], o2['siaf']),
                    xytext=(o1['aa_mean'], o1['siaf']),
                    arrowprops=dict(arrowstyle='->', color='#b5000a', lw=1.6,
                                    linestyle='dashed', connectionstyle='arc3,rad=0.18'))
ax.set_xlabel(r'Arctic Amplification ($\Delta T_\mathrm{Arctic}\ /\ \Delta T_\mathrm{global}$)',
              fontsize=11)
ax.set_ylabel(r'Pan-Arctic Annual SIAF (W m$^{-2}$ K$^{-1}$)', fontsize=11)
ax.tick_params(axis='both', labelsize=10)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
ax.grid(True, alpha=0.2, zorder=0)
for sp_ in ['top','right']: ax.spines[sp_].set_visible(False)
fig.tight_layout(); fig.savefig(PLOT_OUT, facecolor=fig.get_facecolor()); plt.close(fig)
print(f'Saved → {PLOT_OUT}')