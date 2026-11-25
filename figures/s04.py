#!/usr/bin/env python
"""Extended Data — SIAF vs Arctic Amplification, family-grouped, 2-sigma bars."""
import os, glob
from pathlib import Path
import numpy as np, pandas as pd
try:
    import xarray as xr
except ImportError:
    xr = None
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.size': 14.8,
    'axes.labelsize': 14.8,
    'axes.titlesize': 17.7,
    'xtick.labelsize': 14.8,
    'ytick.labelsize': 14.8,
    'legend.fontsize': 14.8,
})

R_FIG = Path(__file__).resolve().parent
_D_CMIP6 = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
_D_OBS = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
DIR_CMIP6_CSV = _D_CMIP6 if os.path.exists(_D_CMIP6) else str(R_FIG / 'siaf_cmip6_summary_final.csv')
DIR_OBS_CSV   = str(R_FIG / 'siaf_pixelwise_summary.csv') if os.path.exists(str(R_FIG / 'siaf_pixelwise_summary.csv')) else _D_OBS
BOOT_CSV      = R_FIG.parent / 'source_data' / 'SourceData_Supplementary_Figure4_Bootstrap.csv'
NCEP_PATH     = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB     = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                 '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')
_DERECHO_PLOT = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/siaf_cmip6_vs_AA_family.png'
PLOT_OUT      = _DERECHO_PLOT if os.path.exists(os.path.dirname(_DERECHO_PLOT)) else str(R_FIG / 'figures/s04.png')
SCALE         = 0.0347 * 0.5
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
    if xr is not None:
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
    if not aas:
        # Exact NCEP & ERA5 AA values computed from the Derecho datasets above
        if late_years == LATE_YEARS_CMIP6:
            aas = {'NCEP': 4.313431922970221, 'ERA5': 3.947277604266602}
        elif late_years == LATE_YEARS_OBS:
            aas = {'NCEP': 3.437538376524532, 'ERA5': 2.994581481923235}
        else:
            return np.nan, np.nan, []
    v = list(aas.values())
    return float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0., v

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
a1, s1, aa1_vals = compute_obs_AA(LATE_YEARS_CMIP6); a2, s2, aa2_vals = compute_obs_AA(LATE_YEARS_OBS)
if 'CMIP6'   in obs: obs['CMIP6'].update(aa_mean=a1, aa_std=2*s1, aa_vals=aa1_vals)
if 'Obs-era' in obs: obs['Obs-era'].update(aa_mean=a2, aa_std=2*s2, aa_vals=aa2_vals)
print(f'  CMIP6-era AA={a1:.3f}  Obs-era AA={a2:.3f}')

_b_s4 = pd.read_csv(BOOT_CSV)
for pk in ('CMIP6', 'Obs-era'):
    if pk in obs:
        _draws = _b_s4.loc[_b_s4.period == pk, 'siaf_global_W_m2_K'].to_numpy(float)
        assert np.isclose(2 * np.nanstd(_draws, ddof=0), obs[pk]['siaf_err'], atol=1e-4)
        obs[pk]['siaf_draws'] = _draws

aa_arr, si_arr = fam.AA.values, fam.siaf_global.values
fin_ = np.isfinite(aa_arr) & np.isfinite(si_arr)
slope, intercept, r, p_ols, _ = stats.linregress(aa_arr[fin_], si_arr[fin_])
print(f'OLS (family means): r={r:+.3f} r2={r**2:.3f} p={p_ols:.4f}')

pal = list(plt.get_cmap('tab20').colors)
cols = {f: pal[i % len(pal)] for i, f in enumerate(order)}
fig, ax = plt.subplots(figsize=(9.5, 9.6), dpi=300, layout='constrained')
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')

for f in order:
    s_ = df[df.family == f]
    ax.scatter(s_.AA, s_.siaf_global, color=cols[f], s=12, alpha=0.40,
               zorder=1, edgecolors='none')
# Label only outlying families: the extremes in feedback and in Arctic
# amplification. Labelling all 18 made the text overlap illegibly.
_sf = fam.siaf_global.sort_values()
_aa = fam.AA.sort_values()
label_fams = set(list(_sf.index[:3]) + list(_sf.index[-3:])
                 + list(_aa.index[:2]) + list(_aa.index[-2:]))

for f in order:
    x_m, y_m = fam.AA[f], fam.siaf_global[f]
    ax.errorbar(x_m, y_m, xerr=fam.aa_sd[f] or None, yerr=fam.siaf_sd[f] or None,
                fmt='none', color=cols[f], alpha=0.5, lw=1.0, capsize=2.5, zorder=2)
    ax.scatter(x_m, y_m, color=cols[f], s=75, zorder=4,
               edgecolors='white', linewidths=0.6,
               label=f'{f} (n={int(fam.n_mem[f])})')
    # families are identified by the legend outside the axes

xx = np.linspace(aa_arr[fin_].min()*0.9, aa_arr[fin_].max()*1.1, 100)
ax.plot(xx, slope*xx + intercept, color='#333333', lw=1.5, ls='--', zorder=3,
        label=f'OLS (family means): r²={r**2:.2f}, p={p_ols:.3f}')
_rng_s4 = np.random.default_rng(20261125)
for pk, st in (('CMIP6', dict(marker='D', ms=13)), ('Obs-era', dict(marker='*', ms=18))):
    if pk not in obs: continue
    o = obs[pk]
    if not (np.isfinite(o['siaf']) and np.isfinite(o.get('aa_mean', np.nan))): continue
    if 'siaf_draws' in o:
        _dr = o['siaf_draws']
        ax.scatter(o['aa_mean'] + _rng_s4.uniform(-0.045, 0.045, len(_dr)), _dr,
                   s=6, color='#b5000a', alpha=0.10, edgecolors='none', zorder=5, rasterized=True)
    if o.get('aa_vals'):
        ax.scatter(o['aa_vals'], [o['siaf']] * len(o['aa_vals']),
                   s=28, facecolor='white', edgecolor='#b5000a', linewidths=1.3, zorder=8)
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
ax.set_xlabel('Arctic amplification\n' r'($\Delta T_\mathrm{Arctic}\ /\ \Delta T_\mathrm{global}$)',
              fontsize=14.8)
ax.set_ylabel(r'Pan-Arctic annual SIAF (W m$^{-2}$ K$^{-1}$)', fontsize=14.8)
ax.tick_params(axis='both', labelsize=14.8)
# 23 Nov 2026: the 21-entry legend beside the axes was as large as the plot.
# Families go under the axes in three columns; the fit and the two observed
# points keep a small legend in the empty upper-left corner.
_h, _l = ax.get_legend_handles_labels()
_fam = [(a, b) for a, b in zip(_h, _l) if ' (n=' in b]
_ref = [(a, b) for a, b in zip(_h, _l) if ' (n=' not in b]
_ref.append((Line2D([0], [0], marker='o', color='none', markerfacecolor='#b5000a',
                    markeredgecolor='none', markersize=4.5, alpha=0.55),
             'Obs. pairs bootstrap / reanalyses'))
ax.legend([x[0] for x in _ref], [x[1] for x in _ref], loc='upper left',
          fontsize=13.5, framealpha=0.92)
# headroom so that small legend clears the tallest whiskers (INM reaches 0.253)
ax.set_ylim(top=0.33)
fig.legend([x[0] for x in _fam], [x[1] for x in _fam], loc='outside lower center',
           ncol=3, fontsize=14.8, frameon=False, handletextpad=0.3,
           columnspacing=1.0, labelspacing=0.3)
ax.grid(True, alpha=0.2, zorder=0)
for sp_ in ['top','right']: ax.spines[sp_].set_visible(False)
Path(PLOT_OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOT_OUT, facecolor=fig.get_facecolor())
_d_s4 = '/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028/figures/s04.png'
if os.path.exists(os.path.dirname(_d_s4)):
    fig.savefig(_d_s4, facecolor=fig.get_facecolor())
plt.close(fig)
print(f'Saved → {PLOT_OUT}')
