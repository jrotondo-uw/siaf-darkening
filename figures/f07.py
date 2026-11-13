#!/usr/bin/env python
"""
Fig 4 — IS vs RS for the expanded ensemble.

Every ensemble member is plotted and coloured by model family. No per-model
markers or error bars: the member cloud carries the spread directly, and
overlapping bars from models of the same family were unreadable. Only the
observations and the multi-model mean carry error bars.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CMIP6_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
PLOT_OUT  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/f04_family.png'
MIN_MEMBERS = 3

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

PAL = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#e377c2',
       '#17becf','#bcbd22','#7f7f7f','#aec7e8','#ff9896','#98df8a','#ffbb78',
       '#c5b0d5','#c49c94','#f7b6d2','#dbdb8d','#9edae5','#c7c7c7']

OBS_STYLE = {
    'CERES':  dict(color='#b5000a', marker='*', ms=21,
                   label='Obs · CERES APRP ($\\pm2\\sigma$)'),
    'Kernel': dict(color='#5e3c99', marker='P', ms=14,
                   label='Obs · ERA5 kernel ($\\pm2\\sigma$)'),
}


def load_obs(df_obs, method):
    m = df_obs[(df_obs.rs_method == method) & (df_obs.period == 'CMIP6') &
               (np.isclose(df_obs.pval, 0.05))]
    if m.empty:
        print(f'WARNING: no OBS row for {method}'); return None
    r = m.iloc[0]
    o = dict(is_=float(r.IS_tot_arctic), is_2s=float(r.sigma2_IS_arctic),
             rs=float(r.RS_arctic),      rs_2s=float(r.sigma2_RS_arctic))
    print(f'OBS {method:7s}: IS={o["is_"]:.4f}±{o["is_2s"]:.4f}  '
          f'RS={o["rs"]:.2f}±{o["rs_2s"]:.2f}')
    return o


# ── data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CMIP6_CSV)
df = df[df.rs_method == 'full_T07'].copy()
cnt = df.groupby('model').size()
df = df[df.model.isin(cnt[cnt >= MIN_MEMBERS].index)].copy()
df = df.dropna(subset=['rs_arctic', 'is_tot_arctic'])
df['family'] = df.model.map(lambda m: FAMILY.get(m, m))
print(f'{len(df)} members / {df.model.nunique()} models / {df.family.nunique()} families')

# per-model stats: not plotted, but printed for the text
mstat = (df.groupby(['family','model'])
           .agg(rs=('rs_arctic','mean'), is_=('is_tot_arctic','mean'),
                rs_sd=('rs_arctic','std'), is_sd=('is_tot_arctic','std'))
           .fillna(0.).reset_index())

mm  = df.groupby(['family','model'])[['rs_arctic','is_tot_arctic']].mean()
fam = mm.groupby('family').mean()
fam['n_mem'] = df.groupby('family').size()
fam = fam.sort_values('is_tot_arctic', ascending=False)   # legend tracks y-axis

MMM_RS, MMM_IS = fam.rs_arctic.mean(), fam.is_tot_arctic.mean()
MMM_RS_2s = 2 * fam.rs_arctic.std(ddof=1)
MMM_IS_2s = 2 * fam.is_tot_arctic.std(ddof=1)
print(f'family MMM: RS={MMM_RS:.2f}±{MMM_RS_2s:.2f}  IS={MMM_IS:.4f}±{MMM_IS_2s:.4f}')
print(f'family RS range: {fam.rs_arctic.min():.0f}-{fam.rs_arctic.max():.0f}')
print(f'per-model RS range: {mstat.rs.min():.0f}-{mstat.rs.max():.0f}')
print(f'median within-model RS 2-sigma: {2*mstat.rs_sd.median():.2f} W m-2')
print(f'median within-model IS 2-sigma: {2*mstat.is_sd.median():.4f} K-1')

df_obs = pd.read_csv(OBS_CSV)
obs = {m: load_obs(df_obs, m) for m in ('CERES', 'Kernel')}
o_prim = obs['CERES']

# ── figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7.5), facecolor='#FAFAFA', dpi=300)
ax.set_facecolor('#FAFAFA')

fams = list(fam.index)
cols = {f: PAL[i % len(PAL)] for i, f in enumerate(fams)}

# observational IS band (shared by both RS methods)
if o_prim:
    ax.axhspan(o_prim['is_'] - o_prim['is_2s'], o_prim['is_'] + o_prim['is_2s'],
               color='#888888', alpha=0.22, zorder=1)

# every member, coloured by family
for f in fams:
    s = df[df.family == f]
    ax.scatter(s.rs_arctic, s.is_tot_arctic, s=45, color=cols[f], alpha=0.75,
               edgecolors='white', linewidths=0.4, zorder=3,
               label=f'{f} (n={int(fam.n_mem[f])})')

# family-weighted MMM
ax.errorbar(MMM_RS, MMM_IS, xerr=MMM_RS_2s, yerr=MMM_IS_2s, fmt='none',
            color='#222222', capsize=5, lw=1.8, capthick=1.8, zorder=7)
ax.scatter(MMM_RS, MMM_IS, marker='D', color='#222222', s=95,
           edgecolors='white', linewidths=0.9, zorder=8,
           label='Family-weighted MMM ($\\pm2\\sigma$)')

# both observational RS estimates
for method in ('CERES', 'Kernel'):
    o = obs[method]
    if o is None:
        continue
    st = OBS_STYLE[method]
    ax.errorbar(o['rs'], o['is_'], xerr=o['rs_2s'], yerr=o['is_2s'],
                fmt=st['marker'], color=st['color'], markersize=st['ms'],
                markeredgecolor='white', markeredgewidth=0.9, ecolor=st['color'],
                elinewidth=1.8, capsize=4, capthick=1.8, zorder=9, label=st['label'])
if obs['CERES'] and obs['Kernel']:
    rc, rk, yc = obs['CERES']['rs'], obs['Kernel']['rs'], obs['CERES']['is_']
    ax.annotate('', xy=(rk, yc), xytext=(rc, yc),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3), zorder=8)

ax.set_xlabel(r'RS  [W m$^{-2}$]', fontsize=12)
ax.set_ylabel(r'IS  [K$^{-1}$]', fontsize=12)
ax.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=11)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
          framealpha=0.9, handlelength=1.4, borderpad=0.6, labelspacing=0.45)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'-> {PLOT_OUT}')