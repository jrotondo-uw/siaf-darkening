#!/usr/bin/env python
"""
Fig 3 (family-aggregated) — CMIP6 SIAF decomposition by model family.
(a) global-mean SIAF, stacked area-loss + darkening.
(b) LOCAL feedback intensity: mean IS x RS over each component's active
    pixels, i.e. the per-area SIAF with the global normalisation
    (AREA_RATIO x SEASON_FRAC) divided back out. Units are W m-2 K-1 per
    unit area of the active region, so panel b is ~1-2 orders of magnitude
    larger than panel a by construction.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CMIP6_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
PLOT_OUT  = '/glade/work/jrotondo/ncc_paper/figures_final/talk_cmip6_panel_a.png'
MIN_MEMBERS = 3
SCALE = 0.0347 * 0.5          # AREA_RATIO_OCEAN x SEASON_FRAC, as in the pipeline

COL_EXT, COL_DRK, COL_OBS_EDGE, COL_MMM = '#3A6BAF', '#D95F1A', '#b5000a', '#333333'

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

df = pd.read_csv(CMIP6_CSV); df = df[df.rs_method == 'full_T07'].copy()
cnt = df.groupby('model').size()
df = df[df.model.isin(cnt[cnt >= MIN_MEMBERS].index)].copy()
df['family'] = df.model.map(lambda m: FAMILY.get(m, m))

# convert per-area SIAF -> LOCAL intensity
df['ext_loc'] = df.siaf_ext_per_area / SCALE
df['drk_loc'] = df.siaf_drk_per_area / SCALE

mm = df.groupby(['family','model']).agg(
        tot=('siaf_global','mean'), ext=('siaf_ext_global','mean'),
        drk=('siaf_drk_global','mean'),
        ext_loc=('ext_loc','mean'), drk_loc=('drk_loc','mean'))
fam = mm.groupby('family').mean()
spread = df.groupby('family').agg(tot_sd=('siaf_global','std'),
                                  ext_loc_sd=('ext_loc','std'),
                                  drk_loc_sd=('drk_loc','std'))
fam = fam.join(spread)
fam['n_mod'] = mm.groupby('family').size()
fam['n_mem'] = df.groupby('family').size()
fam = fam.sort_values('tot'); fams = list(fam.index)

o = pd.read_csv(OBS_CSV)
o = o[(o.rs_method=='CERES') & (o.period=='CMIP6') & (np.isclose(o.pval,0.05))].iloc[0]
obs = dict(tot=float(o.SIAF_tot_global), ext=float(o.SIAF_ext_global),
           drk=float(o.SIAF_drk_global), err=float(o.sigma2_tot_global),
           ext_loc=float(o.SIAF_ext_per_area)/SCALE,
           drk_loc=float(o.SIAF_drk_per_area)/SCALE,
           ext_loc_e=float(o.sigma2_ext_per_area)/SCALE,
           drk_loc_e=float(o.sigma2_drk_per_area)/SCALE)

mmm      = float(fam.tot.mean());     mmm_2s   = float(2*fam.tot.std(ddof=1))
mmm_e_lc = float(fam.ext_loc.mean()); mmm_e_2s = float(2*fam.ext_loc.std(ddof=1))
mmm_d_lc = float(fam.drk_loc.mean()); mmm_d_2s = float(2*fam.drk_loc.std(ddof=1))

labels = [f'{f}\n({int(fam.n_mod[f])} model{"s" if fam.n_mod[f]>1 else ""}, '
          f'n={int(fam.n_mem[f])})' for f in fams] + ['OBS']
tot = list(fam.tot) + [obs['tot']];  ext = list(fam.ext) + [obs['ext']]
drk = list(fam.drk) + [obs['drk']];  err = list(2*fam.tot_sd.fillna(0)) + [obs['err']]
elc = list(fam.ext_loc) + [obs['ext_loc']]; dlc = list(fam.drk_loc) + [obs['drk_loc']]
ele = list(2*fam.ext_loc_sd.fillna(0)) + [obs['ext_loc_e']]
dle = list(2*fam.drk_loc_sd.fillna(0)) + [obs['drk_loc_e']]

n = len(labels); x = np.arange(n); w = 0.65
_scratch, ax2 = plt.subplots(1, 1)   # panel b drawn here and discarded
fig, ax = plt.subplots(1, 1, figsize=(max(12, n*0.72), 6.2), dpi=300)
fig.patch.set_facecolor('#FAFAFA')
for a in (ax, ax2): a.set_facecolor('#FAFAFA')

ax2.text(-0.05, 1.05, 'b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
ax.set_title('Comparison of Pan-Arctic radiative feedback components across '
             'CMIP6 model families and observations',
             fontsize=10, color='#555555', style='italic', pad=10)
ax2.set_title('Local feedback intensity: SIAF per unit area of each '
              'component\u2019s active region',
              fontsize=9, color='#454545', style='italic', pad=8)

# ---- panel a: global-mean, stacked ----
ax.axhspan(mmm-mmm_2s, mmm+mmm_2s, color=COL_MMM, alpha=0.10, zorder=1)
ax.axhline(mmm, color=COL_MMM, lw=1.8, ls='--', alpha=0.80, zorder=2)
for i in range(n):
    is_obs = (i == n-1)
    edge = COL_OBS_EDGE if is_obs else 'white'; lw = 2.0 if is_obs else 0.8
    hatch = '///' if is_obs else ''
    ax.bar(x[i], ext[i], w, color=COL_EXT, edgecolor=edge, linewidth=lw, hatch=hatch, zorder=3)
    ax.bar(x[i], drk[i], w, bottom=ext[i], color=COL_DRK, edgecolor=edge,
           linewidth=lw, hatch=hatch, zorder=3)
    ax.errorbar(x[i], tot[i], yerr=err[i], fmt='none',
                color=COL_OBS_EDGE if is_obs else '#222222', capsize=5, lw=1.8, zorder=5)
ax.set_ylabel('Pan-Arctic Annual SIAF\n(W m$^{-2}$ K$^{-1}$)', fontsize=12)
ax.legend(handles=[mpatches.Patch(facecolor=COL_EXT, label=r'SIAF$_{area}$'),
                   mpatches.Patch(facecolor=COL_DRK, label=r'SIAF$_{drk}$'),
                   Line2D([0],[0], color=COL_MMM, lw=1.8, ls='--', label='MMM')],
          loc='upper left', fontsize=9.5, framealpha=0.9)

# ---- panel b: LOCAL intensity, grouped ----
w2, gap = 0.28, 0.05
ax2.axhspan(mmm_e_lc-mmm_e_2s, mmm_e_lc+mmm_e_2s, color=COL_EXT, alpha=0.09, zorder=1)
ax2.axhline(mmm_e_lc, color=COL_EXT, lw=1.6, ls='--', alpha=0.80, zorder=2)
ax2.axhspan(mmm_d_lc-mmm_d_2s, mmm_d_lc+mmm_d_2s, color=COL_DRK, alpha=0.09, zorder=1)
ax2.axhline(mmm_d_lc, color=COL_DRK, lw=1.6, ls='--', alpha=0.80, zorder=2)
for i in range(n):
    is_obs = (i == n-1)
    edge = COL_OBS_EDGE if is_obs else 'white'; hatch = '///' if is_obs else ''
    xe, xd = x[i]-w2/2-gap/2, x[i]+w2/2+gap/2
    if np.isfinite(elc[i]):
        ax2.bar(xe, elc[i], w2, color=COL_EXT, edgecolor=edge,
                lw=2.0 if is_obs else 0.8, hatch=hatch, zorder=3)
        ax2.errorbar(xe, elc[i], yerr=ele[i], fmt='none',
                     color=COL_OBS_EDGE if is_obs else '#222222', capsize=4, zorder=5)
    if np.isfinite(dlc[i]):
        ax2.bar(xd, dlc[i], w2, color=COL_DRK, edgecolor=edge,
                lw=2.0 if is_obs else 0.8, hatch=hatch, zorder=3)
        ax2.errorbar(xd, dlc[i], yerr=dle[i], fmt='none',
                     color=COL_OBS_EDGE if is_obs else '#222222', capsize=4, zorder=5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax2.set_ylabel('Local SIAF intensity\n(W m$^{-2}$ K$^{-1}$)', fontsize=12)
ax2.legend(handles=[mpatches.Patch(facecolor=COL_EXT, label=r'SIAF$_{area}$ (local)'),
                    mpatches.Patch(facecolor=COL_DRK, label=r'SIAF$_{drk}$ (local)')],
           loc='upper left', fontsize=9.5, framealpha=0.9)

for a in (ax, ax2):
    a.yaxis.grid(True, alpha=0.35, lw=0.8, zorder=0); a.set_axisbelow(True)
    for s in ('top','right'): a.spines[s].set_visible(False)
fig.savefig(PLOT_OUT, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'-> {PLOT_OUT}')
print(f'   local MMM: area={mmm_e_lc:.2f}  drk={mmm_d_lc:.2f}  '
      f'| obs area={obs["ext_loc"]:.2f} drk={obs["drk_loc"]:.2f} W m-2 K-1')
import matplotlib.pyplot as _plt
_plt.close(_scratch)
fig.savefig('/glade/work/jrotondo/ncc_paper/figures_final/talk_cmip6_panel_a.pdf',
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print('wrote talk_cmip6_panel_a.png and .pdf')
