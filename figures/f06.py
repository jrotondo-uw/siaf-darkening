from pathlib import Path
#!/usr/bin/env python
"""
Figure 3 (revision) — CMIP6 SIAF decomposition by model family.

Changes from the submitted version:
  * individual ensemble members overlaid on every bar, as the editor requires
    for any bar chart showing a mean with error bars; the bar form is kept
    because the stacked area/darkening partition is the point of the figure
  * observational bootstrap replicates overlaid on the OBS bar when available
  * larger type throughout, 300 dpi
  * error bars explicitly identified as internal variability (2 s.d. across
    members), with n reported per family on the axis labels
Everything else - families, ordering, colours, normalisation - is unchanged.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CMIP6_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
BOOT_NPZ  = '/glade/derecho/scratch/jrotondo/rev_out/fig1_bootstrap_replicates.npz'
OUT_DIR   = '/glade/derecho/scratch/jrotondo/paper_revision_20261024/figures'
os.makedirs(OUT_DIR, exist_ok=True)
PLOT_OUT  = os.path.join(OUT_DIR, 'f03.png')

MIN_MEMBERS = 3
SCALE = 0.0347 * 0.5
DPI = 300

matplotlib.rcParams.update({
    'font.size':       13,
    'axes.labelsize':  14,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

COL_EXT, COL_DRK, COL_OBS_EDGE, COL_MMM = '#3A6BAF', '#D95F1A', '#b5000a', '#333333'
COL_PT = '#2b2b2b'

FAMILY = {**{m: 'CESM2' for m in ['CESM2', 'CESM2-WACCM', 'CESM2-FV2', 'CESM2-WACCM-FV2']},
          **{m: 'EC-Earth3' for m in ['EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR',
                                      'EC-Earth3-CC', 'EC-Earth3-AerChem', 'EC-Earth3-ESM-1']},
          **{m: 'CMCC' for m in ['CMCC-CM2-SR5', 'CMCC-ESM2', 'CMCC-CM2-HR4']},
          **{m: 'E3SM' for m in ['E3SM-1-0', 'E3SM-1-1', 'E3SM-1-1-ECA']},
          **{m: 'HadGEM3/UKESM' for m in ['HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
                                          'UKESM1-0-LL', 'UKESM1-1-LL']},
          **{m: 'MPI/ICON' for m in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
                                     'MPI-ESM-1-2-HAM', 'ICON-ESM-LR']},
          **{m: 'NorESM/NorCPM' for m in ['NorESM2-LM', 'NorESM2-MM', 'NorCPM1']},
          **{m: 'CNRM' for m in ['CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1']},
          **{m: 'CanESM5' for m in ['CanESM5', 'CanESM5-CanOE']},
          **{m: 'MIROC' for m in ['MIROC6', 'MIROC-ES2L']},
          **{m: 'IPSL' for m in ['IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'IPSL-CM5A2-INCA']},
          **{m: 'INM' for m in ['INM-CM4-8', 'INM-CM5-0']},
          **{m: 'FGOALS' for m in ['FGOALS-g3', 'FGOALS-f3-L']},
          **{m: 'BCC' for m in ['BCC-CSM2-MR', 'BCC-ESM1']},
          **{m: 'ACCESS' for m in ['ACCESS-CM2', 'ACCESS-ESM1-5']},
          **{m: 'GISS' for m in ['GISS-E2-1-H', 'GISS-E2-1-G']}}

df = pd.read_csv(CMIP6_CSV)
df = df[df.rs_method == 'full_T07'].copy()
cnt = df.groupby('model').size()
df = df[df.model.isin(cnt[cnt >= MIN_MEMBERS].index)].copy()
df['family'] = df.model.map(lambda m: FAMILY.get(m, m))
df['ext_loc'] = df.siaf_ext_per_area / SCALE
df['drk_loc'] = df.siaf_drk_per_area / SCALE

mm = df.groupby(['family', 'model']).agg(
    tot=('siaf_global', 'mean'), ext=('siaf_ext_global', 'mean'),
    drk=('siaf_drk_global', 'mean'),
    ext_loc=('ext_loc', 'mean'), drk_loc=('drk_loc', 'mean'))
fam = mm.groupby('family').mean()
spread = df.groupby('family').agg(tot_sd=('siaf_global', 'std'),
                                  ext_loc_sd=('ext_loc', 'std'),
                                  drk_loc_sd=('drk_loc', 'std'))
fam = fam.join(spread)
fam['n_mod'] = mm.groupby('family').size()
fam['n_mem'] = df.groupby('family').size()
fam = fam.sort_values('tot')
fams = list(fam.index)

o = pd.read_csv(OBS_CSV)
o = o[(o.rs_method == 'CERES') & (o.period == 'CMIP6')
      & (np.isclose(o.pval, 0.05))].iloc[0]
unc=pd.read_csv(Path(__file__).parent/'uncertainty.csv').query("method == 'Common MMM trend' and block_years == 3").set_index('component')
obs = dict(tot=float(o.SIAF_tot_global), ext=float(o.SIAF_ext_global),
           drk=float(o.SIAF_drk_global), err=float(unc.loc['total','joint_2sigma']),
           ext_loc=float(o.SIAF_ext_per_area) / SCALE,
           drk_loc=float(o.SIAF_drk_per_area) / SCALE,
           ext_loc_e=float(o.sigma2_ext_per_area) / SCALE,
           drk_loc_e=float(o.sigma2_drk_per_area) / SCALE)

obs_boot = None
if os.path.exists(BOOT_NPZ):
    z = np.load(BOOT_NPZ)
    if 'CERES_CMIP6' in z:
        obs_boot = z['CERES_CMIP6'][:, 0]
        print(f'loaded {obs_boot.size} observational bootstrap replicates')

mmm = float(fam.tot.mean())
mmm_2s = float(2 * fam.tot.std(ddof=1))
mmm_e_lc = float(fam.ext_loc.mean())
mmm_e_2s = float(2 * fam.ext_loc.std(ddof=1))
mmm_d_lc = float(fam.drk_loc.mean())
mmm_d_2s = float(2 * fam.drk_loc.std(ddof=1))

labels = [f'{f}\n({int(fam.n_mod[f])} model{"s" if fam.n_mod[f] > 1 else ""}, '
          f'n={int(fam.n_mem[f])})' for f in fams] + ['Observations\n(1982-2014)']
tot = list(fam.tot) + [obs['tot']]
ext = list(fam.ext) + [obs['ext']]
drk = list(fam.drk) + [obs['drk']]
err = list(2 * fam.tot_sd.fillna(0)) + [obs['err']]
elc = list(fam.ext_loc) + [obs['ext_loc']]
dlc = list(fam.drk_loc) + [obs['drk_loc']]
ele = list(2 * fam.ext_loc_sd.fillna(0)) + [obs['ext_loc_e']]
dle = list(2 * fam.drk_loc_sd.fillna(0)) + [obs['drk_loc_e']]

n = len(labels)
x = np.arange(n)
w = 0.65
rng = np.random.default_rng(7)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(max(13, n * 0.80), 11.5),
                              sharex=True,
                              gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.08})
fig.patch.set_facecolor('white')
for a in (ax, ax2):
    a.set_facecolor('white')

ax.text(-0.05, 1.05, 'a', transform=ax.transAxes, fontsize=18,
        fontweight='bold', va='top')
ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes, fontsize=18,
         fontweight='bold', va='top')

# ---- panel a: global-mean, stacked ----
ax.axhspan(mmm - mmm_2s, mmm + mmm_2s, color=COL_MMM, alpha=0.10, zorder=1)
ax.axhline(mmm, color=COL_MMM, lw=1.8, ls='--', alpha=0.80, zorder=2)
for i in range(n):
    is_obs = (i == n - 1)
    edge = COL_OBS_EDGE if is_obs else 'white'
    lw = 2.0 if is_obs else 0.8
    hatch = '///' if is_obs else ''
    ax.bar(x[i], ext[i], w, color=COL_EXT, edgecolor=edge, linewidth=lw,
           hatch=hatch, zorder=3)
    ax.bar(x[i], drk[i], w, bottom=ext[i], color=COL_DRK, edgecolor=edge,
           linewidth=lw, hatch=hatch, zorder=3)
    ax.errorbar(x[i], tot[i], yerr=err[i], fmt='none',
                color=COL_OBS_EDGE if is_obs else '#222222',
                capsize=5, lw=1.8, zorder=5)
    # individual data points
    if is_obs:
        pts = obs_boot
    else:
        pts = df.loc[df.family == fams[i], 'siaf_global'].values
    if pts is not None and len(pts):
        if len(pts) > 400:
            pts = rng.choice(pts, 400, replace=False)
        jit = rng.uniform(-w * 0.30, w * 0.30, size=len(pts))
        ax.scatter(x[i] + jit, pts, s=9, facecolor=COL_PT, edgecolor='none',
                   alpha=0.45, zorder=6, rasterized=True)

ax.set_ylabel('Pan-Arctic annual SIAF\n(W m$^{-2}$ K$^{-1}$)')
ax.legend(handles=[mpatches.Patch(facecolor=COL_EXT, label=r'SIAF$_{\mathrm{area}}$'),
                   mpatches.Patch(facecolor=COL_DRK, label=r'SIAF$_{\mathrm{drk}}$'),
                   Line2D([0], [0], color=COL_MMM, lw=1.8, ls='--',
                          label='Multi-model mean'),
                   Line2D([0], [0], marker='o', color='none',
                          markerfacecolor=COL_PT, markeredgecolor='none',
                          markersize=5, alpha=0.6,
                          label='Individual members / bootstrap replicates')],
          loc='upper left', framealpha=0.92)

# ---- panel b: local intensity, grouped ----
w2, gap = 0.28, 0.05
ax2.axhspan(mmm_e_lc - mmm_e_2s, mmm_e_lc + mmm_e_2s, color=COL_EXT,
            alpha=0.09, zorder=1)
ax2.axhline(mmm_e_lc, color=COL_EXT, lw=1.6, ls='--', alpha=0.80, zorder=2)
ax2.axhspan(mmm_d_lc - mmm_d_2s, mmm_d_lc + mmm_d_2s, color=COL_DRK,
            alpha=0.09, zorder=1)
ax2.axhline(mmm_d_lc, color=COL_DRK, lw=1.6, ls='--', alpha=0.80, zorder=2)
for i in range(n):
    is_obs = (i == n - 1)
    edge = COL_OBS_EDGE if is_obs else 'white'
    hatch = '///' if is_obs else ''
    xe, xd = x[i] - w2 / 2 - gap / 2, x[i] + w2 / 2 + gap / 2
    if np.isfinite(elc[i]):
        ax2.bar(xe, elc[i], w2, color=COL_EXT, edgecolor=edge,
                lw=2.0 if is_obs else 0.8, hatch=hatch, zorder=3)
        ax2.errorbar(xe, elc[i], yerr=ele[i], fmt='none',
                     color=COL_OBS_EDGE if is_obs else '#222222',
                     capsize=4, zorder=5)
    if np.isfinite(dlc[i]):
        ax2.bar(xd, dlc[i], w2, color=COL_DRK, edgecolor=edge,
                lw=2.0 if is_obs else 0.8, hatch=hatch, zorder=3)
        ax2.errorbar(xd, dlc[i], yerr=dle[i], fmt='none',
                     color=COL_OBS_EDGE if is_obs else '#222222',
                     capsize=4, zorder=5)
    if not is_obs:
        sub = df.loc[df.family == fams[i]]
        for xpos, vals in ((xe, sub.ext_loc.values), (xd, sub.drk_loc.values)):
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            if len(vals) > 400:
                vals = rng.choice(vals, 400, replace=False)
            jit = rng.uniform(-w2 * 0.32, w2 * 0.32, size=len(vals))
            ax2.scatter(xpos + jit, vals, s=7, facecolor=COL_PT,
                        edgecolor='none', alpha=0.40, zorder=6, rasterized=True)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.set_ylabel('Local SIAF intensity\n(W m$^{-2}$ K$^{-1}$)')
ax2.legend(handles=[mpatches.Patch(facecolor=COL_EXT,
                                   label=r'SIAF$_{\mathrm{area}}$ (local)'),
                    mpatches.Patch(facecolor=COL_DRK,
                                   label=r'SIAF$_{\mathrm{drk}}$ (local)')],
           loc='upper left', framealpha=0.92)

for a in (ax, ax2):
    a.yaxis.grid(True, alpha=0.35, lw=0.8, zorder=0)
    a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)

plt.savefig(PLOT_OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'-> {PLOT_OUT}  (dpi={DPI})')
print(f'   families={len(fams)}  members={len(df)}  '
      f'local MMM: area={mmm_e_lc:.2f} drk={mmm_d_lc:.2f}')
