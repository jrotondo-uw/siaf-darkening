#!/usr/bin/env python
"""
Supplementary Fig. 6 (was Extended Data Fig. 6) and Supplementary Table 4.

Both are regenerated with sigma_obs/2 as the Gaussian scale (R2 c2), on the
same 21-model / >=5-member ensemble as Fig. 5 so that every PDF-based result
shares one ensemble and one convention.  300 dpi, larger type.
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CMIP6_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
OUT_DIR   = '/glade/derecho/scratch/jrotondo/paper_revision_20261024/figures'
os.makedirs(OUT_DIR, exist_ok=True)

MIN_MEMBERS = 5
TOP_PCT     = 90
SELECT_ON   = 'tot'
DPI         = 300

matplotlib.rcParams.update({
    'font.size':        13,
    'axes.labelsize':   14,
    'axes.titlesize':   15,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  11,
})

COL_EXT, COL_DRK = '#3A6BAF', '#D95F1A'
COL_UW, COL_BMA, COL_VC, COL_OBS = '#999999', '#111111', '#2166ac', '#e8262c'
COL_MDL = ['#7EB2DD', '#F0A860', '#8FBF7F', '#C08BC0']

d = pd.read_csv(CMIP6_CSV)
d = d[d.rs_method == 'full_T07']
cnt = d.groupby('model').size()
d = d[d.model.isin(cnt[cnt >= MIN_MEMBERS].index)]
print(f'pool: {d.model.nunique()} models, {len(d)} members')

o = pd.read_csv(OBS_CSV)
o = o[(o.rs_method == 'CERES') & (o.period == 'CMIP6')
      & (np.isclose(o.pval, 0.05))].iloc[0]

COLS = {'area': 'siaf_ext_global', 'drk': 'siaf_drk_global', 'tot': 'siaf_global'}
# sigma2_* are 2-sigma bootstrap half-widths -> Gaussian scale is half of that
OBS = {
    'area': (float(o.SIAF_ext_global), float(o.sigma2_ext_global) / 2.),
    'drk':  (float(o.SIAF_drk_global), float(o.sigma2_drk_global) / 2.),
    'tot':  (float(o.SIAF_tot_global), float(o.sigma2_tot_global) / 2.),
}


unc=pd.read_csv(os.path.join(os.path.dirname(__file__),'uncertainty.csv')).query("method == 'Common MMM trend' and block_years == 3").set_index('component')
OBS={k:(v[0],float(unc.loc[{'area':'area_loss','drk':'darkening','tot':'total'}[k],'joint_2sigma'])/2) for k,v in OBS.items()}

def norm(x, mu, sd):
    sd = max(float(sd), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def bma_w(g, mu_o, sd_o):
    w = np.array([norm(mu_o, m, np.sqrt(s ** 2 + sd_o ** 2))
                  for m, s in zip(g['mean'], g['std'])])
    return w / w.sum() if w.sum() > 0 else np.ones(len(g)) / len(g)


# ── subset selection ──────────────────────────────────────────────────────
g_sel = d.groupby('model')[COLS[SELECT_ON]].agg(['mean', 'std']).dropna()
w_sel = bma_w(g_sel, *OBS[SELECT_ON])
wmap = dict(zip(g_sel.index, w_sel))
keep = g_sel.index[w_sel >= np.percentile(w_sel, TOP_PCT)].tolist()
pmap = {m: 100 * (w_sel <= wmap[m]).mean() for m in keep}
keep = sorted(keep, key=lambda m: -wmap[m])
CMAP = {m: COL_MDL[j % len(COL_MDL)] for j, m in enumerate(keep)}
sub = d[d.model.isin(keep)]
print(f'subset on SIAF_{SELECT_ON}: {len(keep)} of {len(g_sel)} models, '
      f'{len(sub)} members')
for m in keep:
    print(f'   {m:18s} w={wmap[m]:.3f}  {pmap[m]:.0f}th pct')

fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.5))
fig.patch.set_facecolor('white')

for i, (ax, key) in enumerate(zip(axes, ['area', 'drk', 'tot'])):
    col = COLS[key]
    mu_o, sd_o = OBS[key]
    g = sub.groupby('model')[col].agg(['mean', 'std']).dropna()
    mus, sds = g['mean'].values, g['std'].values

    lo = min(mus.min() - 3 * max(sds.max(), sd_o), mu_o - 4 * sd_o)
    hi = max(mus.max() + 3 * max(sds.max(), sd_o), mu_o + 4 * sd_o)
    x = np.linspace(lo, hi, 1200)

    w = bma_w(g, mu_o, sd_o)
    p_uw = np.mean([norm(x, m, s) for m, s in zip(mus, sds)], axis=0)
    p_bma = np.sum([wi * norm(x, m, s) for wi, m, s in zip(w, mus, sds)], axis=0)
    p_vc = np.mean([norm(x, m, sd_o) for m in mus], axis=0)
    p_obs = norm(x, mu_o, sd_o)

    ax.set_facecolor('white')
    for m in g.index:
        ax.plot(x, norm(x, g.loc[m, 'mean'], g.loc[m, 'std']),
                color=CMAP[m], lw=1.8, zorder=2)
    ax.plot(x, p_uw,  color=COL_UW,  ls='--', lw=2.2, zorder=3)
    ax.plot(x, p_bma, color=COL_BMA, lw=2.6, zorder=4)
    ax.plot(x, p_vc,  color=COL_VC,  lw=2.2, zorder=4)
    ax.plot(x, p_obs, color=COL_OBS, lw=2.6, zorder=5)
    ax.fill_between(x, p_obs, color=COL_OBS, alpha=0.16, zorder=1)
    ax.axvline(mu_o, color=COL_OBS, ls=':', lw=1.6, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Pan-Arctic SIAF (W m$^{-2}$ K$^{-1}$)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(-0.02, 1.10, chr(97 + i), transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='right')
    if key == 'area':
        ax.set_title(r'SIAF$_{\mathrm{area}}$', color=COL_EXT, pad=10)
    elif key == 'drk':
        ax.set_title(r'SIAF$_{\mathrm{drk}}$', color=COL_DRK, pad=10)
    else:
        ax.set_title(r'SIAF$_{\mathrm{area}}$ + SIAF$_{\mathrm{drk}}$',
                     color='#333333', pad=10)

    print(f'\n  {key}: obs={mu_o:.4f}+/-{sd_o:.4f} (1 sigma)  '
          f'subset mean={mus.mean():.4f} ({100*(mu_o-mus.mean())/mu_o:.1f}% low)')

axes[0].set_ylabel('Probability density')
handles = [Line2D([0], [0], color=COL_UW,  ls='--', lw=2.2, label='Unweighted ensemble'),
           Line2D([0], [0], color=COL_BMA, lw=2.6, label='BMA-weighted ensemble'),
           Line2D([0], [0], color=COL_VC,  lw=2.2, label='Variance-constrained'),
           Patch(facecolor=COL_OBS, alpha=0.34, edgecolor=COL_OBS,
                 label='Observed center, model-based 2 SD')]
handles += [Line2D([0], [0], color=CMAP[m], lw=2.0,
                   label=f'{m} (w={wmap[m]:.3f})') for m in keep]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, framealpha=0.95, fontsize=9)

out = os.path.join(OUT_DIR, 'fe06.png')
fig.tight_layout(rect=[0,.12,1,1])
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'\n-> {out}  (dpi={DPI})')

# ======================================================================
# Supplementary Table 4 — BMA weights, same ensemble and convention
# ======================================================================
models = sorted(d.model.unique())
recs = []
for col, tag in (('siaf_ext_global', 'ext'), ('siaf_drk_global', 'drk'),
                 ('siaf_global', 'tot')):
    key = {'ext': 'area', 'drk': 'drk', 'tot': 'tot'}[tag]
    obs_mu, obs_sig = OBS[key]
    stats_list, raw = [], []
    for m in models:
        v = d[d.model == m][col].dropna()
        mu_i = float(v.mean())
        sig_i = float(v.std(ddof=1)) if len(v) > 1 else 0.
        rw = 0. if sig_i <= 0 else float(
            stats.norm.pdf(obs_mu, loc=mu_i,
                           scale=np.sqrt(sig_i ** 2 + obs_sig ** 2)))
        stats_list.append((m, mu_i, sig_i, len(v), rw))
        raw.append(rw)
    tw = sum(raw)
    for (m, mu_i, sig_i, n_mem, rw) in stats_list:
        recs.append(dict(component=tag, model=m, n_members=n_mem,
                         model_mean=round(mu_i, 4),
                         model_sigma=round(sig_i, 4),
                         tebaldi_weight=round(rw / tw if tw > 0 else np.nan, 5)))

res = pd.DataFrame(recs)
csv_out = os.path.join(OUT_DIR, 'siaf_tebaldi_weights_revised.csv')
res.to_csv(csv_out, index=False)

piv = res.pivot_table(index='model', columns='component',
                      values=['n_members', 'model_mean', 'model_sigma',
                              'tebaldi_weight'], aggfunc='first').round(4)
piv.columns = ['_'.join(c) for c in piv.columns]
piv = piv.reset_index().sort_values('tebaldi_weight_tot', ascending=False)

lines = [
    r'\begin{sidewaystable}[htbp]', r'\centering', r'\footnotesize',
    r'\caption{\textbf{Bayesian model averaging weights and model statistics',
    r'    for the CMIP6 SIAF ensemble.} Each row is one CMIP6 model with',
    r'    $\geq ' + str(MIN_MEMBERS) + r'$ ensemble members ($n = ' + str(len(d)) +
    r'$ members from $' + str(len(models)) + r'$ models). $\mu_i$ and',
    r'    $\sigma_i$ are the ensemble mean and standard deviation for each',
    r'    component (W\,m$^{-2}$\,K$^{-1}$); $w_i^{\mathrm{BMA}}$ is the',
    r'    normalized weight \citep{Tebaldi2007UseCMIP} measuring consistency',
    r'    with the observational target, computed with the Gaussian scale',
    r'    $\sigma_{\mathrm{obs}}/2$ (Eq.~\ref{eq:tebaldi_weight}). Models are',
    r'    sorted by descending total $w_i^{\mathrm{BMA}}$.}',
    r'\label{tab:tebaldi_weights}',
    r'\begin{tabular}{l r  r r r  r r r  r r r}', r'\toprule',
    r'& & \multicolumn{3}{c}{Area loss ($\mathrm{SIAF}_{\mathrm{area}}$)}',
    r'  & \multicolumn{3}{c}{Darkening ($\mathrm{SIAF}_{\mathrm{drk}}$)}',
    r'  & \multicolumn{3}{c}{Total ($\mathrm{SIAF}_{\mathrm{tot}}$)} \\',
    r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
    r'Model & $N$ & $\mu_i$ & $\sigma_i$ & $w_i^{\mathrm{BMA}}$'
    r' & $\mu_i$ & $\sigma_i$ & $w_i^{\mathrm{BMA}}$'
    r' & $\mu_i$ & $\sigma_i$ & $w_i^{\mathrm{BMA}}$ \\',
    r'\midrule',
]
for _, r in piv.iterrows():
    name = r['model'].replace('_', r'\_')
    nm = int(r.get('n_members_tot', 0))
    parts = [f'{name} & {nm}']
    for tag in ('ext', 'drk', 'tot'):
        mu = r.get(f'model_mean_{tag}', np.nan)
        sg = r.get(f'model_sigma_{tag}', np.nan)
        wv = r.get(f'tebaldi_weight_{tag}', np.nan)
        parts.append(f'{mu:.3f} & {sg:.3f} & {100*wv:.1f}\\%')
    lines.append(' & '.join(parts) + r' \\')
lines.append(r'\midrule')
obs_row = [r'\textbf{OBS target} & ---']
for key in ('area', 'drk', 'tot'):
    mu, sg = OBS[key]
    obs_row.append(f'\\textbf{{{mu:.3f}}} & {sg:.3f} & ---')
lines.append(' & '.join(obs_row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{sidewaystable}']

tex_out = os.path.join(OUT_DIR, 'si_table4_bma_weights.tex')
open(tex_out, 'w').write('\n'.join(lines) + '\n')
print(f'-> {csv_out}\n-> {tex_out}')
print(piv[['model', 'tebaldi_weight_tot']].head(6).to_string(index=False))
