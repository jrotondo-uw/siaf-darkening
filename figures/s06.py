# %%
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24.9,
    'axes.labelsize': 24.9,
    'axes.titlesize': 29.8,
    'xtick.labelsize': 24.9,
    'ytick.labelsize': 24.9,
    'legend.fontsize': 24.9,
})


# ── Config ────────────────────────────────────────────────────────────
EXP         = 'obs'   # change to 'cmip6' for the other experiment
OUTPUT_ROOT = "/glade/work/jrotondo/j_of_clim_dk_sea_ice_paper/output/albedo_disc/"
_R_FIG      = Path(__file__).resolve().parent
_DEFAULT_PKL = f"{OUTPUT_ROOT}/{EXP}/plot_data_{EXP}.pkl"
PKL_PATH = os.environ.get('SIAF_ROC_PKL', _DEFAULT_PKL)
CSV_CURVES = _R_FIG.parent / 'source_data' / 'SourceData_Supplementary_Figure6_AnnualCurves.csv'

N_SIGMA = 2   # shading half-width, matching the 2-sigma convention used throughout


# ── Load ──────────────────────────────────────────────────────────────
MONTH_ORDER = ['April', 'May', 'June', 'July', 'August', 'September']
if EXP == 'obs' and not os.environ.get('SIAF_ROC_PKL') and CSV_CURVES.exists():
    exp_key = 'obs'
    _cdf = pd.read_csv(CSV_CURVES)
    roc_store = {m: {} for m in MONTH_ORDER}
    roc_edge_store = {m: {} for m in MONTH_ORDER}
    for (dom, mo, prod, yr), g in _cdf.groupby(['domain', 'month', 'product', 'year'], sort=False):
        target = roc_store if dom == 'Full Arctic' else roc_edge_store
        target.setdefault(mo, {}).setdefault(prod, {})[int(yr)] = (
            g['tpr'].to_numpy(float),
            g['fpr'].to_numpy(float),
        )
else:
    with open(PKL_PATH, 'rb') as fh:
        d = pickle.load(fh)
    exp_key        = d['exp_key']
    roc_store      = d['roc_store']
    roc_edge_store = d['roc_edge_store']
    MONTH_ORDER    = d['MONTH_ORDER']


# ── Helpers ───────────────────────────────────────────────────────────
MONTH_COLORS = {
    'April':     '#1f77b4',
    'May':       '#ff7f00',
    'June':      '#2ca02c',
    'July':      '#d62728',
    'August':    '#9467bd',
    'September': '#8c564b',
}
FPR_GRID = np.linspace(0, 1, 500)


FS = 14   # master font size — change one place to rescale everything


def mean_roc(roc_dict):
    """Interpolate per-year curves to common FPR grid, return mean ± std."""
    tprs, aucs = [], []
    for tpr, fpr in roc_dict.values():
        order = np.argsort(fpr)
        tprs.append(np.interp(FPR_GRID, fpr[order], tpr[order]))
        aucs.append(float(np.trapezoid(tpr[order], fpr[order])))
    if not tprs:
        return None
    tprs = np.array(tprs)
    m = np.nanmean(tprs, axis=0)
    s = np.nanstd(tprs,  axis=0)
    i = int(np.argmax(m - FPR_GRID))          # Youden index
    return dict(mean=m, std=s, auc=float(np.mean(aucs)),
                opt_fpr=FPR_GRID[i], opt_tpr=m[i])


# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'Interannual Mean ROC Curves (AMJJAS) — {exp_key.upper()}',
             fontsize=FS + 2, fontweight='bold')


for ax, store, panel_label, title in [
    (axes[0], roc_store,      'a', 'Total ice (full Arctic)'),
    (axes[1], roc_edge_store, 'b', 'Ice edge (MIZ)'),
]:
    leg_handles, leg_labels = [], []


    for month in MONTH_ORDER:
        col = MONTH_COLORS[month]
        for product, ls in [('CLARA-A3', '-'), ('APP-x', '--')]:
            roc_dict = store[month][product]
            if not roc_dict:
                continue
            s = mean_roc(roc_dict)
            if s is None:
                continue

            # Faint individual annual ROC curves behind mean and 2-sigma band
            for tpr, fpr in roc_dict.values():
                order = np.argsort(fpr)
                tpr_i = np.interp(FPR_GRID, fpr[order], tpr[order])
                ax.plot(FPR_GRID, tpr_i, color=col, ls=ls, lw=0.5, alpha=0.14,
                        zorder=1, rasterized=True)

            # +/- N_SIGMA band, clipped to [0, 1] since TPR is bounded
            lo = np.clip(s['mean'] - N_SIGMA * s['std'], 0, 1)
            hi = np.clip(s['mean'] + N_SIGMA * s['std'], 0, 1)
            ax.fill_between(FPR_GRID, lo, hi,
                            color=col, alpha=0.10, linewidth=0, zorder=2)
            line, = ax.plot(FPR_GRID, s['mean'],
                            color=col, ls=ls, lw=1.8, zorder=3,
                            label=f"{month} {product} (AUC={s['auc']:.3f})")
            ax.scatter(s['opt_fpr'], s['opt_tpr'],
                       color=col, marker='o', s=55, zorder=5,
                       edgecolors='k', linewidths=0.5)
            if product == 'CLARA-A3':
                leg_handles.append(line)
                leg_labels.append(f"{month}  (AUC={s['auc']:.3f})")


    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=FS)
    ax.set_ylabel('True Positive Rate', fontsize=FS)
    ax.set_title(f'({panel_label}) {title}', fontsize=FS, loc='left')
    ax.tick_params(labelsize=FS)
    ax.legend(leg_handles, leg_labels,
              prop={'size': FS - 2},
              loc='lower right', framealpha=0.9,
              title='Solid = CLARA-A3 | Dashed = APP-x',
              title_fontsize=FS - 2)


plt.tight_layout()
_out_dir = Path(f"{OUTPUT_ROOT}/{EXP}")
out_png = str(_out_dir / f"roc_curves_{EXP}.png") if _out_dir.exists() else str(Path(__file__).resolve().parent / "figures/s06.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved → {out_png}")