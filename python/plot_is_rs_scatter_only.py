#!/usr/bin/env python
"""
plot_is_rs_scatter_only.py
───────────────────────────
Single-panel IS vs RS scatter (full >=60N ocean domain) for CMIP6 members,
with the observational SIAF +/- 2 sigma band and BOTH observational RS methods
(CERES APRP and ERA5 kernel) shown as separate targets. The two methods share
an identical IS but differ in full-domain RS (they diverge over open water,
where IS = 0 and the SIAF is unaffected); this figure makes that explicit.

OBS values are loaded from siaf_pixelwise_summary.csv (CMIP6 / p=0.05 rows).
CMIP6 members use APRP RS only. Nothing is hardcoded.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR_CMIP6_CSV = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/siaf_cmip6_summary.csv'
DIR_OBS_CSV   = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
PLOT_OUT      = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/siaf_is_rs_scatter.png'

SCALE = 0.0347 * 0.5   # must match siaf_obs_analysis.py

MODEL_COLORS = {
    'CESM2':'#1f77b4','E3SM-1-0':'#ff7f0e','MPI-ESM1-2-HR':'#2ca02c',
    'MPI-ESM1-2-LR':'#98df8a','IPSL-CM6A-LR':'#d62728','MIROC6':'#9467bd',
    'CanESM5':'#8c564b','ACCESS-ESM1-5':'#e377c2','GFDL-ESM4':'#7f7f7f',
    'UKESM1-0-LL':'#17becf','HadGEM3-GC31-LL':'#aec7e8','CNRM-CM6-1':'#ffbb78',
    'BCC-CSM2-MR':'#ff9896',
}
KNOWN_PROXY_MODELS = {'MPI-ESM1-2-HR','MPI-ESM1-2-LR','CanESM5',
                      'ACCESS-ESM1-5','GFDL-ESM4','UKESM1-0-LL','HadGEM3-GC31-LL'}

# observational RS-method styling
OBS_STYLE = {
    'CERES':  dict(color='#b5000a', marker='*', ms=19,
                   label='Obs · CERES APRP ($\\pm2\\sigma$)'),
    'Kernel': dict(color='#5e3c99', marker='P', ms=13,
                   label='Obs · ERA5 kernel ($\\pm2\\sigma$)'),
}


def load_obs_row(df_obs, method):
    """One CMIP6-period, p=0.05 observational row for the given RS method."""
    m = df_obs[(df_obs['rs_method'] == method) &
               (df_obs['period'] == 'CMIP6') &
               (np.isclose(df_obs['pval'], 0.05))]
    if m.empty:
        print(f"WARNING: no OBS row for rs_method={method}")
        return None
    r = m.iloc[0]
    o = dict(siaf=float(r['SIAF_tot_global']), siaf_2s=float(r['sigma2_tot_global']),
             is_=float(r['IS_tot_arctic']),   is_2s=float(r['sigma2_IS_arctic']),
             rs=float(r['RS_arctic']),         rs_2s=float(r['sigma2_RS_arctic']))
    print(f"OBS {method:7s}:  SIAF={o['siaf']:.4f}+/-{o['siaf_2s']:.4f}  "
          f"IS={o['is_']:.4f}+/-{o['is_2s']:.4f}  RS={o['rs']:.2f}+/-{o['rs_2s']:.2f}")
    return o


def draw_siaf_band(ax, rs_range, siaf_lo, siaf_hi, scale,
                   color='#2166ac', alpha=0.18, n=400):
    rs_vals = np.linspace(rs_range[0], rs_range[1], n)
    lo = siaf_lo / (rs_vals * scale)
    hi = siaf_hi / (rs_vals * scale)
    ax.fill_between(rs_vals, lo, hi, color=color, alpha=alpha, zorder=1,
                    label='Obs SIAF $\\pm2\\sigma$ (both RS methods)')
    for curve in (lo, hi):
        ax.plot(rs_vals, curve, color=color, lw=1.0,
                linestyle=(0, (4, 3)), alpha=0.6, zorder=2)
    mid = (siaf_lo + siaf_hi) / 2.
    ax.plot(rs_vals, mid / (rs_vals * scale), color=color, lw=1.4,
            linestyle='--', alpha=0.8, zorder=2)


def main():
    if not os.path.exists(DIR_CMIP6_CSV):
        print(f"ERROR: Cannot find {DIR_CMIP6_CSV}"); return

    df      = pd.read_csv(DIR_CMIP6_CSV)
    df_obs  = pd.read_csv(DIR_OBS_CSV)
    obs     = {m: load_obs_row(df_obs, m) for m in ('CERES', 'Kernel')}
    o_prim  = obs['CERES']                     # SIAF band from CERES (methods agree on SIAF)

    # ── filter proxy members, require >= 3 members ────────────────────────────
    df['_proxy'] = (df['rs_method'] == 'rsds_proxy') if 'rs_method' in df.columns \
                   else df['model'].isin(KNOWN_PROXY_MODELS)
    df = df[~df['_proxy']].copy()
    counts       = df['model'].value_counts()
    valid_models = counts[counts >= 3].index.tolist()
    df           = df[df['model'].isin(valid_models)].copy()
    print(f"\nRetained {len(valid_models)} models (>=3 full/approx T07 members)")

    # ── equal-weight MMM ──────────────────────────────────────────────────────
    rs_means = [df[df.model == m]['rs_arctic'].mean() for m in valid_models]
    is_means = [df[df.model == m]['is_tot_arctic'].mean() for m in valid_models]
    rs_means = [v for v in rs_means if np.isfinite(v)]
    is_means = [v for v in is_means if np.isfinite(v)]
    MMM_RS, MMM_IS = np.mean(rs_means), np.mean(is_means)
    MMM_RS_2s = 2. * np.std(rs_means, ddof=1)
    MMM_IS_2s = 2. * np.std(is_means, ddof=1)
    print(f"MMM RS={MMM_RS:.2f}+/-{MMM_RS_2s:.2f}   MMM IS={MMM_IS:.4f}+/-{MMM_IS_2s:.4f}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    all_rs = df['rs_arctic'].dropna()
    rs_lo  = min(max(all_rs.min() - 5., 80.),  MMM_RS - MMM_RS_2s - 5.)
    rs_hi  = max(min(all_rs.max() + 5., 220.), MMM_RS + MMM_RS_2s + 5.)
    # make sure both obs RS targets are in view
    for o in obs.values():
        if o:
            rs_lo = min(rs_lo, o['rs'] - o['rs_2s'] - 5.)
            rs_hi = max(rs_hi, o['rs'] + o['rs_2s'] + 5.)
    rs_range = (rs_lo, rs_hi)

    if o_prim:
        draw_siaf_band(ax, rs_range, o_prim['siaf'] - o_prim['siaf_2s'],
                       o_prim['siaf'] + o_prim['siaf_2s'], SCALE)
        # shared IS target band (IS is identical for both methods)
        ax.axhspan(o_prim['is_'] - o_prim['is_2s'], o_prim['is_'] + o_prim['is_2s'],
                   color='#888888', alpha=0.08, zorder=1)

    # CMIP6 members (APRP)
    for m in df['model'].unique():
        sub = df[df.model == m]
        ax.scatter(sub['rs_arctic'], sub['is_tot_arctic'],
                   color=MODEL_COLORS.get(m, '#888888'), alpha=0.85, marker='o',
                   edgecolors='white', linewidths=0.5, s=70, zorder=4, label=m)

    # MMM (APRP)
    ax.errorbar(MMM_RS, MMM_IS, xerr=MMM_RS_2s, yerr=MMM_IS_2s, fmt='none',
                color='#222222', capsize=5, linewidth=1.6, capthick=1.6, zorder=7)
    ax.scatter(MMM_RS, MMM_IS, marker='D', color='#222222', s=70,
               edgecolors='white', linewidths=0.8, zorder=8, label='MMM ($\\pm2\\sigma$)')

    # both observational RS methods (same IS, different full-domain RS)
    for method in ('CERES', 'Kernel'):
        o = obs[method]
        if o is None:
            continue
        st = OBS_STYLE[method]
        ax.errorbar(o['rs'], o['is_'], xerr=o['rs_2s'], yerr=o['is_2s'],
                    fmt=st['marker'], color=st['color'], markersize=st['ms'],
                    markeredgecolor='white', markeredgewidth=0.9,
                    ecolor=st['color'], elinewidth=1.6, capsize=4, capthick=1.6,
                    zorder=9, label=st['label'])
    # connector + label for the full-domain RS disagreement (lifted clear of points)
    if obs['CERES'] and obs['Kernel']:
        rc, rk, yc = obs['CERES']['rs'], obs['Kernel']['rs'], obs['CERES']['is_']
        ax.annotate('', xy=(rk, yc), xytext=(rc, yc),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3), zorder=8)
        ax.text(0.5*(rc+rk), yc+0.020,
                f"full-domain RS gap: {abs(rk-rc):.0f} W m$^{{-2}}$",
                ha='center', va='bottom', fontsize=8.5, color='#222',
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#999', lw=0.6, alpha=0.92),
                zorder=11)

    ax.set_xlim(rs_range)
    ax.set_xlabel(r'RS  [W m$^{-2}$]  (full $\geq\!60^\circ$N ocean domain)', fontsize=11)
    ax.set_ylabel(r'IS  [K$^{-1}$]', fontsize=11)
    ax.set_title('Member distribution: IS vs RS', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.tick_params(axis='both', labelsize=12)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8,
              framealpha=0.9, handlelength=1.4, borderpad=0.7, labelspacing=0.55)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nPlot saved -> {PLOT_OUT}')


if __name__ == '__main__':
    main()
