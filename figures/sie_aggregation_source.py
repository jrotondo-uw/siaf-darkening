#!/usr/bin/env python
"""
Extended Data — SIE bias vs SIAF structure, family-grouped.

SIE from sie_df_common.pkl: all 31 models, computed by remapping siconc onto a
common 0.5-degree grid (nearest neighbour) and summing the analytic spherical
cell area where SIC >= 15%. This removes any dependence on model-specific
areacello fields -- unpublished for several models, and unreliable when
reconstructed from cell vertices -- and measures every model identically.
Validated against native-grid areacello values: agreement 0.1-3.7%.

Error bars are +/-2 sigma across ensemble members, matching Figs 3 and 4.
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/sie_comparison/'
CACHE_DIR = os.path.join(PLOT_DIR, 'cache')
OUT_DIR   = os.path.join(PLOT_DIR, 'family')
os.makedirs(OUT_DIR, exist_ok=True)

CMIP6_SIAF_CSV = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_SIAF_CSV   = ('/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
                  'siaf_pixelwise_summary.csv')
SIE_PKL        = '/glade/derecho/scratch/jrotondo/sie_df_common.pkl'   # CHANGED

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
EARLY_YEARS = list(range(1982, 1992))
TREND_YEARS = list(range(1982, 2015))
FOCUS_RS_METHOD, MIN_MEMBERS = 'full_T07', 3
OBS_PERIOD, OBS_RS, OBS_PVAL = 'CMIP6', 'CERES', 0.05
COL_OBS = '#b5000a'

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

def load_cache(name):
    p = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(p, 'rb') as f: obj = pickle.load(f)
    print(f'  [cache] loaded ← {p}')
    return obj

def _style_ax(ax):
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.xaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

def _ols_line(ax, xs, ys, color='#222', ls='--'):
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 3: return None, None
    r, p = stats.pearsonr(xs[ok], ys[ok])
    sl, b = np.polyfit(xs[ok], ys[ok], 1)
    xr_ = np.linspace(xs[ok].min(), xs[ok].max(), 100)
    ax.plot(xr_, sl*xr_ + b, color=color, linestyle=ls, linewidth=1.8,
            label=f'r = {r:.2f}, p = {p:.3f}', zorder=3)
    return r, p

def _season_mean(row, key):
    v = [row.get(f'm{mn}_{key}', np.nan) for mn in MONTH_NUMS]
    f = [x for x in v if np.isfinite(x)]
    return float(np.nanmean(f)) if f else np.nan

def _scatter(ax, xs, ys, labels, colors, x_errs=None, y_errs=None,
             obs_xy=None, obs_err_x=None, obs_err_y=None, obs_label='OBS'):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if x_errs is not None or y_errs is not None:
        xe = np.asarray(x_errs, float) if x_errs is not None else np.zeros_like(xs)
        ye = np.asarray(y_errs, float) if y_errs is not None else np.zeros_like(ys)
        for xi, yi, xei, yei, c in zip(xs, ys, xe, ye, colors):
            ax.errorbar(xi, yi,
                        xerr=float(xei) if np.isfinite(xei) and xei > 0 else None,
                        yerr=float(yei) if np.isfinite(yei) and yei > 0 else None,
                        fmt='none', color=c, alpha=0.55, linewidth=1.1,
                        capsize=3.0, capthick=1.1, zorder=2)
    ax.scatter(xs, ys, c=colors, s=95, edgecolors='white', linewidths=0.8, zorder=4)
    for xi, yi, l in zip(xs, ys, labels):
        ax.annotate(l, (xi, yi), textcoords='offset points', xytext=(4,3),
                    fontsize=7.5, zorder=5)
    _ols_line(ax, xs, ys)
    if obs_xy is not None:
        ox, oy = obs_xy
        if np.isfinite(ox) and np.isfinite(oy):
            ax.scatter([ox],[oy], marker='*', s=280, c=COL_OBS,
                       edgecolors='#800000', linewidths=0.8, zorder=7, label=obs_label)
            xv = float(obs_err_x) if obs_err_x and np.isfinite(obs_err_x) else None
            yv = float(obs_err_y) if obs_err_y and np.isfinite(obs_err_y) else None
            if xv or yv:
                ax.errorbar(ox, oy, xerr=xv, yerr=yv, fmt='none', color=COL_OBS,
                            linewidth=1.4, capsize=4, capthick=1.4, zorder=6)

# ── data ──────────────────────────────────────────────────────────────────
sie_df = pd.read_pickle(SIE_PKL)
print(f'SIE: {len(sie_df)} members / {sie_df.model.nunique()} models')
if 'area_source' in sie_df:
    print('  areas:', sie_df.area_source.value_counts().to_dict())
obs_sie_by_month = load_cache('obs_sie_by_month')

o = pd.read_csv(OBS_SIAF_CSV)
row = o[(o.period == OBS_PERIOD) & (o.rs_method == OBS_RS) &
        np.isclose(o.pval, OBS_PVAL)].iloc[0]
OBS_TOT, OBS_EXT = float(row.SIAF_tot_global), float(row.SIAF_ext_global)
OBS_ERR = float(row.sigma2_tot_global)          # CHANGED: already 2-sigma
obs_prop = OBS_EXT / OBS_TOT
obs_prop_err = OBS_ERR / OBS_TOT * obs_prop

s = pd.read_csv(CMIP6_SIAF_CSV); s = s[s.rs_method == FOCUS_RS_METHOD]
c = s.groupby('model').size(); s = s[s.model.isin(c[c >= MIN_MEMBERS].index)]

msie = {}
for m in sie_df.model.unique():
    sub = sie_df[sie_df.model == m]
    msie[m] = dict(trend=float(np.nanmean([_season_mean(r,'trend_slope')
                                           for _, r in sub.iterrows()])),
                   early=float(np.nanmean([_season_mean(r,'early_mean')
                                           for _, r in sub.iterrows()])))
msiaf = {}
for m in s.model.unique():
    sub = s[s.model == m]
    sv = sub.siaf_global.values.astype(float); ev = sub.siaf_ext_global.values.astype(float)
    pr = np.clip(np.where(sv > 0, ev/sv, np.nan), 0, 1)
    msiaf[m] = dict(siaf=float(np.nanmean(sv)), prop=float(np.nanmean(pr)), n=len(sub))

common  = sorted(set(msie) & set(msiaf))
excluded = sorted(set(msiaf) - set(msie))
print(f'{len(common)} models with SIE'
      + (f'; {len(excluded)} without: {excluded}' if excluded else '; none missing'))

md = pd.DataFrame([dict(model=m, family=FAMILY.get(m, m), n=msiaf[m]['n'],
                        trend=msie[m]['trend'], early=msie[m]['early'],
                        siaf=msiaf[m]['siaf'], prop=msiaf[m]['prop'])
                   for m in common])

# family means from model means (one vote per model)
fam = md.groupby('family')[['trend','early','siaf','prop']].mean()
fam['n_mod'] = md.groupby('family').size()
fam['n_mem'] = md.groupby('family')['n'].sum()

# CHANGED: spread across MEMBERS (2 sigma), so every family gets an error bar
sie_mem = sie_df[sie_df.model.isin(common)].copy()
sie_mem['trend'] = [_season_mean(r, 'trend_slope') for _, r in sie_mem.iterrows()]
sie_mem['early'] = [_season_mean(r, 'early_mean')  for _, r in sie_mem.iterrows()]
sie_mem['family'] = sie_mem.model.map(lambda m: FAMILY.get(m, m))

siaf_mem = s[s.model.isin(common)].copy()
siaf_mem['prop'] = np.clip(np.where(siaf_mem.siaf_global > 0,
                                    siaf_mem.siaf_ext_global / siaf_mem.siaf_global,
                                    np.nan), 0, 1)
siaf_mem['family'] = siaf_mem.model.map(lambda m: FAMILY.get(m, m))

fam_sd = pd.DataFrame(index=fam.index)
fam_sd['trend'] = 2 * sie_mem.groupby('family')['trend'].std()
fam_sd['early'] = 2 * sie_mem.groupby('family')['early'].std()
fam_sd['siaf']  = 2 * siaf_mem.groupby('family')['siaf_global'].std()
fam_sd['prop']  = 2 * siaf_mem.groupby('family')['prop'].std()
fam_sd = fam_sd.fillna(0.)

fam = fam.sort_values('siaf'); fam_sd = fam_sd.reindex(fam.index)
print(f'{len(fam)} families')

# ── OBS SIE ───────────────────────────────────────────────────────────────
sl_, sl_e, ea_, ea_e = [], [], [], []
for mn in MONTH_NUMS:
    ser = obs_sie_by_month.get(mn)
    if ser is None or len(ser) < 5: continue
    st = ser[(ser.index >= TREND_YEARS[0]) & (ser.index <= TREND_YEARS[-1])].dropna()
    if len(st) < 5: continue
    res = stats.linregress(st.index.values.astype(float), st.values)
    sl_.append(float(res.slope)); sl_e.append(float(res.stderr))
    se = ser[ser.index.isin(EARLY_YEARS)].dropna()
    if len(se):
        ea_.append(float(se.mean()))
        ea_e.append(float(se.std(ddof=1)) if len(se) > 1 else 0.)
obs_slope, obs_slope_err = np.nanmean(sl_), np.nanmean(sl_e)
obs_early, obs_early_err = np.nanmean(ea_), np.nanmean(ea_e)
print(f'OBS SIE slope {obs_slope:+.4f} ± {obs_slope_err:.4f}, '
      f'early {obs_early:.3f} ± {obs_early_err:.3f}')

lbls = list(fam.index)
pal  = list(plt.get_cmap('tab20').colors)
cols = [pal[i % len(pal)] for i in range(len(lbls))]
SLOPE_LABEL = 'SIE trend  (10⁶ km² yr⁻¹)'
EARLY_LABEL = 'Early-period SIE, 1982–1991  (10⁶ km²)'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')
specs = [
 (axes[0,0], 'trend','siaf', SLOPE_LABEL, 'Pan-Arctic SIAF  (W m⁻² K⁻¹)',
  'SIE Trend vs SIAF', (obs_slope, OBS_TOT), obs_slope_err, OBS_ERR, None),
 (axes[0,1], 'early','siaf', EARLY_LABEL, 'Pan-Arctic SIAF  (W m⁻² K⁻¹)',
  'Early SIE vs SIAF', (obs_early, OBS_TOT), obs_early_err, OBS_ERR, None),
 (axes[1,0], 'trend','prop', SLOPE_LABEL, r'Pan-Arctic SIAF$_{\mathrm{area}}$ Proportion',
  r'SIE Trend vs SIAF$_{\mathrm{area}}$ Proportion',
  (obs_slope, obs_prop), obs_slope_err, obs_prop_err, 1.0),
 (axes[1,1], 'early','prop', EARLY_LABEL, r'Pan-Arctic SIAF$_{\mathrm{area}}$ Proportion',
  r'Early SIE vs SIAF$_{\mathrm{area}}$ Proportion',
  (obs_early, obs_prop), obs_early_err, obs_prop_err, 1.0),
]
for i, (ax, xk, yk, xl, yl, ttl, oxy, oex, oey, top) in enumerate(specs):
    ax.set_facecolor('#FAFAFA')
    _scatter(ax, fam[xk].values, fam[yk].values, lbls, cols,
             x_errs=fam_sd[xk].values, y_errs=fam_sd[yk].values,
             obs_xy=oxy, obs_err_x=oex, obs_err_y=oey, obs_label='OBS')
    ax.set_xlabel(xl, fontsize=14); ax.set_ylabel(yl, fontsize=14)
    ax.set_title(ttl, fontsize=14, fontweight='bold', pad=8)
    ax.set_title(f'{chr(97+i)})', loc='left', fontsize=16, fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.85); _style_ax(ax)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if top is not None: ax.set_ylim(top=top)
plt.tight_layout(pad=2.0, w_pad=3.5, h_pad=4.0)
out = os.path.join(OUT_DIR, 'ed_sie_siaf_family.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f'  → {out}')

print('\n=== stats for the Results paragraph ===')
for xk, yk, lab in (('trend','siaf','SIE trend vs SIAF_tot'),
                    ('early','siaf','early SIE vs SIAF_tot'),
                    ('trend','prop','SIE trend vs area prop'),
                    ('early','prop','early SIE vs area prop')):
    r, p = stats.pearsonr(fam[xk], fam[yk])
    print(f'  {lab:24s} r={r:+.3f}  r2={r**2:.3f}  p={p:.4f}')