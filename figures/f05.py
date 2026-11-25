"""f05.py -- Figure 5, four panels.

a,b  Total SIAF against the family SIE trend and the early-period SIE.
     Unchanged from the two-panel version; panel a is the retreat RATE and
     panel b the initial extent, which is the order the running text now uses.
c,d  The concentration-adjusted derivatives, restored from the 12 Nov 2026
     historical-only rerun.  The two-panel file that was delivered by mistake
     is kept as f05_twopanel_superseded.py.

Provenance.  Only the OBSERVATIONS were rerun on 12 Nov: the archived
observational bivariate CSV covered 1982-2024 while the models cover
1982-2014, so the comparison was mixed-period.  The model side is unchanged.
Do NOT fall back to ncc_paper/plots/bivariate_diag/stage3_bivariate.csv for
observations -- that is the superseded mixed-period file.  The three inputs
below are local copies, taken because two of the originals sit on purgeable
scratch:
  sdfilter_test_20261123/obs_nofilter/stage3_bivariate.csv    -> f05_cd_obs_stage3.csv
                                                     (and f05_cd_panels.csv)
  fig5_uncertainty_nofilter_20261123/uncertainty_summary_direct_nofilter.csv
                                                              -> f05_cd_obs_sigma.csv
  fig5_rerun31_patched_20261123/model_bivariate_by_month_nofilter.csv
                                                              -> f05_cd_models.csv
    (23 Nov 2026: one retained member from each of the 31 headline models,
    grid_patch applied, and NO SIC-sigma filter -- Joey dropped it the same
    day; restage_f05_nofilter.py did the swap. The filtered tables are kept as
    *.filtered_superseded and the 24-model ones as *.24model_superseded.)

Conventions.  Every interval is written as sigma, never "s.d.".  Model spread
in c-d is +/-2 sigma across models, matching the manuscript caption; the
observational interval is 2 sigma from observed-minus-MMM residual block resampling,
pooled member -> model -> family as in that run's summarize.py.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
OUT = R / 'figures'
plt.rcParams.update({'font.size': 16, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

OBS_C = '#b5000a'      # observations, reserved paper-wide
MOD_C = '#444444'      # CMIP6, reserved paper-wide
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']
MLAB = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# ---------------------------------------------------------------- a, b
# Reuse the saved aggregation without its plotting section; data and
# weighting unchanged.
src = (R / 'sie_aggregation_source.py').read_text()
src = src[:src.index('lbls = list(fam.index)')]
src = src.replace("os.makedirs(OUT_DIR, exist_ok=True)", '')
src = (src.replace("pd.read_pickle(SIE_PKL)", "pd.read_pickle(R/'sie_df_common.pkl')")
          .replace("load_cache('obs_sie_by_month')", "pd.read_pickle(R/'obs_sie_by_month.pkl')")
          .replace('pd.read_csv(OBS_SIAF_CSV)', "pd.read_csv(R/'siaf_pixelwise_summary.csv')")
          .replace('pd.read_csv(CMIP6_SIAF_CSV)', "pd.read_csv(R/'siaf_cmip6_summary_final.csv')"))
exec(compile(src, 'saved_sie_aggregation', 'exec'))
fam.to_csv(OUT / 'f05_family_means.csv')
fam_sd.to_csv(OUT / 'f05_family_spread.csv')

mem_ab = pd.merge(
    siaf_mem[['model', 'member', 'siaf_global']],
    sie_mem[['model', 'member', 'trend', 'early']],
    on=['model', 'member'], how='inner'
)
assert len(mem_ab) == 378 and mem_ab.model.nunique() == 31

# Reconstruct the 10,000 historical total-SIAF residual-bootstrap draws
AN_DIR = R.parent / 'analysis_data' / 'observed_uncertainty_20261125'
_m_tr = pd.read_csv(AN_DIR / 'SIAF_model_member_calendar_trends.csv')
_b_tot = float(_m_tr.groupby(['family', 'model'])['total'].mean().groupby('family').mean().mean())
_a_c = pd.read_csv(AN_DIR / 'advisor_annual_CERES.csv').sort_values('year')

def _block_boot_slopes(y, T, year, b, L=3, B=10000):
    y = np.asarray(y, dtype=float); y = y - y.mean()
    T = np.asarray(T, dtype=float); T = T - T.mean()
    t = np.asarray(year, dtype=float); t = t - t.mean()
    n = len(t)
    res = y - float(b) * t
    starts = np.random.default_rng(20261023 + L).integers(0, n, size=(B, int(np.ceil(n / L))))
    idx = ((starts[:, :, None] + np.arange(L)) % n).reshape(B, -1)[:, :n]
    return res[idx] @ T / (T @ T)

_sl_tot = _block_boot_slopes(_a_c.total, _a_c.temperature, _a_c.year, _b_tot)
obs_tot_draws = OBS_TOT + (_sl_tot - _sl_tot.mean())
assert np.isclose(2 * obs_tot_draws.std(ddof=1), 0.0537936108477549, atol=1e-12)

# ---------------------------------------------------------------- c, d
cd = pd.read_csv(R / 'f05_cd_panels.csv')
sig = pd.read_csv(R / 'f05_cd_obs_sigma.csv')
mem = pd.read_csv(R / 'f05_cd_models.csv')
obs_cd_ser = pd.read_csv(AN_DIR / 'observed_Fig5_series.csv')
sig['month_name'] = sig.month.map(dict(zip(range(4, 10), MONTHS)))

fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
x = np.arange(6)
rng_ab = np.random.default_rng(20261125)
ab_ymin = min(-0.012, float(mem_ab.siaf_global.min()) - 0.008, float(obs_tot_draws.min()) - 0.008)
ab_ymax = max(0.315, float(mem_ab.siaf_global.max()) + 0.012, float(obs_tot_draws.max()) + 0.012)

for i, ax in enumerate(axes[0]):
    k, ox, xlab, title, ojit = [
        ('trend', obs_slope, 'SIE trend (10$^6$ km$^2$ yr$^{-1}$)', 'Retreat rate', 0.0028),
        ('early', obs_early, '1982–1991 SIE (10$^6$ km$^2$)', 'Initial extent', 0.20),
    ][i]
    ax.scatter(mem_ab[k], mem_ab.siaf_global, s=11, c='0.62', alpha=0.40, lw=0,
               zorder=1, rasterized=True)
    ax.errorbar(fam[k], fam.siaf, xerr=fam_sd[k], yerr=fam_sd.siaf,
                fmt='none', ecolor=MOD_C, alpha=.23, lw=.8, zorder=2)
    ax.scatter(fam[k], fam.siaf, c=MOD_C, s=36, edgecolors='white', linewidths=0.5,
               zorder=3)
    r, p = stats.pearsonr(fam[k], fam.siaf)
    sl, b = np.polyfit(fam[k], fam.siaf, 1)
    xr = np.linspace(fam[k].min(), fam[k].max(), 100)
    ax.plot(xr, sl * xr + b, color=MOD_C, ls='--', lw=1.2, zorder=3.5)
    ax.scatter(ox + rng_ab.uniform(-ojit, ojit, len(obs_tot_draws)), obs_tot_draws,
               s=5, c=OBS_C, alpha=0.025, lw=0, zorder=2.8, rasterized=True)
    ax.errorbar(ox, OBS_TOT, yerr=np.float64(0.0537936108477549), fmt='*', color=OBS_C,
                markersize=15, markeredgecolor='white', markeredgewidth=0.7,
                capsize=4, elinewidth=1.8, capthick=1.6, zorder=5)
    ax.text(.04, .95, f'r = {r:.2f}\np = {p:.3f}'.replace('-', '\u2212'),
            va='top', transform=ax.transAxes)
    ax.set_title(chr(97 + i) + '  ' + title, loc='left', weight='bold', fontsize=15)
    ax.set_xlabel(xlab)
    ax.set_ylim(ab_ymin, ab_ymax)
axes[0, 0].set_xticks([-.15, -.10, -.05, 0])
axes[0, 0].set_ylabel('Total SIAF (W m$^{-2}$ K$^{-1}$)')

SPEC = [('area_loss', 'L', '$-$dSIC/dT (K$^{-1}$)',
         'Concentration loss per K', 'Area-loss region'),
        ('darkening', 'I', r'$-$d$\alpha$/d$T$ (K$^{-1}$)',
         'Albedo loss per K', 'Darkening region')]

for i, (ax, (region, var, ylab, title, where)) in enumerate(zip(axes[1], SPEC)):
    sub = cd[cd.region == region].set_index('month').loc[MONTHS]
    ss = sig[sig.region == region].set_index('month_name').loc[MONTHS]
    rng = np.random.default_rng(0)
    rng_obs = np.random.default_rng(100 + i)
    for j, m in enumerate(MONTHS):
        v = mem[(mem.region == region) & (mem.month == m)][var].astype(float).dropna()
        ax.scatter(j + rng.uniform(-.14, .14, len(v)), v, s=14, c='0.6',
                   alpha=.45, lw=0, zorder=1)
        z_obs = obs_cd_ser[(obs_cd_ser.component == region) & (obs_cd_ser.month == j + 4)].sort_values('year')
        sl_m = _block_boot_slopes(z_obs.anomaly, z_obs.temperature, z_obs.year, z_obs.MMM_trend.iloc[0])
        assert np.isclose(2 * sl_m.std(ddof=1), float(ss.loc[m, 'halfwidth_2sigma']), atol=1e-12)
        obs_m_draws = float(sub.loc[m, 'observed']) + (sl_m - sl_m.mean())
        ax.scatter(j + rng_obs.uniform(-.06, .06, len(obs_m_draws)), obs_m_draws,
                   s=5, c=OBS_C, alpha=0.025, lw=0, zorder=2, rasterized=True)
    ax.errorbar(x, sub.model_mean, yerr=2 * sub.model_sd, fmt='o-', color=MOD_C,
                lw=1.6, markersize=7, capsize=4, zorder=3)
    ax.errorbar(x, sub.observed, yerr=ss.halfwidth_2sigma, fmt='D-', color=OBS_C,
                lw=2.0, markersize=7.5, markeredgecolor='white', markeredgewidth=0.7,
                capsize=4, elinewidth=1.8, capthick=1.6, zorder=4)
    ax.axhline(0, color='0.5', lw=.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(MLAB)
    ax.set_xlim(-.5, 5.5)
    ax.set_ylabel(ylab)
    ax.set_title(chr(99 + i) + '  ' + title, loc='left', weight='bold', fontsize=15)
    ax.text(.03, .95, where, va='top', transform=ax.transAxes,
            color='0.35', weight='bold', fontsize=13)
    ax.set_xlabel('Month')

for ax in axes.ravel():
    ax.grid(color='.92')
    ax.set_axisbelow(True)

# Legends sit inside the panels: a top-of-figure legend costs vertical room and
# forces the type down, and shorter strings are the only lever on on-page size.
leg_ab = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=MOD_C, markeredgecolor='white',
           markeredgewidth=0.5, markersize=6.5, label='CMIP6 family mean'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='0.55', markeredgecolor='none',
           markersize=4.5, alpha=0.65, label='CMIP6 member'),
    Line2D([0], [0], marker='*', color=OBS_C, markerfacecolor=OBS_C, markeredgecolor='white',
           markeredgewidth=0.5, markersize=12, lw=1.6, label='Observed, 1982–2014'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=OBS_C, markeredgecolor='none',
           markersize=4.5, alpha=0.45, label='Obs. residual bootstrap'),
]
axes[0, 0].legend(handles=leg_ab, fontsize=11, framealpha=.92, facecolor='white', loc='lower left')
leg_cd = [
    Line2D([0], [0], marker='o', color=MOD_C, markerfacecolor=MOD_C, markersize=7, lw=1.6,
           label='CMIP6 model mean'),
    Line2D([0], [0], marker='D', color=OBS_C, markerfacecolor=OBS_C, markeredgecolor='white',
           markeredgewidth=0.6, markersize=7, lw=2.0, label='Observed'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='0.55', markeredgecolor='none',
           markersize=4.5, alpha=0.65, label='CMIP6 model'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=OBS_C, markeredgecolor='none',
           markersize=4.5, alpha=0.45, label='Obs. residual bootstrap'),
]
axes[1, 0].legend(handles=leg_cd, fontsize=11, framealpha=.92, facecolor='white', loc='upper left',
                  bbox_to_anchor=(0, .88))
# No footer: the editor does not allow text in a figure file that is not part
# of the figure. Period, member, weighting and interval definitions are all
# in the caption.
fig.subplots_adjust(top=.955, bottom=.085, left=.093, right=.985,
                    wspace=.28, hspace=.40)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'f05.{ext}', dpi=300)

chk = cd[(cd.region == 'darkening') & (cd.month == 'May')].iloc[0]
s5 = float(sig[(sig.region == 'darkening') & (sig.month == 5)].halfwidth_2sigma.iloc[0])
print(f'May darkening: obs {chk.observed:.4f} +/- {s5:.4f}, model {chk.model_mean:.4f} '
      f'(direct -dalpha/dT, 1982–2014, corrected spatial weighting)')
print('wrote', OUT / 'f05.png')
