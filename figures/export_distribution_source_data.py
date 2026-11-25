#!/usr/bin/env python
"""Export distributional source-data tables backing revised main and supplementary plots.

Writes compact, self-contained CSVs into source_data/ for every distribution,
member cloud, family mean, reanalysis value, and annual ROC curve added to the
revised figures so figure scripts run offline without binary archives or HPC access:
  - SourceData_Figure2_Bootstrap.csv
  - SourceData_Figure3_Bootstrap.csv
  - SourceData_Figure5_Members.csv
  - SourceData_Figure5_Bootstrap.csv
  - SourceData_Figure6_Bootstrap.csv
  - SourceData_Figure7_FamilyMeans.csv
  - SourceData_Figure7_Bootstrap.csv
  - SourceData_Supplementary_Figure1_Bootstrap.csv
  - SourceData_Supplementary_Figure4_Observed.csv
  - SourceData_Supplementary_Figure4_FamilyMeans.csv
  - SourceData_Supplementary_Figure4_Bootstrap.csv
  - SourceData_Supplementary_Figure6_AnnualCurves.csv
"""
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
REPO = FIG_DIR.parent
SD = REPO / 'source_data'
AN_DIR = REPO / 'analysis_data' / 'observed_uncertainty_20261125'
SCALE = 0.0347 * 0.5
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']

BOOT_NPZ = Path(os.environ.get(
    'SIAF_PAIRS_BOOT_NPZ',
    str(REPO.parents[1] / 'output' / 'fig2_bootstrap_export' / 'all_pairs_bootstrap.npz')
))
ROC_PKL = Path(os.environ.get(
    'SIAF_ROC_PKL',
    str(REPO.parents[1] / 'output' / 'fig2_bootstrap_export' / 'plot_data_obs.pkl')
))

FAMILY = {
    **{m: 'CESM2' for m in ['CESM2', 'CESM2-WACCM', 'CESM2-FV2', 'CESM2-WACCM-FV2']},
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
    **{m: 'GISS' for m in ['GISS-E2-1-H', 'GISS-E2-1-G']},
}


def block_boot_slopes(y, T, year, b, L=3, B=10000):
    y = np.asarray(y, dtype=float); y = y - y.mean()
    T = np.asarray(T, dtype=float); T = T - T.mean()
    t = np.asarray(year, dtype=float); t = t - t.mean()
    n = len(t)
    res = y - float(b) * t
    starts = np.random.default_rng(20261023 + L).integers(0, n, size=(B, int(np.ceil(n / L))))
    idx = ((starts[:, :, None] + np.arange(L)) % n).reshape(B, -1)[:, :n]
    return res[idx] @ T / (T @ T)


boot_npz = np.load(BOOT_NPZ)
px = pd.read_csv(FIG_DIR / 'siaf_pixelwise_summary.csv')
px_m = pd.read_csv(FIG_DIR / 'siaf_pixelwise_summary_monthly.csv')
u_mm = pd.read_csv(AN_DIR / 'observed_minus_MMM_uncertainty.csv')
m_tr = pd.read_csv(AN_DIR / 'SIAF_model_member_calendar_trends.csv')
mmm_tr = m_tr.groupby(['family', 'model'])[['total', 'area_loss', 'darkening']].mean().groupby('family').mean().mean()

# 1. Figure 2 distributions (historical residual 10,000 + Obs-era pairs 1,000)
f2_frames = []
residual_cache = {}
for method in ('CERES', 'Kernel'):
    adv = pd.read_csv(AN_DIR / f'advisor_annual_{method}.csv').sort_values('year')
    prow = px[(px.rs_method == method) & (px.period == 'CMIP6') & np.isclose(px.pval, 0.05)].iloc[0]
    comp_draws = {}
    for comp, col in (('total', 'tot'), ('area_loss', 'ext'), ('darkening', 'drk')):
        sl = block_boot_slopes(adv[comp], adv.temperature, adv.year, mmm_tr[comp])
        draws = float(prow[f'SIAF_{col}_global']) + (sl - sl.mean())
        ref_2s = float(u_mm[(u_mm.method == method) & (u_mm.component == comp)].block3_2sigma.iloc[0])
        np.testing.assert_allclose(2 * draws.std(ddof=1), ref_2s, atol=1e-12, rtol=0)
        comp_draws[comp] = draws
    residual_cache[method] = comp_draws
    f2_frames.append(pd.DataFrame({
        'rs_method': method,
        'period': '1982-2014',
        'interval_method': 'observed-minus-MMM residual, 3-year blocks',
        'replicate': np.arange(1, 10001),
        'total': comp_draws['total'],
        'area_loss': comp_draws['area_loss'],
        'darkening': comp_draws['darkening'],
    }))

    arr_obs = np.nanmean(boot_npz[f'{method}__Obs-era__0.05'][:, :, :3], axis=1) * SCALE
    prow_obs = px[(px.rs_method == method) & (px.period == 'Obs-era') & np.isclose(px.pval, 0.05)].iloc[0]
    for i, col in enumerate(('tot', 'ext', 'drk')):
        np.testing.assert_allclose(2 * np.nanstd(arr_obs[:, i], ddof=0), float(prow_obs[f'sigma2_{col}_global']), atol=1e-4, rtol=0)
    f2_frames.append(pd.DataFrame({
        'rs_method': method,
        'period': '1982-2024',
        'interval_method': 'observational pairs bootstrap, p<0.05',
        'replicate': np.arange(1, len(arr_obs) + 1),
        'total': arr_obs[:, 0],
        'area_loss': arr_obs[:, 1],
        'darkening': arr_obs[:, 2],
    }))
f2_boot = pd.concat(f2_frames, ignore_index=True)
f2_boot.to_csv(SD / 'SourceData_Figure2_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure2_Bootstrap.csv: {len(f2_boot)} rows')

# 2. Figure 3 pairs-bootstrap monthly total SIAF (p < 0.05)
f3_frames = []
for method in ('CERES', 'Kernel'):
    for period in ('CMIP6', 'Obs-era'):
        s = px_m[(px_m.rs_method == method) & (px_m.period == period) & np.isclose(px_m.pval, 0.05)].set_index('month').loc[MONTHS]
        draws = boot_npz[f'{method}__{period}__0.05'][:, :, 0] * SCALE
        for m_idx, mo in enumerate(MONTHS):
            np.testing.assert_allclose(2 * np.nanstd(draws[:, m_idx], ddof=0), float(s.sigma2_tot_global.iloc[m_idx]), atol=1e-4, rtol=0)
            f3_frames.append(pd.DataFrame({
                'rs_method': method,
                'period': period,
                'pval': 0.05,
                'month': mo,
                'replicate': np.arange(1, len(draws) + 1),
                'SIAF_tot_global': draws[:, m_idx],
                'figure_monthly_to_annual_divisor': 6,
            }))
f3_boot = pd.concat(f3_frames, ignore_index=True)
f3_boot.to_csv(SD / 'SourceData_Figure3_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure3_Bootstrap.csv: {len(f3_boot)} rows')

# 3. Figure 5 individual members (378) and residual-bootstrap draws (10,000)
R = FIG_DIR
src = (R / 'sie_aggregation_source.py').read_text()
src = src[:src.index('lbls = list(fam.index)')]
src = src.replace("os.makedirs(OUT_DIR, exist_ok=True)", '')
src = (src.replace("pd.read_pickle(SIE_PKL)", "pd.read_pickle(R/'sie_df_common.pkl')")
          .replace("load_cache('obs_sie_by_month')", "pd.read_pickle(R/'obs_sie_by_month.pkl')")
          .replace('pd.read_csv(OBS_SIAF_CSV)', "pd.read_csv(R/'siaf_pixelwise_summary.csv')")
          .replace('pd.read_csv(CMIP6_SIAF_CSV)', "pd.read_csv(R/'siaf_cmip6_summary_final.csv')"))
ns = {'R': R, 'np': np, 'pd': pd, '__file__': str(R / 'sie_aggregation_source.py')}
exec(compile(src, 'saved_sie_aggregation', 'exec'), ns)
mem_ab = pd.merge(
    ns['siaf_mem'][['model', 'member', 'siaf_global']],
    ns['sie_mem'][['model', 'member', 'trend', 'early']],
    on=['model', 'member'], how='inner'
)
mem_ab['family'] = mem_ab.model.map(lambda m: FAMILY.get(m, m))
assert len(mem_ab) == 378 and mem_ab.model.nunique() == 31 and mem_ab.family.nunique() == 18
mem_ab_out = mem_ab[['family', 'model', 'member', 'trend', 'early', 'siaf_global']].rename(columns={
    'trend': 'sie_trend_1e6km2_per_yr',
    'early': 'sie_1982_1991_1e6km2',
}).sort_values(['family', 'model', 'member']).reset_index(drop=True)
mem_ab_out.to_csv(SD / 'SourceData_Figure5_Members.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure5_Members.csv: {len(mem_ab_out)} rows')

f5_cd = pd.read_csv(FIG_DIR / 'f05_cd_panels.csv')
f5_sig = pd.read_csv(FIG_DIR / 'f05_cd_obs_sigma.csv')
f5_ser = pd.read_csv(AN_DIR / 'observed_Fig5_series.csv')
f5_boot_cols = {
    'replicate': np.arange(1, 10001),
    'ab_total_siaf_W_m2_K': residual_cache['CERES']['total'],
}
for region, prefix, suffix in (('area_loss', 'c', 'L_per_K'), ('darkening', 'd', 'IS_per_K')):
    sub_r = f5_cd[f5_cd.region == region].set_index('month').loc[MONTHS]
    sig_r = f5_sig[f5_sig.region == region].set_index('month').loc[range(4, 10)]
    for m_num, mo in zip(range(4, 10), MONTHS):
        z = f5_ser[(f5_ser.component == region) & (f5_ser.month == m_num)].sort_values('year')
        sl = block_boot_slopes(z.anomaly, z.temperature, z.year, z.MMM_trend.iloc[0])
        ref_w = float(sig_r.loc[m_num, 'halfwidth_2sigma'])
        np.testing.assert_allclose(2 * sl.std(ddof=1), ref_w, atol=1e-12, rtol=0)
        draws = float(sub_r.loc[mo, 'observed']) + (sl - sl.mean())
        f5_boot_cols[f'{prefix}_{mo}_{suffix}'] = draws
f5_boot = pd.DataFrame(f5_boot_cols)
f5_boot.to_csv(SD / 'SourceData_Figure5_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure5_Bootstrap.csv: {len(f5_boot)} rows')

# 4. Figure 6 observational draws (panel a total residual + panel b local pairs)
ceres_cmip6 = boot_npz['CERES__CMIP6__0.05']
obs_ext_loc = np.nanmean(ceres_cmip6[:, :, 7], axis=1)
obs_drk_loc = np.nanmean(ceres_cmip6[:, :, 8], axis=1)
f6_obs = pd.read_csv(SD / 'SourceData_Figure6_Observed.csv').set_index('component')
np.testing.assert_allclose(2 * np.nanstd(obs_ext_loc, ddof=0), float(f6_obs.loc['area_loss', 'local_2sigma']), atol=1e-4, rtol=0)
np.testing.assert_allclose(2 * np.nanstd(obs_drk_loc, ddof=0), float(f6_obs.loc['darkening', 'local_2sigma']), atol=1e-4, rtol=0)
f6_boot = pd.concat([
    pd.DataFrame({
        'panel': 'a',
        'component': 'total',
        'metric': 'pan_arctic_annual_siaf',
        'interval_method': 'observed-minus-MMM residual, 3-year blocks',
        'replicate': np.arange(1, 10001),
        'value_W_m2_K': residual_cache['CERES']['total'],
    }),
    pd.DataFrame({
        'panel': 'b',
        'component': 'area_loss',
        'metric': 'local_siaf_intensity',
        'interval_method': 'observational pairs bootstrap, p<0.05',
        'replicate': np.arange(1, len(obs_ext_loc) + 1),
        'value_W_m2_K': obs_ext_loc,
    }),
    pd.DataFrame({
        'panel': 'b',
        'component': 'darkening',
        'metric': 'local_siaf_intensity',
        'interval_method': 'observational pairs bootstrap, p<0.05',
        'replicate': np.arange(1, len(obs_drk_loc) + 1),
        'value_W_m2_K': obs_drk_loc,
    }),
], ignore_index=True)
f6_boot.to_csv(SD / 'SourceData_Figure6_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure6_Bootstrap.csv: {len(f6_boot)} rows')

# 5. Figure 7 family means (18) + marginal observational draws
df7 = pd.read_csv(SD / 'SourceData_Figure7.csv')
df7 = df7[df7.rs_method == 'full_T07'].copy()
c7 = df7.groupby('model').size()
df7 = df7[df7.model.isin(c7[c7 >= 3].index)].dropna(subset=['rs_arctic', 'is_tot_arctic']).copy()
df7['family'] = df7.model.map(lambda m: FAMILY.get(m, m))
mm7 = df7.groupby(['family', 'model'])[['rs_arctic', 'is_tot_arctic']].mean()
fam7 = mm7.groupby('family').mean()
fam7['n_models'] = mm7.groupby('family').size()
fam7['n_members'] = df7.groupby('family').size()
fam7 = fam7.sort_values('is_tot_arctic', ascending=False).reset_index()
fam7_out = fam7[['family', 'n_models', 'n_members', 'rs_arctic', 'is_tot_arctic']].rename(columns={
    'rs_arctic': 'rs_arctic_Wm2',
    'is_tot_arctic': 'is_tot_arctic_perK',
})
f7_obs = pd.read_csv(SD / 'SourceData_Figure7_Observed.csv')
mmm_row = f7_obs[f7_obs.element == 'Family-weighted multi-model mean'].iloc[0]
np.testing.assert_allclose(fam7_out.rs_arctic_Wm2.mean(), float(mmm_row.rs_arctic_Wm2), atol=1e-10)
np.testing.assert_allclose(2 * fam7_out.rs_arctic_Wm2.std(ddof=1), float(mmm_row.rs_arctic_2sigma_Wm2), atol=1e-10)
np.testing.assert_allclose(fam7_out.is_tot_arctic_perK.mean(), float(mmm_row.is_tot_arctic_perK), atol=1e-10)
np.testing.assert_allclose(2 * fam7_out.is_tot_arctic_perK.std(ddof=1), float(mmm_row.is_tot_arctic_2sigma_perK), atol=1e-10)
fam7_out.to_csv(SD / 'SourceData_Figure7_FamilyMeans.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure7_FamilyMeans.csv: {len(fam7_out)} rows')

ois = pd.read_csv(AN_DIR / 'observed_IS_monthly_series.csv')
bis = float(pd.read_csv(AN_DIR / 'IS_MMM_trends.csv', index_col=0).MMM_trend.loc['total'])
z_is = ois[ois.component == 'total'].groupby('year')[['anomaly', 'temperature']].mean().reset_index().sort_values('year')
sl_is = block_boot_slopes(z_is.anomaly, z_is.temperature, z_is.year, bis)
is_center = float(f7_obs[f7_obs.rs_method == 'CERES-APRP'].is_tot_arctic_perK.iloc[0])
is_2s = float(f7_obs[f7_obs.rs_method == 'CERES-APRP'].is_tot_arctic_2sigma_perK.iloc[0])
np.testing.assert_allclose(2 * sl_is.std(ddof=1), is_2s, atol=1e-12, rtol=0)
obs_is_draws = is_center + (sl_is - sl_is.mean())
obs_rs_draws = np.nanmean(ceres_cmip6[:, :, 12], axis=1)
np.testing.assert_allclose(2 * np.nanstd(obs_rs_draws, ddof=0), float(f7_obs[f7_obs.rs_method == 'CERES-APRP'].rs_arctic_2sigma_Wm2.iloc[0]), atol=1e-2, rtol=0)
f7_boot = pd.concat([
    pd.DataFrame({
        'axis': 'y',
        'rs_method': 'CERES_and_Kernel',
        'metric': 'IS_tot_arctic_perK',
        'interval_method': 'observed-minus-MMM residual, 3-year blocks',
        'replicate': np.arange(1, 10001),
        'value': obs_is_draws,
    }),
    pd.DataFrame({
        'axis': 'x',
        'rs_method': 'CERES',
        'metric': 'RS_arctic_Wm2',
        'interval_method': 'observational pairs bootstrap, p<0.05',
        'replicate': np.arange(1, len(obs_rs_draws) + 1),
        'value': obs_rs_draws,
    }),
], ignore_index=True)
f7_boot.to_csv(SD / 'SourceData_Figure7_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Figure7_Bootstrap.csv: {len(f7_boot)} rows')

# 6. Supplementary Figure 1 pairs-bootstrap distributions
s1_sum = pd.read_csv(SD / 'SourceData_Supplementary_Figure1.csv')
s1_frames = []
for method in ('CERES', 'Kernel'):
    for period in ('CMIP6', 'Obs-era'):
        for pv in (0.01, 0.05, 0.10):
            arr = np.nanmean(boot_npz[f'{method}__{period}__{pv:.2f}'][:, :, :3], axis=1) * SCALE
            r_ = s1_sum[(s1_sum.rs_method == method) & (s1_sum.period == period) & np.isclose(s1_sum.pval, pv)].iloc[0]
            for i, col in enumerate(('tot', 'ext', 'drk')):
                np.testing.assert_allclose(2 * np.nanstd(arr[:, i], ddof=0), float(r_[f'sigma2_{col}_global']), atol=1e-5, rtol=0)
            s1_frames.append(pd.DataFrame({
                'rs_method': method,
                'period': period,
                'pval': pv,
                'replicate': np.arange(1, len(arr) + 1),
                'SIAF_tot_global': arr[:, 0],
                'SIAF_ext_global': arr[:, 1],
                'SIAF_drk_global': arr[:, 2],
            }))
s1_boot = pd.concat(s1_frames, ignore_index=True)
s1_boot.to_csv(SD / 'SourceData_Supplementary_Figure1_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Supplementary_Figure1_Bootstrap.csv: {len(s1_boot)} rows')

# 7. Supplementary Figure 4 observed reanalyses, 18 family means, and bootstrap draws
df_s4_raw = pd.read_csv(FIG_DIR / 'siaf_cmip6_summary_final.csv')
df_s4 = df_s4_raw[df_s4_raw.rs_method == 'full_T07'].copy()
c_s4 = df_s4.groupby('model').size()
df_s4 = df_s4[df_s4.model.isin(c_s4[c_s4 >= 3].index)].copy()
df_s4['AA'] = df_s4.delta_t_arctic / df_s4.delta_t
df_s4 = df_s4.dropna(subset=['AA', 'siaf_global'])
df_s4 = df_s4[df_s4.AA > 0].copy()
df_s4['family'] = df_s4.model.map(lambda m: FAMILY.get(m, m))
mm_s4 = df_s4.groupby(['family', 'model'])[['AA', 'siaf_global']].mean()
fam_s4 = mm_s4.groupby('family').mean()
sp_s4 = 2 * df_s4.groupby('family')[['AA', 'siaf_global']].std().fillna(0.)
fam_s4['aa_2sigma'] = sp_s4.AA
fam_s4['siaf_global_2sigma'] = sp_s4.siaf_global
fam_s4['n_models'] = mm_s4.groupby('family').size()
fam_s4['n_members'] = df_s4.groupby('family').size()
fam_s4 = fam_s4.sort_values('siaf_global').reset_index()
fam_s4_out = fam_s4[['family', 'n_models', 'n_members', 'AA', 'aa_2sigma', 'siaf_global', 'siaf_global_2sigma']].rename(columns={
    'AA': 'aa_mean',
    'siaf_global': 'siaf_global_mean',
})
fam_s4_out.to_csv(SD / 'SourceData_Supplementary_Figure4_FamilyMeans.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Supplementary_Figure4_FamilyMeans.csv: {len(fam_s4_out)} rows')

aa_cmip6 = {'NCEP': 4.313431922970221, 'ERA5': 3.947277604266602}
aa_obsera = {'NCEP': 3.437538376524532, 'ERA5': 2.994581481923235}
o_c = px[(px.period == 'CMIP6') & (px.rs_method == 'CERES') & np.isclose(px.pval, 0.05)].iloc[0]
o_o = px[(px.period == 'Obs-era') & (px.rs_method == 'CERES') & np.isclose(px.pval, 0.05)].iloc[0]
s4_obs_df = pd.DataFrame([dict(
    a1=float(np.mean(list(aa_cmip6.values()))),
    s1=float(np.std(list(aa_cmip6.values()))),
    a2=float(np.mean(list(aa_obsera.values()))),
    s2=float(np.std(list(aa_obsera.values()))),
    aa_ncep_cmip6=aa_cmip6['NCEP'],
    aa_era5_cmip6=aa_cmip6['ERA5'],
    siaf_cmip6=float(o_c.SIAF_tot_global),
    siaf_2sigma_cmip6=float(o_c.sigma2_tot_global),
    aa_ncep_obsera=aa_obsera['NCEP'],
    aa_era5_obsera=aa_obsera['ERA5'],
    siaf_obsera=float(o_o.SIAF_tot_global),
    siaf_2sigma_obsera=float(o_o.sigma2_tot_global),
)])
s4_obs_df.to_csv(SD / 'SourceData_Supplementary_Figure4_Observed.csv', index=False, float_format='%.12g')
print('updated SourceData_Supplementary_Figure4_Observed.csv')

s4_frames = []
for pk, row_o in (('CMIP6', o_c), ('Obs-era', o_o)):
    draws = np.nanmean(boot_npz[f'CERES__{pk}__0.05'][:, :, 0], axis=1) * SCALE
    np.testing.assert_allclose(2 * np.nanstd(draws, ddof=0), float(row_o.sigma2_tot_global), atol=1e-4, rtol=0)
    s4_frames.append(pd.DataFrame({
        'period': pk,
        'rs_method': 'CERES',
        'pval': 0.05,
        'interval_method': 'observational pairs bootstrap, p<0.05',
        'replicate': np.arange(1, len(draws) + 1),
        'siaf_global_W_m2_K': draws,
    }))
s4_boot = pd.concat(s4_frames, ignore_index=True)
s4_boot.to_csv(SD / 'SourceData_Supplementary_Figure4_Bootstrap.csv', index=False, float_format='%.8g')
print(f'wrote SourceData_Supplementary_Figure4_Bootstrap.csv: {len(s4_boot)} rows')

# 8. Supplementary Figure 6 annual ROC curves (20 years x 6 months x 2 products x 2 domains)
with open(ROC_PKL, 'rb') as fh:
    pkl = pickle.load(fh)
s6_sum = pd.read_csv(SD / 'SourceData_Supplementary_Figure6.csv')
FPR_GRID = np.linspace(0, 1, 500)
roc_rows = []
for domain_label, sk in (('Full Arctic', 'roc_store'), ('MIZ (Marginal Ice Zone)', 'roc_edge_store')):
    st = pkl[sk]
    for mo in pkl['MONTH_ORDER']:
        for product in ('CLARA-A3', 'APP-x'):
            yd = st[mo][product]
            tprs_interp, aucs = [], []
            for yr in sorted(yd.keys()):
                tpr_y, fpr_y = yd[yr]
                order = np.argsort(fpr_y)
                tprs_interp.append(np.interp(FPR_GRID, fpr_y[order], tpr_y[order]))
                aucs.append(float(np.trapezoid(tpr_y[order], fpr_y[order])))
                for f_val, t_val in zip(fpr_y, tpr_y):
                    roc_rows.append((domain_label, mo, product, int(yr), float(f_val), float(t_val)))
            mean_tpr = np.nanmean(np.array(tprs_interp), axis=0)
            i_opt = int(np.argmax(mean_tpr - FPR_GRID))
            ref_s6 = s6_sum[(s6_sum['domain'] == domain_label) & (s6_sum['month'] == mo) & (s6_sum['product'] == product)].iloc[0]
            np.testing.assert_allclose(float(np.mean(aucs)), float(ref_s6.auc_mean), atol=1e-12)
            np.testing.assert_allclose(float(mean_tpr[i_opt]), float(ref_s6.optimal_tpr), atol=1e-12)
            np.testing.assert_allclose(float(FPR_GRID[i_opt]), float(ref_s6.optimal_fpr), atol=1e-12)

s6_curves = pd.DataFrame(roc_rows, columns=['domain', 'month', 'product', 'year', 'fpr', 'tpr'])
s6_curves.to_csv(SD / 'SourceData_Supplementary_Figure6_AnnualCurves.csv', index=False, float_format='%.7g')
print(f'wrote SourceData_Supplementary_Figure6_AnnualCurves.csv: {len(s6_curves)} rows')
print('ALL DISTRIBUTION SOURCE DATA VERIFIED AND WRITTEN SUCCESSFULLY.')
