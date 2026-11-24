"""Re-stage Figure 5 panels c-d without the SIC-sigma filter (23 Nov 2026).

Joey's decision (23 Nov): drop the 0.005 SIC-sigma condition. It existed only
for the partial slope b, which the paper no longer uses, and it removed about
a third of the April footprints (24-30% of the observed April-May darkening
pixels) -- the consolidated pack the darkening diagnostic is about. All three
c-d inputs are swapped; the filtered tables are kept as *.filtered_superseded:
  models -> f05_cd_models.csv
      fig5_rerun31_patched_20261123/model_bivariate_by_month_nofilter.csv
      (bivariate_sdcheck_31.py with SAVE_FULL=1: the same 31 members and
      grid_patch as the headline run, refused pixels kept; I and L only)
  observations -> f05_cd_obs_stage3.csv (durable copy) and f05_cd_panels.csv
      sdfilter_test_20261123/obs_nofilter/stage3_bivariate.csv
      (bivariate_historical.py with the filter removed from the selection)
  widths -> f05_cd_obs_sigma.csv
      fig5_uncertainty_nofilter_20261123/uncertainty_summary_direct_nofilter.csv
      (extract_nf.py + boot_nf.py: the 378 members, seeds and block draws of
      12 Nov, direct slopes on region-mean series -- exact, see
      check_series_shortcut.py)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

NB = Path('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028')
SCR = Path('/glade/derecho/scratch/jrotondo')
MOD = SCR / 'fig5_rerun31_patched_20261123/model_bivariate_by_month_nofilter.csv'
OBS3 = SCR / 'sdfilter_test_20261123/obs_nofilter/stage3_bivariate.csv'
SIG = SCR / 'fig5_uncertainty_nofilter_20261123/uncertainty_summary_direct_nofilter.csv'
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']

mod = pd.read_csv(MOD)
assert mod.model.nunique() == 31 and mod.n_years.min() == 33 and mod.member.groupby(mod.model).nunique().max() == 1
obs = pd.read_csv(OBS3).set_index(['region', 'month'])
sig = pd.read_csv(SIG)
assert len(sig) == 12 and sig.n_models.min() >= 30
old = pd.read_csv(NB / 'f05_cd_panels.csv')

rows = []
for region, var in (('area_loss', 'L'), ('darkening', 'I')):
    for m in MONTHS:
        v = mod[(mod.region == region) & (mod.month == m)][var].astype(float).dropna()
        rows.append(dict(region=region, month=m, variable=var, observed=float(obs.loc[(region, m), var]),
                         model_mean=float(v.mean()), model_sd=float(v.std(ddof=1)), n_models=int(len(v))))
panels = pd.DataFrame(rows)[list(old.columns)]

for f in ('f05_cd_panels.csv', 'f05_cd_models.csv', 'f05_cd_obs_sigma.csv'):
    bak = NB / (f + '.filtered_superseded')
    if not bak.exists():
        shutil.copy(NB / f, bak)
panels.to_csv(NB / 'f05_cd_panels.csv', index=False)
shutil.copy(MOD, NB / 'f05_cd_models.csv')
shutil.copy(SIG, NB / 'f05_cd_obs_sigma.csv')
shutil.copy(OBS3, NB / 'f05_cd_obs_stage3.csv')
print(panels.round(4).to_string(index=False))
print(sig[['month', 'region', 'halfwidth_2sigma', 'n_members', 'n_models', 'n_families']].round(4).to_string(index=False))

# ---------------------------------------------------------------- f05.py
p = NB / 'f05.py'
src = p.read_text()
bak = NB / 'f05.py.prenofilter_bak'
if not bak.exists():
    shutil.copy(p, bak)


def sub(old_s, new_s):
    global src
    assert src.count(old_s) == 1, f'not found once: {old_s[:60]!r}'
    src = src.replace(old_s, new_s)


sub("""  fig5_review_20261112/preview/Figure5_new_panels_source.csv  -> f05_cd_panels.csv
  fig5_uncertainty_20261112/uncertainty_summary.csv           -> f05_cd_obs_sigma.csv
  fig5_rerun31_patched_20261123/model_bivariate_by_month.csv  -> f05_cd_models.csv
    (23 Nov 2026: one retained member from each of the 31 headline models,
    grid_patch applied; supersedes the 24-model MODEL_INFO run, whose tables
    are kept as *.24model_superseded. restage_f05_31.py did the swap.)
""", """  sdfilter_test_20261123/obs_nofilter/stage3_bivariate.csv    -> f05_cd_obs_stage3.csv
                                                     (and f05_cd_panels.csv)
  fig5_uncertainty_nofilter_20261123/uncertainty_summary_direct_nofilter.csv
                                                              -> f05_cd_obs_sigma.csv
  fig5_rerun31_patched_20261123/model_bivariate_by_month_nofilter.csv
                                                              -> f05_cd_models.csv
    (23 Nov 2026: one retained member from each of the 31 headline models,
    grid_patch applied, and NO SIC-sigma filter -- Joey dropped it the same
    day; restage_f05_nofilter.py did the swap. The filtered tables are kept as
    *.filtered_superseded and the 24-model ones as *.24model_superseded.)
""")
sub("""      f'-> paper says 0.088 and 0.027 (direct -dalpha/dT, not the partial slope)')""",
    """      f'-> paper says 0.075 and 0.024 (direct -dalpha/dT, no SIC-sigma filter)')""")
p.write_text(src)
print('patched f05.py')
