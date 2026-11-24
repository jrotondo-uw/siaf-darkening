"""Re-stage Figure 5 panels c-d on the 31-model headline ensemble.

The central-value diagnostic was rerun on 23 Nov 2026 (bivariate_model_31.py,
with grid_patch, one retained member from each of the 31 models with >= 3
valid members). This replaces the 24-model MODEL_INFO run in the two staging
tables (originals kept with a .24model_superseded suffix). The observational
values and the model-based observational widths are unchanged.

Also patches f05.py:
  * the three-line footer is removed -- the editor does not allow text in a
    figure file that is not part of the figure, and the caption carries it;
  * the r/p annotation in a-b goes on two lines with a true minus sign, so it
    no longer runs through the observed error bar.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

NB = Path('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028')
NEW = Path('/glade/derecho/scratch/jrotondo/fig5_rerun31_patched_20261123/model_bivariate_by_month.csv')
OBS3 = Path('/glade/derecho/scratch/jrotondo/fig5_review_20261112/observations/stage3_bivariate.csv')
MONTHS = ['April', 'May', 'June', 'July', 'August', 'September']

new = pd.read_csv(NEW)
assert new.model.nunique() == 31 and new.n_years.min() == 33
obs = pd.read_csv(OBS3).set_index(['region', 'month'])
old = pd.read_csv(NB / 'f05_cd_panels.csv')

rows = []
for region, var in (('area_loss', 'L'), ('darkening', 'I')):
    for m in MONTHS:
        v = new[(new.region == region) & (new.month == m)][var].astype(float).dropna()
        ob = float(obs.loc[(region, m), var])
        was = old[(old.region == region) & (old.month == m)].iloc[0]
        assert was.variable == var and np.isclose(was.observed, ob, rtol=0, atol=1e-9), (region, m)
        rows.append(dict(region=region, month=m, variable=var, observed=ob,
                         model_mean=float(v.mean()), model_sd=float(v.std(ddof=1)),
                         n_models=int(len(v))))
panels = pd.DataFrame(rows)[list(old.columns)]

for f in ('f05_cd_panels.csv', 'f05_cd_models.csv'):
    bak = NB / (f + '.24model_superseded')
    if not bak.exists():
        shutil.copy(NB / f, bak)
panels.to_csv(NB / 'f05_cd_panels.csv', index=False)
shutil.copy(NEW, NB / 'f05_cd_models.csv')
print(panels.to_string(index=False))

# ---------------------------------------------------------------- f05.py
p = NB / 'f05.py'
src = p.read_text()
bak = NB / 'f05.py.pre31_bak'
if not bak.exists():
    shutil.copy(p, bak)


def sub(old_s, new_s):
    global src
    assert src.count(old_s) == 1, f'not found once: {old_s[:60]!r}'
    src = src.replace(old_s, new_s)


sub("""  ncc_paper/plots/bivariate_diag/model_bivariate_by_month.csv -> f05_cd_models.csv
""", """  fig5_rerun31_patched_20261123/model_bivariate_by_month.csv  -> f05_cd_models.csv
    (23 Nov 2026: one retained member from each of the 31 headline models,
    grid_patch applied; supersedes the 24-model MODEL_INFO run, whose tables
    are kept as *.24model_superseded. restage_f05_31.py did the swap.)
""")

sub("""    ax.text(.04, .95, f'r = {r:.2f}; p = {p:.3f}', va='top', transform=ax.transAxes)""",
    """    ax.text(.04, .95, f'r = {r:.2f}\\np = {p:.3f}'.replace('-', '\\u2212'),
            va='top', transform=ax.transAxes)""")

sub("""fig.text(.5, .012,
         '1982–2014 throughout  ·  monthly regions in c–d, one member per model, '
         'cosine-latitude weighting\\n'
         'a–b: family whiskers 2$\\\\sigma$, observed model-based 2$\\\\sigma$\\n'
         'c–d: model spread 2$\\\\sigma$ across models, '
         'observed 2$\\\\sigma$ from CMIP6 paired block resampling',
         ha='center', va='bottom', fontsize=10.5, color='0.3')
fig.subplots_adjust(top=.955, bottom=.165, left=.093, right=.985,
                    wspace=.28, hspace=.40)""",
    """# No footer: the editor does not allow text in a figure file that is not part
# of the figure. Period, member, weighting and interval definitions are all
# in the caption.
fig.subplots_adjust(top=.955, bottom=.085, left=.093, right=.985,
                    wspace=.28, hspace=.40)""")

p.write_text(src)
print('patched f05.py')
