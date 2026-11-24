"""prepare + bootstrap + summarize for the no-filter Fig. 5c-d observational
2-sigma widths (23 Nov 2026), on the region-mean series from extract_nf.py.

Mirrors fig5_uncertainty_20261112/{prepare,bootstrap,summarize_direct}.py:
 1. common MMM calendar trends of the regional albedo and SIC series for each
    month/region (members -> models -> families -> equally over families), and
    one global temperature trend through the same hierarchy;
 2. 4,000 circular 3-year-block replicates per member, seeded from model/member
    exactly as before (so the draws are identical to 12 Nov), refitting the
    direct slopes L (area loss) and I (darkening) only;
 3. replicate variances pooled members -> models -> families -> equally over
    families; half-width = 2 sqrt(variance); first and last 2,000 as a
    Monte Carlo check.
Identity resampling must reproduce each member's per-pixel baseline I and L.
usage: python boot_nf.py [--partial]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = Path('/glade/derecho/scratch/jrotondo/fig5_uncertainty_20261112')
sys.path.insert(0, str(ROOT))
from series_core import block_indices, refits_direct   # noqa: E402

families = json.loads((SRC / 'families.json').read_text())
req = pd.read_csv(SRC / 'requested_members.csv')
st = pd.DataFrame([json.loads(p.read_text()) for p in (ROOT / 'status').glob('*.json')])
print('extraction status:', st.status.value_counts().to_dict(), 'of', len(req), 'requested', flush=True)
if len(st) != len(req) and '--partial' not in sys.argv:
    raise SystemExit('extraction incomplete')
bad = st[st.status != 'complete']
if len(bad):
    print(bad[['model', 'member', 'error']].to_string(index=False))

rows = []
for p in sorted((ROOT / 'series').glob('*.npz')):
    z = np.load(p)
    yrs = np.asarray(z['years'], float); x = yrs - yrs.mean()
    rows.append(dict(path=p.name, model=str(z['model']), member=str(z['member']), month=int(z['month']),
                     region=str(z['region']), n_years=len(yrs),
                     trend_a=float(x @ z['a'] / (x @ x)), trend_c=float(x @ z['c'] / (x @ x)),
                     trend_t=float(x @ z['t'] / (x @ x)),
                     **{k: float(z[k]) for k in ('baseline_L', 'baseline_I', 'original_L', 'original_I',
                                                 'kept_weight_fraction')},
                     n_pixels=int(z['n_pixels']), n_complete=int(z['n_complete'])))
s = pd.DataFrame(rows)
s.to_csv(ROOT / 'manifest_nf.csv', index=False)
print('series:', len(s), 'records,', s.model.nunique(), 'models,',
      s[['model', 'member']].drop_duplicates().shape[0], 'members; n_years',
      s.n_years.min(), '-', s.n_years.max(), '; kept weight min', round(s.kept_weight_fraction.min(), 3), flush=True)

cols = ['trend_a', 'trend_c', 'trend_t']
model = s.groupby(['month', 'region', 'model'])[cols].mean().reset_index()
model['family'] = model.model.map(lambda m: families.get(m, m))
fam = model.groupby(['month', 'region', 'family'])[cols].mean().reset_index()
mmm = fam.groupby(['month', 'region'])[cols].mean().reset_index()
T = s.groupby(['model', 'member']).trend_t.first().groupby('model').mean().to_frame()
T['family'] = [families.get(m, m) for m in T.index]
mmm['trend_t'] = float(T.groupby('family').trend_t.mean().mean())
mmm.to_csv(ROOT / 'mmm_trends_nf.csv', index=False)
mmm = mmm.set_index(['month', 'region'])

out, worst = [], 0.0
ident = np.arange(33)[None, :]
for (mdl, mbr), g in s.groupby(['model', 'member']):
    seed = int(hashlib.sha256(f'{mdl}/{mbr}'.encode()).hexdigest()[:8], 16)
    idx = block_indices(seed=seed)
    for _, r in g.iterrows():
        z = np.load(ROOT / 'series' / r.path)
        sl = mmm.loc[(r.month, r.region), cols].to_numpy(float)
        L, I = refits_direct(z['a'], z['c'], z['t'], z['years'], sl, idx)
        l0, i0 = refits_direct(z['a'], z['c'], z['t'], z['years'], sl, ident)
        assert np.isclose(l0[0], r.baseline_L, atol=3e-6, rtol=1e-4), (r.path, l0[0], r.baseline_L)
        assert np.isclose(i0[0], r.baseline_I, atol=3e-6, rtol=1e-4), (r.path, i0[0], r.baseline_I)
        worst = max(worst, abs(l0[0] - r.baseline_L), abs(i0[0] - r.baseline_I))
        v = L if r.region == 'area_loss' else I
        assert len(v) == 4000 and np.isfinite(v).all()
        out.append(dict(model=mdl, member=mbr, month=r.month, region=r.region,
                        variable='L' if r.region == 'area_loss' else 'I',
                        variance=float(np.var(v, ddof=1)), variance_first=float(np.var(v[:2000], ddof=1)),
                        variance_last=float(np.var(v[2000:], ddof=1)), mean=float(v.mean())))
print(f'identity resampling reproduces every per-pixel baseline (max |diff| {worst:.1e})', flush=True)
member = pd.DataFrame(out)
member.to_csv(ROOT / 'member_variances_nf.csv', index=False)

vc = ['variance', 'variance_first', 'variance_last']
mv = member.groupby(['month', 'region', 'model'])[vc].mean().reset_index()
mv['family'] = mv.model.map(lambda m: families.get(m, m))
fv = mv.groupby(['month', 'region', 'family'])[vc].mean().reset_index()
agg = fv.groupby(['month', 'region'])[vc].mean().reset_index()
for a_, b_ in (('halfwidth_2sigma', 'variance'), ('halfwidth_first', 'variance_first'),
               ('halfwidth_last', 'variance_last')):
    agg[a_] = 2 * np.sqrt(agg[b_])
cov = member.groupby(['month', 'region']).agg(n_members=('member', 'size'), n_models=('model', 'nunique')).reset_index()
fc = fv.groupby(['month', 'region']).family.nunique().rename('n_families').reset_index()
agg = agg.merge(cov).merge(fc)
agg.to_csv(ROOT / 'uncertainty_summary_direct_nofilter.csv', index=False)
old = pd.read_csv(SRC / 'uncertainty_summary_direct.csv').set_index(['month', 'region'])
cmp_ = agg.set_index(['month', 'region'])[['halfwidth_2sigma', 'halfwidth_first', 'halfwidth_last',
                                           'n_members', 'n_models', 'n_families']]
cmp_['filtered_12Nov'] = old.halfwidth_2sigma
cmp_['ratio'] = cmp_.halfwidth_2sigma / cmp_.filtered_12Nov
print(cmp_.round(4).to_string(), flush=True)
