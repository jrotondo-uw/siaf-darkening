"""Fig. 5c-d observational 2-sigma widths, re-extracted WITHOUT the SIC-sigma
filter (23 Nov 2026; Joey: the filter is arbitrary and no longer needed).

Same 378 members / 31 models (requested_members.csv), native-grid monthly
regions, 10-year minimum, cos-latitude weights, grid_patch and complete-support
rule as fig5_uncertainty_20261112/extract.py. One change: ols_bivariate's
0.005 SIC-sigma condition no longer removes pixels, so pixels whose
concentration barely varies stay in both footprints. (That condition existed
only for the partial slope b, which the paper no longer uses; the direct slopes
I and L are defined at every pixel with a valid record.)

Stores, per member/month/region, the complete-support area-weighted mean
albedo and SIC series. With one temperature series per member and fixed
weights, the weighted mean of per-pixel slopes IS the slope of the weighted
mean series, so the direct-slope bootstrap needs nothing more; this is checked
against the per-pixel baseline (identity resampling) in boot_nf.py and against
the 12 Nov per-pixel replicates in check_series_shortcut.py.
"""
from pathlib import Path
import os, sys, json, traceback, gc, importlib.util
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = Path('/glade/derecho/scratch/jrotondo/fig5_uncertainty_20261112')
(ROOT / 'series').mkdir(exist_ok=True); (ROOT / 'status').mkdir(exist_ok=True)
sys.path.insert(0, '/glade/derecho/scratch/jrotondo/uncertainty_mmm_20261023/deps')
sys.path.insert(0, '/glade/derecho/scratch/jrotondo')
spec = importlib.util.spec_from_file_location('bivariate_diag', SRC / 'bivariate_diag_source.py')
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd); sys.modules['bivariate_diag'] = bd

# ---- the one change: keep the pixels the SIC-sigma condition refuses ----------
_orig = bd.ols_bivariate
def _keep_all(y, x1, x2, *a, **k):
    b, d, n, r, uy, ux = _orig(y, x1, x2, *a, **k)
    fill = np.isfinite(uy) & np.isfinite(ux) & ~(np.isfinite(b) & np.isfinite(d))
    b, d = b.copy(), d.copy()
    b[fill], d[fill] = 0.0, 0.0   # placeholders so the region selection keeps them; b, D unused
    return b, d, n, r, uy, ux
bd.ols_bivariate = _keep_all      # must precede the exec below (it imports the name)

src = (SRC / 'bivariate_model_source.py').read_text().split('def _one_model')[0]
src = src.replace('workers=-1', 'workers=1')
needle = '            w = coslat[sel]'
assert src.count(needle) == 1
src = src.replace(needle, needle + "\n            capture(model,member,mn,reg,yrs,A[:,sel],Cc[:,sel],T,w,L[sel],I[sel])")
ns = {'__name__': 'isolated_capture'}; exec(compile(src, 'captured_bivariate_model', 'exec'), ns)
assert ns['ols_bivariate'] is _keep_all
import grid_patch
grid_patch.apply(ns['C'])
spec = importlib.util.spec_from_file_location('legacy', '/glade/derecho/scratch/jrotondo/mask_audit_20261111/legacy_diagnostics.py')
leg = importlib.util.module_from_spec(spec); spec.loader.exec_module(leg)
for k, v in leg.EXTRA_MODEL_INFO.items():
    ns['C'].MODEL_INFO.setdefault(k, v)


def capture(model, member, month, region, years, a, c, t, w, L, I):
    if region not in ('area_loss', 'darkening') or month not in range(4, 10):
        return
    finite = np.isfinite(a).all(0) & np.isfinite(c).all(0) & np.isfinite(w) & (w > 0)
    if finite.sum() < 20:
        raise ValueError('Fewer than 20 complete-support pixels')
    ww = w[finite].astype(float); ww /= ww.sum()
    key = f'{model}__{member}__{month}__{region}'
    np.savez(ROOT / 'series' / f'{key}.npz', model=model, member=member, month=month, region=region,
             years=np.asarray(years), t=np.asarray(t, float),
             a=np.asarray(a[:, finite], np.float32).astype(float) @ ww,
             c=np.asarray(c[:, finite], np.float32).astype(float) @ ww,
             baseline_L=float(ww @ L[finite]), baseline_I=float(ww @ I[finite]),
             original_L=float(np.average(L, weights=w)), original_I=float(np.average(I, weights=w)),
             kept_weight_fraction=float(w[finite].sum() / w.sum()),
             n_pixels=int(len(w)), n_complete=int(finite.sum()))
    print('SAVED', key, int(finite.sum()), flush=True)


ns['capture'] = capture
pairs = pd.read_csv(os.environ.get('PAIRS_FILE', SRC / 'requested_members.csv'))
worker = int(os.environ.get('WORKER', '0')); stride = int(os.environ.get('WORKERS', '8'))
only = os.environ.get('ONLY')   # 'model/member' for a timing test
for idx in range(worker, len(pairs), stride):
    row = pairs.iloc[idx]; model, member = row.model, row.member
    if only and only != f'{model}/{member}':
        continue
    path = ROOT / 'status' / f'{model}__{member}.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        continue
    try:
        temps = ns['C'].compute_T_annual_global(model, member, ns['C'].TREND_YEARS)
        if not temps:
            raise ValueError('No temperature')
        rows = ns['member_bivariate'](model, member, temps)
        paths = list((ROOT / 'series').glob(f'{model}__{member}__*.npz'))
        if len(paths) < 6:
            raise ValueError(f'Only {len(paths)} month-region records')
        pd.DataFrame(rows).to_csv(ROOT / 'status' / f'{model}__{member}.csv', index=False)
        status = dict(model=model, member=member, status='complete', n_records=len(paths))
    except Exception as e:
        status = dict(model=model, member=member, status='failed', error=str(e), traceback=traceback.format_exc())
        print(status['traceback'], flush=True)
    path.write_text(json.dumps(status, indent=2)); gc.collect(); print('STATUS', idx, status['status'], flush=True)
