"""Model-based 2-sigma width for the observed ice sensitivity IS (24 Nov 2026).

Same procedure as the SIAF model-based uncertainty (Methods; uncertainty.csv,
reproduced to ~1% by recon.py), applied to the unweighted albedo series
Q_IS(t) = <-[alpha(t) - alpha_bar]> saved by extract_is.py. There is no
radiative-sensitivity term, so the variance is the albedo-temperature slope
variance alone. Subtract family-weighted MMM calendar trends (or within-model
ensemble-mean trends), resample paired residuals in circular blocks (4,000
replicates), restore the trends, refit, pool members -> models -> families ->
equal, report 2*sqrt(V).
"""
import glob
import numpy as np
import pandas as pd

R = '/glade/derecho/scratch/jrotondo/uncertainty_is_20261124'
OUT = '/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028'
SUMMARY = '/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv'
OBS_CSV = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv'
FAM = {'ACCESS': ['ACCESS-ESM1-5', 'ACCESS-CM2'], 'BCC': ['BCC-CSM2-MR', 'BCC-ESM1'],
       'CESM2': ['CESM2', 'CESM2-WACCM', 'CESM2-FV2'], 'CMCC': ['CMCC-CM2-SR5'],
       'CNRM': ['CNRM-CM6-1', 'CNRM-ESM2-1'], 'CanESM5': ['CanESM5'], 'E3SM': ['E3SM-1-0'],
       'EC-Earth3': ['EC-Earth3-Veg', 'EC-Earth3-Veg-LR'], 'FGOALS': ['FGOALS-f3-L', 'FGOALS-g3'],
       'GISS': ['GISS-E2-1-H'], 'HadGEM3/UKESM': ['UKESM1-0-LL', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM'],
       'INM': ['INM-CM5-0'], 'IPSL': ['IPSL-CM6A-LR'], 'MIROC': ['MIROC-ES2L', 'MIROC6'],
       'MPI/ICON': ['MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'MPI-ESM-1-2-HAM'], 'MRI': ['MRI-ESM2-0'],
       'NESM3': ['NESM3'], 'NorESM': ['NorCPM1', 'NorESM2-LM']}
m2f = {m: f for f, v in FAM.items() for m in v}
COMPS = ['total', 'area_loss', 'darkening']

# ---- per-member series and the check against the saved point estimates
S = pd.read_csv(SUMMARY)
S = S[S.rs_method == 'full_T07'].set_index(['model', 'member'])
recs, chk = [], []
for f in sorted(glob.glob(R + '/members/*.npz')):
    z = np.load(f)
    k = (str(z['model']), str(z['member']))
    yrs = z['years'].astype(int)
    recs.append(dict(model=k[0], member=k[1], years=yrs, T=z['temperature'].astype(float),
                     Q=z['q_is'].astype(float)))
    sl = np.polyfit(z['temperature'], z['q_is'][:, 0], 1)[0]
    chk.append(dict(model=k[0], member=k[1], n=len(yrs), pt=z['original_point'][0],
                    csv=S.loc[k, 'is_tot_arctic'], pt_drk=z['original_point'][2],
                    csv_drk=S.loc[k, 'is_drk_arctic'], pt_ext=z['original_point'][1],
                    csv_ext=S.loc[k, 'is_ext_arctic'], slope=sl))
C = pd.DataFrame(chk)
print(f'{len(C)} members, {C.model.nunique()} models; years per member: {C.n.value_counts().to_dict()}')
for a, b in [('pt', 'csv'), ('pt_ext', 'csv_ext'), ('pt_drk', 'csv_drk'), ('slope', 'pt')]:
    r = C[a] / C[b] - 1
    print(f'  {a:6s} vs {b:7s}: max |rel| = {np.nanmax(np.abs(r)):.2e}, median rel = {np.nanmedian(r):+.2e}')

models = np.array([r['model'] for r in recs])
fam = np.array([m2f[m] for m in models])
nm = pd.Series(models).value_counts()
nf = pd.Series({f: len(v) for f, v in FAM.items()})
w = np.array([1 / (len(FAM) * nf[m2f[m]] * nm[m]) for m in models])
assert abs(w.sum() - 1) < 1e-12


def tc(yrs):
    return yrs - 1997.5


def slope_t(y, X):
    t = tc(y)
    tt = t - t.mean()
    Xc = X - X.mean(0)
    return (tt @ Xc) / (tt @ tt) if X.ndim == 2 else (tt @ Xc) / (tt @ tt)


sT = np.array([slope_t(r['years'], r['T']) for r in recs])
sQ = np.stack([slope_t(r['years'], r['Q']) for r in recs])


def hier(v):
    d = pd.DataFrame({'model': models, 'fam': fam, 'v': v})
    return d.groupby(['fam', 'model']).v.mean().groupby(level=0).mean().mean()


def boot_idx(rng, n, L, B):
    nb = -(-n // L)
    s = rng.integers(0, n, size=(B, nb))
    idx = (s[:, :, None] + np.arange(L)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def run(method, L, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    if method == 'Common MMM trend':
        bT = np.full(len(models), (w * sT).sum())
        bQ = np.tile((w[:, None] * sQ).sum(0), (len(models), 1))
    else:
        bT = np.empty(len(models))
        bQ = np.empty_like(sQ)
        for mod in np.unique(models):
            ix = np.where(models == mod)[0]
            bT[ix] = sT[ix].mean()
            bQ[ix] = sQ[ix].mean(0)
    va = np.zeros((len(models), 3))
    for i, r in enumerate(recs):
        t = tc(r['years'])
        n = len(t)
        rT = r['T'] - r['T'].mean() - bT[i] * (t - t.mean())
        rQ = r['Q'] - r['Q'].mean(0) - np.outer(t - t.mean(), bQ[i])
        idx = boot_idx(rng, n, L, B)
        Ts = bT[i] * t[None, :] + rT[idx]
        Qs = bQ[i][None, None, :] * t[None, :, None] + rQ[idx]
        Tc = Ts - Ts.mean(1, keepdims=True)
        Qc = Qs - Qs.mean(1, keepdims=True)
        sl = np.einsum('bn,bnc->bc', Tc, Qc) / np.einsum('bn,bn->b', Tc, Tc)[:, None]
        d = sl - sl.mean(0)
        va[i] = (d ** 2).mean(0)
    return {c: hier(va[:, ci]) for ci, c in enumerate(COMPS)}


rows = []
for method in ['Common MMM trend', 'Within-model ensemble-mean trend']:
    for L in [1, 3, 5]:
        res = run(method, L)
        for c, v in res.items():
            rows.append(dict(method=method, block_years=L, component=c, var_IS=v,
                             two_sigma_IS=2 * np.sqrt(v)))
        print(f'{method[:6]} L={L}: ' + '  '.join(f'{c} 2sig={2*np.sqrt(v):.5f}' for c, v in res.items()))
U = pd.DataFrame(rows)
U.to_csv(OUT + '/uncertainty_is.csv', index=False)
print('wrote', OUT + '/uncertainty_is.csv')

# ---- observed IS, intervals and how many models/members reach the lower bound
O = pd.read_csv(OBS_CSV)
o = O[(O.rs_method == 'CERES') & (O.period == 'CMIP6') & np.isclose(O.pval, 0.05)].iloc[0]
obs = {'total': o.IS_tot_arctic, 'area_loss': o.IS_ext_arctic, 'darkening': o.IS_drk_arctic}
boot = {'total': o.sigma2_IS_arctic, 'area_loss': o.sigma2_IS_ext_arctic, 'darkening': o.sigma2_IS_drk_arctic}
col = {'total': 'is_tot_arctic', 'area_loss': 'is_ext_arctic', 'darkening': 'is_drk_arctic'}
D = S.reset_index()
D = D[D.model.isin(m2f)].copy()
D['fam'] = D.model.map(m2f)
print(f'\nsummary rows used for counts: {len(D)} members, {D.model.nunique()} models, {D.fam.nunique()} families')
mm = D.groupby('model')[list(col.values())].mean()
ff = D.groupby(['fam', 'model'])[list(col.values())].mean().groupby(level=0).mean()
for c in COMPS:
    m2 = U[(U.method == 'Common MMM trend') & (U.block_years == 3) & (U.component == c)].two_sigma_IS.iloc[0]
    for lab, hw in [('bootstrap', boot[c]), ('model-based', m2)]:
        lo, hi = obs[c] - hw, obs[c] + hw
        x = D[col[c]]
        print(f'{c:9s} obs {obs[c]:.6f} {lab:11s} 2sig {hw:.6f} -> [{lo:.5f}, {hi:.5f}]: '
              f'members >= lo {100*(x >= lo).mean():.1f}% ({(x >= lo).sum()}/{len(x)}), '
              f'model means >= lo {(mm[col[c]] >= lo).sum()}/{len(mm)}, '
              f'family means >= lo {(ff[col[c]] >= lo).sum()}/{len(ff)}, '
              f'members >= obs {(x >= obs[c]).sum()}, model means >= obs {(mm[col[c]] >= obs[c]).sum()}')
    print(f'{"":9s} family-weighted MMM {ff[col[c]].mean():.6f}; family range {ff[col[c]].min():.5f}-{ff[col[c]].max():.5f}; '
          f'model range {mm[col[c]].min():.5f}-{mm[col[c]].max():.5f}; deficit {100*(1-ff[col[c]].mean()/obs[c]):.1f}%')
