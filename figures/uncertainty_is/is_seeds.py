"""Monte Carlo stability of the primary IS width (common MMM, 3-yr blocks): 2sigma is
0.018512 with seed 0, next to the 0.0185 rounding boundary, so repeat with ten seeds."""
import sys, numpy as np, pandas as pd
sys.argv = ['x']
src = open('/glade/derecho/scratch/jrotondo/uncertainty_is_20261124/is_boot.py').read()
src = src.split("rows = []")[0]          # definitions and data only
exec(src)
v = {c: [] for c in COMPS}
for seed in range(10):
    r = run('Common MMM trend', 3, seed=seed)
    for c in COMPS:
        v[c].append(r[c])
for c in COMPS:
    a = np.array(v[c]); s2 = 2*np.sqrt(a)
    print(f'{c:9s} 2sig per seed: ' + ' '.join(f'{x:.5f}' for x in s2) +
          f' | pooled-variance 2sig {2*np.sqrt(a.mean()):.6f}  seed SE {s2.std(ddof=1)/np.sqrt(len(s2)):.6f}')
# margin of members around the new lower bound
D = S.reset_index(); D = D[D.model.isin(m2f)]
lo = 0.0824 - 2*np.sqrt(np.mean(v['total']))
x = np.sort(D.is_tot_arctic.values); print('lower bound', lo, 'nearest members', x[np.argsort(np.abs(x-lo))[:4]])
mm = D.groupby('model').is_tot_arctic.mean().sort_values(); print('nearest model means', mm.iloc[np.argsort(np.abs(mm.values-lo))[:3]].to_dict())
print('model means >= lo:', mm[mm >= lo].round(5).to_dict())
