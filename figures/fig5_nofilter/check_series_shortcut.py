"""Does the region-mean-series bootstrap reproduce the 12 Nov per-pixel replicates?
Uses the 12 Nov stacks (filtered pixels) and their MMM trends and seeds."""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from series_core import block_indices, refits_direct
R = Path('/glade/derecho/scratch/jrotondo/fig5_uncertainty_20261112')
mmm = pd.read_csv(R / 'mmm_trends.csv').set_index(['month', 'region'])
man = pd.read_csv(R / 'manifest.csv')
rng = np.random.default_rng(1)
pick = man.iloc[rng.choice(len(man), 24, replace=False)]
worst = 0.0
for _, r in pick.iterrows():
    z = np.load(R / 'stacks' / r.path); rep = np.load(R / 'replicates' / r.path)
    seed = int(hashlib.sha256(f'{r.model}/{r.member}'.encode()).hexdigest()[:8], 16)
    idx = block_indices(seed=seed)
    sl = mmm.loc[(r.month, r.region), ['trend_a', 'trend_c', 'trend_t']].to_numpy(float)
    a = np.asarray(z['a'], float) @ z['w']; c = np.asarray(z['c'], float) @ z['w']
    L, I = refits_direct(a, c, z['t'], z['years'], sl, idx)
    dL = np.max(np.abs(L - rep['L'])) / np.std(rep['L']); dI = np.max(np.abs(I - rep['I'])) / np.std(rep['I'])
    worst = max(worst, dL, dI)
    print(f'{r.path:45s} max|dL|/sd {dL:.1e}  max|dI|/sd {dI:.1e}  var ratio L {np.var(L)/np.var(rep["L"]):.6f} I {np.var(I)/np.var(rep["I"]):.6f}')
print('WORST', worst)
