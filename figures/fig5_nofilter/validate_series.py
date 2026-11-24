"""Remove unreadable/partial series files after the duplicate-worker race and
reset the members they belong to, so the relaunched workers redo them."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
bad = []
for p in sorted((R / 'series').glob('*.npz')):
    try:
        z = np.load(p)
        ok = (all(k in z.files for k in ('a', 'c', 't', 'years', 'baseline_L', 'baseline_I'))
              and np.isfinite(z['a']).all() and np.isfinite(z['c']).all()
              and len(z['a']) == len(z['years']) == len(z['t']))
    except Exception:
        ok = False
    if not ok:
        bad.append(p)
members = sorted({tuple(p.name.split('__')[:2]) for p in bad})
for p in bad:
    p.unlink()
for m, mb in members:
    s = R / 'status' / f'{m}__{mb}.json'
    if s.exists():
        s.unlink()
redo = []
for s in (R / 'status').glob('*.json'):
    d = json.loads(s.read_text())
    n = len(list((R / 'series').glob(f"{d['model']}__{d['member']}__*.npz")))
    if d.get('status') == 'complete' and n != d.get('n_records'):
        redo.append(s.name)
        s.unlink()
print('series files:', len(list((R / 'series').glob('*.npz'))), '| bad removed:', len(bad),
      '| members reset:', members, '| count mismatches reset:', redo)
