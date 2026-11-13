#!/usr/bin/env python
"""Probe one cached MOD08_M3 HDF to find the correct water vapour variable name."""
import glob
from pyhdf.SD import SD, SDC

SCRATCH = '/glade/derecho/scratch/jrotondo/modis_cod_raw/'
files   = sorted(glob.glob(f'{SCRATCH}/MOD08_M3_*.hdf'))
if not files:
    raise SystemExit(f'No cached HDF files found in {SCRATCH}')

hdf = SD(files[0], SDC.READ)
datasets = hdf.datasets()
print(f'Probing: {files[0]}\nTotal SDS: {len(datasets)}\n')

print('── Water / vapour variables ─────────────────────────────────────────')
for name, info in sorted(datasets.items()):
    if any(k in name.lower() for k in ['water', 'vapor', 'vapour', 'pwv', 'tpw', 'precip']):
        sds   = hdf.select(name)
        attrs = sds.attributes()
        sds.endaccess()
        print(f'  {name}')
        print(f'    shape={info[1]}  scale={attrs.get("scale_factor","?")}  '
              f'units={attrs.get("units","")}')

hdf.end()
