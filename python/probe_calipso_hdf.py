#!/usr/bin/env python
"""
probe_calipso_hdf.py
─────────────────────
List every dataset in the first CALIPSO L3 HDF file in scratch.
Also tries HDF5 in case the collection version uses that format.
Run interactively on Derecho to confirm variable names before the full download.

Usage:
  python probe_calipso_hdf.py
"""
import glob, sys
from pathlib import Path

SCRATCH = '/glade/derecho/scratch/jrotondo/calipso_raw/'

files = (sorted(glob.glob(f'{SCRATCH}/CALIPSO*.hdf')) +
         sorted(glob.glob(f'{SCRATCH}/CAL_LID*.hdf'))  +
         sorted(glob.glob(f'{SCRATCH}/CALIPSO*.h5'))   +
         sorted(glob.glob(f'{SCRATCH}/CAL_LID*.h5')))

if not files:
    # Nothing downloaded yet — download one file manually to probe:
    print('No files found.  Downloading one granule to probe ...')
    try:
        import earthaccess
        earthaccess.login()
        r = earthaccess.search_data(
            short_name='CAL_LID_L3_Cloud_Occurrence-Standard-V1-00',
            temporal=('2010-07-01', '2010-07-31'),
            count=1,
        )
        if not r:
            r = earthaccess.search_data(
                short_name='CAL_LID_L3_Cloud_Occurrence-Standard-V1-00',
                provider='LARC_ASDC',
                temporal=('2010-07-01', '2010-07-31'),
                count=1,
            )
        if r:
            import os; os.makedirs(SCRATCH, exist_ok=True)
            earthaccess.download(r[:1], local_path=SCRATCH)
            files = (sorted(glob.glob(f'{SCRATCH}/*.hdf')) +
                     sorted(glob.glob(f'{SCRATCH}/*.h5')))
        if not files:
            sys.exit('Download attempt failed — check credentials and short_name.')
    except Exception as e:
        sys.exit(f'Could not auto-download: {e}')

fname = files[0]
print(f'Probing: {fname}\n')

# ── Try HDF4 ──────────────────────────────────────────────────────────────────
if fname.endswith('.hdf'):
    try:
        from pyhdf.SD import SD, SDC
        hdf = SD(fname, SDC.READ)
        datasets = hdf.datasets()
        print(f'Format: HDF4   |  Total SDS datasets: {len(datasets)}\n')

        print('── Cloud-related ─────────────────────────────────────────────')
        cloud_vars = sorted(k for k in datasets if 'cloud' in k.lower() or 'Cloud' in k)
        for name in cloud_vars:
            info = datasets[name]
            try:
                sds   = hdf.select(name)
                attrs = sds.attributes()
                sds.endaccess()
                scale  = attrs.get('scale_factor', 'N/A')
                offset = attrs.get('add_offset',   'N/A')
                fill   = attrs.get('_FillValue',   'N/A')
                units  = attrs.get('units', '')
                print(f'  {name}')
                print(f'    shape={info[1]}  scale={scale}  '
                      f'offset={offset}  fill={fill}  units={units}')
            except Exception as e:
                print(f'  {name}  (attr error: {e})')

        print('\n── All other SDS names ───────────────────────────────────────')
        other_vars = sorted(k for k in datasets if 'cloud' not in k.lower() and 'Cloud' not in k)
        for name in other_vars:
            print(f'  {name}')

        hdf.end()
        sys.exit(0)
    except ImportError:
        print('pyhdf not available, trying h5py ...')
    except Exception as e:
        print(f'HDF4 open failed ({e}), trying h5py ...')

# ── Try HDF5 ──────────────────────────────────────────────────────────────────
try:
    import h5py

    def _print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            tag = '[cloud]' if 'cloud' in name.lower() else ''
            print(f'  {name}  shape={obj.shape}  dtype={obj.dtype}  {tag}')

    with h5py.File(fname, 'r') as f:
        print(f'Format: HDF5   |  Top-level keys: {list(f.keys())}\n')
        print('── All datasets (filtered to cloud-related) ──────────────────')
        f.visititems(lambda n, o: _print_item(n, o)
                     if 'cloud' in n.lower() else None)
        print('\n── All datasets ──────────────────────────────────────────────')
        f.visititems(_print_item)

except ImportError:
    sys.exit('Neither pyhdf nor h5py available.  '
             'conda install -c conda-forge pyhdf -y  OR  pip install h5py')
except Exception as e:
    sys.exit(f'HDF5 open failed: {e}')

