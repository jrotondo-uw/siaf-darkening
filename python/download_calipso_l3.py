#!/usr/bin/env python
"""
download_calipso_l3.py
──────────────────────
Download CALIPSO CALIOP Level-3 monthly cloud occurrence for the Arctic
(≥55°N), AMJJAS 2006–2023, and save derived cloud fraction fields to NetCDF.

Product : CAL_LID_L3_Cloud_Occurrence-Standard-V1-00
Format  : HDF4, 3-D profile (altitude × lat × lon)
Provider: NASA ASDC

The product stores raw sample counts at each altitude level.  This script
computes cloud fraction from counts — no pre-baked CF variable exists.

CF formula at each altitude level z:
  valid(z)  = Cloud_Accepted_Samples(z) + Cloud_Free_Samples(z)
  CF(z)     = Cloud_Accepted_Samples(z) / valid(z)   [0-1, NaN where no data]

Output variables (all collapsed to 2-D lat×lon by mean over altitude):
  cf_total   — any-cloud occurrence fraction (all types)
  cf_water   — liquid water cloud fraction   (most relevant for SW CRE)
  cf_ice     — ice cloud fraction
  cf_opaque  — opaque cloud fraction only (thick clouds that dominate SWCRE)

Why CALIPSO: active lidar is NOT biased by bright ice/snow surfaces, unlike
MODIS passive optical.  Use this to cross-check MODIS CF trends.

Limitation: 2006–2023 only.

Setup (run once):
  module load conda && conda activate npl
  conda install -c conda-forge pyhdf -y
  pip install earthaccess --quiet

Earthdata login (free at urs.earthdata.nasa.gov):
  export EARTHDATA_USERNAME=jrotondo
  export EARTHDATA_PASSWORD=YOUR_PASSWORD

Output: /glade/work/jrotondo/calipso_cf_arctic_amjjas.nc
"""

import gc, glob, os, re, sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_FILE   = '/glade/work/jrotondo/calipso_cf_arctic_amjjas.nc'
SCRATCH    = '/glade/derecho/scratch/jrotondo/calipso_raw/'
MONTH_NUMS = [4, 5, 6, 7, 8, 9]
YEARS      = list(range(2006, 2024))   # CALIPSO: June 2006 – 2023
LAT_MIN    = 55.0

# Grid will be read from the file's Latitude_Midpoint / Longitude_Midpoint
# coordinate arrays.  These placeholders are filled on first file read.
# Confirmed dimension order from probe: (lat=85, lon=144, alt=344)
#   lat: -84 → 84 at 2.0° spacing (85 bins, S→N)
#   lon: -178.75 → 178.75 at 2.5° spacing (144 bins)
#   alt: -0.38 → 20.167 km (344 levels)
LAT_GRID = None
LON_GRID = None
LAT_ROWS = None   # indices into LAT_GRID (axis 0) where lat >= LAT_MIN

os.makedirs(SCRATCH, exist_ok=True)

# ── CHECK DEPENDENCIES ────────────────────────────────────────────────────────
try:
    import earthaccess
except ImportError:
    sys.exit('earthaccess not installed.  Run: pip install earthaccess')

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    sys.exit('pyhdf not installed.  Run: conda install -c conda-forge pyhdf -y')

# ── AUTHENTICATE ──────────────────────────────────────────────────────────────
print('Authenticating with NASA Earthdata ...')
earthaccess.login()
print('  OK\n')


# ── HELPER: parse (year, month) from filename ─────────────────────────────────
# Filename: CAL_LID_L3_Cloud_Occurrence-Standard-V1-00.YYYY-MMD.hdf
# "D" suffix = daytime-only granule (we want those for Arctic summer)
# "N" suffix = nighttime; no suffix = combined day+night
_DATE_RE = re.compile(r'\.(\d{4})-(\d{2})')

def _fname_to_ym(fname):
    m = _DATE_RE.search(Path(fname).name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _fname_type(fname):
    """Return 'D', 'N', or 'A' (all) based on filename suffix."""
    name = Path(fname).stem   # strip .hdf
    if name.endswith('D'): return 'D'
    if name.endswith('N'): return 'N'
    return 'A'


# ── HELPER: read one HDF4 file → dict of 2-D Arctic arrays ───────────────────
def read_calipso_arctic(hdf_path):
    """
    Reads sample count arrays, computes cloud fraction at each altitude,
    then collapses to 2-D (lat × lon) by averaging over altitude.

    Returns dict of {varname: 2-D float32 (n_arctic_lat × n_lon)}
    """
    global LAT_GRID, LON_GRID, LAT_ROWS

    result = {}
    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception as e:
        print(f'    ERROR opening {hdf_path}: {e}')
        return result

    try:
        # ── Read coordinate arrays (first time only) ──────────────────────────
        # Dimension order confirmed by probe: (lat, lon, alt)
        if LAT_GRID is None:
            lat_raw = hdf.select('Latitude_Midpoint').get()   # shape (85,) S→N
            lon_raw = hdf.select('Longitude_Midpoint').get()  # shape (144,)
            LAT_GRID = lat_raw.astype(np.float32)   # keep S→N; LAT_ROWS selects ≥55°N
            LON_GRID = lon_raw.astype(np.float32)
            LAT_ROWS = np.where(LAT_GRID >= LAT_MIN)[0]
            print(f'  Grid: lat {LAT_GRID[0]:.2f}→{LAT_GRID[-1]:.2f} '
                  f'({len(LAT_GRID)} bins, 2°)  '
                  f'lon {LON_GRID[0]:.2f}→{LON_GRID[-1]:.2f} '
                  f'({len(LON_GRID)} bins, 2.5°)')
            print(f'  Arctic rows (≥{LAT_MIN}°N): {len(LAT_ROWS)}')

        # ── Read sample count arrays — shape is (lat, lon, alt) ──────────────
        def _read(name):
            return hdf.select(name).get().astype(np.float64)  # (85, 144, 344)

        cloud_acc  = _read('Cloud_Accepted_Samples')
        cloud_free = _read('Cloud_Free_Samples')
        water_op   = _read('Water_Cloud_Opaque_Samples')
        water_tr   = _read('Water_Cloud_Transparent_Samples')
        ice_op     = _read('Ice_Cloud_Opaque_Samples')
        ice_tr     = _read('Ice_Cloud_Transparent_Samples')

        # ── Compute CF at each altitude level ─────────────────────────────────
        valid = cloud_acc + cloud_free
        with np.errstate(invalid='ignore', divide='ignore'):
            cf_any    = np.where(valid > 0, cloud_acc / valid,             np.nan)
            cf_water  = np.where(valid > 0, (water_op + water_tr) / valid, np.nan)
            cf_ice    = np.where(valid > 0, (ice_op   + ice_tr)   / valid, np.nan)
            cf_opaque = np.where(valid > 0, (water_op + ice_op)   / valid, np.nan)

        # ── Collapse altitude (axis=2) → 2-D (lat, lon) ──────────────────────
        # Use nanmax over altitude, NOT nanmean.
        # nanmean gives ~5% because most altitude levels (0-20 km) are clear;
        # nanmax gives the peak occurrence across all levels = column CF proxy
        # (~70-90% for Arctic summer), which is what MODIS/ERA5 report.
        def _collapse(arr3d):
            # arr3d shape: (85 lat, 144 lon, 344 alt)
            arctic_slice = arr3d[LAT_ROWS, :, :]      # (n_arctic_lat, 144, 344)
            return np.nanmax(arctic_slice, axis=2).astype(np.float32)   # (n_arctic_lat, 144)

        result['cf_total']  = _collapse(cf_any)
        result['cf_water']  = _collapse(cf_water)
        result['cf_ice']    = _collapse(cf_ice)
        result['cf_opaque'] = _collapse(cf_opaque)

    except Exception as e:
        print(f'    ERROR reading {hdf_path}: {e}')
    finally:
        hdf.end()

    return result


# ── STEP 1: ONE bulk search ────────────────────────────────────────────────────
# We prefer the Daytime product (suffix D) for Arctic summer — CALIPSO has
# better spatial sampling in daytime for AMJJAS.  Download all then prefer D.
print('Searching for CALIPSO L3 Cloud Occurrence granules (2006–2023) ...')
all_granules = earthaccess.search_data(
    short_name='CAL_LID_L3_Cloud_Occurrence-Standard-V1-00',
    temporal=('2006-01-01', '2023-12-31'),
    count=800,   # day + night + combined = up to 3× per month
)
print(f'  Found {len(all_granules)} total granules')

if len(all_granules) == 0:
    print('  Retrying with provider=LARC_ASDC ...')
    all_granules = earthaccess.search_data(
        short_name='CAL_LID_L3_Cloud_Occurrence-Standard-V1-00',
        provider='LARC_ASDC',
        temporal=('2006-01-01', '2023-12-31'),
        count=800,
    )
    print(f'  Found {len(all_granules)} granules (LARC_ASDC)')

if len(all_granules) == 0:
    sys.exit('No granules found.  Check short_name with '
             'earthaccess.search_datasets(short_name="CAL_LID*", count=10)')

# Build (year, month, type) → granule map; prefer Daytime (D) over All (A)
granule_map = {}   # (yr, mo) → granule  [D preferred, else A, else N]
PREF = {'D': 0, 'A': 1, 'N': 2}

for g in all_granules:
    try:
        links = g.data_links(access='external')
        if not links:
            links = g.data_links()
        fname = Path(links[0]).name
        yr, mo = _fname_to_ym(fname)
        typ    = _fname_type(fname)
        if yr and mo:
            key = (yr, mo)
            if key not in granule_map or PREF[typ] < PREF[_fname_type(
                    Path(granule_map[key].data_links()[0]).name)]:
                granule_map[key] = g
    except Exception:
        pass

target_keys = {(y, m) for y in YEARS for m in MONTH_NUMS}
found_keys  = target_keys & set(granule_map.keys())
missing     = target_keys - found_keys

print(f'  {len(found_keys)}/{len(target_keys)} target months matched')
if missing:
    print(f'  Missing (first 10): '
          f'{[f"{y}-{m:02d}" for y,m in sorted(missing)[:10]]}')


# ── STEP 2: Download + extract ────────────────────────────────────────────────
records = {}
n_total = len(found_keys)
n_done  = 0

for (year, month) in sorted(found_keys):
    n_done += 1
    tag = f'{year}-{month:02d}'

    # Check if already cached (any suffix)
    existing = (glob.glob(f'{SCRATCH}/CALIPSO_L3_{year}{month:02d}*.hdf') +
                glob.glob(f'{SCRATCH}/CALIPSO_L3_{year}{month:02d}*.h5'))
    if existing:
        cached = Path(existing[0])
    else:
        g = granule_map[(year, month)]
        try:
            earthaccess.download([g], local_path=SCRATCH)
        except Exception as e:
            print(f'  [{n_done}/{n_total}] {tag}  download error: {e}')
            continue

        # Rename to predictable filename preserving D/N/A suffix
        raw = (sorted(glob.glob(f'{SCRATCH}/CAL_LID_L3_Cloud_Occurrence*'
                                f'{year}-{month:02d}*.hdf')) +
               sorted(glob.glob(f'{SCRATCH}/CAL_LID_L3_Cloud_Occurrence*'
                                f'{year}-{month:02d}*.h5')))
        if not raw:
            print(f'  [{n_done}/{n_total}] {tag}  downloaded file not found')
            continue
        typ    = _fname_type(raw[-1])
        suffix = Path(raw[-1]).suffix
        cached = Path(SCRATCH) / f'CALIPSO_L3_{year}{month:02d}{typ}{suffix}'
        os.rename(raw[-1], cached)

    row = read_calipso_arctic(cached)
    if not row:
        print(f'  [{n_done}/{n_total}] {tag}  EMPTY read')
        continue

    records[(year, month)] = row
    cf_str = f'CF_total={np.nanmean(row["cf_total"]):.3f}  '  \
             f'CF_water={np.nanmean(row["cf_water"]):.3f}'
    print(f'  [{n_done}/{n_total}] {tag}  {cf_str}')
    gc.collect()


# ── STEP 3: Package into NetCDF ───────────────────────────────────────────────
print(f'\n{len(records)} month-fields extracted.  Building NetCDF ...')

if LAT_GRID is None:
    sys.exit('No files were read successfully — cannot build output grid.')

keys_sorted = sorted(records.keys())
times       = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for (y, m) in keys_sorted])
arctic_lat  = LAT_GRID[LAT_ROWS]

VAR_DESCS = {
    'cf_total':  'Total column cloud fraction (all types, mean over altitude)',
    'cf_water':  'Liquid water cloud fraction (opaque + transparent)',
    'cf_ice':    'Ice cloud fraction (opaque + transparent)',
    'cf_opaque': 'Opaque cloud fraction (water + ice, optically thick only)',
}

data_vars = {}
for vname in ['cf_total', 'cf_water', 'cf_ice', 'cf_opaque']:
    arr = np.full((len(times), len(arctic_lat), len(LON_GRID)),
                  np.nan, dtype=np.float32)
    for i, key in enumerate(keys_sorted):
        if vname in records[key]:
            arr[i] = records[key][vname]
    data_vars[vname] = xr.DataArray(
        arr,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': times,
            'lat':  xr.DataArray(arctic_lat, dims='lat',
                                 attrs={'units': 'degrees_north'}),
            'lon':  xr.DataArray(LON_GRID, dims='lon',
                                 attrs={'units': 'degrees_east'}),
        },
        attrs={'long_name': VAR_DESCS[vname],
               'units': '1',
               'source': 'CALIPSO CALIOP L3 Cloud Occurrence V1-00'}
    )

ds = xr.Dataset(
    data_vars,
    attrs={
        'description': ('CALIPSO CALIOP L3 Cloud Occurrence — Arctic (≥55°N) '
                        'AMJJAS 2006–2023'),
        'product':   'CAL_LID_L3_Cloud_Occurrence-Standard-V1-00',
        'platform':  'CALIPSO',
        'retrieval': 'Active lidar — not biased by ice/snow surface reflectance',
        'cf_method': ('CF computed from sample counts: '
                      'Cloud_Accepted / (Cloud_Accepted + Cloud_Free), '
                      'then averaged over all altitude levels'),
        'key_var':   'cf_water (liquid clouds) most relevant for SW CRE comparison',
    }
)

encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
            for v in data_vars}
ds.to_netcdf(OUT_FILE, encoding=encoding)

print(f'\nSaved → {OUT_FILE}')
print(f'  {len(times)} times × {len(arctic_lat)} lat × {len(LON_GRID)} lon')
print(f'  Variables: {list(data_vars.keys())}')
print(f'\nKey comparison:')
print(f'  cf_water  — liquid clouds → compare with MODIS CF and your SW CRE signal')
print(f'  cf_total  — all clouds    → compare with ERA5 TCC to see reanalysis bias')
print(f'\nDelete scratch when done:  rm -rf {SCRATCH}')
