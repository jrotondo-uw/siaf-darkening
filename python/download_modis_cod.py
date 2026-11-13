#!/usr/bin/env python
"""
download_modis_cod.py
─────────────────────
Download MOD08_M3 v6.1 (MODIS/Terra Monthly 1°×1° Atmosphere) and extract
cloud optical depth + cloud fraction for the Arctic (≥55°N), AMJJAS 2000–2025.

Key fix vs previous version: ONE search for the full date range, then filter
client-side. This avoids NASA CMR rate-limiting that kills per-month searches.

Setup (run once on Derecho before submitting the PBS job):
  module load conda && conda activate npl
  conda install -c conda-forge pyhdf -y    # bundles libhdf4 — easier than pip
  pip install earthaccess --quiet

Earthdata credentials (free account at urs.earthdata.nasa.gov):
  export EARTHDATA_USERNAME=your_username
  export EARTHDATA_PASSWORD=your_password
  (or stored in ~/.netrc — earthaccess will find them automatically)

Output: /glade/work/jrotondo/modis_cod_arctic_amjjas.nc
"""

import gc, glob, os, re, sys, calendar
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_FILE   = '/glade/work/jrotondo/modis_cod_arctic_amjjas.nc'
SCRATCH    = '/glade/derecho/scratch/jrotondo/modis_cod_raw/'
MONTH_NUMS = [4, 5, 6, 7, 8, 9]
YEARS      = list(range(2000, 2026))
LAT_MIN    = 55.0

# MOD08_M3 1°×1° global grid (cell centres, 90N → 90S)
LAT_GRID = np.arange(89.5,  -90.5, -1.0)   # 180 values
LON_GRID = np.arange(-179.5, 180.5,  1.0)  # 360 values
LAT_ROWS = np.where(LAT_GRID >= LAT_MIN)[0]  # 35 rows (55–89.5°N)

# Variable names as they actually appear in MOD08_M3 v6.1 HDF4 files.
# Note: Collection 6.1 does NOT store COD as a simple 2D mean — only in
# joint histograms.  Cloud Water Path (CWP) is proportional to optical depth
# (τ ∝ CWP/r_eff) and is the right proxy for "thicker clouds" vs "more clouds".
VARS_WANT = [
    'Cloud_Retrieval_Fraction_Combined_FMean',  # cloud fraction 0-1 (scale=1e-4)
    'Cloud_Water_Path_Liquid_Mean_Mean',         # liquid CWP g/m² (key for SW CRE)
    'Cloud_Water_Path_Ice_Mean_Mean',            # ice CWP g/m²
    'Cloud_Top_Pressure_Mean_Mean',              # cloud top pressure hPa (scale=0.1)
    'High_Cloud_Fraction_Infrared_FMean',        # high cloud fraction (scale=1e-4)
]

os.makedirs(SCRATCH, exist_ok=True)

# ── CHECK DEPENDENCIES ────────────────────────────────────────────────────────
try:
    import earthaccess
except ImportError:
    sys.exit('earthaccess not installed.  Run: pip install earthaccess')

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    sys.exit('pyhdf not installed.\n'
             'Run: conda install -c conda-forge pyhdf -y\n'
             '  or: pip install pyhdf   (needs libhdf4 system library)')

# ── AUTHENTICATE ──────────────────────────────────────────────────────────────
print('Authenticating with NASA Earthdata ...')
earthaccess.login()   # tries env vars, .netrc, then interactive prompt
print('  OK\n')


# ── HELPER: parse (year, month) from MOD08_M3 filename ───────────────────────
# Filename pattern: MOD08_M3.AYYYYDDD.061.YYYYDDDHHMMSS.hdf
# YYYYDDD = year + day-of-year of the first day of the month
_DOY_RE = re.compile(r'MOD08_M3\.A(\d{4})(\d{3})\.')

def _fname_to_ym(fname):
    m = _DOY_RE.search(Path(fname).name)
    if not m:
        return None, None
    yr  = int(m.group(1))
    doy = int(m.group(2))
    # Convert DOY → month (day-of-year of Jan 1 = 1)
    import datetime
    dt = datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)
    return yr, dt.month


# ── HELPER: read one HDF4 file ────────────────────────────────────────────────
def read_hdf4_arctic(hdf_path):
    """Return dict of {varname: 2-D float32 (n_arctic_lat × 360)} or {}."""
    result = {}
    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception as e:
        print(f'    ERROR opening {hdf_path}: {e}')
        return result

    for vname in VARS_WANT:
        try:
            sds   = hdf.select(vname)
            raw   = sds.get().astype(np.float64)
            attrs = sds.attributes()
            sds.endaccess()

            scale  = float(attrs.get('scale_factor', 1.0))
            offset = float(attrs.get('add_offset',   0.0))
            fill   = attrs.get('_FillValue', None)

            if fill is not None:
                raw = np.where(raw == fill, np.nan, raw)
            data = (scale * (raw - offset)).astype(np.float32)

            result[vname] = data[LAT_ROWS, :]   # Arctic subset
        except Exception as e:
            print(f'    WARNING: {vname} → {e}')

    hdf.end()
    return result


# ── STEP 1: ONE bulk search for all MOD08_M3 granules 2000-2025 ───────────────
# This avoids the per-month API rate-limiting that killed the previous approach.
print('Searching for all MOD08_M3 granules (2000-2025) in one request ...')
all_granules = earthaccess.search_data(
    concept_id='C1443766359-LAADS',   # MOD08_M3 v6.1 — version='6.1' also works
    temporal=('2000-01-01', '2025-12-31'),
    count=400,   # 26 yr × 12 mon = 312 max; 400 is safe headroom
)
print(f'  Found {len(all_granules)} total granules')

# Build a (year, month) → granule lookup using the filename DOY
granule_map = {}
for g in all_granules:
    try:
        links = g.data_links(access='external')
        if not links:
            links = g.data_links()
        fname = Path(links[0]).name
        yr, mo = _fname_to_ym(fname)
        if yr and mo:
            granule_map[(yr, mo)] = g
    except Exception:
        pass

# Filter to AMJJAS within our year range
target_keys = {(y, m) for y in YEARS for m in MONTH_NUMS}
found_keys  = target_keys & set(granule_map.keys())
missing     = target_keys - found_keys

print(f'  {len(found_keys)}/{len(target_keys)} target months matched')
if missing:
    missing_list = sorted(missing)[:10]
    print(f'  Missing (first 10): {[f"{y}-{m:02d}" for y,m in missing_list]}')


# ── STEP 2: Download + extract, skipping already-present files ────────────────
records  = {}   # (year, month) → dict of {varname: 2-D array}
n_total  = len(found_keys)
n_done   = 0

for (year, month) in sorted(found_keys):
    n_done += 1
    tag    = f'{year}-{month:02d}'
    cached = Path(SCRATCH) / f'MOD08_M3_{year}{month:02d}.hdf'

    if not cached.exists():
        g = granule_map[(year, month)]
        try:
            downloaded = earthaccess.download([g], local_path=SCRATCH)
        except Exception as e:
            print(f'  [{n_done}/{n_total}] {tag}  download error: {e}')
            continue

        # earthaccess saves with original NASA filename; rename for predictability
        matches = sorted(glob.glob(f'{SCRATCH}/MOD08_M3.A{year}*.hdf'))
        if matches:
            os.rename(matches[-1], cached)
        else:
            print(f'  [{n_done}/{n_total}] {tag}  downloaded file not found in scratch')
            continue

    # Read and extract Arctic subset
    row = read_hdf4_arctic(cached)
    if not row:
        print(f'  [{n_done}/{n_total}] {tag}  EMPTY read')
        continue

    records[(year, month)] = row

    cwp = row.get('Cloud_Water_Path_Liquid_Mean_Mean')
    cf  = row.get('Cloud_Retrieval_Fraction_Combined_FMean')
    cwp_str = f'CWP_liq={np.nanmean(cwp):.1f} g/m²' if cwp is not None else 'CWP=n/a'
    cf_str  = f'CF={np.nanmean(cf):.3f}'              if cf  is not None else 'CF=n/a'
    print(f'  [{n_done}/{n_total}] {tag}  {cwp_str}  {cf_str}')

    gc.collect()


# ── STEP 3: Package into NetCDF ───────────────────────────────────────────────
print(f'\n{len(records)} month-fields extracted.  Building NetCDF ...')

keys_sorted = sorted(records.keys())
times       = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for (y, m) in keys_sorted])
arctic_lat  = LAT_GRID[LAT_ROWS]

data_vars = {}
for vname in VARS_WANT:
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
                                 attrs={'units': 'degrees_north',
                                        'long_name': 'latitude'}),
            'lon':  xr.DataArray(LON_GRID, dims='lon',
                                 attrs={'units': 'degrees_east',
                                        'long_name': 'longitude'}),
        },
        attrs={'source': 'MODIS Terra MOD08_M3 v6.1', 'long_name': vname}
    )

ds = xr.Dataset(
    data_vars,
    attrs={
        'description': ('MODIS/Terra MOD08_M3 v6.1 — Arctic (≥55°N) '
                        'AMJJAS 2000–2025, 1°×1°'),
        'product':  'MOD08_M3.061',
        'platform': 'Terra',
        'lat_min':  str(LAT_MIN),
        'months':   str(MONTH_NUMS),
    }
)

encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
            for v in data_vars}
ds.to_netcdf(OUT_FILE, encoding=encoding)

print(f'\nSaved → {OUT_FILE}')
print(f'  {len(times)} times  ×  {len(arctic_lat)} lat  ×  {len(LON_GRID)} lon')
print(f'  Variables: {list(data_vars.keys())}')
print(f'\nYou can now delete the raw HDF4 scratch files:')
print(f'  rm -rf {SCRATCH}')