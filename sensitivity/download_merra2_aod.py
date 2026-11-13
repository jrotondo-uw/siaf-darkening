#!/usr/bin/env python
"""
download_merra2_aod.py
──────────────────────
Download MERRA-2 monthly total aerosol optical depth (TOTEXTTAU, 550 nm)
for the Arctic, AMJJAS 2000–2024, and save to NetCDF.

Product : M2TMNXAER  (MERRA-2 Monthly Mean, 2D Aerosol Diagnostics)
Grid    : 0.5° × 0.625° global
Variable: TOTEXTTAU — total aerosol extinction optical depth at 550 nm
          (dust + sea salt + black carbon + OC + sulfate)
Source  : NASA GES DISC via earthaccess

Why MERRA-2 over MODIS: passive satellite AOD (Dark Target) returns NaN
over bright Arctic ice/snow. MERRA-2 assimilates MODIS where available
and fills with a coupled chemistry model elsewhere — the only full-Arctic
AOD product available monthly back to 2000.

Output: /glade/work/jrotondo/merra2_aod_arctic_amjjas.nc

Setup (run once):
  pip install earthaccess --quiet
  # Earthdata login at urs.earthdata.nasa.gov (same account as CALIPSO)
  export EARTHDATA_USERNAME=jrotondo
  export EARTHDATA_PASSWORD=YOUR_PASSWORD
"""

import gc, glob, os, sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_FILE   = '/glade/work/jrotondo/merra2_aod_arctic_amjjas.nc'
SCRATCH    = '/glade/derecho/scratch/jrotondo/merra2_aod_raw/'
MONTH_NUMS = [4, 5, 6, 7, 8, 9]
YEARS      = list(range(2000, 2025))   # CERES overlap period
LAT_MIN    = 50.0                      # generous south boundary for regridding

os.makedirs(SCRATCH, exist_ok=True)

try:
    import earthaccess
except ImportError:
    sys.exit('pip install earthaccess')

# ── AUTHENTICATE ──────────────────────────────────────────────────────────────
print('Authenticating with NASA Earthdata ...')
earthaccess.login()
print('  OK\n')

# ── SEARCH ───────────────────────────────────────────────────────────────────
# M2TMNXAER: MERRA-2 Monthly Mean, 2D Aerosol Diagnostics, 0.5°×0.625°
# One file per month, ~4 MB each (small — all variables bundled).
print('Searching for MERRA-2 M2TMNXAER granules (2000–2024) ...')
all_granules = earthaccess.search_data(
    short_name='M2TMNXAER',
    temporal=('2000-01-01', '2024-12-31'),
    count=400,
)
print(f'  Found {len(all_granules)} granules')

if len(all_granules) == 0:
    sys.exit('No granules found. Check short_name or credentials.')

# Build (year, month) → granule map from temporal metadata
granule_map = {}
for g in all_granules:
    try:
        # MERRA-2 granule temporal info is reliable from metadata
        t_start = g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        yr = int(t_start[:4]); mo = int(t_start[5:7])
        if mo in MONTH_NUMS and yr in YEARS:
            granule_map[(yr, mo)] = g
    except Exception:
        # Fallback: try parsing from data link filename
        try:
            links = g.data_links()
            fname = Path(links[0]).name   # e.g. MERRA2_400.tavgM_2d_aer_Nx.200001.nc4
            parts = fname.split('.')
            date_part = next(p for p in parts if len(p) == 6 and p.isdigit())
            yr = int(date_part[:4]); mo = int(date_part[4:6])
            if mo in MONTH_NUMS and yr in YEARS:
                granule_map[(yr, mo)] = g
        except Exception:
            pass

target_keys = {(y, m) for y in YEARS for m in MONTH_NUMS}
found_keys  = target_keys & set(granule_map.keys())
missing     = target_keys - found_keys
print(f'  {len(found_keys)}/{len(target_keys)} target months matched')
if missing:
    print(f'  Missing (first 10): {[f"{y}-{m:02d}" for y,m in sorted(missing)[:10]]}')

# ── DOWNLOAD + EXTRACT ────────────────────────────────────────────────────────
records = {}
n_total = len(found_keys)
n_done  = 0

for (year, month) in sorted(found_keys):
    n_done += 1
    tag = f'{year}-{month:02d}'

    cached_pat = list(Path(SCRATCH).glob(f'MERRA2_*{year}{month:02d}*.nc4'))
    if cached_pat:
        cached = cached_pat[0]
    else:
        g = granule_map[(year, month)]
        try:
            earthaccess.download([g], local_path=SCRATCH)
        except Exception as e:
            print(f'  [{n_done}/{n_total}] {tag}  download error: {e}')
            continue
        nc4s = sorted(Path(SCRATCH).glob(f'MERRA2_*{year}{month:02d}*.nc4'))
        if not nc4s:
            print(f'  [{n_done}/{n_total}] {tag}  file not found after download')
            continue
        cached = nc4s[0]

    # Extract Arctic subset of TOTEXTTAU
    try:
        with xr.open_dataset(cached, engine='netcdf4') as ds:
            # Standardise lat/lon name
            lat_name = next(n for n in ['lat', 'latitude'] if n in ds.coords)
            lon_name = next(n for n in ['lon', 'longitude'] if n in ds.coords)
            lats = ds[lat_name].values
            row_mask = lats >= LAT_MIN
            arctic_lats = lats[row_mask]

            aod = ds['TOTEXTTAU'].values   # (time, lat, lon) or (lat, lon)
            if aod.ndim == 3:
                aod = aod[0]              # squeeze single time dim if present
            aod_arctic = aod[row_mask, :].astype(np.float32)
            aod_arctic = np.where(aod_arctic >= 0., aod_arctic, np.nan)

            if 'lat' not in records:
                records['lat'] = arctic_lats
                records['lon'] = ds[lon_name].values
            records[(year, month)] = aod_arctic

        mu = np.nanmean(aod_arctic[aod_arctic <= 2.])   # exclude fill
        print(f'  [{n_done}/{n_total}] {tag}  AOD_mean={mu:.4f}')
    except Exception as e:
        print(f'  [{n_done}/{n_total}] {tag}  read error: {e}')

    gc.collect()

# ── PACKAGE ───────────────────────────────────────────────────────────────────
print(f'\n{len(records) - 2} month-fields extracted.  Building NetCDF ...')  # -2 for lat/lon keys

keys_sorted = sorted(k for k in records if isinstance(k, tuple))
times       = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for (y, m) in keys_sorted])
arctic_lats = records['lat']
lons        = records['lon']

arr = np.full((len(times), len(arctic_lats), len(lons)), np.nan, dtype=np.float32)
for i, key in enumerate(keys_sorted):
    arr[i] = records[key]

da = xr.DataArray(
    arr,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': times,
        'lat':  xr.DataArray(arctic_lats, dims='lat',
                             attrs={'units': 'degrees_north'}),
        'lon':  xr.DataArray(lons, dims='lon',
                             attrs={'units': 'degrees_east'}),
    },
    attrs={
        'long_name': 'Total aerosol extinction optical depth at 550 nm',
        'units'    : '1',
        'source'   : 'MERRA-2 M2TMNXAER (TOTEXTTAU)',
        'note'     : ('Sum of dust, sea salt, black carbon, organic carbon, '
                      'sulfate AOD. Includes model-filled values over ice/snow '
                      'where passive satellite retrieval fails.'),
    }
)

ds_out = da.to_dataset(name='aod')
ds_out.attrs = {
    'description': 'MERRA-2 total AOD 550nm — Arctic (≥50°N) AMJJAS 2000–2024',
    'product'    : 'M2TMNXAER',
    'variable'   : 'TOTEXTTAU',
}
encoding = {'aod': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
ds_out.to_netcdf(OUT_FILE, encoding=encoding)

print(f'\nSaved → {OUT_FILE}')
print(f'  {len(times)} times × {len(arctic_lats)} lat × {len(lons)} lon')
print(f'\nDelete scratch when done:  rm -rf {SCRATCH}')
