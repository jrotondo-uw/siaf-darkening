#!/usr/bin/env python
"""
download_era5_tcwv.py
─────────────────────
Download ERA5 total column water vapour (TCWV, kg m⁻²) for the Arctic,
AMJJAS 1979–2025, and save to NetCDF.

Variable: total_column_water_vapour  (single-level, monthly means)
Dataset : reanalysis-era5-single-levels-monthly-means
          product_type: monthly_averaged_reanalysis

Units   : kg m⁻²  (equivalent to mm precipitable water)
Grid    : 0.25° × 0.25° global  (ERA5 native; subset to ≥50°N here)

Why TCWV: water vapour absorbs SW in near-IR bands (1.4, 1.9, 2.7 µm).
          Arctic moistening (more evaporation from open ocean) may reduce
          surface SW even with flat cloud fraction.

Output: /glade/work/jrotondo/era5_tcwv_arctic_amjjas.nc

Setup (run once):
  module load conda && conda activate npl
  pip install cdsapi --quiet

CDS credentials (~/.cdsapirc):
  url: https://cds.climate.copernicus.eu/api
  key: YOUR_CDS_API_KEY
  (Get key free at cds.climate.copernicus.eu → your profile)
"""

import os, sys
import numpy as np
import xarray as xr

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_FILE   = '/glade/work/jrotondo/era5_tcwv_arctic_amjjas.nc'
MONTH_NUMS = [4, 5, 6, 7, 8, 9]           # AMJJAS
YEARS      = list(range(1979, 2026))
LAT_MIN    = 50.0                          # download a little south of analysis
                                           # domain so regridding has no edge gaps

# ── CHECK CDS API ─────────────────────────────────────────────────────────────
try:
    import cdsapi
except ImportError:
    sys.exit('cdsapi not installed.  Run: pip install cdsapi')

if not os.path.exists(os.path.expanduser('~/.cdsapirc')):
    sys.exit(
        'No ~/.cdsapirc found.\n'
        'Create it with:\n'
        '  url: https://cds.climate.copernicus.eu/api\n'
        '  key: YOUR_CDS_API_KEY\n'
        'Get your key at https://cds.climate.copernicus.eu (free account).'
    )

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
# One request for all years/months at once — CDS handles the aggregation.
# Requesting monthly-averaged reanalysis (not ensemble members).
print('Submitting CDS request for ERA5 TCWV ...')
print(f'  Years  : {YEARS[0]}–{YEARS[-1]}')
print(f'  Months : {MONTH_NUMS}')
print(f'  Domain : ≥{LAT_MIN}°N  (global lon)')
print(f'  Output : {OUT_FILE}')
print('  (This may queue; typical wait 5–30 min on first request)\n')

c = cdsapi.Client()

tmp = OUT_FILE.replace('.nc', '_raw.nc')

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable'    : 'total_column_water_vapour',
        'year'        : [str(y) for y in YEARS],
        'month'       : [f'{m:02d}' for m in MONTH_NUMS],
        'time'        : '00:00',          # monthly means have only one time step
        'area'        : [90, -180, LAT_MIN, 180],   # N, W, S, E
        'format'      : 'netcdf',
    },
    tmp,
)
print(f'\nDownload complete → {tmp}')

# ── TIDY UP ───────────────────────────────────────────────────────────────────
# ERA5 CDS output uses 'tcwv' or 'total_column_water_vapour' as the var name
# and 'valid_time' or 'time' as the time coord depending on API version.
print('Tidying output ...')
with xr.open_dataset(tmp) as ds:
    print(f'  Raw vars   : {list(ds.data_vars)}')
    print(f'  Raw coords : {list(ds.coords)}')

    # Standardise variable name
    vname = next((v for v in ['tcwv', 'total_column_water_vapour',
                               'p72.162', 'VAR_72'] if v in ds), None)
    if vname is None:
        vname = list(ds.data_vars)[0]
        print(f'  WARNING: unexpected var name, using first: {vname}')

    # Standardise time coord
    tname = 'valid_time' if 'valid_time' in ds.coords else 'time'

    da = ds[vname].rename('tcwv')
    da.attrs.update({
        'long_name': 'Total column water vapour',
        'units'    : 'kg m-2',
        'source'   : 'ERA5 reanalysis-era5-single-levels-monthly-means',
        'note'     : ('AMJJAS monthly means, ≥50°N. '
                      '1 kg/m² ≈ 1 mm precipitable water.'),
    })

    # Rename coords to standard names
    rename = {}
    if 'valid_time' in da.coords: rename['valid_time'] = 'time'
    if 'latitude'   in da.coords: rename['latitude']   = 'lat'
    if 'longitude'  in da.coords: rename['longitude']  = 'lon'
    if rename:
        da = da.rename(rename)

    ds_out = da.to_dataset()
    ds_out.attrs = {
        'description': 'ERA5 total column water vapour — Arctic AMJJAS 1979–2025',
        'product'    : 'reanalysis-era5-single-levels-monthly-means',
        'variable'   : 'total_column_water_vapour',
    }

    encoding = {'tcwv': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
    ds_out.to_netcdf(OUT_FILE, encoding=encoding)
    print(f'  Times   : {len(da.time)} steps')
    print(f'  Lat     : {float(da.lat.min()):.2f} → {float(da.lat.max()):.2f}')

os.remove(tmp)
print(f'\nSaved → {OUT_FILE}')
print(f'Delete tmp if it lingers: rm -f {tmp}')
