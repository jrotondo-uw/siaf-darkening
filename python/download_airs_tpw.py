#!/usr/bin/env python
"""
download_airs_tpw.py
────────────────────
Download AIRS Level-3 monthly total precipitable water (TPW) for the Arctic,
AMJJAS 2002–2024, and save to NetCDF.

Product : AIRS3STM v7 (AIRS/Aqua L3 Monthly Standard Physical Retrieval)
Variable: TotH2OVap_A  — total column water vapor, ascending (daytime)
          Units in the file: cm precipitable water → converted to kg/m² here
          (1 cm PW = 10 kg/m²)
Grid    : 1°×1° global
Source  : NASA GES DISC via earthaccess

Why AIRS: infrared sounder — completely independent retrieval from ERA5
(reanalysis) and from MODIS (near-IR, fails over ice/snow).  AIRS infrared
channels work day and night and have better Arctic coverage than visible/NIR.
Limitation: 2002-present only (Aqua launch), and quality degrades in very
cold dry conditions typical of winter Arctic.

Output: /glade/work/jrotondo/airs_tpw_arctic_amjjas.nc

Prerequisite: accept GES DISC EULA once at
  https://urs.earthdata.nasa.gov → Applications → Authorized Apps
  → approve "NASA GESDISC DATA ARCHIVE"
"""

import gc, os, sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_FILE   = '/glade/work/jrotondo/airs_tpw_arctic_amjjas.nc'
SCRATCH    = '/glade/derecho/scratch/jrotondo/airs_tpw_raw/'
MONTH_NUMS = [4, 5, 6, 7, 8, 9]
YEARS      = list(range(2002, 2025))   # AIRS: 2002-present
LAT_MIN    = 50.0

os.makedirs(SCRATCH, exist_ok=True)

try:
    import earthaccess
except ImportError:
    sys.exit('pip install earthaccess')

print('Authenticating with NASA Earthdata ...')
earthaccess.login()
print('  OK\n')

# ── SEARCH ────────────────────────────────────────────────────────────────────
# AIRS3STM: AIRS/Aqua L3 Monthly Standard Physical Retrieval (AIRS+AMSU)
# ~1 file per month, ~50 MB each (global 1×1°, many variables).
print('Searching AIRS3STM granules (2002–2024) ...')
all_granules = earthaccess.search_data(
    short_name='AIRS3STM',
    temporal=('2002-01-01', '2024-12-31'),
    count=300,
)
print(f'  Found {len(all_granules)} granules')

if not all_granules:
    # Try alternate short name used in older API versions
    all_granules = earthaccess.search_data(
        short_name='AIRS3STM',
        version='7.0',
        temporal=('2002-01-01', '2024-12-31'),
        count=300,
    )
    print(f'  Retry (v7): {len(all_granules)} granules')

if not all_granules:
    sys.exit(
        'No granules found. Check that GES DISC EULA is accepted at:\n'
        '  https://urs.earthdata.nasa.gov → Applications → Authorized Apps'
    )

# Map granules to (year, month)
granule_map = {}
for g in all_granules:
    try:
        t_start = g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        yr = int(t_start[:4]); mo = int(t_start[5:7])
        if mo in MONTH_NUMS and yr in YEARS:
            granule_map[(yr, mo)] = g
    except Exception:
        try:
            links = g.data_links()
            fname = Path(links[0]).name   # AIRS.2010.07.L3.RetStd001.v7.0...hdf
            parts = fname.split('.')
            yr = int(parts[1]); mo = int(parts[2])
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
n_total = len(found_keys); n_done = 0

for (year, month) in sorted(found_keys):
    n_done += 1
    tag = f'{year}-{month:02d}'

    # Check cache (AIRS files have long names, glob by year/month)
    cached_pat = list(Path(SCRATCH).glob(f'AIRS.*{year}.{month:02d}*.hdf')) + \
                 list(Path(SCRATCH).glob(f'AIRS.*{year}.{month:02d}*.nc'))
    if cached_pat:
        cached = cached_pat[0]
    else:
        g = granule_map[(year, month)]
        try:
            earthaccess.download([g], local_path=SCRATCH)
        except Exception as e:
            print(f'  [{n_done}/{n_total}] {tag}  download error: {e}')
            continue
        # Find the downloaded file
        new_files = (list(Path(SCRATCH).glob(f'AIRS.*{year}.{month:02d}*.hdf')) +
                     list(Path(SCRATCH).glob(f'AIRS.*{year}.{month:02d}*.nc')))
        if not new_files:
            print(f'  [{n_done}/{n_total}] {tag}  file not found after download')
            continue
        cached = new_files[0]

    # Extract TotH2OVap_A (ascending/daytime) — convert cm → kg/m²
    try:
        suffix = Path(cached).suffix.lower()
        if suffix == '.hdf':
            try:
                from pyhdf.SD import SD, SDC
                hdf = SD(str(cached), SDC.READ)
                datasets = hdf.datasets()
                # Try ascending first, fall back to combined
                var_name = next((v for v in ['TotH2OVap_A', 'TotH2OVap_D',
                                             'TotH2OVap'] if v in datasets), None)
                if var_name is None:
                    print(f'  [{n_done}/{n_total}] {tag}  no TPW var found. '
                          f'Available: {list(datasets.keys())[:10]}')
                    hdf.end(); continue
                sds   = hdf.select(var_name)
                raw   = sds.get().astype(np.float64)
                attrs = sds.attributes()
                sds.endaccess()
                fill  = attrs.get('_FillValue', -9999.)
                scale = float(attrs.get('scale_factor', 1.0))
                off   = float(attrs.get('add_offset',   0.0))
                lat_sds = hdf.select('Latitude')
                lats    = lat_sds.get(); lat_sds.endaccess()
                lon_sds = hdf.select('Longitude')
                lons    = lon_sds.get(); lon_sds.endaccess()
                hdf.end()
                data = np.where(raw == fill, np.nan, scale * (raw - off))
            except ImportError:
                print(f'  [{n_done}/{n_total}] {tag}  pyhdf not available for HDF4')
                continue
        else:
            # NetCDF format (v7 sometimes delivers nc)
            with xr.open_dataset(cached) as ds:
                var_name = next((v for v in ['TotH2OVap_A', 'TotH2OVap_D',
                                             'TotH2OVap'] if v in ds), None)
                if var_name is None:
                    print(f'  [{n_done}/{n_total}] {tag}  no TPW var. '
                          f'Vars: {list(ds.data_vars)[:10]}')
                    continue
                data = ds[var_name].values.astype(np.float64)
                lats = ds['Latitude'].values  if 'Latitude'  in ds.coords else ds['lat'].values
                lons = ds['Longitude'].values if 'Longitude' in ds.coords else ds['lon'].values

        # cm → kg/m²  (1 cm PW = 10 kg/m²)
        data = np.where(np.isfinite(data) & (data > 0.) & (data < 200.),
                        data * 10., np.nan).astype(np.float32)

        # Subset to Arctic (lats may be 1-D or 2-D depending on format)
        if lats.ndim == 1:
            row_mask = lats >= LAT_MIN
            arctic_lats = lats[row_mask]
            arctic_data = data[row_mask, :]
            arctic_lons = lons
        else:
            # 2-D lat/lon — flatten, will be regridded later in analysis
            # For now take the rows where mean lat ≥ LAT_MIN
            row_mask = lats.mean(axis=1) >= LAT_MIN
            arctic_lats = lats[row_mask, :].mean(axis=1)
            arctic_data = data[row_mask, :]
            arctic_lons = lons[row_mask, :].mean(axis=0)

        if 'lat' not in records:
            records['lat'] = arctic_lats
            records['lon'] = arctic_lons
        records[(year, month)] = arctic_data

        mu = np.nanmean(arctic_data)
        print(f'  [{n_done}/{n_total}] {tag}  TPW_mean={mu:.1f} kg/m²  var={var_name}')

    except Exception as e:
        print(f'  [{n_done}/{n_total}] {tag}  read error: {e}')

    gc.collect()

# ── PACKAGE ───────────────────────────────────────────────────────────────────
data_keys = [k for k in records if isinstance(k, tuple)]
print(f'\n{len(data_keys)} month-fields extracted.  Building NetCDF ...')

if not data_keys:
    sys.exit('No data extracted — check download errors above.')

keys_sorted = sorted(data_keys)
times       = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for (y, m) in keys_sorted])
arctic_lats = records['lat']
lons        = records['lon']

arr = np.full((len(times), len(arctic_lats), len(lons)), np.nan, dtype=np.float32)
for i, key in enumerate(keys_sorted):
    d = records[key]
    arr[i, :d.shape[0], :d.shape[1]] = d

da = xr.DataArray(
    arr, dims=['time', 'lat', 'lon'],
    coords={
        'time': times,
        'lat':  xr.DataArray(arctic_lats, dims='lat',
                             attrs={'units': 'degrees_north'}),
        'lon':  xr.DataArray(lons, dims='lon',
                             attrs={'units': 'degrees_east'}),
    },
    attrs={
        'long_name': 'Total column water vapour (AIRS ascending/daytime)',
        'units'    : 'kg m-2',
        'source'   : 'AIRS/Aqua AIRS3STM v7, TotH2OVap_A, cm→kg/m² (×10)',
        'note'     : ('Independent IR-sounder retrieval; compare with ERA5 TCWV '
                      'to confirm reanalysis moistening signal in observations.'),
    }
)

ds_out = da.to_dataset(name='tpw')
ds_out.attrs = {
    'description': 'AIRS L3 Monthly TPW — Arctic (≥50°N) AMJJAS 2002–2024',
    'product'    : 'AIRS3STM v7',
}
encoding = {'tpw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
ds_out.to_netcdf(OUT_FILE, encoding=encoding)

print(f'\nSaved → {OUT_FILE}')
print(f'  {len(times)} times × {len(arctic_lats)} lat × {len(lons)} lon')
print(f'\nDelete scratch when done:  rm -rf {SCRATCH}')
