#!/usr/bin/env python
"""
download_modis_tpw.py
─────────────────────
Download MOD08_M3 v6.1 and extract total precipitable water vapor (TPW)
for the Arctic, AMJJAS 2000–2025.

Reuses the same MOD08_M3 files as download_modis_cod.py (cloud properties).
If raw HDF files are still in scratch, this runs in seconds.
If scratch was cleaned, it re-downloads (same files, same product).

Variable: Precipitable_Water_Vapor_Mean_Mean
Units   : cm  →  converted to kg/m² (×10)
Grid    : 1°×1° global

Caveat: MODIS near-IR TPW retrieval requires clear sky and works best over
land. Over dark ocean it falls back to IR; over sea ice it may be sparse.
Use alongside ERA5 and AIRS to triangulate.

Output: /glade/work/jrotondo/modis_tpw_arctic_amjjas.nc
"""

import gc, glob, os, re, sys, calendar
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

OUT_FILE   = '/glade/work/jrotondo/modis_tpw_arctic_amjjas.nc'
SCRATCH    = '/glade/derecho/scratch/jrotondo/modis_cod_raw/'  # same as cloud download
MONTH_NUMS = [4, 5, 6, 7, 8, 9]
YEARS      = list(range(2000, 2026))
LAT_MIN    = 50.0

LAT_GRID = np.arange(89.5,  -90.5, -1.0)
LON_GRID = np.arange(-179.5, 180.5,  1.0)
LAT_ROWS = np.where(LAT_GRID >= LAT_MIN)[0]

# Confirmed by probe: MOD08_M3 v6.1 uses 'Atmospheric_Water_Vapor_Mean_Mean'
# (combined IR+NIR retrieval, scale=0.001, units=cm → ×10 gives kg/m²)
# High-QA version also available: 'Atmospheric_Water_Vapor_High_QA_Mean_Mean'
TPW_VAR = 'Atmospheric_Water_Vapor_Mean_Mean'

os.makedirs(SCRATCH, exist_ok=True)

try:
    import earthaccess
except ImportError:
    sys.exit('pip install earthaccess')

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    sys.exit('conda install -c conda-forge pyhdf -y')

# ── HELPER ────────────────────────────────────────────────────────────────────
_DOY_RE = re.compile(r'MOD08_M3\.A(\d{4})(\d{3})\.')
def _fname_to_ym(fname):
    import datetime
    m = _DOY_RE.search(Path(fname).name)
    if not m: return None, None
    yr = int(m.group(1)); doy = int(m.group(2))
    dt = datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)
    return yr, dt.month

def read_tpw(hdf_path):
    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception as e:
        print(f'    ERROR opening {hdf_path}: {e}'); return None
    try:
        sds   = hdf.select(TPW_VAR)
        raw   = sds.get().astype(np.float64)
        attrs = sds.attributes()
        sds.endaccess()
        scale = float(attrs.get('scale_factor', 1.0))
        off   = float(attrs.get('add_offset',   0.0))
        fill  = attrs.get('_FillValue', None)
        if fill is not None:
            raw = np.where(raw == fill, np.nan, raw)
        data = (scale * (raw - off)).astype(np.float32)
        # cm → kg/m²
        data = np.where(np.isfinite(data) & (data > 0.), data * 10., np.nan)
        return data[LAT_ROWS, :]   # Arctic subset
    except Exception as e:
        print(f'    WARNING {TPW_VAR}: {e}'); return None
    finally:
        hdf.end()

# ── CHECK CACHE FIRST ─────────────────────────────────────────────────────────
print('Checking for cached MOD08_M3 HDF files ...')
cached_files = sorted(glob.glob(f'{SCRATCH}/MOD08_M3_*.hdf'))
print(f'  Found {len(cached_files)} cached HDF files')

records = {}

if cached_files:
    print('  Reading TPW from cached files ...')
    for hdf_path in cached_files:
        stem = Path(hdf_path).stem   # MOD08_M3_YYYYMM
        try:
            yr = int(stem[-6:-2]); mo = int(stem[-2:])
        except ValueError:
            continue
        if mo not in MONTH_NUMS or yr not in YEARS: continue
        data = read_tpw(hdf_path)
        if data is not None:
            records[(yr, mo)] = data
            print(f'  {yr}-{mo:02d}  TPW_mean={np.nanmean(data):.1f} kg/m²')
    print(f'  Read {len(records)} months from cache')

# ── DOWNLOAD MISSING ──────────────────────────────────────────────────────────
target_keys = {(y, m) for y in YEARS for m in MONTH_NUMS}
missing     = target_keys - set(records.keys())

if missing:
    print(f'\nDownloading {len(missing)} missing months from NASA ...')
    earthaccess.login()

    all_granules = earthaccess.search_data(
        concept_id='C1443766359-LAADS',
        temporal=('2000-01-01', '2025-12-31'),
        count=400,
    )
    granule_map = {}
    for g in all_granules:
        try:
            links = g.data_links(access='external') or g.data_links()
            yr, mo = _fname_to_ym(Path(links[0]).name)
            if yr and mo: granule_map[(yr, mo)] = g
        except Exception: pass

    n_total = len(missing); n_done = 0
    for (year, month) in sorted(missing):
        n_done += 1
        tag = f'{year}-{month:02d}'
        if (year, month) not in granule_map:
            print(f'  [{n_done}/{n_total}] {tag}  not in granule map, skipping')
            continue

        cached = Path(SCRATCH) / f'MOD08_M3_{year}{month:02d}.hdf'
        if not cached.exists():
            try:
                earthaccess.download([granule_map[(year, month)]], local_path=SCRATCH)
            except Exception as e:
                print(f'  [{n_done}/{n_total}] {tag}  download error: {e}'); continue
            matches = sorted(glob.glob(f'{SCRATCH}/MOD08_M3.A{year}*.hdf'))
            if matches:
                os.rename(matches[-1], cached)
            else:
                print(f'  [{n_done}/{n_total}] {tag}  file not found after download')
                continue

        data = read_tpw(cached)
        if data is not None:
            records[(year, month)] = data
            print(f'  [{n_done}/{n_total}] {tag}  TPW_mean={np.nanmean(data):.1f} kg/m²')
        gc.collect()

# ── PACKAGE ───────────────────────────────────────────────────────────────────
print(f'\n{len(records)} month-fields extracted.  Building NetCDF ...')

keys_sorted = sorted(records.keys())
times       = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for y, m in keys_sorted])
arctic_lat  = LAT_GRID[LAT_ROWS]

arr = np.full((len(times), len(arctic_lat), len(LON_GRID)), np.nan, dtype=np.float32)
for i, key in enumerate(keys_sorted):
    arr[i] = records[key]

da = xr.DataArray(
    arr, dims=['time', 'lat', 'lon'],
    coords={
        'time': times,
        'lat':  xr.DataArray(arctic_lat, dims='lat', attrs={'units': 'degrees_north'}),
        'lon':  xr.DataArray(LON_GRID,   dims='lon', attrs={'units': 'degrees_east'}),
    },
    attrs={
        'long_name': 'Total precipitable water vapour (MODIS near-IR)',
        'units'    : 'kg m-2',
        'source'   : 'MODIS Terra MOD08_M3 v6.1, Precipitable_Water_Vapor_Mean_Mean, cm→kg/m²',
        'caveat'   : ('Near-IR retrieval; may be sparse/biased over sea ice. '
                      'Use ERA5 and AIRS for cross-check over ice-covered areas.'),
    }
)
ds_out = da.to_dataset(name='tpw')
ds_out.attrs = {'description': 'MODIS TPW — Arctic (≥50°N) AMJJAS 2000–2025',
                'product': 'MOD08_M3.061'}
encoding = {'tpw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
ds_out.to_netcdf(OUT_FILE, encoding=encoding)

print(f'\nSaved → {OUT_FILE}')
print(f'  {len(times)} times × {len(arctic_lat)} lat × {len(LON_GRID)} lon')
