#!/usr/bin/env python
"""Source data for Figure 4: the gridded SIAF maps and their 2-sigma uncertainty, as plotted.

Reads the same files as f04.py (siaf_spatial_boot_{ceres,huang}.npz; observational
era 1982-2024, p < 0.05), verifies complete coverage within the
analysis domain, and writes one row per domain cell of the 25 km NSIDC polar
stereographic grid to source_data/SourceData_Figure4_maps.csv.

Columns
  x_m, y_m          polar stereographic coordinates of the cell centre (m)
  lat, lon          cell centre (degrees north, degrees east)
  siaf_<c>_ceres    component <c> with the CERES APRP radiative sensitivity (left column)
  siaf_<c>_era5     component <c> with the ERA5 radiative kernel (centre column)
  siaf_<c>_2sigma   2 sigma of the April-September mean from bootstrap draws,
                    averaged over the two radiative-sensitivity methods (right column)
                    <c> is area (area loss, top row), drk (darkening, middle row) or
                    tot (total, bottom row); all in W m-2 K-1
  filled_ceres, filled_era5
                    False for every cell: corrected fields have complete domain coverage
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from pyproj import Transformer

OUT_DIR = str(Path(__file__).resolve().parent.parent/'analysis_data/weighting_revision_20261125')
NPZ = {'ceres': os.path.join(OUT_DIR, 'siaf_spatial_boot_ceres.npz'),
       'era5': os.path.join(OUT_DIR, 'siaf_spatial_boot_huang.npz')}
PREFIX = 'Obs-era_0.05'
PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
DEST = Path(__file__).resolve().parent.parent / 'source_data' / 'SourceData_Figure4_maps.csv'


def fill_sparse_map(arr, domain_mask):
    """Full-domain corrected field; no display gap filling."""
    assert np.isfinite(arr[domain_mask]).all()
    return np.where(domain_mask, arr, np.nan)


z = {k: np.load(p) for k, p in NPZ.items()}
dom = z['ceres']['domain'].astype(bool)
assert np.array_equal(dom, z['era5']['domain'].astype(bool)), 'domains differ'
xx, yy = z['ceres']['psn_xx'], z['ceres']['psn_yy']
lon, lat = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True).transform(xx, yy)

cols = {'x_m': xx[dom], 'y_m': yy[dom], 'lat': lat[dom], 'lon': lon[dom]}
flags = {}
for comp, tag in (('area', 'ext'), ('drk', 'drk'), ('tot', 'tot')):
    sig = {}
    for meth in ('ceres', 'era5'):
        mean_raw = z[meth][f'{PREFIX}_mean_{tag}']
        sig_raw = z[meth][f'{PREFIX}_sigma2_{tag}']
        for raw in (mean_raw, sig_raw):
            v = ~np.isnan(raw) & dom
            if meth in flags:
                assert np.array_equal(flags[meth], v), f'{meth}: valid cells differ between fields'
            else:
                flags[meth] = v
        cols[f'siaf_{comp}_{meth}'] = fill_sparse_map(mean_raw, dom)[dom]
        sig[meth] = fill_sparse_map(sig_raw, dom)
    cols[f'siaf_{comp}_2sigma'] = ((sig['ceres'] + sig['era5']) / 2.)[dom]
for meth in ('ceres', 'era5'):
    cols[f'filled_{meth}'] = ~flags[meth][dom]

df = pd.DataFrame(cols)
df[['x_m', 'y_m']] = df[['x_m', 'y_m']].round().astype('int64')   # cell centres are whole metres
assert df.drop(columns=['filled_ceres', 'filled_era5']).notna().all().all(), 'unfilled cells remain'
df.to_csv(DEST, index=False, float_format='%.6g')
print(f'wrote {DEST.name}: {len(df)} cells; filled {df.filled_ceres.sum()} (CERES), {df.filled_era5.sum()} (ERA5)')
print(df.describe().T[['mean', 'min', 'max']].to_string())
