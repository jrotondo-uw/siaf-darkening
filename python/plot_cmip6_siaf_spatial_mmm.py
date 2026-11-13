#!/usr/bin/env python
"""
plot_cmip6_siaf_spatial_mmm.py
──────────────────────────────
Spatial comparison of CMIP6 ensemble vs OBS for SIAF (ext / drk / tot).

Strategy
────────
CMIP6 SIAF was computed on each model's native atmospheric grid (to avoid
inflating N via PSN25 upsampling).  For *visualization only*, we regrid the
seasonal-mean spatial SIAF field from every qualifying member to the PSN25
25-km grid using Nearest-Neighbor, then average:
  1. across members → per-model spatial mean
  2. across models  → Multi-Model Mean (MMM)

OBS spatial fields are loaded directly from the .npz produced by
plot_siaf_spatial_huang.py (already on PSN25).

Figure layout  (3 rows × 3 cols + narrow cbar cols):
  row = component  (ext | drk | tot)
  col = OBS mean  |  CMIP6 MMM  |  MMM − OBS

Filters (matching histogram script):
  • rs_method == 'full_T07'
  • ≥ 3 qualifying members per model

Per-model spatial cache: CACHE_SPATIAL_DIR / {model}_spatial.npz
  Keys: mean_ext, mean_drk, mean_tot  (PSN25 shape, W m⁻² K⁻¹ per pixel,
        i.e. IS * RS monthly mean, NOT yet scaled by AREA_RATIO_OCEAN / 12)
  Re-computed only if cache file is absent or FORCE_RECOMPUTE = True.

Outputs:
  siaf_cmip6_spatial_mmm.png
  siaf_cmip6_spatial_mmm.npz   (mmm_ext/drk/tot + obs fields + psn metadata)
"""

import re, gc, os, glob, warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as mgridspec
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

from scipy.spatial import cKDTree
from scipy.stats  import t as t_dist

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('WARNING: cartopy not found.')

try:
    from pyproj import Transformer
except ImportError:
    raise ImportError('pyproj required.')

try:
    from global_land_mask import globe as _gm_globe
    _HAS_LAND_MASK = True
except ImportError:
    _HAS_LAND_MASK = False


# ============================================================
# CONFIG  — edit paths to match your environment
# ============================================================
CMIP6_ROOT  = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT  = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

# CSV produced by cmip6_siaf_pistone_only.py
CMIP6_CSV   = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/siaf_cmip6_pistone_summary.csv'

# .npz produced by plot_siaf_spatial_huang.py  (contains OBS PSN25 spatial fields)
OBS_NPZ     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_spatial_boot_huang.npz'

# PSN25 SIC file (for grid coordinates)
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'

OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
CACHE_DIR   = os.path.join(OUT_DIR, 'spatial_cache')
PLOT_OUT    = os.path.join(OUT_DIR, 'siaf_cmip6_spatial_mmm.png')
NPZ_OUT     = os.path.join(OUT_DIR, 'siaf_cmip6_spatial_mmm.npz')
os.makedirs(CACHE_DIR, exist_ok=True)

FORCE_RECOMPUTE = False   # set True to ignore cached per-model .npz files

# OBS NPZ key prefix  (period 'Obsera', pval 0.05)
OBS_PREFIX = 'Obsera_p005'   # adjust if needed; see keys in the .npz

PVAL        = 0.05
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}
EARLY_YEARS = list(range(1982, 1992))
LATE_YEARS  = list(range(2005, 2015))
TREND_YEARS = list(range(1982, 2015))
SIC_THRESH  = 0.15

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

RAD_VARS = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs', 'rsuscs']
VAR_TABLE_MAP = {
    'rsus': ['Amon'], 'rsds': ['Amon'], 'rsdt': ['Amon'], 'rsut': ['Amon'],
    'rsutcs': ['Amon', 'CFmon'], 'rsdscs': ['Amon', 'CFmon'],
    'rsuscs': ['Amon', 'CFmon'], 'siconc': ['SImon', 'SIday'],
}
# Copied from CMIP6 main script so this file is self-contained
MODEL_INFO = {
    'CESM2'          : dict(institution='NCAR',                member='r1i1p1f1'),
    'E3SM-1-0'       : dict(institution='E3SM-Project',        member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',                member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',               member='r1i1p1f1'),
    'MIROC-ES2L'     : dict(institution='MIROC',               member='r1i1p1f2'),
    'CanESM5'        : dict(institution='CCCma',               member='r1i1p1f1'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',               member='r1i1p1f1'),
    'GFDL-ESM4'      : dict(institution='NOAA-GFDL',           member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',                member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',                member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS',        member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',                 member='r1i1p1f1'),
    'ACCESS-CM2'     : dict(institution='CSIRO-ARCCSS',        member='r1i1p1f1'),
    'CNRM-ESM2-1'    : dict(institution='CNRM-CERFACS',        member='r1i1p1f2'),
    'EC-Earth3'      : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'EC-Earth3-Veg'  : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'GFDL-CM4'       : dict(institution='NOAA-GFDL',           member='r1i1p1f2'),
    'GISS-E2-1-G'    : dict(institution='NASA-GISS',           member='r1i1p1f1'),
    'GISS-E2-1-H'    : dict(institution='NASA-GISS',           member='r1i1p1f1'),
    'INM-CM4-8'      : dict(institution='INM',                 member='r1i1p1f1'),
    'INM-CM5-0'      : dict(institution='INM',                 member='r1i1p1f1'),
    'MRI-ESM2-0'     : dict(institution='MRI',                 member='r1i1p1f1'),
    'NorESM2-LM'     : dict(institution='NCC',                 member='r1i1p1f1'),
    'NorESM2-MM'     : dict(institution='NCC',                 member='r1i1p1f1'),
}


# ============================================================
# PSN25 GRID  (same setup as main OBS script)
# ============================================================
print('Setting up PSN25 grid...', flush=True)
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)
    psn_y = ds['y'].values.astype(np.float64)

psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
PSN_SHAPE = psn_xx.shape

_tfm_psn2ll = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
psn_lon2d, psn_lat2d = _tfm_psn2ll.transform(psn_xx, psn_yy)

if _HAS_LAND_MASK:
    lp = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    DOMAIN_PSN = (psn_lat2d >= 60.) & _gm_globe.is_ocean(psn_lat2d, lp)
else:
    DOMAIN_PSN = psn_lat2d >= 60.

# imshow extent
_psn_dx = float(psn_x[1] - psn_x[0])
_psn_dy = float(psn_y[1] - psn_y[0])
_imshow_x0 = float(psn_x[0])  - abs(_psn_dx) / 2
_imshow_x1 = float(psn_x[-1]) + abs(_psn_dx) / 2
_y_lo = min(float(psn_y[0]), float(psn_y[-1])) - abs(_psn_dy) / 2
_y_hi = max(float(psn_y[0]), float(psn_y[-1])) + abs(_psn_dy) / 2
_imshow_y0, _imshow_y1 = _y_lo, _y_hi
_imshow_origin = 'upper' if _psn_dy < 0 else 'lower'

# Map display extent (58°N bounding circle)
_tfm_ll2psn = Transformer.from_crs('EPSG:4326', PSN25_PROJ4, always_xy=True)
_lons_rim = np.linspace(-180, 180, 361)
_xs_rim, _ys_rim = _tfm_ll2psn.transform(_lons_rim, np.full_like(_lons_rim, 57.))
_rim_r = max(np.abs(_xs_rim).max(), np.abs(_ys_rim).max()) * 1.04
_ext_xmin, _ext_xmax = -_rim_r, _rim_r
_ext_ymin, _ext_ymax = -_rim_r, _rim_r

# Cartopy CRS
if HAS_CARTOPY:
    _globe = ccrs.Globe(semimajor_axis=6378273., semiminor_axis=6356889.449)
    PSN_CRS = ccrs.Stereographic(central_latitude=90., central_longitude=-45.,
                                  true_scale_latitude=70., globe=_globe)
else:
    PSN_CRS = None

print(f'  PSN25 shape: {PSN_SHAPE}  domain: {DOMAIN_PSN.sum():,} cells', flush=True)


# ============================================================
# UTILITIES  (NN regrid, file discovery — mirrors main script)
# ============================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])


def build_nn_index_to_psn(src_lat2d, src_lon2d):
    """Return NN index: for each PSN25 cell, index of nearest source cell."""
    tree = cKDTree(latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel()))
    _, idx = tree.query(latlon_to_xyz(psn_lat2d.ravel(), psn_lon2d.ravel()),
                        workers=-1)
    del tree; gc.collect()
    return idx  # shape: (PSN_SHAPE flat,)


def regrid_to_psn(arr2d, nn_idx):
    """
    Nearest-neighbour regrid arr2d (native model grid) → PSN25.
    For display only — not used in any area-weighted calculation.
    """
    out = arr2d.ravel()[nn_idx].reshape(PSN_SHAPE).astype(np.float32)
    # Mask non-domain pixels
    out = np.where(DOMAIN_PSN, out, np.nan)
    return out


def _prefer_regular_grid(files):
    if not files: return files
    bns = [os.path.basename(f) for f in files]
    is_gr = lambda bn: ('_gr_' in bn) or ('_gr1_' in bn) or ('_gr2_' in bn)
    is_gn = lambda bn: '_gn_' in bn
    if any(is_gr(bn) for bn in bns) and any(is_gn(bn) for bn in bns):
        return [f for f, bn in zip(files, bns) if is_gr(bn)]
    return files


_ALL_NEEDED = set(range(1982, 2015))

def find_var_files(inst, model, member, variable):
    tables = VAR_TABLE_MAP.get(variable, ['Amon'])
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in tables:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, variable)
            if not os.path.isdir(d): continue
            files = sorted(glob.glob(os.path.join(d, '**', f'{variable}_*.nc'),
                                     recursive=True))
            if not files: continue
            files = _prefer_regular_grid(files)
            relevant = [f for f in files if _covers_any(f, _ALL_NEEDED)]
            if relevant: return relevant
    return []


def _covers_any(f, yr_set):
    m = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', os.path.basename(f))
    if m:
        s, e = int(m.group(1)[:4]), int(m.group(2)[:4])
        return any(s <= y <= e for y in yr_set)
    return True


def _is_hdf_error(exc):
    msg = str(exc).lower()
    return ('hdf error' in msg or 'errno -101' in msg or
            ('netcdf' in msg and 'error' in msg) or
            'did not find a match' in msg)


def _triage(files, variable):
    good = []
    for f in files:
        try:
            ds = xr.open_dataset(f, use_cftime=True)
            _ = ds[variable].isel(time=0).values
            ds.close(); good.append(f)
        except Exception:
            print(f'    [triage] skip corrupt: {os.path.basename(f)}', flush=True)
    return good


def _open_robust(files, variable):
    strategies = [
        dict(combine='by_coords'),
        dict(combine='nested', concat_dim='time'),
        dict(combine='nested', concat_dim='time', coords='minimal', compat='override'),
    ]
    last = None
    for kw in strategies:
        try:
            kw['chunks'] = {'time': 12}
            kw['use_cftime'] = True
            ds = xr.open_mfdataset(files, **kw)
            return ds
        except Exception as e:
            last = e
    if _is_hdf_error(last) and len(files) > 1:
        clean = _triage(files, variable)
        if clean and len(clean) < len(files):
            return _open_robust(clean, variable)
    raise RuntimeError(f'Cannot open {variable}: {last}')


def _std_latlon(da):
    aliases = {'lat': ['latitude','nav_lat','rlat','y'],
               'lon': ['longitude','nav_lon','rlon','x']}
    for new, als in aliases.items():
        for c in list(da.coords):
            if c.lower() in als and c != new:
                try: da = da.rename({c: new})
                except Exception: pass
                break
    lat  = da['lat'].values if 'lat' in da.coords else None
    lon  = da['lon'].values if 'lon' in da.coords else None
    is2d = (lat.ndim == 2 if lat is not None else False)
    return da, lat, lon, is2d


def load_filtered(files, variable, years, months):
    try:
        ds   = _open_robust(files, variable)
        da   = ds[variable]
        ys, ms = set(years), set(months)
        mask = np.array([int(t.dt.year) in ys and int(t.dt.month) in ms
                          for t in da.time], dtype=bool)
        da   = da.isel(time=mask).load(); ds.close()
        da, _, _, _ = _std_latlon(da)
        return da
    except Exception as e:
        print(f'  load_filtered({variable}): {e}', flush=True)
        return None


# ============================================================
# PIXEL-WISE OLS TREND (SIC)
# ============================================================

def pixel_ols_trend(years_1d, sic_3d):
    x  = years_1d - years_1d.mean()
    ss = float(np.dot(x, x))
    dy = sic_3d - np.nanmean(sic_3d, axis=0)[None]
    slope = np.nansum(x[:, None, None] * dy, axis=0) / ss
    resid = sic_3d - (np.nanmean(sic_3d, axis=0) + slope * x[:, None, None])
    sse   = np.nansum(resid**2, axis=0)
    df    = years_1d.shape[0] - 2
    with np.errstate(invalid='ignore', divide='ignore'):
        se     = np.sqrt(sse / df) / np.sqrt(ss)
        t_stat = slope / se
        pval   = 2. * (1. - t_dist.cdf(np.abs(t_stat), df))
    return slope, pval


# ============================================================
# T07 INVERSION
# ============================================================

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S  = np.where(rsdt > 1., rsdt, np.nan)
        Sd = np.where(rsds > 1., rsds, np.nan)
        a  = np.clip(rsus / Sd, 0., 1.)
        p  = rsds / S;  q = rsut / S
        den = q + p - 2.*p*a
        R   = np.clip((q - p*a) / np.where(np.abs(den)>1e-6, den, np.nan), 0., .99)
        A   = np.clip(1. - p*(1.-a*R) / np.where((1.-R)>.01, 1.-R, np.nan), 0., .99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S
        dc  = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc - pc*ac) / np.where(np.abs(dc)>1e-6, dc, np.nan), 0., .99)
        Rl  = np.clip(R - Rc, 0., .99)
        RS  = (S*(1.-A)*(1.-Rl)**2
               / np.where((1.-a*Rl)**2>1e-9, (1.-a*Rl)**2, np.nan))
    return RS.astype(np.float32)


def compute_rs_month(inst, model, member, mn, rs_years=list(range(1985,2015))):
    """Full-T07 RS for one month on native grid. Returns (arr, lat, lon, is2d)."""
    rf = {v: find_var_files(inst, model, member, v) for v in RAD_VARS}
    if not all(rf[v] for v in RAD_VARS):
        return None, None, None, None

    def _vm(v):
        files = rf[v]
        try:
            ds = _open_robust(files, v)
            da = ds[v]
            mask = np.array([int(t.dt.year) in set(rs_years) and
                              int(t.dt.month) == mn for t in da.time], dtype=bool)
            if not mask.any(): ds.close(); return None
            sub = da.isel(time=mask).mean('time').load()
            sub, _, _, _ = _std_latlon(sub)
            ds.close()
            return sub.values.astype(float)
        except Exception as e:
            print(f'    RS {v}: {e}', flush=True); return None

    data = {v: _vm(v) for v in RAD_VARS}
    if any(d is None for d in data.values()):
        return None, None, None, None

    rs_arr = t07_invert(data['rsdt'], data['rsut'], data['rsds'], data['rsus'],
                         data['rsutcs'], data['rsdscs'], data['rsuscs'])

    # Extract lat/lon from rsds (always available)
    ds_tmp = _open_robust(rf['rsds'], 'rsds')
    da_tmp, lat, lon, is2d = _std_latlon(ds_tmp['rsds'])
    ds_tmp.close()
    return rs_arr, lat, lon, is2d


# ============================================================
# PER-MEMBER SPATIAL SIAF  (IS * RS on native grid → PSN25)
# ============================================================

def compute_member_spatial_psn(model, member):
    """
    Returns dict keyed (mn) → {'ext','drk','tot'} on PSN25,
    or None on failure.  full_T07 only.
    """
    inst = MODEL_INFO[model]['institution']
    rsus_f = find_var_files(inst, model, member, 'rsus')
    rsds_f = find_var_files(inst, model, member, 'rsds')
    sic_f  = find_var_files(inst, model, member, 'siconc')
    if not all([rsus_f, rsds_f, sic_f]):
        return None

    # Check all T07 vars available
    if not all(find_var_files(inst, model, member, v) for v in RAD_VARS):
        print(f'  [{member}] not full_T07 — skip', flush=True)
        return None

    all_yrs = list(range(1980, 2016))
    ru = load_filtered(rsus_f, 'rsus',   all_yrs, MONTH_NUMS)
    rd = load_filtered(rsds_f, 'rsds',   all_yrs, MONTH_NUMS)
    si = load_filtered(sic_f,  'siconc', all_yrs, MONTH_NUMS)
    if any(x is None for x in [ru, rd, si]):
        return None

    _, alb_lat, alb_lon, alb_is2d = _std_latlon(ru)
    _, sic_lat, sic_lon, sic_is2d = _std_latlon(si)
    if alb_lat is None or sic_lat is None:
        return None

    def _ym(da):
        return set(zip([int(t.dt.year) for t in da.time],
                        [int(t.dt.month) for t in da.time]))
    comm = _ym(ru) & _ym(rd) & _ym(si)
    if not comm: return None

    def _tmask(da):
        return np.array([(int(t.dt.year), int(t.dt.month)) in comm
                          for t in da.time], dtype=bool)
    ru = ru.isel(time=_tmask(ru))
    rd = rd.isel(time=_tmask(rd))
    si = si.isel(time=_tmask(si))

    sv = si.values.astype(float)
    if np.nanmax(sv) > 1.5: sv /= 100.

    alb_sh = alb_lat.shape if alb_is2d else (len(alb_lat), len(alb_lon))

    # Regrid SIC to albedo grid if needed
    if sv.shape[1:] != alb_sh:
        if sic_is2d:
            sla, slo = sic_lat.ravel(), sic_lon.ravel()
        else:
            slo2d, sla2d = np.meshgrid(sic_lon, sic_lat)
            sla, slo = sla2d.ravel(), slo2d.ravel()
        if alb_is2d:
            ala, alo = alb_lat.ravel(), alb_lon.ravel()
        else:
            alo2d, ala2d = np.meshgrid(alb_lon, alb_lat)
            ala, alo = ala2d.ravel(), alo2d.ravel()
        sic_v  = np.isfinite(sla) & np.isfinite(slo)
        _, idx = cKDTree(latlon_to_xyz(sla[sic_v], slo[sic_v])).query(
            latlon_to_xyz(ala, alo).astype(np.float32), workers=-1)
        sv = sv.reshape(sv.shape[0], -1)[:, sic_v][:, idx].reshape(
            (sv.shape[0],) + alb_sh)

    ru_yrs = np.array([int(t.dt.year)  for t in ru.time])
    ru_mon = np.array([int(t.dt.month) for t in ru.time])
    si_yrs = np.array([int(t.dt.year)  for t in si.time])
    si_mon = np.array([int(t.dt.month) for t in si.time])
    del si; gc.collect()

    # Build NN index to PSN25 (once per grid shape)
    if alb_is2d:
        a_lat2d, a_lon2d = alb_lat, alb_lon
    else:
        a_lon2d, a_lat2d = np.meshgrid(alb_lon, alb_lat)
    nn_idx = build_nn_index_to_psn(a_lat2d.astype(np.float32),
                                    a_lon2d.astype(np.float32))

    result_by_month = {}

    for mn in MONTH_NUMS:
        def epoch_alb(yr_list):
            mask = np.isin(ru_yrs, yr_list) & (ru_mon == mn)
            if not mask.any(): return None
            u = ru.isel(time=mask).mean('time').values
            d = rd.isel(time=mask).mean('time').values
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(d > 1., u / d, np.nan)
            return np.clip(alb.astype(np.float32), 0., 1.)

        ae = epoch_alb(EARLY_YEARS)
        al = epoch_alb(LATE_YEARS)
        if ae is None or al is None: continue

        dalpha = al - ae
        tm = (si_mon == mn) & np.isin(si_yrs, TREND_YEARS)
        em = (si_mon == mn) & np.isin(si_yrs, EARLY_YEARS)
        lm = (si_mon == mn) & np.isin(si_yrs, LATE_YEARS)
        if tm.sum() < 10 or em.sum() == 0 or lm.sum() == 0: continue

        sic_e   = np.nanmean(sv[em], axis=0)
        sic_l   = np.nanmean(sv[lm], axis=0)
        slope, pval = pixel_ols_trend(si_yrs[tm].astype(float), sv[tm])

        # dT per-member is stored in the CSV — we just need to regrid IS fields
        # dT normalisation is applied AFTER regridding (we use dT=1 as placeholder;
        # the final per-pixel IS * RS product is scaled by true dT below)
        # Actually: IS = -Δα / ΔT.  We keep Δα here and divide by dT in the caller.
        all_ice  = (a_lat2d >= 60.) & (sic_e >= SIC_THRESH)
        changing = (all_ice & np.isfinite(slope) & np.isfinite(pval)
                    & (slope < 0) & (pval < PVAL))
        stable   = all_ice & ~changing & (np.fmax(sic_e, sic_l) >= SIC_THRESH)

        # Store Δα (not yet divided by ΔT — division happens per-member in caller)
        dalpha_ext = np.where(changing, dalpha, np.nan).astype(np.float32)
        dalpha_drk = np.where(stable,   dalpha, np.nan).astype(np.float32)
        dalpha_tot = np.where(
            np.isfinite(dalpha_ext) | np.isfinite(dalpha_drk),
            np.where(np.isfinite(dalpha_ext), dalpha_ext, 0.)
            + np.where(np.isfinite(dalpha_drk), dalpha_drk, 0.),
            np.nan).astype(np.float32)
        dalpha_ext = np.where(np.isfinite(dalpha_tot),
                               np.where(np.isfinite(dalpha_ext), dalpha_ext, 0.),
                               np.nan).astype(np.float32)
        dalpha_drk = np.where(np.isfinite(dalpha_tot),
                               np.where(np.isfinite(dalpha_drk), dalpha_drk, 0.),
                               np.nan).astype(np.float32)

        # RS for this month
        rs_arr, _, _, _ = compute_rs_month(inst, model, member, mn)
        if rs_arr is None or rs_arr.shape != alb_sh:
            print(f'    RS shape mismatch for {model}/{member} mn={mn}; skip',
                  flush=True)
            continue

        result_by_month[mn] = {
            'dalpha_ext': dalpha_ext, 'dalpha_drk': dalpha_drk,
            'dalpha_tot': dalpha_tot, 'rs': rs_arr.astype(np.float32),
            'nn_idx': nn_idx,
        }
        gc.collect()

    del ru, rd, sv; gc.collect()
    return result_by_month if result_by_month else None


def member_to_psn_spatial(model, member, delta_t):
    """
    Combine IS * RS monthly fields on PSN25.
    Returns {'ext','drk','tot'} on PSN25 (seasonal mean, W m⁻² K⁻¹ per pixel).
    """
    monthly = compute_member_spatial_psn(model, member)
    if monthly is None:
        return None

    maps_ext, maps_drk, maps_tot = [], [], []
    for mn, d in monthly.items():
        nn  = d['nn_idx']
        rs  = d['rs']
        dT  = max(delta_t, 0.01)

        # IS = -Δα / ΔT  (negative Δα → positive IS for retreating ice)
        IS_ext = (-d['dalpha_ext'] / dT).astype(np.float32)
        IS_drk = (-d['dalpha_drk'] / dT).astype(np.float32)
        IS_tot = (-d['dalpha_tot'] / dT).astype(np.float32)

        SIAF_ext = (IS_ext * rs).astype(np.float32)
        SIAF_drk = (IS_drk * rs).astype(np.float32)
        SIAF_tot = (IS_tot * rs).astype(np.float32)

        # fill NaN → 0 within arctic domain before regridding
        arctic = (np.meshgrid(
            np.arange(rs.shape[1]), np.arange(rs.shape[0]))[0] >= 0)  # placeholder
        # use lat2d from build step
        # For fill: arctic cells with finite IS_tot → already handled in IS logic
        maps_ext.append(regrid_to_psn(SIAF_ext, nn))
        maps_drk.append(regrid_to_psn(SIAF_drk, nn))
        maps_tot.append(regrid_to_psn(SIAF_tot, nn))
        gc.collect()

    if not maps_tot:
        return None
    return {
        'ext': np.nanmean(np.stack(maps_ext, 0), 0).astype(np.float32),
        'drk': np.nanmean(np.stack(maps_drk, 0), 0).astype(np.float32),
        'tot': np.nanmean(np.stack(maps_tot, 0), 0).astype(np.float32),
    }


# ============================================================
# LOAD / COMPUTE PER-MODEL SPATIAL MEAN  (with cache)
# ============================================================

def get_model_spatial(model, member_rows):
    """
    Load from cache or recompute spatial PSN25 fields for qualifying members,
    then return the per-model ensemble mean on PSN25.
    member_rows: rows from CMIP6 CSV for this model (full_T07, filtered already)
    """
    cache_path = os.path.join(CACHE_DIR, f'{model}_spatial.npz')
    if not FORCE_RECOMPUTE and os.path.exists(cache_path):
        print(f'  [{model}] Loading from cache...', flush=True)
        npz = np.load(cache_path)
        return {k: npz[k] for k in ['mean_ext', 'mean_drk', 'mean_tot']}

    print(f'\n  [{model}] Computing spatial fields '
          f'({len(member_rows)} members)...', flush=True)
    mem_maps = []
    for _, row in member_rows.iterrows():
        mem  = row['member']
        dT   = float(row['delta_t'])
        print(f'    member {mem}  dT={dT:+.3f} K', flush=True)
        try:
            sp = member_to_psn_spatial(model, mem, dT)
        except Exception as e:
            print(f'    FAILED: {e}', flush=True)
            sp = None
        if sp is not None:
            mem_maps.append(sp)

    if not mem_maps:
        print(f'  [{model}] No valid spatial fields.', flush=True)
        return None

    result = {
        'mean_ext': np.nanmean([m['ext'] for m in mem_maps], axis=0).astype(np.float32),
        'mean_drk': np.nanmean([m['drk'] for m in mem_maps], axis=0).astype(np.float32),
        'mean_tot': np.nanmean([m['tot'] for m in mem_maps], axis=0).astype(np.float32),
    }
    np.savez_compressed(cache_path, **result)
    print(f'  [{model}] Saved → {cache_path}', flush=True)
    return result


# ============================================================
# OBS LOADER
# ============================================================

def load_obs_spatial(npz_path, prefix):
    """Load OBS spatial fields from the .npz produced by the OBS script."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    npz = np.load(npz_path)
    # Try to find keys — prefix may differ across runs
    def _find(tag):
        candidates = [f'{prefix}_mean_{tag}', f'mean_{tag}']
        for c in candidates:
            if c in npz: return npz[c].astype(np.float32)
        # Fallback: search for any key containing the tag
        for k in npz.files:
            if f'mean_{tag}' in k:
                print(f'  OBS key fallback: {k}', flush=True)
                return npz[k].astype(np.float32)
        available = [k for k in npz.files if 'mean' in k]
        raise KeyError(f'Cannot find OBS mean_{tag}. Available: {available}')
    return {
        'ext': _find('ext'),
        'drk': _find('drk'),
        'tot': _find('tot'),
    }


# ============================================================
# PLOTTING  (PSN25 + cartopy, same style as OBS script)
# ============================================================

CMAPS = {
    'val' : 'YlOrRd',
    'diff': 'RdBu_r',
}
ROW_DEFS = [
    ('ext', r'SIAF$_{\rm ext}$', '#3A6BAF'),
    ('drk', r'SIAF$_{\rm drk}$', '#D95F1A'),
    ('tot', r'SIAF$_{\rm tot}$', '#6A3FA5'),
]


def add_map_features(ax):
    ax.add_feature(cfeature.LAND, facecolor='#CCCCCC', zorder=6, edgecolor='none')
    ax.coastlines(resolution='50m', linewidth=0.5, color='#333333', zorder=7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.35,
                      color='#AAAAAA', linestyle='--', zorder=8)
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))


def make_ax(fig, gs_cell):
    if HAS_CARTOPY:
        ax = fig.add_subplot(gs_cell, projection=PSN_CRS)
        ax.set_extent([_ext_xmin, _ext_xmax, _ext_ymin, _ext_ymax], crs=PSN_CRS)
        add_map_features(ax)
    else:
        ax = fig.add_subplot(gs_cell)
        ax.set_aspect('equal')
    return ax


def plot_field(ax, data, cmap, vmin, vmax):
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad(alpha=0.0)
    arr = np.where(DOMAIN_PSN, data, np.nan).astype(np.float32)
    extent_m = [_imshow_x0, _imshow_x1, _imshow_y0, _imshow_y1]
    if HAS_CARTOPY:
        im = matplotlib.axes.Axes.imshow(
            ax, arr, origin=_imshow_origin, extent=extent_m,
            cmap=cmap_obj, vmin=vmin, vmax=vmax,
            interpolation='nearest', zorder=5)
    else:
        im = ax.imshow(arr, origin=_imshow_origin,
                       cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
    return im


def make_figure(obs, mmm, model_list, out_path):
    """
    3-row × 3-col figure:
      col 0 = OBS
      col 1 = CMIP6 MMM
      col 2 = MMM − OBS (diverging, symmetric)

    Shared colourbars: one for cols 0–1, one for col 2.
    """
    # --- colour limits ---
    fields_val = [obs[t] for t in ('ext','drk','tot')] + \
                 [mmm[t] for t in ('ext','drk','tot')]
    vmax_val = max(np.nanpercentile(f[DOMAIN_PSN], 98)
                   for f in fields_val if np.isfinite(f[DOMAIN_PSN]).any())
    vmax_val = max(vmax_val, 0.05)

    diffs = {t: mmm[t] - obs[t] for t in ('ext','drk','tot')}
    vmax_diff = max(np.nanpercentile(np.abs(d[DOMAIN_PSN]), 98)
                    for d in diffs.values() if np.isfinite(d[DOMAIN_PSN]).any())
    vmax_diff = max(vmax_diff, 0.02)

    # Layout: 3r × 7c  [map_obs | cbar_01 | map_mmm | gap | map_diff | cbar_diff | gap]
    # Simpler: 3r × 5c with width_ratios
    fig = plt.figure(figsize=(18, 18), facecolor='#F4F4F4')
    gs  = mgridspec.GridSpec(
        3, 7, figure=fig,
        left=0.08, right=0.96,
        top=0.93,  bottom=0.03,
        hspace=0.55, wspace=0.06,
        width_ratios=[1, 0.04, 1, 0.06, 1, 0.04, 0.02],
    )
    # cols: 0=OBS map, 1=shared cbar 0-1, 2=MMM map,
    #       3=spacer,  4=DIFF map,  5=diff cbar,  6=spacer

    im_val_last = im_diff_last = None

    col_titles = ['OBS  (Huang/ERA5 kernel)', 'CMIP6 MMM', 'MMM − OBS']

    for row, (tag, label, accent) in enumerate(ROW_DEFS):
        ax_obs  = make_ax(fig, gs[row, 0])
        ax_mmm  = make_ax(fig, gs[row, 2])
        ax_diff = make_ax(fig, gs[row, 4])

        im_val_last  = plot_field(ax_obs,  obs[tag], CMAPS['val'],  0, vmax_val)
        plot_field(ax_mmm,  mmm[tag], CMAPS['val'],  0, vmax_val)
        im_diff_last = plot_field(ax_diff, diffs[tag], CMAPS['diff'],
                                   -vmax_diff, vmax_diff)

        # Row label
        ax_obs.text(-0.10, 0.5, label,
                     transform=ax_obs.transAxes,
                     fontsize=15, fontweight='bold', color=accent,
                     va='center', ha='right', rotation=90, rotation_mode='anchor')

        if row == 0:
            for ax, ttl in zip([ax_obs, ax_mmm, ax_diff], col_titles):
                ax.set_title(ttl, fontsize=13, fontweight='bold', pad=7)

    # Shared cbar for OBS + MMM  (between col 0 and col 2 → place at col 1)
    cax_val  = fig.add_subplot(gs[:, 1])
    cax_diff = fig.add_subplot(gs[:, 5])

    cb_val = fig.colorbar(im_val_last, cax=cax_val)
    cb_val.set_label('W m⁻² K⁻¹', fontsize=12)
    cb_val.ax.tick_params(labelsize=11)

    cb_diff = fig.colorbar(im_diff_last, cax=cax_diff)
    cb_diff.set_label('ΔW m⁻² K⁻¹', fontsize=12)
    cb_diff.ax.tick_params(labelsize=11)

    n_models = len(model_list)
    fig.suptitle(
        f'SIAF spatial comparison  |  Apr–Sep  |  p < {PVAL}\n'
        f'CMIP6 MMM: {n_models} models (full T07, ≥3 members, ensemble mean per model)',
        fontsize=13, fontweight='bold', y=0.965)

    fig.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out_path}', flush=True)


# ============================================================
# OPTIONAL: per-model anomaly figure  (model - MMM)
# ============================================================

def make_model_anomaly_figure(model_spatial, mmm, model_list, out_path):
    """
    Small-multiples: each panel = one model's spatial field minus MMM.
    One figure per SIAF component (ext / drk / tot).
    """
    n = len(model_list)
    if n == 0: return

    # Compute symmetric colour limit
    all_diffs = np.concatenate([
        (model_spatial[m]['mean_tot'] - mmm['tot'])[DOMAIN_PSN]
        for m in model_list if model_spatial.get(m) is not None
    ])
    vmax = max(np.nanpercentile(np.abs(all_diffs[np.isfinite(all_diffs)]), 97), 0.02)

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    for comp_tag, comp_label in [('tot', 'Total'), ('ext', 'Extent'), ('drk', 'Darkening')]:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4.5, nrows * 4.2),
            subplot_kw={'projection': PSN_CRS} if HAS_CARTOPY else {},
            facecolor='#F4F4F4',
        )
        axes_flat = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

        cmap_obj = matplotlib.colormaps[CMAPS['diff']].copy()
        cmap_obj.set_bad(alpha=0.0)
        extent_m = [_imshow_x0, _imshow_x1, _imshow_y0, _imshow_y1]
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)

        im_last = None
        for i, model in enumerate(model_list):
            ax = axes_flat[i]
            if HAS_CARTOPY:
                ax.set_extent([_ext_xmin, _ext_xmax, _ext_ymin, _ext_ymax],
                               crs=PSN_CRS)
                add_map_features(ax)
            sp = model_spatial.get(model)
            if sp is None:
                ax.set_title(f'{model}\n(no data)', fontsize=8)
                continue
            diff = np.where(DOMAIN_PSN, sp[f'mean_{comp_tag}'] - mmm[comp_tag], np.nan)
            if HAS_CARTOPY:
                im_last = matplotlib.axes.Axes.imshow(
                    ax, diff.astype(np.float32), origin=_imshow_origin,
                    extent=extent_m, cmap=cmap_obj, norm=norm,
                    interpolation='nearest', zorder=5)
            else:
                im_last = ax.imshow(diff.astype(np.float32), origin=_imshow_origin,
                                    cmap=cmap_obj, norm=norm, interpolation='nearest')
            ax.set_title(model, fontsize=9, fontweight='bold')

        # Hide spare axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if im_last is not None:
            fig.colorbar(im_last, ax=axes_flat[:n], shrink=0.5, pad=0.02,
                         label='ΔW m⁻² K⁻¹ (model − MMM)')

        fig.suptitle(
            f'SIAF {comp_label} — model anomaly from MMM  |  Apr–Sep  |  p < {PVAL}',
            fontsize=13, fontweight='bold')

        anom_path = out_path.replace('.png', f'_anomaly_{comp_tag}.png')
        fig.savefig(anom_path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Saved → {anom_path}', flush=True)


# ============================================================
# MAIN
# ============================================================

print('=' * 70, flush=True)
print('LOADING CMIP6 CSV...', flush=True)
print('=' * 70, flush=True)

if not os.path.exists(CMIP6_CSV):
    raise FileNotFoundError(f'CMIP6 CSV not found: {CMIP6_CSV}')

df = pd.read_csv(CMIP6_CSV)

# Apply same filters as histogram script
df = df[df['rs_method'] == 'full_T07'].copy()
n_per_model = df.groupby('model').size()
valid_models = sorted(n_per_model[n_per_model >= 3].index.tolist())
df = df[df['model'].isin(valid_models)].copy()
print(f'  {len(valid_models)} qualifying models, {len(df)} member rows', flush=True)
print(f'  Models: {valid_models}', flush=True)

# ── Compute / load per-model spatial means ────────────────────────────────────
print('\n' + '=' * 70, flush=True)
print('PER-MODEL SPATIAL FIELDS', flush=True)
print('=' * 70, flush=True)

model_spatial = {}
for model in valid_models:
    sub = df[df['model'] == model]
    sp  = get_model_spatial(model, sub)
    if sp is not None:
        model_spatial[model] = sp

present = sorted(model_spatial.keys())
print(f'\n  {len(present)}/{len(valid_models)} models with spatial fields', flush=True)

if not present:
    raise RuntimeError('No model spatial fields available — check data paths.')

# ── Multi-Model Mean  (equal weight per model) ────────────────────────────────
print('\nComputing MMM...', flush=True)
mmm = {
    tag: np.nanmean(
        np.stack([model_spatial[m][f'mean_{tag}'] for m in present], axis=0), axis=0
    ).astype(np.float32)
    for tag in ('ext', 'drk', 'tot')
}

# ── Load OBS ──────────────────────────────────────────────────────────────────
print('\nLoading OBS spatial fields...', flush=True)
obs = load_obs_spatial(OBS_NPZ, OBS_PREFIX)
print(f'  OBS tot max (domain): '
      f'{np.nanmax(obs["tot"][DOMAIN_PSN]):.4f} W m⁻² K⁻¹', flush=True)

# ── Figures ───────────────────────────────────────────────────────────────────
print('\nGenerating figures...', flush=True)
make_figure(obs, mmm, present, PLOT_OUT)
make_model_anomaly_figure(model_spatial, mmm, present,
                          PLOT_OUT.replace('.png', '_anomaly.png'))

# ── Save .npz ─────────────────────────────────────────────────────────────────
save_dict = {
    'psn_lat': psn_lat2d, 'psn_lon': psn_lon2d,
    'psn_xx': psn_xx, 'psn_yy': psn_yy, 'domain': DOMAIN_PSN,
    'mmm_ext': mmm['ext'], 'mmm_drk': mmm['drk'], 'mmm_tot': mmm['tot'],
    'obs_ext': obs['ext'], 'obs_drk': obs['drk'], 'obs_tot': obs['tot'],
    'diff_ext': mmm['ext'] - obs['ext'],
    'diff_drk': mmm['drk'] - obs['drk'],
    'diff_tot': mmm['tot'] - obs['tot'],
    'n_models': np.array([len(present)]),
    'model_list': np.array(present),
}
for m in present:
    for tag in ('ext', 'drk', 'tot'):
        save_dict[f'{m}_mean_{tag}'] = model_spatial[m][f'mean_{tag}']
np.savez_compressed(NPZ_OUT, **save_dict)
print(f'\nSaved .npz → {NPZ_OUT}', flush=True)

print('\nAll done.', flush=True)