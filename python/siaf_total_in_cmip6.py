#!/usr/bin/env python
"""
cmip6_IS_RS_SIAF_comparison_pistone.py
--------------------------------------

Same as cmip6_IS_RS_SIAF_comparison.py, **augmented** with a parallel
Pistone-2014-style pipeline.  Nothing from the original pipeline is
removed; all original scalars, plots and CSV columns are preserved.

What the Pistone-style layer adds (per member, per month, per OBS):

  * A polar-stereographic PSN25 grid (25x25 km, Arctic-focused, nearly
    equal-area).  Every native-grid field that gets collapsed to a
    scalar is also scatter-regridded to PSN25.
  * An ocean-only mask on PSN25 (global-land-mask) + the six Pistone
    Fig S2 longitudinal wedges spanning 60-90 deg N.
  * Regional (6 wedges) area-weighted means on PSN25, using a true
    polar-stereographic scale-factor correction so the weights are
    surface-area weights, not projected-area weights.
  * An Arctic-OCEAN aggregate computed as the area-weighted sum across
    the six wedges (equivalent to one area-weighted mean over all
    ocean cells >= 60 N, by construction).
  * A separate AREA_RATIO_60N_OCEAN = 0.0347 (Arctic *ocean* area / global
    surface area) for converting the Arctic-ocean mean to a global
    annual W m^-2 K^-1.
  * Results written to an additional CSV
    `siaf_detailed_ensemble_results_pistone.csv` that carries, per row,
    the six regional SIAF values plus the Arctic-ocean aggregate and
    the global-annual conversion.  The original CSV is still written.

Per-model Delta T is preserved (physically correct for feedback
analysis).  OBS still uses OBS_DELTA_T = 0.581 K.

Why bother doing this for models, since they have no retrieval
contamination?  Three reasons:

  1. Methodological parity with the OBS pipeline (apples-to-apples).
  2. Reduced sensitivity to the per-pixel p<0.05 trend-classification
     cut that splits IS into extent vs darkening.
  3. Pistone's ocean-only mask gives an Arctic-ocean mean that isn't
     diluted by land (~47% of the >=60 N cap).  This is what primarily
     makes the reported Arctic-mean SIAF "physically larger."

The global SIAF should be nearly invariant to the choice, because
mean_all_arctic * 0.0650 and mean_ocean_arctic * 0.0347 are two
bookkeeping paths to the same integral when SIAF = 0 over land.  Both
are reported so you can confirm this empirically.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
from scipy.stats import t as t_dist
from datetime import datetime as _dt
from netCDF4 import Dataset as NC4Dataset
from pykrige.ok import OrdinaryKriging
import glob, os, gc, re, warnings
warnings.filterwarnings('ignore')

# --- PISTONE: extra imports ------------------------------------------------
from pyproj import Transformer
try:
    from global_land_mask import globe as _gm_globe
    _HAS_LAND_MASK = True
except Exception as _e:
    print(f'[PISTONE] global-land-mask unavailable ({_e}); '
          f'ocean mask will fall back to ALL cells. Install with: '
          f'pip install global-land-mask', flush=True)
    _HAS_LAND_MASK = False

# =============================================================================
# CONFIG  (original)
# =============================================================================

CMIP6_ROOT  = '/glade/collections/cmip/CMIP6/CMIP'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
CERES_FILE  = ('/glade/derecho/scratch/jrotondo/ceres_data_updated/'
               'CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc')
PLOT_DIR    = '/glade/work/jrotondo/ncc_paper/plots/cmip6_IS_RS_SIAF/'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- PISTONE: cache dir for SIC PSN25 files --------------------------------
DIR_SIC = '/glade/derecho/scratch/jrotondo/SIC_data/'

MONTH_NAMES = {4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
MONTH_NUMS  = [4, 5, 6, 7, 8, 9]

CMIP6_EARLY_YEARS = list(range(1982, 1992))
CMIP6_LATE_YEARS  = list(range(2005, 2015))
TREND_YEARS       = list(range(1982, 2015))
RS_YEARS          = list(range(1985, 2015))
CERES_RS_YEARS    = list(range(2000, 2015))

OBS_DELTA_T            = 0.581
OBS_DELTA_T_STD        = 0.025
SIC_ICE_THRESHOLD      = 0.15
PVAL                   = 0.05

# Geometric fraction of Earth's surface poleward of 60 N: (1 - sin 60)/2 ~ 0.067.
# Use 0.065 (paper convention).  Correct when the Arctic mean includes ALL >=60 N
# pixels in the denominator, with non-ice pixels contributing 0 to the numerator.
AREA_RATIO_60N = 0.0650

# --- PISTONE: ocean-only >=60 N fraction of Earth's surface ---------------
# Use this with the Pistone-style Arctic-ocean mean.
AREA_RATIO_60N_OCEAN = 0.0347

SKIP_MODELS = {'NorESM2-LM'}

RS_METHOD_FULL_T07   = 'FULL_T07'
RS_METHOD_APPROX_T07 = 'APPROX_T07'
RS_METHOD_RSDS_PROXY = 'RSDS_PROXY'
RS_METHOD_NONE       = 'NONE'
RAD_VARS = ['rsdt', 'rsut', 'rsds', 'rsus', 'rsutcs', 'rsdscs', 'rsuscs']

MODEL_INFO = {
    'CESM2'          : dict(institution='NCAR',         member='r1i1p1f1'),
    'E3SM-1-0'       : dict(institution='E3SM-Project', member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',        member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',        member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',         member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',        member='r1i1p1f1'),
    'CanESM5'        : dict(institution='CCCma',        member='r1i1p1f1'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',        member='r1i1p1f1'),
    'GFDL-ESM4'      : dict(institution='NOAA-GFDL',    member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',         member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',         member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS', member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',          member='r1i1p1f1'),
}

MODEL_COLORS = {
    'CESM2':'#1f77b4', 'E3SM-1-0':'#ff7f0e', 'MPI-ESM1-2-HR':'#2ca02c',
    'MPI-ESM1-2-LR':'#98df8a', 'IPSL-CM6A-LR':'#d62728', 'MIROC6':'#9467bd',
    'CanESM5':'#8c564b', 'ACCESS-ESM1-5':'#e377c2', 'GFDL-ESM4':'#7f7f7f',
    'NorESM2-LM':'#bcbd22', 'UKESM1-0-LL':'#17becf', 'HadGEM3-GC31-LL':'#aec7e8',
    'CNRM-CM6-1':'#ffbb78', 'BCC-CSM2-MR':'#ff9896',
}

RS_METHOD_LABEL = {
    RS_METHOD_FULL_T07:'T07', RS_METHOD_APPROX_T07:'T07~',
    RS_METHOD_RSDS_PROXY:'rsds', RS_METHOD_NONE:'none',
}

PLOT_SPECS = {
    'IS_total'      : dict(nk='is_tot_nat',    cmap=plt.cm.RdBu_r, vmin=-0.10, vmax=0.10,
                           cbar_label='albedo K^-1',        units='albedo K^-1',        fmt='{:+.4f}',
                           ylabel='AMJJAS Arctic IS$_{\\rm total}$  (albedo K$^{-1}$)'),
    'IS_extent'     : dict(nk='is_ext_nat',    cmap=plt.cm.RdBu_r, vmin=-0.10, vmax=0.10,
                           cbar_label='albedo K^-1',        units='albedo K^-1',        fmt='{:+.4f}',
                           ylabel='AMJJAS Arctic IS$_{\\rm extent}$  (albedo K$^{-1}$)'),
    'IS_darkening'  : dict(nk='is_drk_nat',    cmap=plt.cm.RdBu_r, vmin=-0.10, vmax=0.10,
                           cbar_label='albedo K^-1',        units='albedo K^-1',        fmt='{:+.4f}',
                           ylabel='AMJJAS Arctic IS$_{\\rm dark}$  (albedo K$^{-1}$)'),
    'RS'            : dict(nk='rs_nat',         cmap=plt.cm.YlOrRd, vmin=0.,    vmax=350.,
                           cbar_label='W m^-2 per unit a', units='W m^-2 per unit a', fmt='{:.1f}',
                           ylabel='AMJJAS Arctic RS  (W m$^{-2}$ per unit $\\alpha$)'),
    'SIAF_total'    : dict(nk='siaf_tot_nat',  cmap=plt.cm.RdBu_r, vmin=-30.,  vmax=30.,
                           cbar_label='W m^-2 K^-1',         units='W m^-2 K^-1',         fmt='{:+.2f}',
                           ylabel='AMJJAS Global SIAF$_{\\rm total}$  (W m$^{-2}$ K$^{-1}$)'),
    'SIAF_extent'   : dict(nk='siaf_ext_nat',  cmap=plt.cm.RdBu_r, vmin=-30.,  vmax=30.,
                           cbar_label='W m^-2 K^-1',         units='W m^-2 K^-1',         fmt='{:+.2f}',
                           ylabel='AMJJAS Global SIAF$_{\\rm extent}$  (W m$^{-2}$ K$^{-1}$)'),
    'SIAF_darkening': dict(nk='siaf_drk_nat',  cmap=plt.cm.RdBu_r, vmin=-30.,  vmax=30.,
                           cbar_label='W m^-2 K^-1',         units='W m^-2 K^-1',         fmt='{:+.2f}',
                           ylabel='AMJJAS Global SIAF$_{\\rm dark}$  (W m$^{-2}$ K$^{-1}$)'),
}

METRIC_STEMS = ['is_tot', 'is_ext', 'is_drk', 'rs', 'siaf_tot', 'siaf_ext', 'siaf_drk']

# =============================================================================
# PISTONE: PSN25 grid + 6 wedges + ocean mask
# =============================================================================

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# Six Pistone Fig S2 longitudinal wedges between 60-90 N, CCW from 0 E.
PISTONE_REGIONS = [
    # code, name,                              lat_min, lat_max, lon_min, lon_max
    ('EU', 'Europe_Barents',              60, 90,    0,   60),
    ('WA', 'WestAsia_KaraLaptev',         60, 90,   60,  120),
    ('WP', 'WestPacific_EastSiberian',    60, 90,  120,  180),
    ('EP', 'EastPacific_ChukchiBeaufort', 60, 90, -180, -120),
    ('NA', 'NorthAmerica_CdnArchBaffin',  60, 90, -120,  -60),
    ('AO', 'Atlantic_LabradorGreenland',  60, 90,  -60,    0),
]
PISTONE_REGION_CODES = [r[0] for r in PISTONE_REGIONS]


def _build_psn25_grid():
    """
    Build a PSN25 (25x25 km polar-stereographic) grid, compute lat/lon,
    polar-stereographic surface-area weights (1/k_ps^2, nominal 25x25 km^2
    at lat_ts=70 N), an ocean mask, and the six Pistone regional masks.

    Returns dict with keys:
      psn_lat2d, psn_lon2d, psn_shape,
      area_weight, ocean_mask, arctic_mask, domain_mask (ocean AND >=60 N),
      region_masks: {code: 2-D bool},
      transformer (for reuse).
    """
    sic_files = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))
    if not sic_files:
        raise FileNotFoundError(
            f'[PISTONE] no PSN25 SIC file found in {DIR_SIC}.  '
            'Cannot initialise PSN25 grid.')
    with xr.open_dataset(sic_files[0]) as ds:
        psn_x = ds['x'].values.astype(np.float64)
        psn_y = ds['y'].values.astype(np.float64)
    psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)
    psn_shape      = psn_xx.shape

    transformer = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
    psn_lon2d, psn_lat2d = transformer.transform(psn_xx, psn_yy)

    # True surface-area weight: 1/k_ps^2, with k_ps = (1+sin phi_ts)/(1+sin phi)
    phi       = np.deg2rad(psn_lat2d)
    phi_ts    = np.deg2rad(70.0)
    k_ps      = (1.0 + np.sin(phi_ts)) / np.maximum(1.0 + np.sin(phi), 1e-9)
    area_weight = (1.0 / k_ps**2).astype(np.float32)

    # Ocean mask (global-land-mask expects lon in [-180, 180])
    if _HAS_LAND_MASK:
        lon_pm     = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
        ocean_mask = _gm_globe.is_ocean(psn_lat2d, lon_pm)
    else:
        ocean_mask = np.ones(psn_shape, dtype=bool)

    arctic_mask = (psn_lat2d >= 60.0)
    domain_mask = ocean_mask & arctic_mask

    # 6 Pistone regional masks (restricted to ocean AND >=60 N)
    region_masks = {}
    lon_pm = np.where(psn_lon2d > 180, psn_lon2d - 360, psn_lon2d)
    for (code, rname, lo, hi, lmn, lmx) in PISTONE_REGIONS:
        lat_ok = (psn_lat2d >= lo) & (psn_lat2d <= hi)
        if lmn <= lmx:
            lon_ok = (lon_pm >= lmn) & (lon_pm <= lmx)
        else:
            lon_ok = (lon_pm >= lmn) | (lon_pm <= lmx)
        region_masks[code] = lat_ok & lon_ok & domain_mask

    return dict(
        psn_lat2d=psn_lat2d, psn_lon2d=psn_lon2d, psn_shape=psn_shape,
        area_weight=area_weight, ocean_mask=ocean_mask,
        arctic_mask=arctic_mask, domain_mask=domain_mask,
        region_masks=region_masks, transformer=transformer,
    )


def _latlon_to_xyz32(lat_deg, lon_deg):
    """float32 unit-sphere coords for KDTree nearest-neighbour."""
    lat = np.deg2rad(lat_deg.astype(np.float32))
    lon = np.deg2rad(lon_deg.astype(np.float32))
    xyz = np.empty((lat.size, 3), dtype=np.float32)
    xyz[:, 0] = np.cos(lat) * np.cos(lon)
    xyz[:, 1] = np.cos(lat) * np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    return xyz


def _build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d,
                         chunk_size=100_000):
    """Each src pixel votes into its nearest tgt cell (scatter average)."""
    tgt_xyz = _latlon_to_xyz32(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = _latlon_to_xyz32(src_lat2d.ravel(), src_lon2d.ravel())
    idx     = np.empty(src_xyz.shape[0], dtype=np.int32)
    tree    = cKDTree(tgt_xyz)
    for i in range(0, src_xyz.shape[0], chunk_size):
        e = min(i + chunk_size, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx


def _scatter_avg(arr_flat, idx, out_shape):
    accum = np.zeros(np.prod(out_shape), dtype=np.float32)
    count = np.zeros(np.prod(out_shape), dtype=np.float32)
    valid = np.isfinite(arr_flat)
    np.add.at(accum, idx[valid], arr_flat[valid].astype(np.float32))
    np.add.at(count, idx[valid], np.float32(1.0))
    out = np.full(np.prod(out_shape), np.nan, dtype=np.float32)
    m   = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)


# Cache of scatter indices from arbitrary src grid -> PSN25, keyed by a
# grid-identity tuple so repeated calls for the same model are free.
_PSN_INDEX_CACHE = {}


def _src_grid_key(src_lat, src_lon, is_2d):
    """A hashable fingerprint of a src grid (shape + first/last lat/lon)."""
    la = np.asarray(src_lat); lo = np.asarray(src_lon)
    if is_2d:
        key_lat = (la.shape, float(la.ravel()[0]), float(la.ravel()[-1]),
                   float(np.nanmin(la)), float(np.nanmax(la)))
        key_lon = (lo.shape, float(lo.ravel()[0]), float(lo.ravel()[-1]),
                   float(np.nanmin(lo)), float(np.nanmax(lo)))
    else:
        key_lat = (la.shape, float(la[0]), float(la[-1]))
        key_lon = (lo.shape, float(lo[0]), float(lo[-1]))
    return ('2d' if is_2d else '1d', key_lat, key_lon)


def _get_src_to_psn_index(src_lat, src_lon, is_2d, psn_ctx):
    """Return a cached (or newly built) scatter index src -> PSN25."""
    key = _src_grid_key(src_lat, src_lon, is_2d)
    if key in _PSN_INDEX_CACHE:
        return _PSN_INDEX_CACHE[key]
    if is_2d:
        src_lat2d, src_lon2d = src_lat, src_lon
    else:
        src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    idx = _build_scatter_index(
        src_lat2d.astype(np.float32), src_lon2d.astype(np.float32),
        psn_ctx['psn_lat2d'].astype(np.float32),
        psn_ctx['psn_lon2d'].astype(np.float32))
    _PSN_INDEX_CACHE[key] = idx
    return idx


def regrid_to_psn(arr2d, src_lat, src_lon, is_2d, psn_ctx):
    """Scatter-average a 2-D src field to PSN25."""
    idx = _get_src_to_psn_index(src_lat, src_lon, is_2d, psn_ctx)
    return _scatter_avg(arr2d.astype(np.float32).ravel(), idx,
                        psn_ctx['psn_shape'])


def _region_mean_psn(arr_psn, rmask, area_weight):
    """Pistone "weighted by grid cell area" within one regional mask."""
    v = arr_psn[rmask]
    w = area_weight[rmask]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan, 0.0
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])), float(np.sum(w[ok]))


def pistone_arctic_mean(arr_psn, psn_ctx, treat_nan_as_zero=False):
    """
    Compute the 6 Pistone regional means and the Arctic-ocean aggregate.

    If treat_nan_as_zero is True, NaN within the domain (ocean AND >=60 N)
    is replaced with 0.  This is the correct behaviour for SIAF-type
    fields whose "non-ice" pixels are represented as NaN but physically
    contribute zero to the Arctic integral -- so the mean is diluted by
    the full Arctic-ocean area, matching AREA_RATIO_60N_OCEAN = 0.0347.

    If False, NaN pixels are excluded from both numerator and denominator
    (appropriate for ice-pixel-only averages of IS or RS as diagnostics).

    Returns a dict:
        { 'EU': val, 'WA': val, 'WP': val, 'EP': val, 'NA': val, 'AO': val,
          'Arctic_ocean': area-weighted aggregate across the 6 wedges,
          'area_weight': total w_sum inside the domain }
    """
    if treat_nan_as_zero:
        arr = np.where(psn_ctx['domain_mask'] & ~np.isfinite(arr_psn),
                       0.0, arr_psn).astype(np.float32)
    else:
        arr = arr_psn.astype(np.float32)

    vals  = {}
    w_sum = {}
    num   = 0.0
    den   = 0.0
    for code, rmask in psn_ctx['region_masks'].items():
        v, w = _region_mean_psn(arr, rmask, psn_ctx['area_weight'])
        vals[code]  = v
        w_sum[code] = w
        if np.isfinite(v) and w > 0:
            num += v * w
            den += w
    vals['Arctic_ocean'] = (num / den) if den > 0 else np.nan
    vals['area_weight']  = den
    vals['_regions_weights'] = w_sum  # per-region area weights (diagnostic)
    return vals


def pistone_global_annual(monthly_dict):
    """
    Convert a dict {month_num: arctic_ocean_mean_SIAF [W m^-2 K^-1]}
    to a global annual mean [W m^-2 K^-1] using
        global = sum_m( val_m * 1/12 * AREA_RATIO_60N_OCEAN ).
    Missing (NaN) months contribute 0, matching global_siaf_from_monthly.
    """
    return float(sum(
        v * (1.0 / 12.0) * AREA_RATIO_60N_OCEAN
        for v in monthly_dict.values()
        if np.isfinite(v)
    ))


# Module-level handle, populated in __main__.
PSN_CTX = None


# =============================================================================
# GLOBAL SIAF CONVERSION (original)
# =============================================================================

def global_siaf_from_monthly(monthly_dict):
    """
    Convert {month: arctic_mean_SIAF [W m^-2 K^-1]} to a global annual
    mean [W m^-2 K^-1] by summing each month's contribution:

        global = sum_m( val_m * (1/12) * AREA_RATIO_60N )

    Missing (NaN) months contribute 0, not the mean of the others.
    """
    return float(sum(
        v * (1.0 / 12.0) * AREA_RATIO_60N
        for v in monthly_dict.values()
        if np.isfinite(v)
    ))


def scale_obs_monthly_to_global(monthly_dict, area_ratio=AREA_RATIO_60N):
    """
    Scale a per-month Arctic SIAF dict to per-month global annual contribution.
    """
    if monthly_dict is None:
        return None
    return {mn: (v * (1.0 / 12.0) * area_ratio
                 if np.isfinite(v) else np.nan)
            for mn, v in monthly_dict.items()}

# =============================================================================
# GEOMETRY / REGRIDDING (original)
# =============================================================================

def make_circle_boundary(ax, n=100):
    theta = np.linspace(0, 2*np.pi, n)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon), np.sin(lat)])

def regrid_for_display(arr2d, src_lat, src_lon, src_is_2d, tgt_lat, tgt_lon):
    """Nearest-neighbour to 1x1 for display only (not computation)."""
    if src_is_2d:
        flat_lat, flat_lon = src_lat.ravel(), src_lon.ravel()
    else:
        lon2d, lat2d = np.meshgrid(src_lon, src_lat)
        flat_lat, flat_lon = lat2d.ravel(), lon2d.ravel()
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    src_xyz = latlon_to_xyz(flat_lat, flat_lon)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    _, idx  = cKDTree(src_xyz).query(tgt_xyz, workers=-1)
    result  = arr2d.ravel().copy()[idx]
    valid   = np.isfinite(arr2d.ravel())
    if valid.any():
        chord = 2.*np.sin(np.deg2rad(200./111.2/2.))
        dists, _ = cKDTree(src_xyz[valid]).query(tgt_xyz, workers=-1)
        result[dists > chord] = np.nan
    return result.reshape(tgt_lat2d.shape)

def regrid_scatter_fn(arr, src_lat2d, src_lon2d, tgt_lat, tgt_lon):
    """Scatter-average: each src pixel votes into its nearest tgt cell."""
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    _, idx  = cKDTree(tgt_xyz).query(src_xyz, workers=-1)
    flat    = arr.ravel()
    accum   = np.zeros(tgt_lat2d.size)
    count   = np.zeros(tgt_lat2d.size)
    valid   = np.isfinite(flat)
    np.add.at(accum, idx[valid], flat[valid])
    np.add.at(count, idx[valid], 1)
    out     = np.full(tgt_lat2d.size, np.nan)
    m       = count > 0; out[m] = accum[m] / count[m]
    return out.reshape(tgt_lat2d.shape)

def regrid_ts(src_3d, src_lat, src_lon, tgt_lat, tgt_lon):
    if src_lat.ndim == 1:
        src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    else:
        src_lat2d, src_lon2d = src_lat, src_lon
    src_xyz              = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)
    tgt_xyz              = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    _, idx               = cKDTree(src_xyz).query(tgt_xyz, workers=-1)
    n_t                  = src_3d.shape[0]
    return src_3d.reshape(n_t, -1)[:, idx].reshape((n_t,) + tgt_lat2d.shape)

def arctic_wmean_1d(arr2d, lat_1d):
    lat2d = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (float(np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid]))
            if valid.any() else np.nan)

def arctic_wmean_2d(arr2d, lat2d):
    w     = np.cos(np.deg2rad(lat2d))
    valid = (lat2d >= 60.) & np.isfinite(arr2d)
    return (float(np.nansum(arr2d[valid]*w[valid]) / np.nansum(w[valid]))
            if valid.any() else np.nan)

def arctic_wmean(arr2d, lat, is_2d):
    if is_2d:
        return arctic_wmean_2d(arr2d, lat)
    lat2d = np.tile(lat[:, None], (1, arr2d.shape[1]))
    return arctic_wmean_1d(np.where(lat2d >= 60., arr2d, np.nan), lat)

def arctic_wmean_common(arr2d, common_lat):
    return arctic_wmean_1d(arr2d, common_lat)

def arctic_wmean_allArctic_2d(arr2d, lat2d):
    w      = np.cos(np.deg2rad(lat2d))
    arctic = lat2d >= 60.
    if not arctic.any():
        return np.nan
    num = float(np.sum(np.where(np.isfinite(arr2d) & arctic, arr2d * w, 0.)))
    den = float(np.sum(w[arctic]))
    return num / den if den > 0 else np.nan


def arctic_wmean_allArctic_1d(arr2d, lat_1d):
    lat2d  = np.tile(lat_1d[:, None], (1, arr2d.shape[1]))
    w      = np.cos(np.deg2rad(lat2d))
    arctic = lat2d >= 60.
    if not arctic.any():
        return np.nan
    num = float(np.sum(np.where(np.isfinite(arr2d) & arctic, arr2d * w, 0.)))
    den = float(np.sum(w[arctic]))
    return num / den if den > 0 else np.nan


def arctic_wmean_allArctic(arr2d, lat, is_2d):
    if is_2d:
        return arctic_wmean_allArctic_2d(arr2d, lat)
    lat2d = np.tile(lat[:, None], (1, arr2d.shape[1]))
    return arctic_wmean_allArctic_1d(
        np.where(lat2d >= 60., arr2d, np.nan), lat)

# =============================================================================
# VECTORIZED OLS PIXEL-WISE TREND
# =============================================================================

def pixel_ols_trend(years_1d, sic_3d):
    T   = years_1d.shape[0]
    x   = years_1d - years_1d.mean()
    ss  = float(np.dot(x, x))
    y_mean = np.nanmean(sic_3d, axis=0)
    dy     = sic_3d - y_mean[None, :, :]
    slope  = np.nansum(x[:, None, None] * dy, axis=0) / ss
    resid  = sic_3d - (y_mean + slope * x[:, None, None])
    sse    = np.nansum(resid**2, axis=0)
    df     = T - 2
    with np.errstate(invalid='ignore', divide='ignore'):
        se     = np.sqrt(sse / df) / np.sqrt(ss)
        t_stat = slope / se
        pval   = 2. * (1. - t_dist.cdf(np.abs(t_stat), df))
    return slope, pval

# =============================================================================
# T07 INVERSION
# =============================================================================

def t07_invert(rsdt, rsut, rsds, rsus, rsutcs, rsdscs, rsuscs):
    with np.errstate(invalid='ignore', divide='ignore'):
        S   = np.where(rsdt  > 1., rsdt,  np.nan)
        Sd  = np.where(rsds  > 1., rsds,  np.nan)
        a   = np.clip(rsus / Sd, 0., 1.)
        p   = rsds / S;  q = rsut / S;  den = q + p - 2.*p*a
        R   = np.clip((q - p*a) / np.where(np.abs(den) > 1e-6, den, np.nan), 0., 0.99)
        A   = np.clip(1. - p*(1. - a*R) / np.where((1.-R) > 0.01, 1.-R, np.nan), 0., 0.99)
        Sc  = np.where(rsdscs > 1., rsdscs, np.nan)
        ac  = np.clip(rsuscs / Sc, 0., 1.)
        pc  = rsdscs / S;  qc = rsutcs / S;  dc = qc + pc - 2.*pc*ac
        Rc  = np.clip((qc - pc*ac) / np.where(np.abs(dc) > 1e-6, dc, np.nan), 0., 0.99)
        Rl  = np.clip(R - Rc, 0., 0.99)
        RS  = (S * (1.-A) * (1.-Rl)**2
               / np.where((1.-a*Rl)**2 > 1e-9, (1.-a*Rl)**2, np.nan))
    return RS

# =============================================================================
# FILE FINDERS / LOADERS
# =============================================================================

def find_var_files(inst, model, member, variable):
    tables = ['SImon', 'SIday'] if variable == 'siconc' else ['Amon', 'CFmon']
    for table in tables:
        var_dir = os.path.join(CMIP6_ROOT, inst, model, 'historical',
                               member, table, variable)
        if os.path.isdir(var_dir):
            files = sorted(glob.glob(
                os.path.join(var_dir, '**', f'{variable}_*.nc'), recursive=True))
            if files:
                return files
    return []

def find_all_members(inst, model):
    base_dir = os.path.join(CMIP6_ROOT, inst, model, 'historical')
    if not os.path.isdir(base_dir):
        return []
    members = []
    try:
        for d in sorted(os.listdir(base_dir)):
            if (find_var_files(inst, model, d, 'rsus') and
                    find_var_files(inst, model, d, 'rsds') and
                    find_var_files(inst, model, d, 'siconc')):
                members.append(d)
    except Exception:
        pass
    return members

def _open_robust(files, variable):
    for kw in [{'combine': 'by_coords'},
               {'combine': 'nested', 'concat_dim': 'time'}]:
        try:
            return xr.open_mfdataset(files, use_cftime=True,
                                      chunks={'time': 12}, **kw)
        except Exception:
            pass
    raise RuntimeError(f'Cannot open {variable}')

def _std_latlon(da):
    aliases = {'lat': ['latitude', 'nav_lat', 'rlat', 'y'],
               'lon': ['longitude', 'nav_lon', 'rlon', 'x']}
    for new, als in aliases.items():
        if new in da.coords and da[new].ndim == 1 and new in da.dims:
            continue
        for c in list(da.coords):
            if c.lower() in als and c != new:
                try:
                    if new in da.coords:
                        if new in da.dims:
                            da = da.drop_vars(new)
                        da = da.rename({c: new})
                    else:
                        da = da.rename({c: new})
                except Exception:
                    pass
                break
    lat = da['lat'].values if 'lat' in da.coords else None
    lon = da['lon'].values if 'lon' in da.coords else None
    return da, lat, lon, (lat.ndim == 2 if lat is not None else False)

def load_filtered(files, variable, years, months):
    try:
        ds   = _open_robust(files, variable)
        da   = ds[variable]
        ys   = set(years); ms = set(months)
        mask = np.array([int(t.dt.year) in ys and int(t.dt.month) in ms
                         for t in da.time], dtype=bool)
        da   = da.isel(time=mask).load(); ds.close()
        da, _, _, _ = _std_latlon(da)
        return da
    except Exception as e:
        print(f'    load_filtered error ({variable}): {e}', flush=True)
        return None

def _ym_set(files, variable, years, months):
    if not files:
        return set()
    try:
        ds = _open_robust(files, variable)
        da = ds[variable]; ys = set(years); ms = set(months)
        ym = {(int(t.dt.year), int(t.dt.month)) for t in da.time
              if int(t.dt.year) in ys and int(t.dt.month) in ms}
        ds.close(); return ym
    except Exception:
        return set()

# =============================================================================
# RS DETERMINATION AND COMPUTATION
# =============================================================================

def determine_rs_method(inst, model, member):
    rf  = {v: find_var_files(inst, model, member, v) for v in RAD_VARS}
    av  = {v for v, f in rf.items() if f}
    overlap = False
    if 'rsdt' in av and 'rsds' in av:
        ym_dt   = _ym_set(rf['rsdt'], 'rsdt', RS_YEARS, list(range(1, 13)))
        ym_ds   = _ym_set(rf['rsds'], 'rsds', RS_YEARS, list(range(1, 13)))
        overlap = bool(ym_dt & ym_ds)
    if overlap:
        if set(RAD_VARS) <= av:
            return RS_METHOD_FULL_T07, rf
        if (set(RAD_VARS) - {'rsuscs'}) <= av:
            return RS_METHOD_APPROX_T07, rf
    if 'rsds' in av:
        return RS_METHOD_RSDS_PROXY, rf
    return RS_METHOD_NONE, {}

def _load_mean(files, variable, years, month_num):
    if not files:
        return None
    try:
        ds   = _open_robust(files, variable)
        da   = ds[variable]
        mask = np.array([int(t.dt.year) in set(years) and
                         int(t.dt.month) == month_num for t in da.time], dtype=bool)
        if not mask.any():
            ds.close(); return None
        sub = da.isel(time=mask).mean('time').load()
        sub, lat, lon, is2d = _std_latlon(sub)
        ds.close()
        return sub.values.astype(float), lat, lon, is2d
    except Exception as e:
        print(f'    _load_mean ({variable}): {e}', flush=True)
        return None

def compute_rs_maps(inst, model, member, method, rf):
    rs     = {}
    lat    = lon = None
    is2d   = False
    actual = method

    for mn in MONTH_NUMS:
        if method == RS_METHOD_RSDS_PROXY:
            r = _load_mean(rf.get('rsds', []), 'rsds', RS_YEARS, mn)
            if r is None:
                rs[mn] = None; continue
            arr, lat, lon, is2d = r; rs[mn] = arr; continue

        req  = (RAD_VARS if method == RS_METHOD_FULL_T07
                else [v for v in RAD_VARS if v != 'rsuscs'])
        yms  = [_ym_set(rf.get(v, []), v, RS_YEARS, [mn]) for v in req]
        comm = yms[0].copy() if yms else set()
        for s in yms[1:]:
            comm &= s

        if not comm:
            r = _load_mean(rf.get('rsds', []), 'rsds', RS_YEARS, mn)
            if r is not None:
                arr, lat, lon, is2d = r; rs[mn] = arr; actual = RS_METHOD_RSDS_PROXY
            else:
                rs[mn] = None
            continue

        yrs  = sorted({y for y, _ in comm})
        data = {}; ok = True
        for v in req:
            r = _load_mean(rf[v], v, yrs, mn)
            if r is None:
                ok = False; break
            arr, lat, lon, is2d = r; data[v] = arr
        if not ok:
            rs[mn] = None; continue

        if method == RS_METHOD_APPROX_T07:
            with np.errstate(invalid='ignore', divide='ignore'):
                a_s = np.where(data['rsds'] > 1.,
                               data['rsus'] / data['rsds'], 0.)
                data['rsuscs'] = data['rsdscs'] * a_s

        RS = t07_invert(data['rsdt'], data['rsut'], data['rsds'], data['rsus'],
                        data['rsutcs'], data['rsdscs'], data['rsuscs'])

        lat2d_t = lat if is2d else np.tile(lat[:, None], (1, lon.shape[0]))
        am = (arctic_wmean_2d(RS, lat2d_t) if is2d
              else arctic_wmean_1d(np.where(lat2d_t >= 60., RS, np.nan), lat))
        if np.isfinite(am) and (am < 0 or am > 600):
            print(f'    RS sanity mn={mn} am={am:.1f} -> rsds proxy', flush=True)
            r = _load_mean(rf.get('rsds', []), 'rsds', RS_YEARS, mn)
            if r is not None:
                RS, lat, lon, is2d = r; actual = RS_METHOD_RSDS_PROXY
        rs[mn] = RS

    return rs, lat, lon, is2d, actual

# =============================================================================
# DELTA T
# =============================================================================

def compute_delta_T(model, member):
    inst = MODEL_INFO[model]['institution']
    tf   = find_var_files(inst, model, member, 'tas')
    if not tf:
        return None
    da   = load_filtered(tf, 'tas',
                         CMIP6_EARLY_YEARS + CMIP6_LATE_YEARS,
                         list(range(1, 13)))
    if da is None:
        return None
    _, lat, lon, is2d = _std_latlon(da)
    if lat is None or is2d or lat.ndim != 1:
        return None

    yrs  = np.array([int(t.dt.year) for t in da.time])
    vals = da.values.astype(float)
    w    = np.cos(np.deg2rad(np.tile(lat[:, None], (1, len(lon)))))

    def gmst(yr_list):
        mask = np.isin(yrs, yr_list)
        if not mask.any():
            return np.nan
        return np.mean([
            np.nansum(vals[i][np.isfinite(vals[i])] * w[np.isfinite(vals[i])])
            / np.nansum(w[np.isfinite(vals[i])])
            for i in np.where(mask)[0] if np.isfinite(vals[i]).any()])

    eT, lT = gmst(CMIP6_EARLY_YEARS), gmst(CMIP6_LATE_YEARS)
    del da; gc.collect()
    if not (np.isfinite(eT) and np.isfinite(lT)):
        return None
    dT = lT - eT
    return dT if dT > 0 else None

# =============================================================================
# OBS ALBEDO (CLARA + APP-x fallback + Kriging gap-fill)
# =============================================================================

def _build_appx_mapping(sial_shape, cache_dir):
    try:
        sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
        with xr.open_dataset(sample_sial) as ds:
            sial_lat1d = ds['lat'].values
            sial_lon1d = ds['lon'].values
        if sial_lat1d.ndim == 1:
            sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
        else:
            sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
        sial_xyz = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())

        sample_appx = glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True)[0]
        with NC4Dataset(sample_appx, 'r') as ds:
            appx_lat = np.ma.filled(ds.variables['latitude'][:], np.nan).astype(float)
            appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
        if appx_lat.ndim == 1:
            appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
        else:
            appx_lat2d, appx_lon2d = appx_lat, appx_lon
        appx_xyz = latlon_to_xyz(appx_lat2d.ravel(), appx_lon2d.ravel())

        _, idx = cKDTree(appx_xyz).query(sial_xyz, workers=-1)
        return idx, sial_lat2d, sial_lon2d
    except Exception as e:
        print(f'  APP-x mapping failed: {e}', flush=True)
        return None, None, None


def load_and_fill_obs_albedo(mname, year, sial_shape, arctic_sial,
                              idx_appx_to_sial):
    clara_path = f'{CACHE_CMIP6}/clara_alb_{mname}_{year}.npy'
    if not os.path.exists(clara_path):
        return None
    merged = np.load(clara_path).astype(float)

    appx_path = f'{CACHE_CMIP6}/appx_alb_{mname}_{year}.npy'
    if os.path.exists(appx_path) and idx_appx_to_sial is not None:
        appx_reg = np.load(appx_path).astype(float).ravel()[idx_appx_to_sial].reshape(sial_shape)
        merged   = np.where(np.isnan(merged), appx_reg, merged)

    missing = np.isnan(merged) & arctic_sial
    if not missing.any():
        return merged

    valid   = ~np.isnan(merged) & arctic_sial
    gy, gx  = np.indices(merged.shape)
    ty, tx, tv = gy[valid].astype(float), gx[valid].astype(float), merged[valid]

    if len(tv) > 2500:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(tv), 2500, replace=False)
        ty, tx, tv = ty[idx], tx[idx], tv[idx]

    ok = OrdinaryKriging(tx, ty, tv, variogram_model='spherical',
                         enable_statistics=False)
    kv, _ = ok.execute('points',
                        gx[missing].astype(float),
                        gy[missing].astype(float))
    merged[missing] = kv
    return merged

# =============================================================================
# OBS IS (Total / Extent / Darkening)
# =============================================================================

def compute_obs_IS(common_lat, common_lon, sial_shape, arctic_sial,
                   sial_lat2d, sial_lon2d, idx_appx_to_sial):
    print('Computing OBS IS...', flush=True)
    out_tot,  out_ext,  out_drk  = {}, {}, {}
    out_tot_nat, out_ext_nat, out_drk_nat = {}, {}, {}

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]

        ae = [a for y in CMIP6_EARLY_YEARS
              if (a := load_and_fill_obs_albedo(mname, y, sial_shape,
                                                arctic_sial, idx_appx_to_sial))
              is not None]
        al = [a for y in CMIP6_LATE_YEARS
              if (a := load_and_fill_obs_albedo(mname, y, sial_shape,
                                                arctic_sial, idx_appx_to_sial))
              is not None]

        if not ae or not al:
            print(f'  OBS {mname}: no albedo data -- skip', flush=True)
            out_tot[mn]  = out_ext[mn]  = out_drk[mn]  = None
            out_tot_nat[mn] = out_ext_nat[mn] = out_drk_nat[mn] = None
            continue

        dalpha = np.nanmean(al, axis=0) - np.nanmean(ae, axis=0)

        slope_path = f'{CACHE_CMIP6}/sic_trend_slope_sial_{mname}.npy'
        pval_path  = f'{CACHE_CMIP6}/sic_trend_pval_sial_{mname}.npy'
        sice_path  = f'{CACHE_CMIP6}/sic_early_sial_{mname}.npy'
        sicl_path  = f'{CACHE_CMIP6}/sic_late_sial_{mname}.npy'

        if not all(os.path.exists(p) for p in
                   [slope_path, pval_path, sice_path, sicl_path]):
            print(f'  OBS {mname}: missing SIC cache -- skip', flush=True)
            out_tot[mn]  = out_ext[mn]  = out_drk[mn]  = None
            out_tot_nat[mn] = out_ext_nat[mn] = out_drk_nat[mn] = None
            continue

        slope = np.load(slope_path)
        pval  = np.load(pval_path)
        sic_e = np.load(sice_path)
        sic_l = np.load(sicl_path)

        valid    = arctic_sial & np.isfinite(dalpha)
        changing = (valid & np.isfinite(slope) & np.isfinite(pval)
                    & (slope < 0) & (pval < PVAL))
        stable   = (valid & ~changing
                    & (np.fmax(sic_e, sic_l) >= SIC_ICE_THRESHOLD))

        IS_ext = np.where(changing, -dalpha / OBS_DELTA_T, np.nan)
        IS_drk = np.where(stable,   -dalpha / OBS_DELTA_T, np.nan)
        IS_tot = np.where(
            np.isfinite(IS_ext) | np.isfinite(IS_drk),
            np.where(np.isfinite(IS_ext), IS_ext, 0.)
            + np.where(np.isfinite(IS_drk), IS_drk, 0.),
            np.nan)

        IS_ext = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
        IS_drk = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)

        out_tot_nat[mn] = IS_tot.copy()
        out_ext_nat[mn] = IS_ext.copy()
        out_drk_nat[mn] = IS_drk.copy()

        out_tot[mn] = regrid_scatter_fn(IS_tot, sial_lat2d, sial_lon2d,
                                         common_lat, common_lon)
        out_ext[mn] = regrid_scatter_fn(IS_ext, sial_lat2d, sial_lon2d,
                                         common_lat, common_lon)
        out_drk[mn] = regrid_scatter_fn(IS_drk, sial_lat2d, sial_lon2d,
                                         common_lat, common_lon)

        am_tot = arctic_wmean_allArctic_2d(IS_tot, sial_lat2d)
        am_ext = arctic_wmean_allArctic_2d(IS_ext, sial_lat2d)
        am_drk = arctic_wmean_allArctic_2d(IS_drk, sial_lat2d)
        print(f'  OBS {mname}: tot={am_tot:+.5f}  ext={am_ext:+.5f}  '
              f'drk={am_drk:+.5f}  [all-Arctic mean]', flush=True)

    return out_tot, out_ext, out_drk, out_tot_nat, out_ext_nat, out_drk_nat

# =============================================================================
# OBS RS FROM CERES T07
# =============================================================================

def compute_obs_RS(common_lat, common_lon, sial_lat2d, sial_lon2d, sial_shape):
    print('Computing OBS RS from CERES...', flush=True)
    ds        = xr.open_dataset(CERES_FILE)
    times     = pd.DatetimeIndex(ds.time.values)
    ceres_lat = ds['lat'].values
    ceres_lon = ds['lon'].values
    clon2d, clat2d = np.meshgrid(ceres_lon, ceres_lat)
    ceres_xyz = latlon_to_xyz(clat2d.ravel(), clon2d.ravel())

    clon2d_cm, clat2d_cm = np.meshgrid(common_lon, common_lat)
    common_xyz = latlon_to_xyz(clat2d_cm.ravel(), clon2d_cm.ravel())
    _, idx_common = cKDTree(common_xyz).query(ceres_xyz, workers=-1)

    sial_xyz   = latlon_to_xyz(sial_lat2d.ravel(), sial_lon2d.ravel())
    _, idx_sial = cKDTree(ceres_xyz).query(sial_xyz, workers=-1)

    def _scatter_common(arr):
        flat  = arr.ravel()
        accum = np.zeros(common_lat.size * common_lon.size)
        count = np.zeros(common_lat.size * common_lon.size)
        valid = np.isfinite(flat)
        np.add.at(accum, idx_common[valid], flat[valid])
        np.add.at(count, idx_common[valid], 1)
        out   = np.full(common_lat.size * common_lon.size, np.nan)
        m     = count > 0; out[m] = accum[m] / count[m]
        return out.reshape(len(common_lat), len(common_lon))

    rs_maps      = {}
    rs_sial_maps = {}
    yr_set = set(CERES_RS_YEARS)
    for mn in MONTH_NUMS:
        mask = np.array([t.year in yr_set and t.month == mn
                         for t in times], dtype=bool)
        if not mask.any():
            rs_maps[mn] = None; rs_sial_maps[mn] = None; continue
        sel = ds.isel(time=mask)
        RS  = t07_invert(
            sel['solar_mon'].mean('time').values,
            sel['toa_sw_all_mon'].mean('time').values,
            sel['sfc_sw_down_all_mon'].mean('time').values,
            sel['sfc_sw_up_all_mon'].mean('time').values,
            sel['toa_sw_clr_c_mon'].mean('time').values,
            sel['sfc_sw_down_clr_c_mon'].mean('time').values,
            sel['sfc_sw_up_clr_c_mon'].mean('time').values)

        rs_maps[mn]      = _scatter_common(RS)
        rs_sial_maps[mn] = RS.ravel()[idx_sial].reshape(sial_shape)

        am = arctic_wmean_1d(np.where(clat2d_cm >= 60., rs_maps[mn], np.nan),
                              common_lat)
        print(f'  OBS RS {MONTH_NAMES[mn]}: {am:.2f} W m^-2 per unit alpha', flush=True)
    ds.close()
    return rs_maps, rs_sial_maps

# =============================================================================
# CORE: COMPUTE ONE MEMBER  (now also produces Pistone-style scalars)
# =============================================================================

def compute_member(model, member, delta_t, rs_maps, rs_lat, rs_lon, rs_is2d,
                   psn_ctx=None):
    """
    Same as the original compute_member, but also (if psn_ctx is provided):

      * Scatter-regrids each native-grid field (IS_tot/ext/drk, RS, SIAF_tot/
        ext/drk) to PSN25.
      * Computes per-region + Arctic-ocean Pistone-style scalars.
      * Returns additional keys per stem:
          {stem}_sc_pistone       : {month: Arctic-ocean scalar}
          {stem}_regional_pistone : {month: {code: value}}
    """
    inst   = MODEL_INFO[model]['institution']
    rsus_f = find_var_files(inst, model, member, 'rsus')
    rsds_f = find_var_files(inst, model, member, 'rsds')
    sic_f  = find_var_files(inst, model, member, 'siconc')
    if not all([rsus_f, rsds_f, sic_f]):
        return None

    all_yrs = list(range(1980, 2016))
    ru = load_filtered(rsus_f, 'rsus',   all_yrs, MONTH_NUMS)
    rd = load_filtered(rsds_f, 'rsds',   all_yrs, MONTH_NUMS)
    si = load_filtered(sic_f,  'siconc', all_yrs, MONTH_NUMS)
    if ru is None or rd is None or si is None:
        return None

    _, alb_lat, alb_lon, alb_is2d = _std_latlon(ru)
    _, sic_lat, sic_lon, sic_is2d = _std_latlon(si)
    if alb_lat is None or sic_lat is None:
        return None
    if not alb_is2d and alb_lat.ndim != 1:
        return None

    def _ym(da):
        return set(zip([int(t.dt.year) for t in da.time],
                        [int(t.dt.month) for t in da.time]))
    comm = _ym(ru) & _ym(rd) & _ym(si)
    if not comm:
        return None

    def _tmask(da):
        return np.array([(int(t.dt.year), int(t.dt.month)) in comm
                          for t in da.time], dtype=bool)
    ru = ru.isel(time=_tmask(ru))
    rd = rd.isel(time=_tmask(rd))
    si = si.isel(time=_tmask(si))

    sv = si.values.astype(float)
    if np.nanmax(sv) > 1.5:
        sv /= 100.

    alb_sh = alb_lat.shape if alb_is2d else (len(alb_lat), len(alb_lon))

    if sv.shape[1:] != alb_sh:
        sl = (sic_lat if sic_is2d else
              (sic_lat[:, 0] if sic_lat.ndim == 2 else sic_lat))
        sw = (sic_lon if sic_is2d else
              (sic_lon[0, :] if sic_lon.ndim == 2 else sic_lon))
        sic_alb = regrid_ts(sv, sl, sw,
                             alb_lat.ravel() if alb_is2d else alb_lat,
                             alb_lon.ravel() if alb_is2d else alb_lon)
        if alb_is2d:
            sic_alb = sic_alb.reshape((sic_alb.shape[0],) + alb_sh)
        del sv; gc.collect()
    else:
        sic_alb = sv

    ru_yrs = np.array([int(t.dt.year)  for t in ru.time])
    ru_mon = np.array([int(t.dt.month) for t in ru.time])
    si_yrs = np.array([int(t.dt.year)  for t in si.time])
    si_mon = np.array([int(t.dt.month) for t in si.time])
    del si; gc.collect()

    if alb_is2d:
        lat2d_a, lon2d_a = alb_lat, alb_lon
    else:
        lon2d_a, lat2d_a = np.meshgrid(alb_lon, alb_lat)

    stems = ['is_tot', 'is_ext', 'is_drk', 'rs',
             'siaf_tot', 'siaf_ext', 'siaf_drk']
    sc          = {s: {} for s in stems}
    nat         = {s: {} for s in stems}
    sc_pistone  = {s: {} for s in stems}          # PISTONE: Arctic-ocean scalars
    reg_pistone = {s: {} for s in stems}          # PISTONE: per-region dicts

    # Helper to compute the Pistone Arctic-ocean mean of any native-grid map,
    # using scatter-regrid to PSN25 and the appropriate NaN treatment.
    def _pistone_scalar(arr_native, treat_nan_as_zero):
        if psn_ctx is None or arr_native is None:
            return np.nan, {c: np.nan for c in PISTONE_REGION_CODES}
        try:
            arr_psn = regrid_to_psn(arr_native, alb_lat, alb_lon,
                                    alb_is2d, psn_ctx)
        except Exception as ex:
            print(f'    [pistone regrid failed: {ex}]', flush=True)
            return np.nan, {c: np.nan for c in PISTONE_REGION_CODES}
        out = pistone_arctic_mean(arr_psn, psn_ctx,
                                  treat_nan_as_zero=treat_nan_as_zero)
        reg = {c: out[c] for c in PISTONE_REGION_CODES}
        return out['Arctic_ocean'], reg

    for mn in MONTH_NUMS:
        def epoch_alb(yr_list):
            mask = np.isin(ru_yrs, yr_list) & (ru_mon == mn)
            if mask.sum() == 0:
                return None
            u = ru.isel(time=mask)
            d = rd.isel(time=mask)
            with np.errstate(invalid='ignore', divide='ignore'):
                alb = np.where(d.mean('time').values > 1.,
                               u.mean('time').values / d.mean('time').values,
                               np.nan)
            return np.clip(alb, 0., 1.)

        ae = epoch_alb(CMIP6_EARLY_YEARS)
        al = epoch_alb(CMIP6_LATE_YEARS)
        if ae is None or al is None:
            for s in stems:
                sc[s][mn] = np.nan; nat[s][mn] = None
                sc_pistone[s][mn] = np.nan
                reg_pistone[s][mn] = {c: np.nan for c in PISTONE_REGION_CODES}
            continue

        dalpha = al - ae

        trend_mask = (si_mon == mn) & np.isin(si_yrs, TREND_YEARS)
        se_mask    = (si_mon == mn) & np.isin(si_yrs, CMIP6_EARLY_YEARS)
        sl_mask    = (si_mon == mn) & np.isin(si_yrs, CMIP6_LATE_YEARS)

        n_trend = trend_mask.sum()
        if n_trend < 10 or se_mask.sum() == 0 or sl_mask.sum() == 0:
            for s in stems:
                sc[s][mn] = np.nan; nat[s][mn] = None
                sc_pistone[s][mn] = np.nan
                reg_pistone[s][mn] = {c: np.nan for c in PISTONE_REGION_CODES}
            continue

        sic_trend  = sic_alb[trend_mask]
        trend_yrs  = si_yrs[trend_mask].astype(float)
        sic_e_mean = np.nanmean(sic_alb[se_mask], axis=0)
        sic_l_mean = np.nanmean(sic_alb[sl_mask], axis=0)

        all_ice = (lat2d_a >= 60.) & (sic_e_mean >= SIC_ICE_THRESHOLD)

        slope, pval = pixel_ols_trend(trend_yrs, sic_trend)

        changing = (all_ice & np.isfinite(slope) & np.isfinite(pval)
                    & (slope < 0) & (pval < PVAL))
        stable   = (all_ice & ~changing
                    & (np.fmax(sic_e_mean, sic_l_mean) >= SIC_ICE_THRESHOLD))

        IS_ext = np.where(changing, -dalpha / delta_t, np.nan)
        IS_drk = np.where(stable,   -dalpha / delta_t, np.nan)
        IS_tot = np.where(
            np.isfinite(IS_ext) | np.isfinite(IS_drk),
            np.where(np.isfinite(IS_ext), IS_ext, 0.)
            + np.where(np.isfinite(IS_drk), IS_drk, 0.),
            np.nan)

        IS_ext = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan)
        IS_drk = np.where(np.isfinite(IS_tot),
                          np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan)

        RS_raw = rs_maps.get(mn)
        if RS_raw is not None and RS_raw.shape == alb_sh:
            RS_m     = np.where(all_ice, RS_raw, np.nan)
            SIAF_ext = np.where(np.isfinite(IS_ext) & np.isfinite(RS_raw),
                                 RS_raw * IS_ext, np.nan)
            SIAF_drk = np.where(np.isfinite(IS_drk) & np.isfinite(RS_raw),
                                 RS_raw * IS_drk, np.nan)
            SIAF_tot = np.where(np.isfinite(IS_tot) & np.isfinite(RS_raw),
                                 RS_raw * IS_tot, np.nan)
            SIAF_ext = np.where(np.isfinite(SIAF_tot),
                                 np.where(np.isfinite(SIAF_ext), SIAF_ext, 0.), np.nan)
            SIAF_drk = np.where(np.isfinite(SIAF_tot),
                                 np.where(np.isfinite(SIAF_drk), SIAF_drk, 0.), np.nan)
        else:
            if RS_raw is not None:
                print(f'    [{model}/{member}] m={mn}: RS shape {RS_raw.shape} '
                      f'!= alb shape {alb_sh} -- SIAF NaN', flush=True)
            nan_map  = np.full(alb_sh, np.nan)
            RS_m     = nan_map.copy()
            SIAF_ext = nan_map.copy()
            SIAF_drk = nan_map.copy()
            SIAF_tot = nan_map.copy()

        # ---- Existing scalars (all-Arctic, cos-lat weighted, 0.065 ratio) ----
        sc['is_tot'][mn]   = arctic_wmean_allArctic(IS_tot,   alb_lat, alb_is2d)
        sc['is_ext'][mn]   = arctic_wmean_allArctic(IS_ext,   alb_lat, alb_is2d)
        sc['is_drk'][mn]   = arctic_wmean_allArctic(IS_drk,   alb_lat, alb_is2d)
        sc['rs'][mn]       = arctic_wmean_allArctic(RS_m,     alb_lat, alb_is2d)
        sc['siaf_tot'][mn] = arctic_wmean_allArctic(SIAF_tot, alb_lat, alb_is2d)
        sc['siaf_ext'][mn] = arctic_wmean_allArctic(SIAF_ext, alb_lat, alb_is2d)
        sc['siaf_drk'][mn] = arctic_wmean_allArctic(SIAF_drk, alb_lat, alb_is2d)

        nat['is_tot'][mn]   = IS_tot.copy()
        nat['is_ext'][mn]   = IS_ext.copy()
        nat['is_drk'][mn]   = IS_drk.copy()
        nat['rs'][mn]       = RS_m.copy()
        nat['siaf_tot'][mn] = SIAF_tot.copy()
        nat['siaf_ext'][mn] = SIAF_ext.copy()
        nat['siaf_drk'][mn] = SIAF_drk.copy()

        # ---- PISTONE scalars (PSN25, 6 wedges, ocean-only, 0.0347 ratio) ----
        # For SIAF and IS variants: treat NaN as 0 within ocean >=60 N so the
        # mean dilutes by the full Arctic-ocean area (matches 0.0347 factor).
        # For RS: use ice-pixel-only mean (diagnostic, no NaN->0).
        s_tot,  r_tot  = _pistone_scalar(IS_tot,   treat_nan_as_zero=True)
        s_ext,  r_ext  = _pistone_scalar(IS_ext,   treat_nan_as_zero=True)
        s_drk,  r_drk  = _pistone_scalar(IS_drk,   treat_nan_as_zero=True)
        s_rs,   r_rs   = _pistone_scalar(RS_m,     treat_nan_as_zero=False)
        s_sft,  r_sft  = _pistone_scalar(SIAF_tot, treat_nan_as_zero=True)
        s_sfe,  r_sfe  = _pistone_scalar(SIAF_ext, treat_nan_as_zero=True)
        s_sfd,  r_sfd  = _pistone_scalar(SIAF_drk, treat_nan_as_zero=True)

        sc_pistone['is_tot'][mn]   = s_tot;  reg_pistone['is_tot'][mn]   = r_tot
        sc_pistone['is_ext'][mn]   = s_ext;  reg_pistone['is_ext'][mn]   = r_ext
        sc_pistone['is_drk'][mn]   = s_drk;  reg_pistone['is_drk'][mn]   = r_drk
        sc_pistone['rs'][mn]       = s_rs;   reg_pistone['rs'][mn]       = r_rs
        sc_pistone['siaf_tot'][mn] = s_sft;  reg_pistone['siaf_tot'][mn] = r_sft
        sc_pistone['siaf_ext'][mn] = s_sfe;  reg_pistone['siaf_ext'][mn] = r_sfe
        sc_pistone['siaf_drk'][mn] = s_sfd;  reg_pistone['siaf_drk'][mn] = r_sfd

    del ru, rd, sic_alb; gc.collect()

    def _amj(d):
        return np.nanmean([d[m] for m in MONTH_NUMS])

    result = {}
    for s in stems:
        result[f'{s}_sc']              = sc[s]
        result[f'{s}_nat']             = nat[s]
        result[f'{s}_amjjas']          = _amj(sc[s])
        result[f'{s}_sc_pistone']      = sc_pistone[s]
        result[f'{s}_regional_pistone']= reg_pistone[s]
        result[f'{s}_amjjas_pistone']  = _amj(sc_pistone[s])
    result.update({'lat': alb_lat, 'lon': alb_lon,
                   'lat_is_2d': alb_is2d, 'delta_t': delta_t})
    return result

# =============================================================================
# SPATIAL MODEL LIST HELPERS
# =============================================================================

def _has_any_data(entry, key):
    if entry is None:
        return False
    maps = entry.get(key, {})
    return any(arr is not None and np.isfinite(arr).any()
               for arr in maps.values())

def _models_with_data(native_store, key):
    names = ['OBS'] if _has_any_data(native_store.get('OBS'), key) else []
    for m in MODEL_INFO:
        if m not in SKIP_MODELS and _has_any_data(native_store.get(m), key):
            names.append(m)
    return names

# =============================================================================
# PLOTTING - SPATIAL
# =============================================================================

def plot_spatial_for_qty(qty, spec, native_store, model_names,
                          common_lat, common_lon, model_info_map):
    if not model_names:
        print(f'  {qty}: no models with data -- skip', flush=True); return

    proj  = ccrs.NorthPolarStereo(central_longitude=-35)
    dcrs  = ccrs.PlateCarree()
    norm  = mcolors.Normalize(vmin=spec['vmin'], vmax=spec['vmax'])
    ncols = min(5, len(model_names))
    nrows = int(np.ceil(len(model_names) / ncols))
    lat2d_c = np.tile(common_lat[:, None], (1, len(common_lon)))
    lon2d_c = np.tile(common_lon[None, :], (len(common_lat), 1))

    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols*5, nrows*4.5),
                                  subplot_kw={'projection': proj},
                                  facecolor='white')
        fig.suptitle(f'{spec["ylabel"]} -- {mname}', fontsize=12,
                     fontweight='bold', y=1.01)
        axes = np.array(axes).ravel()

        for idx, label in enumerate(model_names):
            ax    = axes[idx]
            entry = native_store.get(label)
            info  = model_info_map.get(label, {})
            dTbar = info.get('delta_t_bar', np.nan)
            dT_str = f'dTbar={dTbar:.3f}K' if np.isfinite(dTbar) else 'dTbar=N/A'
            rsm    = RS_METHOD_LABEL.get(info.get('rs_method', ''), '')
            rs_lbl = f' [{rsm}]' if rsm and not qty.startswith('IS') else ''

            ax.set_facecolor('#cce5f0')
            make_circle_boundary(ax)
            ax.set_extent([-180, 180, 60, 90], crs=dcrs)

            arr  = entry.get(spec['nk'], {}).get(mn) if entry else None
            slat = entry.get('lat') if entry else None
            slon = entry.get('lon') if entry else None
            s2d  = entry.get('lat_is_2d', False) if entry else False

            if arr is None or slat is None or not np.isfinite(arr).any():
                ax.set_title(f'{label}{rs_lbl}\nno data  {dT_str}',
                              fontsize=8, color='grey')
                ax.add_feature(cfeature.LAND, facecolor='#d0cfc4', zorder=3)
                ax.coastlines(linewidth=0.4, zorder=4)
                continue

            disp  = regrid_for_display(arr, slat, slon, s2d,
                                        common_lat, common_lon)
            plot_d = np.where(lat2d_c >= 58., disp, np.nan)
            im = ax.pcolormesh(lon2d_c, lat2d_c, plot_d, transform=dcrs,
                                cmap=spec['cmap'], norm=norm,
                                shading='nearest', zorder=2)
            ax.add_feature(cfeature.LAND, facecolor='#d0cfc4', zorder=3)
            ax.coastlines(linewidth=0.4, zorder=4)
            gl = ax.gridlines(crs=dcrs, draw_labels=False, linewidth=0.35,
                               color='#888888', alpha=0.5, linestyle='--',
                               zorder=2)
            gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
            gl.ylocator = mticker.FixedLocator([60, 70, 80])

            am = arctic_wmean_common(np.where(lat2d_c >= 60., disp, np.nan),
                                      common_lat)
            am_str = spec['fmt'].format(am) if np.isfinite(am) else 'nan'
            is_obs = (label == 'OBS')
            ax.set_title(
                f"{'* ' if is_obs else ''}{label}{rs_lbl}\n"
                f'mean={am_str}  {dT_str}',
                fontsize=8,
                color='#b5000a' if is_obs else 'black',
                fontweight='bold' if is_obs else 'normal',
                pad=3)
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                 pad=0.04, fraction=0.046, shrink=0.85)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label(spec['cbar_label'], fontsize=6)

        for j in range(len(model_names), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, f'{qty}_spatial_{mname}.png')
        fig.savefig(fpath, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'    saved {fpath}', flush=True)

# =============================================================================
# HBAR
# =============================================================================

def plot_hbar(qty, all_members_vals, obs_amjjas, obs_2sig,
               model_info_map, ylabel, filename):
    mm = {}; ms = {}; mn_c = {}
    for model, md in all_members_vals.items():
        vals = [v for v in md.values() if np.isfinite(v)]
        if vals:
            mm[model]   = np.mean(vals)
            ms[model]   = np.std(vals) if len(vals) > 1 else 0.
            mn_c[model] = len(vals)
    if obs_amjjas is not None and np.isfinite(obs_amjjas):
        mm['OBS']   = obs_amjjas
        ms['OBS']   = (obs_2sig / 2. if obs_2sig and np.isfinite(obs_2sig) else 0.)
        mn_c['OBS'] = 1
    if not mm:
        print(f'  {qty} hbar: no data', flush=True); return

    sorted_l = sorted(mm, key=lambda l: mm[l] if np.isfinite(mm[l]) else -999)
    fig, ax  = plt.subplots(figsize=(11, max(6, len(sorted_l)*0.55)),
                              facecolor='white')
    if obs_amjjas is not None and obs_2sig is not None and np.isfinite(obs_2sig):
        ax.axvspan(obs_amjjas - obs_2sig, obs_amjjas + obs_2sig,
                   color='#b5000a', alpha=0.12, zorder=0, label='OBS +/-2 sigma')
    for yi, label in enumerate(sorted_l):
        mean = mm.get(label, np.nan); std = ms.get(label, 0.)
        color = '#b5000a' if label == 'OBS' else MODEL_COLORS.get(label, '#4393c3')
        if np.isfinite(mean):
            ax.barh(yi, mean, color=color, alpha=0.85, height=0.7)
            if std > 0:
                ax.errorbar(mean, yi, xerr=std, fmt='none', color='black',
                            capsize=3, linewidth=1.2, zorder=5)
            ax.text(mean + abs(mean)*0.02 + 1e-6, yi,
                    f'{mean:+.4f}' + (f' +/-{std:.4f}' if std > 0 else ''),
                    va='center', ha='left', fontsize=7.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(np.arange(len(sorted_l)))

    ytl = []
    for l in sorted_l:
        info  = model_info_map.get(l, {})
        dTbar = info.get('delta_t_bar', np.nan)
        rsm   = RS_METHOD_LABEL.get(info.get('rs_method', ''), '')
        n     = mn_c.get(l, 0)
        if l == 'OBS':
            ytl.append(f'* OBS  (dT={OBS_DELTA_T:.3f}K)')
        elif np.isfinite(dTbar):
            rs_tag = f' [RS={rsm}]' if rsm and not qty.startswith('IS') else ''
            ytl.append(f'{l}{rs_tag}  (dTbar={dTbar:.3f}K, n={n})')
        else:
            ytl.append(l)
    ax.set_yticklabels(ytl, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(f'CMIP6 ensemble mean vs OBS -- AMJJAS {qty}\n'
                 f'bars=ens. mean +/- 1 sigma  |  per-member dT',
                 fontsize=10, fontweight='bold')
    if obs_amjjas is not None:
        ax.legend(fontsize=9, loc='lower right', framealpha=0.85)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, filename)
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {fpath}', flush=True)

# =============================================================================
# SCATTER
# =============================================================================

def plot_scatter(qty, all_members_vals, all_members_monthly,
                  obs_amjjas, obs_2sig, obs_monthly, obs_monthly_2sig,
                  model_info_map, ylabel, filename_base):
    models_ordered = [m for m in MODEL_INFO
                      if m in all_members_vals
                      and m not in SKIP_MODELS
                      and len(all_members_vals[m]) > 0]
    if not models_ordered:
        print(f'  {qty} scatter: no data', flush=True); return

    def _in_band(v):
        return (obs_amjjas is not None and obs_2sig is not None
                and np.isfinite(obs_2sig) and abs(v - obs_amjjas) <= obs_2sig)

    total  = sum(len(v) for v in all_members_vals.values())
    within = sum(1 for m in all_members_vals
                 for v in all_members_vals[m].values()
                 if np.isfinite(v) and _in_band(v))
    pct    = 100. * within / total if total > 0 else 0.

    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')
    if obs_amjjas is not None and obs_2sig is not None and np.isfinite(obs_2sig):
        ax.axhspan(obs_amjjas - obs_2sig, obs_amjjas + obs_2sig,
                   color='#b5000a', alpha=0.12, zorder=0)
        ax.axhline(obs_amjjas, color='#b5000a', lw=2.5, ls='--', zorder=5,
                   label=f'OBS={obs_amjjas:+.4f}+/-{obs_2sig:.4f}')

    x_cur = 0; xt = []; xl = []
    for model in models_ordered:
        color = MODEL_COLORS.get(model, '#333')
        vals  = [(m, v) for m, v in all_members_vals[model].items()
                 if np.isfinite(v)]
        if not vals:
            continue
        n    = len(vals)
        xpos = np.linspace(x_cur, x_cur + max(1.5, n*0.35), n)
        for xi, (_, v) in zip(xpos, vals):
            in_b = _in_band(v)
            ax.scatter(xi, v, color=color, s=80 if in_b else 35,
                       marker='*' if in_b else 'o',
                       edgecolors='black' if in_b else 'none',
                       linewidths=0.7, zorder=8 if in_b else 4,
                       alpha=1. if in_b else 0.65)
        ax.hlines(np.mean([v for _, v in vals]),
                  xpos[0] - .1, xpos[-1] + .1,
                  colors=color, lw=2.2, alpha=0.85, zorder=6)
        xt.append(np.mean(xpos))
        n_in  = sum(1 for _, v in vals if _in_band(v))
        info  = model_info_map.get(model, {})
        dTbar = info.get('delta_t_bar', np.nan)
        rsm   = RS_METHOD_LABEL.get(info.get('rs_method', ''), '')
        rs_tag = f'[{rsm}]' if rsm and not qty.startswith('IS') else ''
        short = (model.replace('ESM1-2-', 'E1-2-').replace('GC31-', 'G31-')
                      .replace('ESM1-5', 'E1-5').replace('CM6A-', 'CM6-'))
        xl.append(
            (f'{short}{rs_tag}\n(n={n},{n_in}*)\ndTbar={dTbar:.3f}'
             if np.isfinite(dTbar) else f'{short}{rs_tag}\n(n={n},{n_in}*)'))
        x_cur = xpos[-1] + 1.8

    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'CMIP6 {qty} AMJJAS  |  {within}/{total} ({pct:.0f}%) in '
                 f'OBS +/-2 sigma  |  per-member dT',
                 fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, ls='--')

    from matplotlib.lines import Line2D
    leg1 = ax.legend(
        handles=[Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=MODEL_COLORS.get(m, '#333'),
                         markersize=9, label=m)
                 for m in models_ordered],
        fontsize=7.5, loc='upper left', ncol=4, framealpha=0.9, title='Models')
    ax.add_artist(leg1)
    if obs_amjjas is not None:
        ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
        ax.add_artist(leg1)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, f'{filename_base}_scatter.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {fpath}', flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
    fig.suptitle(f'{qty} per month  |  per-member dT',
                 fontsize=12, fontweight='bold')
    for mi, (mn_num, ax) in enumerate(zip(MONTH_NUMS, axes.ravel())):
        mnm = MONTH_NAMES[mn_num]
        ov  = obs_monthly.get(mn_num) if obs_monthly else None
        o2s = obs_monthly_2sig.get(mn_num) if obs_monthly_2sig else None
        if ov is not None and o2s is not None and np.isfinite(o2s):
            ax.axhspan(ov - o2s, ov + o2s, color='#b5000a', alpha=0.15)
            ax.axhline(ov, color='#b5000a', lw=2., ls='--',
                       label=f'OBS={ov:+.4f}+/-{o2s:.4f}')
        x2, xt2, xl2 = 0, [], []
        tot = win = 0
        for model in models_ordered:
            color = MODEL_COLORS.get(model, '#333')
            vals  = [all_members_monthly.get((model, m), {}).get(mn_num, np.nan)
                     for m in sorted(all_members_vals[model])]
            vals  = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            n2  = len(vals)
            xp2 = np.linspace(x2, x2 + max(0.4, n2*0.18), n2)
            for xi, mv in zip(xp2, vals):
                in_b = (ov is not None and o2s is not None
                        and np.isfinite(o2s) and abs(mv - ov) <= o2s)
                ax.scatter(xi, mv, color=color, s=45 if in_b else 18,
                           marker='*' if in_b else 'o',
                           edgecolors='black' if in_b else 'none',
                           linewidths=0.5, zorder=8 if in_b else 3,
                           alpha=1. if in_b else 0.6)
                tot += 1
                if in_b: win += 1
            ax.hlines(np.mean(vals), xp2[0] - .08, xp2[-1] + .08,
                      colors=color, lw=1.5, alpha=0.8)
            xt2.append(np.mean(xp2))
            rsm = RS_METHOD_LABEL.get(
                model_info_map.get(model, {}).get('rs_method', ''), '')
            rs_tag = f'[{rsm}]' if rsm and not qty.startswith('IS') else ''
            xl2.append(model.replace('ESM1-2-', 'E1-2-')
                             .replace('GC31-', 'G31-')
                             .replace('ESM1-5', 'E1-5') + rs_tag)
            x2 = xp2[-1] + 0.6
        pct_m = 100. * win / tot if tot > 0 else 0.
        ax.set_title(f'{mnm}  ({win}/{tot}={pct_m:.0f}%)',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xt2); ax.set_xticklabels(xl2, fontsize=6,
                                                 rotation=35, ha='right')
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, ls='--')
        if mi == 0 and ov:
            ax.legend(fontsize=8, loc='upper left')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fpath = os.path.join(PLOT_DIR, f'{filename_base}_scatter_monthly.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {fpath}', flush=True)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    common_lat = np.arange(-90., 91.,  1.)
    common_lon = np.arange(  0., 360., 1.)
    lat2d_c    = np.tile(common_lat[:, None], (1, len(common_lon)))

    # ── OBS grid metadata ────────────────────────────────────────────────────
    print('Loading OBS grid metadata...', flush=True)
    sial_shape  = tuple(np.load(f'{CACHE_CMIP6}/sial_shape.npy'))
    arctic_sial = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy')
    idx_appx_to_sial, sial_lat2d, sial_lon2d = _build_appx_mapping(
        sial_shape, CACHE_CMIP6)

    # ── PISTONE: build PSN25 grid + regions + ocean mask ONCE ───────────────
    print('\n[PISTONE] Building PSN25 grid, ocean mask, 6 regional wedges ...',
          flush=True)
    try:
        PSN_CTX = _build_psn25_grid()
        print(f'[PISTONE] PSN25 shape={PSN_CTX["psn_shape"]}  '
              f'ocean cells (>=60N)={int(PSN_CTX["domain_mask"].sum()):,}',
              flush=True)
        for code in PISTONE_REGION_CODES:
            print(f'  region {code}: {int(PSN_CTX["region_masks"][code].sum()):>6d} cells',
                  flush=True)
    except Exception as e:
        print(f'[PISTONE] ERROR building PSN25 grid: {e}', flush=True)
        print('[PISTONE] Pistone-style scalars will be NaN.', flush=True)
        PSN_CTX = None

    # ── OBS IS ───────────────────────────────────────────────────────────────
    (obs_is_tot, obs_is_ext, obs_is_drk,
     obs_is_tot_nat, obs_is_ext_nat, obs_is_drk_nat) = compute_obs_IS(
        common_lat, common_lon,
        sial_shape, arctic_sial, sial_lat2d, sial_lon2d, idx_appx_to_sial)

    def _sial_scalar(maps):
        return {mn: (arctic_wmean_allArctic_2d(maps[mn], sial_lat2d)
                     if maps.get(mn) is not None else np.nan)
                for mn in MONTH_NUMS}

    obs_is_tot_mon = _sial_scalar(obs_is_tot_nat)
    obs_is_ext_mon = _sial_scalar(obs_is_ext_nat)
    obs_is_drk_mon = _sial_scalar(obs_is_drk_nat)

    obs_is_tot_amjjas = np.nanmean(list(obs_is_tot_mon.values()))
    obs_is_ext_amjjas = np.nanmean(list(obs_is_ext_mon.values()))
    obs_is_drk_amjjas = np.nanmean(list(obs_is_drk_mon.values()))

    def _2sig(v):
        return (2. * OBS_DELTA_T_STD / OBS_DELTA_T * abs(v)
                if np.isfinite(v) else np.nan)
    def _2sig_mon(d): return {mn: _2sig(v) for mn, v in d.items()}

    obs_is_tot_2sig     = _2sig(obs_is_tot_amjjas)
    obs_is_ext_2sig     = _2sig(obs_is_ext_amjjas)
    obs_is_drk_2sig     = _2sig(obs_is_drk_amjjas)
    obs_is_tot_2sig_mon = _2sig_mon(obs_is_tot_mon)
    obs_is_ext_2sig_mon = _2sig_mon(obs_is_ext_mon)
    obs_is_drk_2sig_mon = _2sig_mon(obs_is_drk_mon)

    print(f'\nOBS IS_total AMJJAS = {obs_is_tot_amjjas:+.4f} +/- {obs_is_tot_2sig:.4f}'
          f'  (ext={obs_is_ext_amjjas:+.4f}  drk={obs_is_drk_amjjas:+.4f}'
          f'  check={obs_is_ext_amjjas+obs_is_drk_amjjas:+.4f})',
          flush=True)

    # ── OBS RS ───────────────────────────────────────────────────────────────
    obs_rs_maps, obs_rs_sial_maps = compute_obs_RS(
        common_lat, common_lon, sial_lat2d, sial_lon2d, sial_shape)

    # ── OBS SIAF on native SIAL (C0) ─────────────────────────────────────────
    obs_siaf_tot,     obs_siaf_ext,     obs_siaf_drk     = {}, {}, {}
    obs_siaf_tot_mon, obs_siaf_ext_mon, obs_siaf_drk_mon = {}, {}, {}
    # PISTONE holders for OBS SIAF on native SIAL (for PSN25 aggregation)
    obs_siaf_tot_nat_sial = {}; obs_siaf_ext_nat_sial = {}; obs_siaf_drk_nat_sial = {}
    obs_rs_nat_sial       = {}
    obs_is_tot_nat_sial   = {}; obs_is_ext_nat_sial   = {}; obs_is_drk_nat_sial   = {}

    for mn in MONTH_NUMS:
        IS_t = obs_is_tot_nat.get(mn)
        IS_e = obs_is_ext_nat.get(mn)
        IS_d = obs_is_drk_nat.get(mn)
        RS_s = obs_rs_sial_maps.get(mn)

        if IS_t is None or RS_s is None:
            obs_siaf_tot[mn]     = obs_siaf_ext[mn]     = obs_siaf_drk[mn]     = None
            obs_siaf_tot_mon[mn] = obs_siaf_ext_mon[mn] = obs_siaf_drk_mon[mn] = np.nan
            obs_siaf_tot_nat_sial[mn] = None; obs_siaf_ext_nat_sial[mn] = None
            obs_siaf_drk_nat_sial[mn] = None
            obs_rs_nat_sial[mn] = RS_s
            obs_is_tot_nat_sial[mn] = IS_t
            obs_is_ext_nat_sial[mn] = IS_e
            obs_is_drk_nat_sial[mn] = IS_d
            continue

        mask  = np.isfinite(IS_t) & np.isfinite(RS_s) & (sial_lat2d >= 60.)
        SF_t  = np.where(mask, RS_s * IS_t, np.nan)
        SF_e  = np.where(mask, RS_s * IS_e, np.nan)
        SF_d  = np.where(mask, RS_s * IS_d, np.nan)

        SF_e = np.where(np.isfinite(SF_t),
                        np.where(np.isfinite(SF_e), SF_e, 0.), np.nan)
        SF_d = np.where(np.isfinite(SF_t),
                        np.where(np.isfinite(SF_d), SF_d, 0.), np.nan)

        obs_siaf_tot_mon[mn] = arctic_wmean_allArctic_2d(SF_t, sial_lat2d)
        obs_siaf_ext_mon[mn] = arctic_wmean_allArctic_2d(SF_e, sial_lat2d)
        obs_siaf_drk_mon[mn] = arctic_wmean_allArctic_2d(SF_d, sial_lat2d)

        obs_siaf_tot[mn] = regrid_scatter_fn(SF_t, sial_lat2d, sial_lon2d,
                                              common_lat, common_lon)
        obs_siaf_ext[mn] = regrid_scatter_fn(SF_e, sial_lat2d, sial_lon2d,
                                              common_lat, common_lon)
        obs_siaf_drk[mn] = regrid_scatter_fn(SF_d, sial_lat2d, sial_lon2d,
                                              common_lat, common_lon)

        # PISTONE: retain native SIAL arrays for PSN25 aggregation
        obs_siaf_tot_nat_sial[mn] = SF_t
        obs_siaf_ext_nat_sial[mn] = SF_e
        obs_siaf_drk_nat_sial[mn] = SF_d
        obs_rs_nat_sial[mn]       = RS_s
        obs_is_tot_nat_sial[mn]   = IS_t
        obs_is_ext_nat_sial[mn]   = IS_e
        obs_is_drk_nat_sial[mn]   = IS_d

    obs_siaf_tot_global = global_siaf_from_monthly(obs_siaf_tot_mon)
    obs_siaf_ext_global = global_siaf_from_monthly(obs_siaf_ext_mon)
    obs_siaf_drk_global = global_siaf_from_monthly(obs_siaf_drk_mon)

    obs_siaf_tot_2sig_global = _2sig(obs_siaf_tot_global)
    obs_siaf_ext_2sig_global = _2sig(obs_siaf_ext_global)
    obs_siaf_drk_2sig_global = _2sig(obs_siaf_drk_global)

    obs_siaf_tot_mon_global = scale_obs_monthly_to_global(obs_siaf_tot_mon)
    obs_siaf_ext_mon_global = scale_obs_monthly_to_global(obs_siaf_ext_mon)
    obs_siaf_drk_mon_global = scale_obs_monthly_to_global(obs_siaf_drk_mon)
    obs_siaf_tot_2sig_mon_global = _2sig_mon(obs_siaf_tot_mon_global)
    obs_siaf_ext_2sig_mon_global = _2sig_mon(obs_siaf_ext_mon_global)
    obs_siaf_drk_2sig_mon_global = _2sig_mon(obs_siaf_drk_mon_global)

    print(f'OBS SIAF_total AMJJAS (Arctic mean) = {np.nanmean(list(obs_siaf_tot_mon.values())):+.4f}'
          f'  ext={np.nanmean(list(obs_siaf_ext_mon.values())):+.4f}'
          f'  drk={np.nanmean(list(obs_siaf_drk_mon.values())):+.4f}',
          flush=True)
    print(f'OBS SIAF_total global annual        = {obs_siaf_tot_global:+.4f} W m^-2 K^-1'
          f'  +/- {obs_siaf_tot_2sig_global:.4f}'
          f'  (ext={obs_siaf_ext_global:+.4f}  drk={obs_siaf_drk_global:+.4f}'
          f'  check={obs_siaf_ext_global+obs_siaf_drk_global:+.4f})',
          flush=True)

    # ── PISTONE: compute OBS Pistone-style scalars from native SIAL maps ────
    #   For each month, regrid the native SIAL field to PSN25 and compute
    #   per-region + Arctic-ocean means.  Treat NaN as 0 for SIAF and IS
    #   variants so the Arctic-ocean mean dilutes to AREA_RATIO_60N_OCEAN.
    obs_pistone = {
        'is_tot'  : dict(mon={}, mon_reg={}),
        'is_ext'  : dict(mon={}, mon_reg={}),
        'is_drk'  : dict(mon={}, mon_reg={}),
        'rs'      : dict(mon={}, mon_reg={}),
        'siaf_tot': dict(mon={}, mon_reg={}),
        'siaf_ext': dict(mon={}, mon_reg={}),
        'siaf_drk': dict(mon={}, mon_reg={}),
    }

    def _obs_pistone_scalar(arr_sial, treat_nan_as_zero):
        if PSN_CTX is None or arr_sial is None:
            return np.nan, {c: np.nan for c in PISTONE_REGION_CODES}
        arr_psn = regrid_to_psn(arr_sial, sial_lat2d, sial_lon2d,
                                True, PSN_CTX)
        out = pistone_arctic_mean(arr_psn, PSN_CTX,
                                  treat_nan_as_zero=treat_nan_as_zero)
        return out['Arctic_ocean'], {c: out[c] for c in PISTONE_REGION_CODES}

    if PSN_CTX is not None:
        for mn in MONTH_NUMS:
            for stem, src, fill0 in [
                ('is_tot',   obs_is_tot_nat_sial.get(mn),   True),
                ('is_ext',   obs_is_ext_nat_sial.get(mn),   True),
                ('is_drk',   obs_is_drk_nat_sial.get(mn),   True),
                ('rs',       obs_rs_nat_sial.get(mn),       False),
                ('siaf_tot', obs_siaf_tot_nat_sial.get(mn), True),
                ('siaf_ext', obs_siaf_ext_nat_sial.get(mn), True),
                ('siaf_drk', obs_siaf_drk_nat_sial.get(mn), True),
            ]:
                s, r = _obs_pistone_scalar(src, fill0)
                obs_pistone[stem]['mon'][mn]     = s
                obs_pistone[stem]['mon_reg'][mn] = r

    # OBS AMJJAS Pistone scalars
    obs_pistone_amjjas = {s: float(np.nanmean(list(obs_pistone[s]['mon'].values())))
                          for s in obs_pistone}

    # OBS global-annual Pistone conversions (ocean area ratio, 0.0347)
    obs_siaf_tot_global_pistone = pistone_global_annual(obs_pistone['siaf_tot']['mon'])
    obs_siaf_ext_global_pistone = pistone_global_annual(obs_pistone['siaf_ext']['mon'])
    obs_siaf_drk_global_pistone = pistone_global_annual(obs_pistone['siaf_drk']['mon'])

    print(f'\n[PISTONE] OBS SIAF_total Arctic-OCEAN AMJJAS = '
          f'{obs_pistone_amjjas["siaf_tot"]:+.4f}'
          f'  ext={obs_pistone_amjjas["siaf_ext"]:+.4f}'
          f'  drk={obs_pistone_amjjas["siaf_drk"]:+.4f}', flush=True)
    print(f'[PISTONE] OBS SIAF_total global annual        = '
          f'{obs_siaf_tot_global_pistone:+.4f} W m^-2 K^-1'
          f'  (ext={obs_siaf_ext_global_pistone:+.4f}  '
          f'drk={obs_siaf_drk_global_pistone:+.4f})', flush=True)
    print(f'[PISTONE] Compare to original (0.065, all-Arctic): '
          f'total={obs_siaf_tot_global:+.4f}  '
          f'difference={obs_siaf_tot_global_pistone - obs_siaf_tot_global:+.4f}',
          flush=True)

    # ── Native store (OBS) ───────────────────────────────────────────────────
    native_store = {}
    native_store['OBS'] = {
        'is_tot_nat':    obs_is_tot,
        'is_ext_nat':    obs_is_ext,
        'is_drk_nat':    obs_is_drk,
        'rs_nat':        obs_rs_maps,
        'siaf_tot_nat':  obs_siaf_tot,
        'siaf_ext_nat':  obs_siaf_ext,
        'siaf_drk_nat':  obs_siaf_drk,
        'lat': common_lat, 'lon': common_lon, 'lat_is_2d': False,
    }
    model_info = {'OBS': {'delta_t_bar': OBS_DELTA_T, 'rs_method': 'T07',
                          'n_members': 1}}

    # ── Storage for hbar / scatter ────────────────────────────────────────────
    all_vals         = {s: {m: {} for m in MODEL_INFO} for s in METRIC_STEMS}
    all_vals_monthly = {s: {} for s in METRIC_STEMS}
    # PISTONE: parallel storage for Pistone-style scalars
    all_vals_pistone         = {s: {m: {} for m in MODEL_INFO} for s in METRIC_STEMS}
    all_vals_monthly_pistone = {s: {} for s in METRIC_STEMS}
    all_regional_pistone     = {s: {} for s in METRIC_STEMS}   # (model,member)->{mn->{code->val}}
    member_dT                = {}

    # ── Model loop ────────────────────────────────────────────────────────────
    for model in MODEL_INFO:
        if model in SKIP_MODELS:
            print(f'\n[{model}] skipped', flush=True); continue

        inst = MODEL_INFO[model]['institution']
        prim = MODEL_INFO[model]['member']
        print(f'\n{"="*60}', flush=True)
        print(f'Model: {model}', flush=True)

        members = find_all_members(inst, model)
        if not members:
            print('  No members found -- skip', flush=True); continue
        if prim in members:
            members = [prim] + [m for m in members if m != prim]
        print(f'  {len(members)} members: '
              f'{members[:5]}{"..." if len(members)>5 else ""}', flush=True)

        rs_meth_p, rf_p = determine_rs_method(inst, model, prim)
        print(f'  RS method: {rs_meth_p}', flush=True)
        rs_p, rsl_p, rsw_p, rsi2d_p, actual_m = compute_rs_maps(
            inst, model, prim, rs_meth_p, rf_p)
        n_rs = sum(1 for v in rs_p.values() if v is not None)
        print(f'  RS: {n_rs}/{len(MONTH_NUMS)} months ({actual_m})', flush=True)

        dT_prim_raw = compute_delta_T(model, prim)
        prim_dT     = (dT_prim_raw)
        print(f'  Primary dT = {prim_dT:+.3f}K', flush=True)

        collected_dTs = []

        for member in members:
            is_prim = (member == prim)

            if is_prim:
                dT_m = prim_dT
            else:
                dT_raw = compute_delta_T(model, member)
                if dT_raw is not None:
                    dT_m = dT_raw
                elif collected_dTs:
                    dT_m = float(np.mean(collected_dTs))
                else:
                    dT_m = prim_dT
                print(f'    [{member}] dT = {dT_m:+.3f}K', flush=True)

            member_dT[(model, member)] = dT_m
            collected_dTs.append(dT_m)

            if is_prim:
                rm, rl, rw, ri2 = rs_p, rsl_p, rsw_p, rsi2d_p
            else:
                rs_meth_m, rf_m = determine_rs_method(inst, model, member)
                if rs_meth_m != RS_METHOD_NONE:
                    rm_full = compute_rs_maps(inst, model, member, rs_meth_m, rf_m)
                    rm, rl, rw, ri2 = rm_full[:4]
                    if sum(1 for v in rm.values() if v is not None) < len(MONTH_NUMS)//2:
                        rm, rl, rw, ri2 = rs_p, rsl_p, rsw_p, rsi2d_p
                else:
                    rm, rl, rw, ri2 = rs_p, rsl_p, rsw_p, rsi2d_p

            print(f'  Processing {model}/{member} (dT={dT_m:+.3f}K)...',
                  flush=True)
            try:
                res = compute_member(model, member, dT_m, rm, rl, rw, ri2,
                                     psn_ctx=PSN_CTX)
                if res is None:
                    print('    no result', flush=True); continue

                for s in METRIC_STEMS:
                    all_vals[s][model][member]           = res[f'{s}_amjjas']
                    all_vals_monthly[s][(model, member)] = res[f'{s}_sc']
                    # PISTONE: parallel Arctic-ocean scalars + per-region dicts
                    all_vals_pistone[s][model][member]   = res[f'{s}_amjjas_pistone']
                    all_vals_monthly_pistone[s][(model, member)] = res[f'{s}_sc_pistone']
                    all_regional_pistone[s][(model, member)]     = res[f'{s}_regional_pistone']

                if is_prim:
                    native_store[model] = {
                        f'{s}_nat': res[f'{s}_nat'] for s in METRIC_STEMS
                    }
                    native_store[model].update({
                        'lat': res['lat'], 'lon': res['lon'],
                        'lat_is_2d': res['lat_is_2d'],
                    })

                is_e = res['is_ext_amjjas']; is_d = res['is_drk_amjjas']
                sf_e = res['siaf_ext_amjjas']; sf_d = res['siaf_drk_amjjas']
                sf_p = res['siaf_tot_amjjas_pistone']
                print(f'    IS_tot={res["is_tot_amjjas"]:+.4f}  '
                      f'IS_ext={is_e:+.4f}  IS_drk={is_d:+.4f}  '
                      f'IS_check={is_e+is_d:+.4f}  '
                      f'SIAF_tot={res["siaf_tot_amjjas"]:+.3f}  '
                      f'SIAF_ext={sf_e:+.3f}  SIAF_drk={sf_d:+.3f}  '
                      f'SIAF_check={sf_e+sf_d:+.3f}  '
                      f'RS={res["rs_amjjas"]:.1f}  '
                      f'[PISTONE] SIAF_tot_ocn={sf_p:+.3f}', flush=True)

            except Exception as e:
                print(f'    FAILED: {e}', flush=True)
            finally:
                gc.collect()

        vdTs  = [member_dT[(model, m)] for m in all_vals['is_tot'][model]
                 if np.isfinite(all_vals['is_tot'][model][m])
                 and (model, m) in member_dT]
        dTbar = float(np.mean(vdTs)) if vdTs else np.nan
        n_is  = sum(1 for v in all_vals['is_tot'][model].values() if np.isfinite(v))
        n_siaf = sum(1 for v in all_vals['siaf_tot'][model].values() if np.isfinite(v))
        model_info[model] = {
            'delta_t_bar': dTbar,
            'rs_method':   actual_m,
            'n_members':   n_is,
        }
        print(f'  [{model}] IS:{n_is}  SIAF:{n_siaf}  '
              f'dTbar={dTbar:.3f}K  RS={actual_m}', flush=True)

    # ── Convert model SIAF to global annual (original 0.065) ─────────────────
    global_siaf         = {s: {} for s in ['siaf_tot', 'siaf_ext', 'siaf_drk']}
    global_siaf_monthly = {s: {} for s in ['siaf_tot', 'siaf_ext', 'siaf_drk']}

    for stem in ['siaf_tot', 'siaf_ext', 'siaf_drk']:
        for model, members in all_vals[stem].items():
            global_siaf[stem][model] = {}
            for mem in members:
                mon_dict = all_vals_monthly[stem].get((model, mem), {})
                global_siaf[stem][model][mem] = global_siaf_from_monthly(mon_dict)
        for mod_mem, monthly_dict in all_vals_monthly[stem].items():
            global_siaf_monthly[stem][mod_mem] = {
                mn: (v * (1.0 / 12.0) * AREA_RATIO_60N
                     if np.isfinite(v) else np.nan)
                for mn, v in monthly_dict.items()
            }

    # ── PISTONE: Convert model SIAF to global annual (ocean ratio 0.0347) ───
    global_siaf_pistone         = {s: {} for s in ['siaf_tot', 'siaf_ext', 'siaf_drk']}
    global_siaf_monthly_pistone = {s: {} for s in ['siaf_tot', 'siaf_ext', 'siaf_drk']}
    for stem in ['siaf_tot', 'siaf_ext', 'siaf_drk']:
        for model, members in all_vals_pistone[stem].items():
            global_siaf_pistone[stem][model] = {}
            for mem in members:
                mon_dict = all_vals_monthly_pistone[stem].get((model, mem), {})
                global_siaf_pistone[stem][model][mem] = pistone_global_annual(mon_dict)
        for mod_mem, monthly_dict in all_vals_monthly_pistone[stem].items():
            global_siaf_monthly_pistone[stem][mod_mem] = {
                mn: (v * (1.0 / 12.0) * AREA_RATIO_60N_OCEAN
                     if np.isfinite(v) else np.nan)
                for mn, v in monthly_dict.items()
            }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"Model summary":─^75}', flush=True)
    print(f'  {"Model":<22} {"n_IS":>5} {"n_SIAF":>6}  {"RS":<6}  '
          f'{"SIAF_tot Arctic":>15}  {"SIAF_tot Global":>15}', flush=True)
    for model in MODEL_INFO:
        if model in SKIP_MODELS: continue
        info = model_info.get(model, {})
        rsm  = RS_METHOD_LABEL.get(info.get('rs_method', ''), '-')
        n_is   = sum(1 for v in all_vals['is_tot'][model].values()  if np.isfinite(v))
        n_siaf = sum(1 for v in all_vals['siaf_tot'][model].values() if np.isfinite(v))

        arc_vals  = [v for v in all_vals['siaf_tot'][model].values() if np.isfinite(v)]
        glob_vals = [v for v in global_siaf['siaf_tot'][model].values() if np.isfinite(v)]
        ens_arc   = float(np.mean(arc_vals))  if arc_vals  else float('nan')
        ens_glob  = float(np.mean(glob_vals)) if glob_vals else float('nan')

        print(f'  {model:<22} {n_is:>5} {n_siaf:>6}  {rsm:<6}  '
              f'{ens_arc:>+15.3f}  {ens_glob:>+15.4f}', flush=True)

    # ── Global SIAF summary (original) ───────────────────────────────────────
    print(f'\n{"Global SIAF summary [original: all-Arctic x 0.0650]":─^64}',
          flush=True)
    print(f'  area_ratio (>=60N cap) = {AREA_RATIO_60N:.4f}', flush=True)
    print(f'  OBS  total={obs_siaf_tot_global:+.4f}'
          f'  ext={obs_siaf_ext_global:+.4f}'
          f'  drk={obs_siaf_drk_global:+.4f}'
          f'  [W m^-2 K^-1]', flush=True)

    def _ens_mmm_global(stem, store=global_siaf):
        per_model = []
        for m in MODEL_INFO:
            if m in SKIP_MODELS: continue
            gv = [v for v in store[stem][m].values() if np.isfinite(v)]
            if gv:
                per_model.append(float(np.mean(gv)))
        return per_model

    for stem, label in [('siaf_tot', 'total'), ('siaf_ext', 'ext  '),
                         ('siaf_drk', 'drk  ')]:
        ev = _ens_mmm_global(stem)
        if ev:
            print(f'  CMIP6 MMM  {label}={np.mean(ev):+.4f}'
                  f'  (+/-{np.std(ev):.4f} 1 sigma across {len(ev)} models)',
                  flush=True)

    # ── PISTONE: Global SIAF summary (ocean ratio, 6-wedge PSN25) ───────────
    print(f'\n{"[PISTONE] Global SIAF summary [ocean-only x 0.0347]":─^64}',
          flush=True)
    print(f'  area_ratio (>=60N ocean) = {AREA_RATIO_60N_OCEAN:.4f}', flush=True)
    print(f'  OBS  total={obs_siaf_tot_global_pistone:+.4f}'
          f'  ext={obs_siaf_ext_global_pistone:+.4f}'
          f'  drk={obs_siaf_drk_global_pistone:+.4f}'
          f'  [W m^-2 K^-1]', flush=True)
    for stem, label in [('siaf_tot', 'total'), ('siaf_ext', 'ext  '),
                         ('siaf_drk', 'drk  ')]:
        ev = _ens_mmm_global(stem, store=global_siaf_pistone)
        if ev:
            print(f'  CMIP6 MMM  {label}={np.mean(ev):+.4f}'
                  f'  (+/-{np.std(ev):.4f} 1 sigma across {len(ev)} models)',
                  flush=True)

    # ── Per-quantity model lists for spatial maps ─────────────────────────────
    qty_to_nk     = {qty: spec['nk'] for qty, spec in PLOT_SPECS.items()}
    spatial_names = {qty: _models_with_data(native_store, nk)
                     for qty, nk in qty_to_nk.items()}
    for qty, names in spatial_names.items():
        print(f'  {qty:<22} ({len(names)} models): {names}')

    # ── Spatial maps ──────────────────────────────────────────────────────────
    print('\nPlotting spatial maps...', flush=True)
    for qty, spec in PLOT_SPECS.items():
        names = spatial_names[qty]
        print(f'\n  {qty} ({len(names)} models):', flush=True)
        plot_spatial_for_qty(qty, spec, native_store, names,
                              common_lat, common_lon, model_info)

    # ── Hbar plots (original) ────────────────────────────────────────────────
    print('\nPlotting hbars...', flush=True)
    hbar_cfg = [
        ('IS_total',       all_vals['is_tot'],       obs_is_tot_amjjas,    obs_is_tot_2sig,
         'IS_total_hbar.png',       PLOT_SPECS['IS_total']['ylabel']),
        ('IS_extent',      all_vals['is_ext'],       obs_is_ext_amjjas,    obs_is_ext_2sig,
         'IS_extent_hbar.png',      PLOT_SPECS['IS_extent']['ylabel']),
        ('IS_darkening',   all_vals['is_drk'],       obs_is_drk_amjjas,    obs_is_drk_2sig,
         'IS_darkening_hbar.png',   PLOT_SPECS['IS_darkening']['ylabel']),
        ('RS',             all_vals['rs'],           None,                 None,
         'RS_hbar.png',             PLOT_SPECS['RS']['ylabel']),
        ('SIAF_total',     global_siaf['siaf_tot'],  obs_siaf_tot_global,  obs_siaf_tot_2sig_global,
         'SIAF_total_hbar.png',     'Global annual SIAF$_{\\rm total}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_extent',    global_siaf['siaf_ext'],  obs_siaf_ext_global,  obs_siaf_ext_2sig_global,
         'SIAF_extent_hbar.png',    'Global annual SIAF$_{\\rm extent}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_darkening', global_siaf['siaf_drk'],  obs_siaf_drk_global,  obs_siaf_drk_2sig_global,
         'SIAF_darkening_hbar.png', 'Global annual SIAF$_{\\rm dark}$  (W m$^{-2}$ K$^{-1}$)'),
    ]
    for qty, data_dict, obs_am, obs_2s, fname, ylabel in hbar_cfg:
        plot_hbar(qty, data_dict, obs_am, obs_2s, model_info, ylabel, fname)

    # ── PISTONE: Hbar plots for Pistone-style scalars ───────────────────────
    print('\n[PISTONE] Plotting hbars (Pistone-style)...', flush=True)
    obs_is_tot_2sig_pistone = _2sig(obs_pistone_amjjas['is_tot'])
    obs_is_ext_2sig_pistone = _2sig(obs_pistone_amjjas['is_ext'])
    obs_is_drk_2sig_pistone = _2sig(obs_pistone_amjjas['is_drk'])
    obs_sft_2sig_pistone    = _2sig(obs_siaf_tot_global_pistone)
    obs_sfe_2sig_pistone    = _2sig(obs_siaf_ext_global_pistone)
    obs_sfd_2sig_pistone    = _2sig(obs_siaf_drk_global_pistone)

    hbar_cfg_pistone = [
        ('IS_total_pistone',       all_vals_pistone['is_tot'],
         obs_pistone_amjjas['is_tot'], obs_is_tot_2sig_pistone,
         'IS_total_hbar_pistone.png',
         '[PISTONE] AMJJAS Arctic-ocean IS$_{\\rm total}$  (albedo K$^{-1}$)'),
        ('IS_extent_pistone',      all_vals_pistone['is_ext'],
         obs_pistone_amjjas['is_ext'], obs_is_ext_2sig_pistone,
         'IS_extent_hbar_pistone.png',
         '[PISTONE] AMJJAS Arctic-ocean IS$_{\\rm extent}$  (albedo K$^{-1}$)'),
        ('IS_darkening_pistone',   all_vals_pistone['is_drk'],
         obs_pistone_amjjas['is_drk'], obs_is_drk_2sig_pistone,
         'IS_darkening_hbar_pistone.png',
         '[PISTONE] AMJJAS Arctic-ocean IS$_{\\rm dark}$  (albedo K$^{-1}$)'),
        ('SIAF_total_pistone',     global_siaf_pistone['siaf_tot'],
         obs_siaf_tot_global_pistone, obs_sft_2sig_pistone,
         'SIAF_total_hbar_pistone.png',
         '[PISTONE] Global annual SIAF$_{\\rm total}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_extent_pistone',    global_siaf_pistone['siaf_ext'],
         obs_siaf_ext_global_pistone, obs_sfe_2sig_pistone,
         'SIAF_extent_hbar_pistone.png',
         '[PISTONE] Global annual SIAF$_{\\rm extent}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_darkening_pistone', global_siaf_pistone['siaf_drk'],
         obs_siaf_drk_global_pistone, obs_sfd_2sig_pistone,
         'SIAF_darkening_hbar_pistone.png',
         '[PISTONE] Global annual SIAF$_{\\rm dark}$  (W m$^{-2}$ K$^{-1}$)'),
    ]
    for qty, data_dict, obs_am, obs_2s, fname, ylabel in hbar_cfg_pistone:
        plot_hbar(qty, data_dict, obs_am, obs_2s, model_info, ylabel, fname)

    # ── Scatter plots (original) ─────────────────────────────────────────────
    print('\nPlotting scatters...', flush=True)
    scatter_cfg = [
        ('IS_total',       all_vals['is_tot'],       all_vals_monthly['is_tot'],
         obs_is_tot_amjjas,    obs_is_tot_2sig,    obs_is_tot_mon,           obs_is_tot_2sig_mon,
         PLOT_SPECS['IS_total']['ylabel']),
        ('IS_extent',      all_vals['is_ext'],       all_vals_monthly['is_ext'],
         obs_is_ext_amjjas,    obs_is_ext_2sig,    obs_is_ext_mon,           obs_is_ext_2sig_mon,
         PLOT_SPECS['IS_extent']['ylabel']),
        ('IS_darkening',   all_vals['is_drk'],       all_vals_monthly['is_drk'],
         obs_is_drk_amjjas,    obs_is_drk_2sig,    obs_is_drk_mon,           obs_is_drk_2sig_mon,
         PLOT_SPECS['IS_darkening']['ylabel']),
        ('SIAF_total',     global_siaf['siaf_tot'],  global_siaf_monthly['siaf_tot'],
         obs_siaf_tot_global,  obs_siaf_tot_2sig_global, obs_siaf_tot_mon_global, obs_siaf_tot_2sig_mon_global,
         'Global annual SIAF$_{\\rm total}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_extent',    global_siaf['siaf_ext'],  global_siaf_monthly['siaf_ext'],
         obs_siaf_ext_global,  obs_siaf_ext_2sig_global, obs_siaf_ext_mon_global, obs_siaf_ext_2sig_mon_global,
         'Global annual SIAF$_{\\rm extent}$  (W m$^{-2}$ K$^{-1}$)'),
        ('SIAF_darkening', global_siaf['siaf_drk'],  global_siaf_monthly['siaf_drk'],
         obs_siaf_drk_global,  obs_siaf_drk_2sig_global, obs_siaf_drk_mon_global, obs_siaf_drk_2sig_mon_global,
         'Global annual SIAF$_{\\rm dark}$  (W m$^{-2}$ K$^{-1}$)'),
    ]
    for qty, data_dict, data_mon, obs_am, obs_2s, obs_mon, obs_2s_mon, ylabel in scatter_cfg:
        plot_scatter(qty, data_dict, data_mon, obs_am, obs_2s,
                     obs_mon, obs_2s_mon, model_info, ylabel, qty)

    # ── Export detailed ensemble results (ORIGINAL) ──────────────────────────
    print('\nExporting detailed ensemble results to CSV...', flush=True)
    detailed_rows = []

    for (model, member) in all_vals_monthly['is_tot'].keys():
        dT = member_dT.get((model, member), np.nan)
        for mn in MONTH_NUMS:
            detailed_rows.append({
                'Model':  model,
                'Member': member,
                'Month':  mn,
                'dT':     dT,
                'IS_tot': all_vals_monthly['is_tot'][(model, member)].get(mn),
                'IS_ext': all_vals_monthly['is_ext'][(model, member)].get(mn),
                'IS_drk': all_vals_monthly['is_drk'][(model, member)].get(mn),
                'SIAF_tot_global': global_siaf_monthly['siaf_tot'].get((model, member), {}).get(mn),
                'SIAF_ext_global': global_siaf_monthly['siaf_ext'].get((model, member), {}).get(mn),
                'SIAF_drk_global': global_siaf_monthly['siaf_drk'].get((model, member), {}).get(mn),
            })

    for mn in MONTH_NUMS:
        detailed_rows.append({
            'Model':  'OBS',
            'Member': 'Satellite',
            'Month':  mn,
            'dT':     OBS_DELTA_T,
            'IS_tot': obs_is_tot_mon.get(mn),
            'IS_ext': obs_is_ext_mon.get(mn),
            'IS_drk': obs_is_drk_mon.get(mn),
            'SIAF_tot_global': obs_siaf_tot_mon_global.get(mn) if obs_siaf_tot_mon_global else np.nan,
            'SIAF_ext_global': obs_siaf_ext_mon_global.get(mn) if obs_siaf_ext_mon_global else np.nan,
            'SIAF_drk_global': obs_siaf_drk_mon_global.get(mn) if obs_siaf_drk_mon_global else np.nan,
        })

    csv_out = os.path.join(PLOT_DIR, 'siaf_detailed_ensemble_results.csv')
    pd.DataFrame(detailed_rows).to_csv(csv_out, index=False)
    print(f'  Detailed scalars saved to: {csv_out}', flush=True)

    # ── PISTONE: Export per-region + Arctic-ocean scalars ────────────────────
    print('\n[PISTONE] Exporting regional ensemble results to CSV...', flush=True)
    pistone_rows = []

    def _pmon(mm, stem, model, member, mn):
        return mm.get((model, member), {}).get(mn)
    def _preg(reg_store, stem, model, member, mn, code):
        return reg_store.get((model, member), {}).get(mn, {}).get(code)

    for (model, member) in all_vals_monthly_pistone['is_tot'].keys():
        dT = member_dT.get((model, member), np.nan)
        for mn in MONTH_NUMS:
            row = {
                'Model':  model,
                'Member': member,
                'Month':  mn,
                'dT':     dT,
                # Arctic-ocean (PSN25, 6-wedge aggregate)
                'IS_tot_arctic_ocean':   _pmon(all_vals_monthly_pistone['is_tot'],   'is_tot',   model, member, mn),
                'IS_ext_arctic_ocean':   _pmon(all_vals_monthly_pistone['is_ext'],   'is_ext',   model, member, mn),
                'IS_drk_arctic_ocean':   _pmon(all_vals_monthly_pistone['is_drk'],   'is_drk',   model, member, mn),
                'RS_arctic_ocean':       _pmon(all_vals_monthly_pistone['rs'],       'rs',       model, member, mn),
                'SIAF_tot_arctic_ocean': _pmon(all_vals_monthly_pistone['siaf_tot'], 'siaf_tot', model, member, mn),
                'SIAF_ext_arctic_ocean': _pmon(all_vals_monthly_pistone['siaf_ext'], 'siaf_ext', model, member, mn),
                'SIAF_drk_arctic_ocean': _pmon(all_vals_monthly_pistone['siaf_drk'], 'siaf_drk', model, member, mn),
                # Global-annual conversion (area ratio 0.0347)
                'SIAF_tot_global_pistone': global_siaf_monthly_pistone['siaf_tot'].get((model, member), {}).get(mn),
                'SIAF_ext_global_pistone': global_siaf_monthly_pistone['siaf_ext'].get((model, member), {}).get(mn),
                'SIAF_drk_global_pistone': global_siaf_monthly_pistone['siaf_drk'].get((model, member), {}).get(mn),
            }
            # Per-region SIAF_tot and IS_tot (6 wedges each)
            for code in PISTONE_REGION_CODES:
                row[f'IS_tot_{code}']   = _preg(all_regional_pistone['is_tot'],   'is_tot',   model, member, mn, code)
                row[f'IS_ext_{code}']   = _preg(all_regional_pistone['is_ext'],   'is_ext',   model, member, mn, code)
                row[f'IS_drk_{code}']   = _preg(all_regional_pistone['is_drk'],   'is_drk',   model, member, mn, code)
                row[f'RS_{code}']       = _preg(all_regional_pistone['rs'],       'rs',       model, member, mn, code)
                row[f'SIAF_tot_{code}'] = _preg(all_regional_pistone['siaf_tot'], 'siaf_tot', model, member, mn, code)
                row[f'SIAF_ext_{code}'] = _preg(all_regional_pistone['siaf_ext'], 'siaf_ext', model, member, mn, code)
                row[f'SIAF_drk_{code}'] = _preg(all_regional_pistone['siaf_drk'], 'siaf_drk', model, member, mn, code)
            pistone_rows.append(row)

    # OBS rows (Pistone-style)
    for mn in MONTH_NUMS:
        row = {
            'Model':  'OBS',
            'Member': 'Satellite',
            'Month':  mn,
            'dT':     OBS_DELTA_T,
            'IS_tot_arctic_ocean':   obs_pistone['is_tot']['mon'].get(mn),
            'IS_ext_arctic_ocean':   obs_pistone['is_ext']['mon'].get(mn),
            'IS_drk_arctic_ocean':   obs_pistone['is_drk']['mon'].get(mn),
            'RS_arctic_ocean':       obs_pistone['rs']['mon'].get(mn),
            'SIAF_tot_arctic_ocean': obs_pistone['siaf_tot']['mon'].get(mn),
            'SIAF_ext_arctic_ocean': obs_pistone['siaf_ext']['mon'].get(mn),
            'SIAF_drk_arctic_ocean': obs_pistone['siaf_drk']['mon'].get(mn),
            'SIAF_tot_global_pistone': (obs_pistone['siaf_tot']['mon'].get(mn, np.nan)
                                        * (1.0 / 12.0) * AREA_RATIO_60N_OCEAN
                                        if np.isfinite(obs_pistone['siaf_tot']['mon'].get(mn, np.nan)) else np.nan),
            'SIAF_ext_global_pistone': (obs_pistone['siaf_ext']['mon'].get(mn, np.nan)
                                        * (1.0 / 12.0) * AREA_RATIO_60N_OCEAN
                                        if np.isfinite(obs_pistone['siaf_ext']['mon'].get(mn, np.nan)) else np.nan),
            'SIAF_drk_global_pistone': (obs_pistone['siaf_drk']['mon'].get(mn, np.nan)
                                        * (1.0 / 12.0) * AREA_RATIO_60N_OCEAN
                                        if np.isfinite(obs_pistone['siaf_drk']['mon'].get(mn, np.nan)) else np.nan),
        }
        for code in PISTONE_REGION_CODES:
            row[f'IS_tot_{code}']   = obs_pistone['is_tot']['mon_reg'].get(mn, {}).get(code)
            row[f'IS_ext_{code}']   = obs_pistone['is_ext']['mon_reg'].get(mn, {}).get(code)
            row[f'IS_drk_{code}']   = obs_pistone['is_drk']['mon_reg'].get(mn, {}).get(code)
            row[f'RS_{code}']       = obs_pistone['rs']['mon_reg'].get(mn, {}).get(code)
            row[f'SIAF_tot_{code}'] = obs_pistone['siaf_tot']['mon_reg'].get(mn, {}).get(code)
            row[f'SIAF_ext_{code}'] = obs_pistone['siaf_ext']['mon_reg'].get(mn, {}).get(code)
            row[f'SIAF_drk_{code}'] = obs_pistone['siaf_drk']['mon_reg'].get(mn, {}).get(code)
        pistone_rows.append(row)

    csv_out_pistone = os.path.join(PLOT_DIR,
                                   'siaf_detailed_ensemble_results_pistone.csv')
    pd.DataFrame(pistone_rows).to_csv(csv_out_pistone, index=False)
    print(f'  [PISTONE] Regional + Arctic-ocean scalars saved to: {csv_out_pistone}',
          flush=True)

    print('\nDONE', flush=True)