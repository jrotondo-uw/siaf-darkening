#!/usr/bin/env python
"""
plot_siaf_spatial_huang.py   (revised 2026-04-19  imshow & kernel fix)
──────────────────────────────────────────────────────────────
Companion to diagnose_siaf_unified.py.

Fix over previous version:
  1. imshow() instead of pcolormesh() for fast, un-distorted cartopy rendering.
  2. coarse-to-fine kernel regridding now uses Nearest Neighbor (build_nn_index)
     instead of scatter_avg to prevent sparse "dot" artifacts.

Figure layout  (3 rows × 2 columns):
  Row 0  SIAF_ext  |  col 0 = Apr–Sep mean map
                   |  col 1 = 2σ bootstrap uncertainty map
  Row 1  SIAF_drk  | …
  Row 2  SIAF_tot  | …

Bootstrap propagates:
  1. Albedo interannual variability  (yearly SIAL/APP-x stacks)
  2. ERA5 kernel RS uncertainty      (multiplicative N(1, σ²), σ = 0.025)
  3. ΔT structural uncertainty       (draw one T-source at random)

Outputs (one figure per p-threshold):
  siaf_spatial_huang_p{pval}.png
  siaf_spatial_boot_huang.npz
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as mgridspec
import glob, os, gc, resource
from scipy import stats
from scipy.spatial import cKDTree
from netCDF4 import Dataset as NC4Dataset

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('WARNING: cartopy not installed.', flush=True)

try:
    from pyproj import Transformer
except ImportError:
    raise ImportError('pyproj required.')

try:
    from global_land_mask import globe as _globe_mod
except ImportError:
    _globe_mod = None


# ============================================================
# CONFIG
# ============================================================
DIR_SIAL    = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
DIR_APPX    = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC     = '/glade/derecho/scratch/jrotondo/SIC_data/'
CACHE_CMIP6 = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_cmip6/'
CACHE_OBS   = '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/'
OUT_DIR     = '/glade/work/jrotondo/ncc_paper/plots/sensitivity/'
PLOT_STEM   = os.path.join(OUT_DIR, 'siaf_spatial_huang')
NPZ_OUT     = os.path.join(OUT_DIR, 'siaf_spatial_boot_huang.npz')
os.makedirs(OUT_DIR, exist_ok=True)

KERNEL_FILE          = '/glade/work/jrotondo/ncc_paper/kernels/ERA5_kernel_alb_SFC.nc'
KERNEL_VAR           = 'SFC_all'
KERNEL_SCALE         = 100.0
KERNEL_RS_FRAC_SIGMA = 0.025   # 1σ → 2σ = ±5%  (Huang & Huang 2023)

GISTEMP_PATH   = '/glade/work/jrotondo/gistemp_global.csv'
COWTANWAY_PATH = '/glade/work/jrotondo/cowtan_way.txt'
NCEP_PATH      = '/glade/work/jrotondo/ncep_air_sig995.nc'
ERA5_GLOB      = ('/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/'
                  '*/e5.moda.an.sfc.128_167_2t.ll025sc.*.nc')

EARLY           = list(range(1982, 1992))
LATE_CMIP6      = list(range(2005, 2015))
LATE_OBS        = list(range(2015, 2025))
TREND_START     = 1982
TREND_END_CMIP6 = 2014
TREND_END_OBS   = 2024

MONTH_NUMS  = [4, 5, 6, 7, 8, 9]
MONTH_NAMES = {4:'April', 5:'May', 6:'June',
               7:'July',  8:'August', 9:'September'}

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

SIC_ICE_THRESHOLD = 0.15
N_BOOTSTRAP       = 1000
BOOTSTRAP_SEED    = 42
PVAL_THRESHOLDS   = [0.01, 0.05, 0.10]
MAP_PERIOD        = 'Obs-era'


# ============================================================
# UTILITIES  (identical to main script)
# ============================================================
def log_mem(tag=''):
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.
    print(f'[MEM] {tag:<55} {gb:.2f} GB', flush=True)

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float32).ravel())
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float32).ravel())
    return np.column_stack([np.cos(lat)*np.cos(lon),
                             np.cos(lat)*np.sin(lon),
                             np.sin(lat)])

def build_scatter_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d, chunk=100_000):
    tgt_xyz = latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    src_xyz = latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel())
    tree = cKDTree(tgt_xyz)
    idx  = np.empty(src_xyz.shape[0], dtype=np.int32)
    for i in range(0, src_xyz.shape[0], chunk):
        e = min(i + chunk, src_xyz.shape[0])
        _, idx[i:e] = tree.query(src_xyz[i:e], workers=1)
    del tree, tgt_xyz, src_xyz; gc.collect()
    return idx, tgt_lat2d.shape

def build_nn_index(src_lat2d, src_lon2d, tgt_lat2d, tgt_lon2d):
    tree = cKDTree(latlon_to_xyz(src_lat2d.ravel(), src_lon2d.ravel()))
    _, idx = tree.query(latlon_to_xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel()), workers=-1)
    del tree; gc.collect()
    return idx

def scatter_avg(arr_flat, idx, out_shape):
    n     = int(np.prod(out_shape))
    accum = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    v     = np.isfinite(arr_flat)
    np.add.at(accum, idx[v], arr_flat[v].astype(np.float32))
    np.add.at(count, idx[v], np.float32(1.))
    out   = np.full(n, np.nan, dtype=np.float32)
    m     = count > 0
    out[m] = accum[m] / count[m]
    return out.reshape(out_shape)

def regrid_sial_to_psn(arr, fill_zero=True):
    if fill_zero:
        src = np.where(arctic_sial,
                       np.where(np.isfinite(arr), arr, np.float32(0.)),
                       np.nan).astype(np.float32)
    else:
        src = arr.astype(np.float32)
        
    # NEW: Use direct indexing instead of scatter_avg
    return src.ravel()[idx_sial_to_psn].reshape(psn_shape)

def trend_dT(years, vals, y0, y1):
    m = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if m.sum() < 5: return np.nan
    return stats.linregress(years[m], vals[m]).slope * (y1 - y0)


# ============================================================
# TEMPERATURE LOADERS
# ============================================================
def _load_gistemp(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, skiprows=1, na_values='***')
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        return df['Year'].values.astype(float), df['J-D'].values.astype(float)
    except Exception: return None, None

def _load_cowtan_way(path):
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None,
                         names=['year','anomaly','u1','u2','u3'])
        return df['year'].values.astype(float), df['anomaly'].values.astype(float)
    except Exception: return None, None

def _load_ncep(path):
    if not os.path.exists(path): return None, None
    try:
        ds = xr.open_dataset(path)
        w  = np.cos(np.deg2rad(ds['air'].lat))
        ta = ds['air'].weighted(w).mean(dim=['lat','lon']).resample(time='1YE').mean()
        ds.close()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except Exception: return None, None

def _load_era5_t2m(glob_pattern):
    files = sorted(glob.glob(glob_pattern))
    if not files: return None, None
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        w  = np.cos(np.deg2rad(ds['VAR_2T'].latitude))
        ta = ds['VAR_2T'].weighted(w).mean(dim=['latitude','longitude']).resample(time='1YE').mean()
        ds.close()
        return ta.time.dt.year.values.astype(float), ta.values.squeeze().astype(float)
    except Exception: return None, None

def build_T_sources():
    sources = []
    for name, (yy, vv) in [
        ('GISTEMP',   _load_gistemp(GISTEMP_PATH)),
        ('CowtanWay', _load_cowtan_way(COWTANWAY_PATH)),
        ('NCEP',      _load_ncep(NCEP_PATH)),
        ('ERA5',      _load_era5_t2m(ERA5_GLOB)),
    ]:
        if yy is not None and np.isfinite(vv).sum() > 10:
            sources.append((name, yy, vv))
            print(f'  T-source {name}: ok ({len(yy)} yrs)', flush=True)
        else:
            print(f'  T-source {name}: FAILED', flush=True)
    if len(sources) < 2: raise RuntimeError('Need ≥2 T-sources.')
    return sources

def get_dT_samples(y0, y1, t_sources):
    vals = [trend_dT(yy, vv, y0, y1) for _, yy, vv in t_sources]
    vals = [v for v in vals if np.isfinite(v)]
    return np.array(vals if vals else [0.84], dtype=np.float32)


# ============================================================
# ALBEDO / SIC LOADERS
# ============================================================
def load_alb_yearly(cache_dir, mname, years):
    use_appx = (mname == 'September')
    result   = []
    for y in years:
        p = (f'{cache_dir}/appx_alb_{mname}_{y}.npy' if use_appx
             else f'{cache_dir}/clara_alb_{mname}_{y}.npy')
        if not os.path.exists(p): continue
        raw = np.load(p).astype(np.float32)
        if use_appx and idx_appx_to_sial is not None:
            a = (raw.ravel()[idx_appx_to_sial].reshape(sial_shape)
                 if raw.shape != sial_shape else raw.copy())
        else:
            a = raw.copy()
        result.append(a); del raw
    return result

def load_sic(cache_dir, mname):
    try:
        return dict(
            slope = np.load(f'{cache_dir}/sic_trend_slope_sial_{mname}.npy').astype(np.float32),
            pval  = np.load(f'{cache_dir}/sic_trend_pval_sial_{mname}.npy').astype(np.float32),
            sic_e = np.load(f'{cache_dir}/sic_early_sial_{mname}.npy').astype(np.float32),
            sic_l = np.load(f'{cache_dir}/sic_late_sial_{mname}.npy').astype(np.float32),
        )
    except Exception: return None


# ============================================================
# IS COMPUTATION
# ============================================================
def compute_IS(ae, al, sic_data, delta_t, pval_thresh):
    dalpha   = al - ae
    valid    = arctic_sial & np.isfinite(dalpha)
    changing = (valid & np.isfinite(sic_data['slope']) & np.isfinite(sic_data['pval'])
                & (sic_data['slope'] < 0) & (sic_data['pval'] < pval_thresh))
    stable   = (valid & ~changing
                & (np.fmax(sic_data['sic_e'], sic_data['sic_l']) >= SIC_ICE_THRESHOLD))
    IS_ext   = np.where(changing, -dalpha / delta_t, np.nan).astype(np.float32)
    IS_drk   = np.where(stable,  -dalpha / delta_t, np.nan).astype(np.float32)
    IS_tot   = np.where(
        np.isfinite(IS_ext) | np.isfinite(IS_drk),
        np.where(np.isfinite(IS_ext), IS_ext, 0.)
        + np.where(np.isfinite(IS_drk), IS_drk, 0.),
        np.nan).astype(np.float32)
    IS_ext = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_ext), IS_ext, 0.), np.nan).astype(np.float32)
    IS_drk = np.where(np.isfinite(IS_tot),
                      np.where(np.isfinite(IS_drk), IS_drk, 0.), np.nan).astype(np.float32)
    return IS_tot, IS_ext, IS_drk


# ============================================================
# KERNEL LOADER
# ============================================================
_kernel_cache = {}

def load_kernel_psn(mn):
    if mn in _kernel_cache: return _kernel_cache[mn]
    if not os.path.exists(KERNEL_FILE):
        raise FileNotFoundError(KERNEL_FILE)
    
    with xr.open_dataset(KERNEL_FILE) as ds:
        tdim = 'month' if 'month' in ds.dims else next(
            d for d, s in ds.dims.items() if s == 12)
        lat_name = 'latitude' if 'latitude' in ds else 'lat'
        lon_name = 'longitude' if 'longitude' in ds else 'lon'
        k_lat = ds[lat_name].values
        k_lon = ds[lon_name].values
        k_lon2d, k_lat2d = np.meshgrid(k_lon, k_lat)
        k_raw = ds[KERNEL_VAR].isel({tdim: mn - 1}).values.astype(np.float32)
        
    k_raw = np.abs(k_raw) * np.float32(KERNEL_SCALE)
    
    # ── THE FIX: Nearest Neighbor Indexing instead of scatter_avg ──
    if '_idx' not in _kernel_cache:
        print('  Building kernel → PSN25 index (Nearest Neighbor)...', flush=True)
        idx_k = build_nn_index(k_lat2d, k_lon2d, psn_lat2d, psn_lon2d)
        _kernel_cache['_idx'] = idx_k
        del k_lat2d, k_lon2d; gc.collect()
        
    idx_k = _kernel_cache['_idx']
    k_psn = k_raw.ravel()[idx_k].reshape(psn_shape).astype(np.float32)
    
    _kernel_cache[mn] = k_psn
    del k_raw; gc.collect()
    return k_psn


# ============================================================
# BOOTSTRAP
# ============================================================
def _stack(arr_list):
    valid = [a for a in arr_list if a is not None]
    return np.stack(valid, axis=0).astype(np.float32) if valid else None

def _resample_mean(stack, rng):
    n = stack.shape[0]
    return np.nanmean(stack[rng.integers(0, n, size=n)], axis=0).astype(np.float32)


def run_spatial_bootstrap(cache_dir, early_yrs, late_yrs, dT_samples, pval_thresh):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_t = len(dT_samples)

    # Pre-load everything before the loop
    k_psns, early_stks, late_stks, sics = [], [], [], []
    for mn in MONTH_NUMS:
        mname = MONTH_NAMES[mn]
        k_psns.append(load_kernel_psn(mn))
        es = _stack(load_alb_yearly(cache_dir, mname, early_yrs))
        ls = _stack(load_alb_yearly(cache_dir, mname, late_yrs))
        sc = load_sic(cache_dir, mname)
        early_stks.append(es); late_stks.append(ls); sics.append(sc)
        ok = es is not None and ls is not None and sc is not None
        print(f'  {mname}: {"ok" if ok else "SKIP"} '
              f'(early={es.shape[0] if es is not None else "–"} '
              f'late={ls.shape[0]  if ls is not None else "–"})', flush=True)

    boot_ext = np.full((N_BOOTSTRAP, *psn_shape), np.nan, dtype=np.float32)
    boot_drk = np.full((N_BOOTSTRAP, *psn_shape), np.nan, dtype=np.float32)
    boot_tot = np.full((N_BOOTSTRAP, *psn_shape), np.nan, dtype=np.float32)

    print(f'  {N_BOOTSTRAP} bootstrap iterations...', flush=True)
    for b in range(N_BOOTSTRAP):
        if b % 200 == 0: print(f'    iter {b}/{N_BOOTSTRAP}', flush=True)
        dT    = float(dT_samples[rng.integers(0, n_t)])
        kscal = float(np.clip(rng.normal(1.0, KERNEL_RS_FRAC_SIGMA), 0.80, 1.20))
        m_ext, m_drk, m_tot = [], [], []
        for j in range(len(MONTH_NUMS)):
            if early_stks[j] is None or late_stks[j] is None or sics[j] is None:
                continue
            ae = _resample_mean(early_stks[j], rng)
            al = _resample_mean(late_stks[j],  rng)
            k  = (k_psns[j] * kscal).astype(np.float32)
            IS_tot, IS_ext, IS_drk = compute_IS(ae, al, sics[j], dT, pval_thresh)
            # fill_zero=True: arctic pixels with no IS → 0, not NaN → continuous field
            m_ext.append(regrid_sial_to_psn(IS_ext, fill_zero=True) * k)
            m_drk.append(regrid_sial_to_psn(IS_drk, fill_zero=True) * k)
            m_tot.append(regrid_sial_to_psn(IS_tot, fill_zero=True) * k)
        if not m_tot: continue
        boot_ext[b] = np.nanmean(np.stack(m_ext, 0), 0)
        boot_drk[b] = np.nanmean(np.stack(m_drk, 0), 0)
        boot_tot[b] = np.nanmean(np.stack(m_tot, 0), 0)

    out = {}
    for tag, boot in [('ext', boot_ext), ('drk', boot_drk), ('tot', boot_tot)]:
        out[f'mean_{tag}']   = np.nanmean(boot, 0).astype(np.float32)
        out[f'sigma2_{tag}'] = (2 * np.nanstd(boot, 0)).astype(np.float32)
    return out


# ============================================================
# CARTOPY / PROJECTION SETUP
# ============================================================
def make_psn_crs():
    if not HAS_CARTOPY: return None
    globe = ccrs.Globe(semimajor_axis=6378273., semiminor_axis=6356889.449)
    return ccrs.Stereographic(central_latitude=90., central_longitude=-45.,
                               true_scale_latitude=70., globe=globe)

PSN_CRS = make_psn_crs()


def add_map_features(ax):
    """Land + coastlines on top of the imshow raster."""
    ax.add_feature(cfeature.LAND,  facecolor='#CCCCCC', zorder=6, edgecolor='none')
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


# ── The key fix: imshow in native PSN25 metre coordinates ───────────────────
def plot_field(ax, data, cmap, vmin, vmax):
    my_cmap = matplotlib.colormaps[cmap].copy()
    my_cmap.set_bad(alpha=0.0)

    arr = np.where(DOMAIN_PSN, data, np.nan).astype(np.float32)
    extent_m = [_imshow_x0, _imshow_x1, _imshow_y0, _imshow_y1]

    if HAS_CARTOPY:
        # Bypass cartopy's imshow override — it would warp PSN_CRS→PSN_CRS
        # pixel-by-pixel, producing sub-pixel dots at Arctic scale.
        # Calling matplotlib's base imshow directly stamps the array as a
        # raster in the GeoAxes native coordinate space (PSN25 metres).
        im = matplotlib.axes.Axes.imshow(
            ax, arr,
            origin=_imshow_origin,
            extent=extent_m,
            cmap=my_cmap, vmin=vmin, vmax=vmax,
            interpolation='nearest',
            zorder=5,
        )
    else:
        im = ax.imshow(arr, origin=_imshow_origin,
                       cmap=my_cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
    return im


# ============================================================
# FIGURE  —  3 rows × 2 cols  (ext / drk / tot)
# ============================================================
ROW_DEFS = [
    ('ext', r'SIAF$_\mathregular{ext}$', '#3A6BAF'),
    ('drk', r'SIAF$_\mathregular{drk}$', '#D95F1A'),
    ('tot', r'SIAF$_\mathregular{tot}$', '#6A3FA5'),
]
CMAP_VAL   = 'YlOrRd'
CMAP_SIGMA = 'Blues'


def make_figure(results, pval, out_path):
    # Shared colour limits across all rows (within each column)
    vmax_val = max(np.nanpercentile(results[f'mean_{t}'][DOMAIN_PSN],   98)
                   for t in ('ext','drk','tot'))
    vmax_sig = max(np.nanpercentile(results[f'sigma2_{t}'][DOMAIN_PSN], 98)
                   for t in ('ext','drk','tot'))
    vmax_val = max(vmax_val, 0.05)
    vmax_sig = max(vmax_sig, 0.01)

    fig = plt.figure(figsize=(13, 18), facecolor='#F4F4F4')

    # 3 rows × 4 cols: [map_val | cbar_val | map_sig | cbar_sig]
    gs = mgridspec.GridSpec(3, 4, figure=fig,
                             left=0.10, right=0.95, # give left margin more room
                             top=0.93,  bottom=0.02,
                             hspace=0.6, wspace=0.04, # increase hspace
                             width_ratios=[1, 0.035, 1, 0.035])

    last_im_val = last_im_sig = None

    for row, (tag, label, accent) in enumerate(ROW_DEFS):
        ax_val = make_ax(fig, gs[row, 0])
        ax_sig = make_ax(fig, gs[row, 2])

        last_im_val = plot_field(ax_val, results[f'mean_{tag}'],
                                 CMAP_VAL,   0, vmax_val)
        last_im_sig = plot_field(ax_sig, results[f'sigma2_{tag}'],
                                 CMAP_SIGMA, 0, vmax_sig)

        # Row label on the left margin
        ax_val.text(-0.09, 0.5, label,
                    transform=ax_val.transAxes,
                    fontsize=16, fontweight='bold', color=accent,
                    va='center', ha='right',
                    rotation=90, rotation_mode='anchor')

        if row == 0:
            ax_val.set_title('Apr–Sep mean  (Obs-era, Huang/ERA5)',
                              fontsize=16, fontweight='bold', pad=8)
            ax_sig.set_title('2σ bootstrap uncertainty',
                              fontsize=16, fontweight='bold', pad=8)

    # Shared colourbars in the narrow columns
    cb_ax_val = fig.add_subplot(gs[:, 1])
    cb_ax_sig = fig.add_subplot(gs[:, 3])

    cb_val = fig.colorbar(last_im_val, cax=cb_ax_val)
    cb_val.set_label('W m⁻² K⁻¹', fontsize=16)
    cb_val.ax.tick_params(labelsize=16)

    cb_sig = fig.colorbar(last_im_sig, cax=cb_ax_sig)
    cb_sig.set_label('W m⁻² K⁻¹', fontsize=16)
    cb_sig.ax.tick_params(labelsize=16)

    '''fig.suptitle(
        f'Huang / ERA5 Kernel — SIAF  Apr–Sep  |  p < {pval}\n'
        'Left: seasonal-mean     Right: 2σ bootstrap uncertainty',
        fontsize=13, fontweight='bold', y=0.965)'''

    fig.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out_path}', flush=True)


# ============================================================
# GRID SETUP
# ============================================================
print('='*70, flush=True)
print('GRID SETUP', flush=True)
print('='*70, flush=True)

# ── SIAL ─────────────────────────────────────────────────────────────────────
sample_sial = sorted(glob.glob(f'{DIR_SIAL}/*.nc'))[0]
with xr.open_dataset(sample_sial) as ds:
    sial_lat1d = ds['lat'].values; sial_lon1d = ds['lon'].values
if sial_lat1d.ndim == 1:
    sial_lon2d, sial_lat2d = np.meshgrid(sial_lon1d, sial_lat1d)
else:
    sial_lat2d, sial_lon2d = sial_lat1d, sial_lon1d
sial_shape = sial_lat2d.shape
try:
    arctic_sial = np.load(f'{CACHE_CMIP6}/arctic_60N_sial.npy').astype(bool)
except Exception:
    arctic_sial = (sial_lat2d >= 60.)

# ── APP-x → SIAL ─────────────────────────────────────────────────────────────
idx_appx_to_sial = None
try:
    sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
    with NC4Dataset(sample_appx, 'r') as ds:
        appx_lat = np.ma.filled(ds.variables['latitude'][:],  np.nan).astype(float)
        appx_lon = np.ma.filled(ds.variables['longitude'][:], np.nan).astype(float)
    if appx_lat.ndim == 1:
        appx_lon2d, appx_lat2d = np.meshgrid(appx_lon, appx_lat)
    else:
        appx_lat2d, appx_lon2d = appx_lat, appx_lon
    idx_appx_to_sial = build_nn_index(appx_lat2d, appx_lon2d,
                                       sial_lat2d, sial_lon2d)
    del appx_lat2d, appx_lon2d, appx_lat, appx_lon; gc.collect()
    print('  APP-x → SIAL: ok', flush=True)
except Exception as e:
    print(f'  APP-x → SIAL: FAILED ({e})', flush=True)

# ── PSN25 in native x/y metres ───────────────────────────────────────────────
sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    psn_x = ds['x'].values.astype(np.float64)   # 1-D, metres
    psn_y = ds['y'].values.astype(np.float64)   # 1-D, metres

psn_xx, psn_yy = np.meshgrid(psn_x, psn_y)     # native PSN25 metres
psn_shape       = psn_xx.shape

# Back-project to lat/lon (for domain mask only)
_tfm_psn2ll = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
psn_lon2d, psn_lat2d = _tfm_psn2ll.transform(psn_xx, psn_yy)

# Area weights
phi_ts          = np.deg2rad(70.)
k_ps            = (1. + np.sin(phi_ts)) / np.maximum(
                      1. + np.sin(np.deg2rad(psn_lat2d)), 1e-9)
AREA_WEIGHT_PSN = (1. / k_ps**2).astype(np.float32)

def make_ocean_mask(lat2d, lon2d):
    if _globe_mod is None: return lat2d >= 60.
    lp = np.where(lon2d > 180, lon2d - 360, lon2d)
    return (lat2d >= 60.) & _globe_mod.is_ocean(lat2d, lp)

DOMAIN_PSN = make_ocean_mask(psn_lat2d, psn_lon2d)
print(f'  PSN25: {psn_shape}  domain cells: {DOMAIN_PSN.sum():,}', flush=True)

# ── imshow extent parameters ─────────────────────────────────────────────────
# PSN25 pixel size (metres)
_psn_dx = float(psn_x[1] - psn_x[0])   # positive  (x increases eastward)
_psn_dy = float(psn_y[1] - psn_y[0])   # sign tells us row orientation

# imshow extent = [left_edge, right_edge, bottom_edge, top_edge]  (metres)
_imshow_x0 = float(psn_x[0])  - abs(_psn_dx) / 2
_imshow_x1 = float(psn_x[-1]) + abs(_psn_dx) / 2
_y_lo = min(float(psn_y[0]), float(psn_y[-1])) - abs(_psn_dy) / 2
_y_hi = max(float(psn_y[0]), float(psn_y[-1])) + abs(_psn_dy) / 2
_imshow_y0, _imshow_y1 = _y_lo, _y_hi  # bottom, top

# origin: if y decreases as row index grows (row 0 = northernmost = top of image)
_imshow_origin = 'upper' if _psn_dy < 0 else 'lower'
print(f'  imshow extent x: [{_imshow_x0/1e6:.2f}, {_imshow_x1/1e6:.2f}] ×10⁶ m', flush=True)
print(f'  imshow extent y: [{_imshow_y0/1e6:.2f}, {_imshow_y1/1e6:.2f}] ×10⁶ m', flush=True)
print(f'  imshow origin: {_imshow_origin}', flush=True)

# ── Map display extent (58°N bounding circle, in PSN25 metres) ───────────────
_tfm_ll2psn = Transformer.from_crs('EPSG:4326', PSN25_PROJ4, always_xy=True)
_lons_rim = np.linspace(-180, 180, 361)
_xs_rim, _ys_rim = _tfm_ll2psn.transform(_lons_rim, np.full_like(_lons_rim, 57.))
_rim_r = max(np.abs(_xs_rim).max(), np.abs(_ys_rim).max()) * 1.04
_ext_xmin, _ext_xmax = -_rim_r, _rim_r
_ext_ymin, _ext_ymax = -_rim_r, _rim_r
print(f'  Map display radius: {_rim_r/1e6:.2f} ×10⁶ m', flush=True)

# ── SIAL → PSN25 scatter index ───────────────────────────────────────────────
# ── SIAL → PSN25 scatter index ───────────────────────────────────────────────
print('  Building SIAL → PSN25 index (Nearest Neighbor)...', flush=True)
idx_sial_to_psn = build_nn_index(sial_lat2d, sial_lon2d, psn_lat2d, psn_lon2d)

# ── Temperature sources ───────────────────────────────────────────────────────
print('\nLoading T sources...', flush=True)
T_SOURCES        = build_T_sources()
dT_samples_CMIP6 = get_dT_samples(TREND_START, TREND_END_CMIP6, T_SOURCES)
dT_samples_OBS   = get_dT_samples(TREND_START, TREND_END_OBS,   T_SOURCES)
print(f'  ΔT CMIP6 samples: {dT_samples_CMIP6}', flush=True)
print(f'  ΔT Obs   samples: {dT_samples_OBS}',   flush=True)

PERIOD_CONFIGS = {
    'CMIP6':   (CACHE_CMIP6, EARLY, LATE_CMIP6, dT_samples_CMIP6),
    'Obs-era': (CACHE_OBS,   EARLY, LATE_OBS,   dT_samples_OBS),
}


# ============================================================
# MAIN LOOP
# ============================================================
boot_cache = {}

for pval in PVAL_THRESHOLDS:
    print(f'\n{"="*70}', flush=True)
    print(f'p < {pval}  —  period: {MAP_PERIOD}', flush=True)
    print(f'{"="*70}', flush=True)

    key = (MAP_PERIOD, pval)
    if key not in boot_cache:
        cache_dir, early_yrs, late_yrs, dT_samp = PERIOD_CONFIGS[MAP_PERIOD]
        boot_cache[key] = run_spatial_bootstrap(
            cache_dir, early_yrs, late_yrs, dT_samp, pval)
    log_mem(f'bootstrap done  p<{pval}')

    make_figure(boot_cache[key], pval, f'{PLOT_STEM}_p{pval}.png')

# ── Save .npz ─────────────────────────────────────────────────────────────────
print('\nSaving .npz...', flush=True)
save_dict = {'psn_lat': psn_lat2d, 'psn_lon': psn_lon2d,
             'psn_xx': psn_xx, 'psn_yy': psn_yy, 'domain': DOMAIN_PSN}
for (period, pval), res in boot_cache.items():
    pfx = f'{period.replace("-","")}_p{str(pval).replace(".","")}'
    for k, v in res.items():
        save_dict[f'{pfx}_{k}'] = v
np.savez_compressed(NPZ_OUT, **save_dict)
print(f'Saved → {NPZ_OUT}', flush=True)

log_mem('DONE')
print('\nAll done.', flush=True)
