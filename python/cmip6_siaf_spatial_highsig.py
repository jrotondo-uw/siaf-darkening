#!/usr/bin/env python
"""
cmip6_siaf_spatial_highsig.py
──────────────────────────────
Spatial average + 2σ spread for the CMIP6 "high-sensitivity" ensemble:
individual ensemble members whose scalar SIAF_global > MMM + 2σ,
where MMM and σ are computed across ALL individual member values in the
scalar CSV (not model means).

Three-phase execution
─────────────────────
  Phase 0 – read scalar CSV, compute threshold, identify qualifying members.
  Phase 1 – parallel per-member spatial computation on native grids,
             scatter-averaged to 1°×1°.  Cached to
             CACHE_DIR/spatial_hisig_{model}_{member}.npz
  Phase 2 – load cache, compute member mean + 2σ spread, generate 3×2 plot.

Outputs
───────
  CACHE_DIR/spatial_hisig_{model}_{member}.npz   per-member spatial fields
  PLOT_DIR/cmip6_siaf_spatial_highsig.png         3×2 polar map
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('WARNING: cartopy not found – falling back to plain lat/lon axes')

# Import computation machinery from the main CMIP6 script
from cmip6_siaf_calculation import (
    MODEL_INFO, MONTH_NUMS, TREND_YEARS,
    compute_T_annual_global,
    compute_member_IS_native,
    compute_model_rs_native,
    latlon_to_xyz,
)

# ============================================================
# CONFIG
# ============================================================
PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
CSV_PATH  = os.path.join(PLOT_DIR, 'siaf_cmip6_pistone_summary.csv')

# Per-member cache goes in its own subdir to avoid confusion with per-model cache
CACHE_DIR = os.path.join(PLOT_DIR, 'spatial_cache_highsig')
os.makedirs(CACHE_DIR, exist_ok=True)

OUT_PATH  = os.path.join(PLOT_DIR, 'cmip6_siaf_spatial_highsig.png')

N_WORKERS       = 4
FORCE_RECOMPUTE = False   # True → redo even if .npz exists
AGREE_THRESH    = 0.70    # dot-stipple threshold

# ── Common target grid: 1°×1° lat-lon, 56.5–89.5 N ─────────────────────────
_LATS = np.arange(56.5, 90., 1.0)    # 33 row centres
_LONS = np.arange(-179.5, 180., 1.0) # 360 col centres
TGT_LON2D, TGT_LAT2D = np.meshgrid(_LONS, _LATS)
TGT_SHAPE = TGT_LAT2D.shape          # (33, 360)
TGT_NTOT  = int(np.prod(TGT_SHAPE))

# ── Polar-stereo raster constants ────────────────────────────────────────────
# The 1°×1° lat-lon grid forms concentric rings (latitude bands) and radial
# spokes (longitude columns) in polar-stereo projection.  To avoid pcolormesh
# drawing those cell edges as visible artefacts, we rasterize each field to a
# regular 600×600 pixel PS grid and display with imshow instead.

_PS_N      = 600        # output raster size (pixels per side)
_PS_EXTENT = 5.5e6      # half-width of domain in metres from the pole

# Lazy-built lookup arrays (built once in the parent process)
_PS_VALID = None   # (600,600) bool  — pixel inside the data domain
_PS_X2D   = None   # (600,600) float — PS x coords in metres
_PS_Y2D   = None   # (600,600) float — PS y coords in metres
_PS_LAT_F = None   # (600,600) float — fractional row index into TGT grid
_PS_LON_F = None   # (600,600) float — fractional col index into TGT grid
# Integer indices (nearest neighbour) used only for the stipple-mask lookup
_PS_LAT_IDX = None
_PS_LON_IDX = None


def _ensure_ps_raster_coords():
    """Build PS-grid ↔ TGT-grid coordinate arrays (once in parent process)."""
    global _PS_VALID, _PS_X2D, _PS_Y2D, _PS_LAT_F, _PS_LON_F
    global _PS_LAT_IDX, _PS_LON_IDX
    if _PS_VALID is not None or not HAS_CARTOPY:
        return
    n = _PS_N
    x_lin = np.linspace(-_PS_EXTENT, _PS_EXTENT, n)
    y_lin = np.linspace(-_PS_EXTENT, _PS_EXTENT, n)
    _PS_X2D, _PS_Y2D = np.meshgrid(x_lin, y_lin)

    pc  = ccrs.PlateCarree()
    ps  = ccrs.NorthPolarStereo(central_longitude=-45)
    pts = pc.transform_points(ps, _PS_X2D.ravel(), _PS_Y2D.ravel())
    lon_ps = pts[:, 0].reshape(n, n)
    lat_ps = pts[:, 1].reshape(n, n)

    # Floating-point indices into the 1°×1° grid for bilinear interpolation
    _PS_LAT_F = (lat_ps - 56.5).astype(np.float32)          # row  0 … 32
    _PS_LON_F = ((lon_ps + 179.5) % 360.).astype(np.float32) # col  0 … 359

    # Integer indices for the stipple-mask lookup
    lat_i = np.round(_PS_LAT_F).astype(np.int32)
    lon_i = np.round(_PS_LON_F).astype(np.int32) % 360
    _PS_VALID   = ((lat_i >= 0) & (lat_i < 33)
                   & (lat_ps >= 55.0) & np.isfinite(lat_ps))
    _PS_LAT_IDX = lat_i
    _PS_LON_IDX = lon_i


def _smooth_std(field_2d, sigma_km=80., lat_1d=None):
    """
    NaN-aware Gaussian smooth of a std/spread field, with the longitude-
    direction kernel width scaled by 1/cos(lat).

    Near the pole, 1 degree of longitude shrinks to near-zero physical
    width, so a constant-degree smoothing kernel (e.g. fixed sigma=2 deg)
    is far too narrow there — leftover grid-scale noise in the longitude
    direction gets geometrically stretched into long radial spokes once
    rasterized to polar stereo.  Scaling the kernel by 1/cos(lat) keeps
    the smoothing scale ~constant in physical km at every latitude.
    Latitude-direction smoothing uses a fixed sigma_km (no pole issue
    there since 1 deg latitude is always ~111 km).
    """
    from scipy.ndimage import gaussian_filter1d
    if lat_1d is None:
        lat_1d = _LATS
    data = np.where(np.isfinite(field_2d), field_2d.astype(np.float64), 0.)
    mask = np.isfinite(field_2d).astype(np.float64)

    sigma_lat = sigma_km / 111.
    data = gaussian_filter1d(data, sigma=sigma_lat, axis=0, mode='nearest')
    mask = gaussian_filter1d(mask, sigma=sigma_lat, axis=0, mode='nearest')

    out  = np.zeros_like(data)
    outm = np.zeros_like(mask)
    for i, lat in enumerate(lat_1d):
        coslat    = max(np.cos(np.deg2rad(lat)), 0.05)
        sigma_lon = min(sigma_km / (111. * coslat), 60.)
        out[i, :]  = gaussian_filter1d(data[i, :], sigma=sigma_lon, mode='wrap')
        outm[i, :] = gaussian_filter1d(mask[i, :], sigma=sigma_lon, mode='wrap')

    result = np.where(outm > 0.1, out / np.maximum(outm, 1e-9), np.nan)
    return result.astype(np.float32)


def _inpaint_gaps(field_2d, n_iter=4):
    """
    Fill isolated NaN cells caused by coarse-resolution source grids
    (e.g. CanESM5 T63 ≈ 2.8°, GISS 2°×2.5°) that leave sparse gaps in
    the 1°×1° target grid.  Only NaN cells with ≥20% valid neighbours are
    filled; large NaN regions (land, far out-of-domain) are left as NaN.
    Valid cells are never modified.
    """
    from scipy.ndimage import uniform_filter
    out = field_2d.copy().astype(np.float64)
    for _ in range(n_iter):
        nan_m = ~np.isfinite(out)
        if not nan_m.any():
            break
        data_f = np.where(np.isfinite(out), out, 0.)
        wt_f   = np.isfinite(out).astype(np.float64)
        sm_d   = uniform_filter(data_f, size=3)
        sm_w   = uniform_filter(wt_f,   size=3)
        can_fill = nan_m & (sm_w > 0.20)          # ≥20% valid neighbours
        out = np.where(can_fill,
                       sm_d / np.maximum(sm_w, 1e-9),
                       out)
    return out.astype(np.float32)


def _smooth_ps_pixels(ps_2d, sigma_px):
    """
    NaN-aware Gaussian smooth directly in PS-pixel space.

    Smoothing in lat-lon space before rasterizing can't fully kill pole-
    region spokes: near the projection's centre, adjacent PS pixels can
    map to widely-separated longitude indices in the (33,360) grid (since
    longitude wraps almost instantly close to the pole), so the bilinear
    resample step itself reintroduces sharp jumps there. PS-pixel space
    has no such singularity — physical distance between pixels is well
    behaved everywhere, including at the pole — so smoothing here is the
    robust fix for whatever the lat-lon-space smoothing leaves behind.
    """
    from scipy.ndimage import gaussian_filter
    finite = np.isfinite(ps_2d)
    data   = np.where(finite, ps_2d, 0.).astype(np.float64)
    mask   = finite.astype(np.float64)
    sm_d   = gaussian_filter(data, sigma=sigma_px)
    sm_m   = gaussian_filter(mask, sigma=sigma_px)
    out    = np.where(sm_m > 0.1, sm_d / np.maximum(sm_m, 1e-9), np.nan)
    return out.astype(np.float32)


def _rasterize(field_2d, smooth_px=0.):
    """
    Bilinear interpolation from (33,360) lat-lon field onto (600,600) PS grid.

    1. Inpaints small NaN gaps (coarse-grid holes) so the bilinear stencil
       doesn't hit NaN walls and produce radial/spoke artefacts.
    2. Pads longitude by PAD columns for dateline continuity.
    3. Uses mode='nearest' (clamps latitude; longitude handled by padding).
    4. Optionally smooths in PS-pixel space (see _smooth_ps_pixels) — pass
       smooth_px>0 for spread/std fields prone to pole-region artefacts.
    """
    from scipy.ndimage import map_coordinates
    _ensure_ps_raster_coords()
    n = _PS_N
    PAD = 3   # longitude padding columns

    filled  = _inpaint_gaps(field_2d)
    field_f = np.where(np.isfinite(filled), filled.astype(np.float64), 0.)
    mask_f  = np.isfinite(filled).astype(np.float64)

    # Pad longitude: [last PAD cols | original 360 | first PAD cols]
    fp  = np.concatenate([field_f[:, -PAD:], field_f, field_f[:, :PAD]], axis=1)
    mp  = np.concatenate([mask_f[:,  -PAD:], mask_f,  mask_f[:,  :PAD]], axis=1)

    v = _PS_VALID
    lat_c = np.clip(_PS_LAT_F[v], 0., 32.).astype(np.float64)
    lon_c = (_PS_LON_F[v] + PAD).astype(np.float64)   # shift for padding

    v_interp = map_coordinates(fp, [lat_c, lon_c], order=1, mode='nearest')
    m_interp = map_coordinates(mp, [lat_c, lon_c], order=1, mode='nearest')

    out    = np.full((n, n), np.nan, np.float32)
    ok     = v.copy()
    ok[v]  = m_interp > 0.5
    out[ok] = (v_interp / np.maximum(m_interp, 1e-6))[m_interp > 0.5].astype(
        np.float32)
    if smooth_px > 0:
        out = _smooth_ps_pixels(out, smooth_px)
    return out


# KD-tree built once in the parent; inherited by fork'd workers on Linux
_TGT_XYZ  = latlon_to_xyz(TGT_LAT2D.ravel(),
                            TGT_LON2D.ravel()).astype(np.float32)
_TGT_TREE = cKDTree(_TGT_XYZ)


# ============================================================
# REGRID UTILITY
# ============================================================

def _scatter_avg(lat2d, lon2d, field):
    """Scatter-average `field` onto the 1°×1° target grid."""
    src    = field.ravel().astype(np.float32)
    xyz    = latlon_to_xyz(lat2d.ravel(), lon2d.ravel()).astype(np.float32)
    _, idx = _TGT_TREE.query(xyz, workers=-1)
    acc = np.zeros(TGT_NTOT, np.float32)
    cnt = np.zeros(TGT_NTOT, np.float32)
    ok  = np.isfinite(src)
    np.add.at(acc, idx[ok], src[ok])
    np.add.at(cnt, idx[ok], np.float32(1.))
    out    = np.full(TGT_NTOT, np.nan, np.float32)
    m      = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out.reshape(TGT_SHAPE)


def _seas_mean(arrs):
    """Seasonal mean of a list of 2-D arrays; None if list is empty."""
    valid = [a for a in arrs if a is not None]
    if not valid:
        return None
    return np.nanmean(np.stack(valid, 0), axis=0).astype(np.float32)


# ============================================================
# PHASE 0 — IDENTIFY HIGH-σ MEMBERS
# ============================================================

def identify_high_members(csv_path, n_sigma=2):
    """
    Read scalar CSV, compute per-member siaf_global, then return members
    with siaf_global > MMM + n_sigma * σ (across ALL individual members).

    Returns
    -------
    high : list of (model, member) tuples
    mmm  : float
    sigma: float
    thr  : float
    """
    df = pd.read_csv(csv_path)

    # Deduplicate: one scalar per (model, member)
    pm = (df.groupby(['model', 'member'])['siaf_global']
            .mean()
            .reset_index())

    vals  = pm['siaf_global'].values
    mmm   = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1))   # sample std, consistent with prior analysis
    thr   = mmm + n_sigma * sigma

    high_df  = pm[pm['siaf_global'] > thr]
    high     = list(zip(high_df['model'], high_df['member']))

    print(f'\nPhase 0  –  threshold identification  (n_sigma={n_sigma})')
    print(f'  N members in CSV : {len(pm)}')
    print(f'  MMM              : {mmm:.4f} W m⁻² K⁻¹')
    print(f'  σ (sample)       : {sigma:.4f} W m⁻² K⁻¹')
    print(f'  Threshold (+{n_sigma}σ) : {thr:.4f} W m⁻² K⁻¹')
    print(f'  Members above thr: {len(high)}')

    # Also print the 1σ threshold for reference
    thr1 = mmm + sigma
    n1   = int((pm['siaf_global'] > thr1).sum())
    print(f'  (reference +1σ = {thr1:.4f},  {n1} members would qualify)')

    if not high:
        print('\n  WARNING: 0 members above +2σ threshold.')
        print('  Falling back to +1σ threshold.')
        high_df = pm[pm['siaf_global'] > thr1]
        high    = list(zip(high_df['model'], high_df['member']))
        thr     = thr1
        n_sigma = 1

    for mdl, mem in sorted(high):
        val = float(pm.loc[(pm['model']==mdl) & (pm['member']==mem),
                            'siaf_global'].iloc[0])
        print(f'    {mdl:<25}  {mem:<15}  {val:.4f}')

    return high, mmm, sigma, thr


# ============================================================
# PHASE 1 — PER-MEMBER WORKER
# ============================================================

def _safe_name(s):
    return s.replace('/', '_').replace(' ', '_')


def process_member_spatial(args):
    """
    Compute seasonal-mean IS/RS/SIAF for ONE ensemble member on its native
    grid, scatter-avg to 1°×1°, and save
    CACHE_DIR/spatial_hisig_{model}_{member}.npz.

    Args: (model, member)  — passed as a single tuple for ProcessPoolExecutor.
    Returns: (model, member, success_bool)
    """
    model, member = args
    tag   = f'{model}_{_safe_name(member)}'
    cache = os.path.join(CACHE_DIR, f'spatial_hisig_{tag}.npz')

    if not FORCE_RECOMPUTE and os.path.exists(cache):
        print(f'  [{tag}] cache hit', flush=True)
        return model, member, True

    # Global annual T for this member; fall back to primary if needed
    prim  = MODEL_INFO[model]['member']
    T     = compute_T_annual_global(model, member, TREND_YEARS)
    if T is None or len(T) < 10:
        T = compute_T_annual_global(model, prim, TREND_YEARS)
    if T is None or len(T) < 10:
        print(f'  [{tag}] insufficient T_annual – skip', flush=True)
        return model, member, False

    try:
        IS_m, ctx = compute_member_IS_native(model, member, T)
    except Exception as exc:
        print(f'  [{tag}] IS error: {exc}', flush=True)
        return model, member, False
    if IS_m is None:
        print(f'  [{tag}] IS returned None – skip', flush=True)
        return model, member, False

    try:
        RS_m, _ = compute_model_rs_native(model, member, ctx)
    except Exception:
        RS_m = {}

    lat2d = ctx['lat2d']
    lon2d = ctx['lon2d']

    tot_l, ext_l, drk_l, rs_l, siaf_l = [], [], [], [], []
    for mn in MONTH_NUMS:
        d = IS_m.get(mn)
        r = RS_m.get(mn)
        if d is not None:
            tot_l.append(d['tot'])
            ext_l.append(d['ext'])
            drk_l.append(d['drk'])
        if r is not None:
            rs_l.append(r)
        if d is not None and r is not None and r.shape == d['tot'].shape:
            siaf_l.append((d['tot'] * r).astype(np.float32))

    is_tot = _seas_mean(tot_l)
    if is_tot is None:
        print(f'  [{tag}] no valid months – skip', flush=True)
        return model, member, False

    def _ra(arr):
        if arr is None:
            return np.full(TGT_SHAPE, np.nan, np.float32)
        return _scatter_avg(lat2d, lon2d, arr)

    np.savez_compressed(
        cache,
        is_tot = _ra(is_tot),
        is_ext = _ra(_seas_mean(ext_l)),
        is_drk = _ra(_seas_mean(drk_l)),
        rs     = _ra(_seas_mean(rs_l)),
        siaf   = _ra(_seas_mean(siaf_l)),
    )
    print(f'  [{tag}] saved', flush=True)
    del IS_m, ctx, RS_m
    gc.collect()
    return model, member, True


# ============================================================
# PHASE 2 — LOAD + AGGREGATE
# ============================================================

FIELDS = ('is_tot', 'is_ext', 'is_drk', 'rs', 'siaf')


def load_and_stack(high_members):
    """
    Load per-member .npz files for the high-σ subset.
    Returns stk3d (N_members × H × W per field) and mean_.
    """
    stacks  = {f: [] for f in FIELDS}
    loaded  = []

    for model, member in high_members:
        tag   = f'{model}_{_safe_name(member)}'
        path  = os.path.join(CACHE_DIR, f'spatial_hisig_{tag}.npz')
        if not os.path.exists(path):
            print(f'  [load] missing: {tag}')
            continue
        with np.load(path) as d:
            for f in FIELDS:
                arr = (d[f].astype(np.float32) if f in d
                       else np.full(TGT_SHAPE, np.nan, np.float32))
                # Inpaint coarse-grid NaN gaps in each member before stacking
                # so that nanmean/nanstd are computed from complete fields
                stacks[f].append(_inpaint_gaps(arr))
        loaded.append((model, member))

    if not loaded:
        raise RuntimeError(f'No .npz files found in {CACHE_DIR}')
    print(f'  Loaded {len(loaded)} member(s)')

    stk3d = {f: np.stack(stacks[f], 0) for f in FIELDS}   # (N, H, W)
    mean_ = {f: np.nanmean(stk3d[f], 0).astype(np.float32) for f in FIELDS}

    # Derive SIAF components from IS × RS
    for comp in ('ext', 'drk'):
        k = f'siaf_{comp}'
        stk3d[k] = stk3d[f'is_{comp}'] * stk3d['rs']
        mean_[k]  = np.nanmean(stk3d[k], 0).astype(np.float32)
    stk3d['siaf_tot'] = stk3d['siaf']
    mean_['siaf_tot']  = mean_['siaf']

    return mean_, stk3d, loaded


# ============================================================
# PLOTTING
# ============================================================

SIAF_ROWS = [
    ('siaf_ext', r'SIAF$_{\rm ext}$', '#3A6BAF'),
    ('siaf_drk', r'SIAF$_{\rm drk}$', '#D95F1A'),
    ('siaf_tot', r'SIAF$_{\rm tot}$', '#2B2B8F'),
]


def _circ_boundary():
    t = np.linspace(0, 2 * np.pi, 200)
    v = np.column_stack([np.sin(t), np.cos(t)])
    return mpath.Path(v * 0.5 + 0.5)


def _siaf_panel(ax, field_2d, stk, vmin, vmax, cmap='YlOrRd', smooth_px=0.):
    """
    Draw one SIAF panel.
    Rasterizes the 1°×1° lat-lon field onto a 600×600 polar-stereo pixel
    grid, then uses imshow — no pcolormesh cell-edge artefacts (rings/spokes).
    Pass stk=None to suppress agreement stippling.
    smooth_px>0 applies PS-pixel-space smoothing (for spread/std panels).
    """
    _ensure_ps_raster_coords()
    if HAS_CARTOPY:
        pc  = ccrs.PlateCarree()
        ext = [-_PS_EXTENT, _PS_EXTENT, -_PS_EXTENT, _PS_EXTENT]
        ax.set_boundary(_circ_boundary(), transform=ax.transAxes)
        ax.set_extent([-180, 180, 55, 90], crs=pc)

        data_ps = np.ma.masked_invalid(_rasterize(field_2d, smooth_px=smooth_px))
        pcm = ax.imshow(data_ps, origin='lower', extent=ext,
                        vmin=vmin, vmax=vmax, cmap=cmap,
                        interpolation='nearest', rasterized=True, zorder=1)

        ax.add_feature(cfeature.LAND,
                       facecolor='#CCCCCC', edgecolor='none', zorder=3)
        ax.add_feature(cfeature.COASTLINE,
                       linewidth=0.45, edgecolor='#444444', zorder=4)
        gl = ax.gridlines(crs=pc, linewidth=0.3,
                          color='#999999', alpha=0.5, draw_labels=False,
                          zorder=2)
        gl.ylocator = mticker.FixedLocator([60, 70, 80])
        gl.xlocator = mticker.FixedLocator(list(range(-180, 181, 30)))

        if stk is not None:
            n_fin    = np.sum(np.isfinite(stk), axis=0)
            n_pos    = np.sum(stk > 0, axis=0)
            frac     = np.where(n_fin > 0, n_pos / np.maximum(n_fin, 1), np.nan)
            agree_ll = (((frac >= AGREE_THRESH) | (frac <= 1. - AGREE_THRESH))
                        & np.isfinite(field_2d))
            # Map agreement mask onto PS raster and scatter in projected metres
            agree_ps = np.zeros((_PS_N, _PS_N), bool)
            v = _PS_VALID
            agree_ps[v] = agree_ll[_PS_LAT_IDX[v], _PS_LON_IDX[v]]
            ys, xs = np.where(agree_ps & ~np.ma.getmaskarray(data_ps))
            step   = max(1, len(ys) // 3500)
            if len(ys):
                ax.scatter(_PS_X2D[ys[::step], xs[::step]],
                           _PS_Y2D[ys[::step], xs[::step]],
                           s=0.6, color='#111111', alpha=0.40,
                           linewidths=0, zorder=5)
    else:
        pcm = ax.pcolormesh(TGT_LON2D, TGT_LAT2D, field_2d,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            shading='auto', rasterized=True)
        ax.set_xlim(-180, 180)
        ax.set_ylim(55, 90)
    return pcm


def plot_highsig(mean_, stk3d, loaded, mmm_scalar, sigma_scalar, thr,
                 out_path):
    """3-row × 2-col plot of high-σ member mean and 2σ spread."""
    sigma2 = {k: _smooth_std((2. * np.nanstd(stk3d[k], axis=0)).astype(np.float32))
              for k, _, _ in SIAF_ROWS}

    proj = ccrs.NorthPolarStereo(central_longitude=-45) if HAS_CARTOPY else None
    kw   = {'projection': proj} if HAS_CARTOPY else {}

    fig, axes = plt.subplots(3, 2, figsize=(13, 19), subplot_kw=kw)
    fig.patch.set_facecolor('white')

    n = len(loaded)
    members_str = ',  '.join(f'{m}/{mb}' for m, mb in sorted(loaded))

    fig.suptitle(
        r'CMIP6 High-Sensitivity Ensemble  $\cdot$  Spatial SIAF  $\cdot$  '
        r'p < 0.05  $\cdot$  Apr–Sep' + '\n'
        r'SIAF$_{\rm global}$ > MMM + 2$\sigma$'
        f'  (threshold = {thr:.3f} W m$^{{-2}}$ K$^{{-1}}$,  '
        f'n = {n} members):\n'
        f'{members_str}',
        fontsize=9.0, fontweight='bold', y=0.999)

    axes[0, 0].set_title(f'High-sensitivity mean\n({n} members)',
                          fontsize=11, fontweight='bold', pad=8)
    axes[0, 1].set_title(r'2$\sigma$ inter-member spread',
                          fontsize=11, fontweight='bold', pad=8)

    pcm_m = pcm_s = None
    for ax_m, ax_s, (key, label, color) in zip(axes[:, 0], axes[:, 1],
                                                SIAF_ROWS):
        pcm_m = _siaf_panel(ax_m, mean_[key],   stk3d[key], 0., 30., 'YlOrRd')
        pcm_s = _siaf_panel(ax_s,  sigma2[key],  None,       0., 20., 'Blues',
                            smooth_px=5.)
        ax_m.text(-0.08, 0.5, label, transform=ax_m.transAxes,
                  fontsize=13, fontweight='bold', color=color,
                  ha='center', va='center', rotation=90)

    cb1 = fig.colorbar(pcm_m, ax=axes[:, 0].tolist(),
                       orientation='vertical', shrink=0.85,
                       fraction=0.03, pad=0.03, aspect=35)
    cb1.set_label(r'W m$^{-2}$ K$^{-1}$', fontsize=10)
    cb1.ax.tick_params(labelsize=8)

    cb2 = fig.colorbar(pcm_s, ax=axes[:, 1].tolist(),
                       orientation='vertical', shrink=0.85,
                       fraction=0.03, pad=0.03, aspect=35)
    cb2.set_label(r'W m$^{-2}$ K$^{-1}$', fontsize=10)
    cb2.ax.tick_params(labelsize=8)

    fig.subplots_adjust(hspace=0.04, wspace=0.08,
                        top=0.96, bottom=0.02,
                        left=0.10, right=0.88)
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_path}')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print('=' * 70)
    print('CMIP6 SIAF SPATIAL  –  HIGH-σ ENSEMBLE  (members > MMM + 2σ)')
    print(f'  Target grid : {TGT_SHAPE[0]} rows × {TGT_SHAPE[1]} cols (1°×1°, 56–90 N)')
    print(f'  N_WORKERS   : {N_WORKERS}')
    print(f'  CACHE_DIR   : {CACHE_DIR}')
    print(f'  CSV         : {CSV_PATH}')
    print(f'  FORCE       : {FORCE_RECOMPUTE}')
    print(f'  cartopy     : {HAS_CARTOPY}')
    print('=' * 70)

    # ── Phase 0: identify qualifying members ──────────────────────────────
    high_members, mmm_s, sigma_s, thr = identify_high_members(CSV_PATH,
                                                               n_sigma=2)
    if not high_members:
        raise SystemExit('No qualifying members found – check CSV and threshold.')

    # ── Phase 1: per-member spatial computation ───────────────────────────
    print(f'\nPhase 1  –  computing spatial fields for {len(high_members)} members')
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_member_spatial, args): args
                   for args in high_members}
        for fut in as_completed(futures):
            m, mb, ok = fut.result()
            tag = f'{m}/{mb}'
            print(f'  {"✓" if ok else "✗"} {tag}', flush=True)

    # ── Phase 2: aggregate ─────────────────────────────────────────────────
    print('\nPhase 2  –  aggregating member fields')
    mean_, stk3d, loaded = load_and_stack(high_members)

    for key in ('siaf_ext', 'siaf_drk', 'siaf_tot'):
        flat = mean_[key][np.isfinite(mean_[key])]
        if flat.size:
            print(f'  {key:10s}  Arctic mean = {flat.mean():.4f}  '
                  f'p2 = {np.percentile(flat,2):.4f}  '
                  f'p98 = {np.percentile(flat,98):.4f}')

    # ── Phase 3: plot ──────────────────────────────────────────────────────
    print('\nPhase 3  –  plotting')
    plot_highsig(mean_, stk3d, loaded, mmm_s, sigma_s, thr, OUT_PATH)

    print(f'\nDone  ->  {OUT_PATH}')
