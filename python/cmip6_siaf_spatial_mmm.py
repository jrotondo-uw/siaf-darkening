#!/usr/bin/env python
"""
cmip6_siaf_spatial_mmm.py
─────────────────────────
Spatial Multi-Model Mean (MMM) of CMIP6 IS / SIAF at p = 0.05.

Each model's IS and RS fields are computed on its own native atmospheric
grid, then scatter-averaged to a common 1°×1° lat-lon target grid
(56–90 N) before the MMM is formed.  Models with curvilinear,
rotated-pole, or regular grids all work — scatter-avg handles any
source resolution without interpolation.

Two-phase execution
───────────────────
  Phase 1 – parallel per-model computation (N_WORKERS processes);
             results cached to CACHE_DIR/spatial_<model>.npz so re-runs
             skip finished models automatically.
  Phase 2 – load cache, stack across models, compute MMM + model-
             agreement fraction, generate 4-panel polar map.

Outputs
───────
  CACHE_DIR/spatial_<model>.npz         per-model member-mean 1°×1° fields
  PLOT_DIR/cmip6_siaf_spatial_mmm.png   4-panel MMM spatial map
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
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

# Import all computation machinery from the main CMIP6 script
from cmip6_siaf_calculation import (
    MODEL_INFO, MONTH_NUMS, TREND_YEARS,
    compute_T_annual_global,
    compute_member_IS_native,
    compute_model_rs_native,
    find_all_members,
    latlon_to_xyz,
)

# ============================================================
# CONFIG
# ============================================================
PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
CACHE_DIR = os.path.join(PLOT_DIR, 'spatial_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

N_WORKERS       = 4
FORCE_RECOMPUTE = False   # True → redo even if .npz exists
AGREE_THRESH    = 0.70    # dot-stipple where >= this fraction agree on sign

# Models excluded from the spatial MMM (must match scalar CSV filter)
# EC-Earth3 T255 members (256×512 grid) are filtered from the scalar analysis
SKIP_MODELS = {'EC-Earth3'}

# ── Common target grid: 1°×1° lat-lon, 56.5–89.5 N ──────────────────────────
_LATS = np.arange(56.5, 90., 1.0)    # 33 row centres
_LONS = np.arange(-179.5, 180., 1.0) # 360 col centres
TGT_LON2D, TGT_LAT2D = np.meshgrid(_LONS, _LATS)
TGT_SHAPE = TGT_LAT2D.shape          # (33, 360)
TGT_NTOT  = int(np.prod(TGT_SHAPE))

# ── Polar-stereo raster constants ────────────────────────────────────────────
_PS_N      = 600
_PS_EXTENT = 5.5e6

_PS_VALID   = None
_PS_X2D     = None
_PS_Y2D     = None
_PS_LAT_F   = None
_PS_LON_F   = None
_PS_LAT_IDX = None
_PS_LON_IDX = None


def _ensure_ps_raster_coords():
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
    _PS_LAT_F = (lat_ps - 56.5).astype(np.float32)
    _PS_LON_F = ((lon_ps + 179.5) % 360.).astype(np.float32)
    lat_i = np.round(_PS_LAT_F).astype(np.int32)
    lon_i = np.round(_PS_LON_F).astype(np.int32) % 360
    _PS_VALID   = ((lat_i >= 0) & (lat_i < 33)
                   & (lat_ps >= 55.0) & np.isfinite(lat_ps))
    _PS_LAT_IDX = lat_i
    _PS_LON_IDX = lon_i


def _inpaint_gaps(field_2d, n_iter=4):
    """Fill isolated NaN gaps from coarse model grids (≤ ~4 cells wide)."""
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
        can_fill = nan_m & (sm_w > 0.20)
        out = np.where(can_fill, sm_d / np.maximum(sm_w, 1e-9), out)
    return out.astype(np.float32)


def _rasterize(field_2d):
    """Bilinear interpolation from (33,360) lat-lon field to (600,600) PS grid."""
    from scipy.ndimage import map_coordinates
    _ensure_ps_raster_coords()
    n = _PS_N
    PAD = 3
    filled  = _inpaint_gaps(field_2d)
    field_f = np.where(np.isfinite(filled), filled.astype(np.float64), 0.)
    mask_f  = np.isfinite(filled).astype(np.float64)
    fp = np.concatenate([field_f[:, -PAD:], field_f, field_f[:, :PAD]], axis=1)
    mp = np.concatenate([mask_f[:,  -PAD:], mask_f,  mask_f[:,  :PAD]], axis=1)
    v       = _PS_VALID
    lat_c   = np.clip(_PS_LAT_F[v], 0., 32.).astype(np.float64)
    lon_c   = (_PS_LON_F[v] + PAD).astype(np.float64)
    v_interp = map_coordinates(fp, [lat_c, lon_c], order=1, mode='nearest')
    m_interp = map_coordinates(mp, [lat_c, lon_c], order=1, mode='nearest')
    out = np.full((n, n), np.nan, np.float32)
    ok  = v.copy()
    ok[v] = m_interp > 0.5
    out[ok] = (v_interp / np.maximum(m_interp, 1e-6))[m_interp > 0.5].astype(
        np.float32)
    return out


# KD-tree built once in the parent; inherited by fork'd workers on Linux
_TGT_XYZ  = latlon_to_xyz(TGT_LAT2D.ravel(),
                            TGT_LON2D.ravel()).astype(np.float32)
_TGT_TREE = cKDTree(_TGT_XYZ)


# ============================================================
# REGRID UTILITY
# ============================================================

def _scatter_avg(lat2d, lon2d, field):
    """
    Scatter-average `field` (native grid, same shape as lat2d/lon2d) onto
    the 1°×1° target grid.  Assigns each source cell to its nearest target
    cell, then averages all source cells that land in the same target cell.
    NaN source cells are excluded.
    """
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
# PER-MODEL WORKER
# ============================================================

def process_model_spatial(model):
    """
    Compute seasonal-mean IS/RS/SIAF for all members of `model` on each
    member's native grid, scatter-avg to 1°×1°, take the member mean,
    and save CACHE_DIR/spatial_<model>.npz.

    Returns (model_name, success_bool).
    """
    if model in SKIP_MODELS:
        print(f'  [{model}] in SKIP_MODELS – skipped', flush=True)
        return model, False

    cache = os.path.join(CACHE_DIR, f'spatial_{model}.npz')
    if not FORCE_RECOMPUTE and os.path.exists(cache):
        print(f'  [{model}] cache hit', flush=True)
        return model, True

    inst    = MODEL_INFO[model]['institution']
    prim    = MODEL_INFO[model]['member']
    members = find_all_members(inst, model)
    if not members:
        print(f'  [{model}] no qualifying members', flush=True)
        return model, False

    if prim in members:
        members = [prim] + [m for m in members if m != prim]

    T_prim = compute_T_annual_global(model, prim, TREND_YEARS)

    accum = {k: [] for k in ('is_tot', 'is_ext', 'is_drk', 'rs', 'siaf')}
    n_ok  = 0

    for member in members:
        # Per-year global T (fall back to primary if unavailable)
        T = compute_T_annual_global(model, member, TREND_YEARS)
        if (T is None or len(T) < 10) and T_prim is not None:
            T = T_prim
        if T is None or len(T) < 10:
            print(f'  [{model}/{member}] insufficient T_annual, skip', flush=True)
            continue

        try:
            IS_m, ctx = compute_member_IS_native(model, member, T)
        except Exception as exc:
            print(f'  [{model}/{member}] IS error: {exc}', flush=True)
            continue
        if IS_m is None:
            continue

        try:
            RS_m, _ = compute_model_rs_native(model, member, ctx)
        except Exception:
            RS_m = {}

        lat2d = ctx['lat2d']
        lon2d = ctx['lon2d']

        # Build seasonal lists (Apr–Sep)
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
            continue

        accum['is_tot'].append(_scatter_avg(lat2d, lon2d, is_tot))

        for key, arr in (('is_ext', _seas_mean(ext_l)),
                          ('is_drk', _seas_mean(drk_l)),
                          ('rs',     _seas_mean(rs_l)),
                          ('siaf',   _seas_mean(siaf_l))):
            if arr is not None:
                accum[key].append(_scatter_avg(lat2d, lon2d, arr))

        n_ok += 1
        del IS_m, ctx, RS_m
        gc.collect()

    if not accum['is_tot']:
        print(f'  [{model}] no valid members after processing', flush=True)
        return model, False

    def _mm(lst):
        if not lst:
            return np.full(TGT_SHAPE, np.nan, np.float32)
        return np.nanmean(np.stack(lst, 0), axis=0).astype(np.float32)

    np.savez_compressed(
        cache,
        is_tot = _mm(accum['is_tot']),
        is_ext = _mm(accum['is_ext']),
        is_drk = _mm(accum['is_drk']),
        rs     = _mm(accum['rs']),
        siaf   = _mm(accum['siaf']),
    )
    print(f'  [{model}] saved ({n_ok} member(s))', flush=True)
    return model, True


# ============================================================
# AGGREGATION
# ============================================================

FIELDS = ('is_tot', 'is_ext', 'is_drk', 'rs', 'siaf')


def load_and_stack():
    """
    Load all per-model .npz files and return:
      mmm   : {field: (H, W) float32}          MMM across models
      stk3d : {field: (n_models, H, W) float32} per-model fields for agreement
      models: list of model names that were loaded

    Also derives siaf_ext = is_ext * rs  and  siaf_drk = is_drk * rs
    from the cached components so Phase 1 does not need to be rerun.
    """
    stacks = {f: [] for f in FIELDS}
    models = []

    for mdl in MODEL_INFO:
        if mdl in SKIP_MODELS:
            continue
        path = os.path.join(CACHE_DIR, f'spatial_{mdl}.npz')
        if not os.path.exists(path):
            print(f'  [load] missing: {mdl}')
            continue
        with np.load(path) as d:
            for f in FIELDS:
                arr = (d[f].astype(np.float32) if f in d
                       else np.full(TGT_SHAPE, np.nan, np.float32))
                stacks[f].append(arr)
        models.append(mdl)

    if not models:
        raise RuntimeError(f'No .npz files found in {CACHE_DIR}')
    print(f'  MMM: {len(models)} models loaded')

    stk3d = {f: np.stack(stacks[f], 0) for f in FIELDS}  # (N, H, W)
    mmm   = {f: np.nanmean(stk3d[f], axis=0).astype(np.float32)
             for f in FIELDS}

    # Derive SIAF components from cached IS and RS — no recompute needed
    for comp in ('ext', 'drk'):
        key = f'siaf_{comp}'
        stk3d[key] = stk3d[f'is_{comp}'] * stk3d['rs']   # (N, H, W)
        mmm[key]   = np.nanmean(stk3d[key], axis=0).astype(np.float32)
    stk3d['siaf_tot'] = stk3d['siaf']   # IS_tot * RS already stored
    mmm['siaf_tot']   = mmm['siaf']

    return mmm, stk3d, models


# ============================================================
# PLOTTING
# ============================================================

# Row definitions: (field_key, left_label, label_color)
SIAF_ROWS = [
    ('siaf_ext', r'SIAF$_{\rm ext}$', '#3A6BAF'),   # extent  – blue
    ('siaf_drk', r'SIAF$_{\rm drk}$', '#D95F1A'),   # darkening – orange
    ('siaf_tot', r'SIAF$_{\rm tot}$', '#2B2B8F'),   # total   – dark indigo
]


def _circ_boundary():
    t = np.linspace(0, 2 * np.pi, 200)
    v = np.column_stack([np.sin(t), np.cos(t)])
    return mpath.Path(v * 0.5 + 0.5)


def _siaf_panel(ax, field_2d, stk, vmin, vmax, cmap='YlOrRd'):
    """
    Draw one SIAF panel.
    Rasterizes the 1°×1° lat-lon field onto a 600×600 polar-stereo pixel
    grid, then uses imshow — eliminates pcolormesh ring/spoke artefacts.
    Pass stk=None to suppress stippling (e.g. spread panels).
    """
    _ensure_ps_raster_coords()
    if HAS_CARTOPY:
        pc  = ccrs.PlateCarree()
        ext = [-_PS_EXTENT, _PS_EXTENT, -_PS_EXTENT, _PS_EXTENT]
        ax.set_boundary(_circ_boundary(), transform=ax.transAxes)
        ax.set_extent([-180, 180, 55, 90], crs=pc)

        data_ps = np.ma.masked_invalid(_rasterize(field_2d))
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


def plot_siaf_mmm(mmm, stk3d, n_models, out_path):
    """
    3-row × 2-column figure matching the obs figure layout:
      col 1 – CMIP6 MMM SIAF         (YlOrRd, 0–30 W m⁻² K⁻¹)
      col 2 – 2σ inter-model spread  (Blues,  0–20 W m⁻² K⁻¹)
    Rows: SIAF_ext / SIAF_drk / SIAF_tot
    """
    # Per-pixel 2σ inter-model spread
    sigma2 = {k: (2. * np.nanstd(stk3d[k], axis=0)).astype(np.float32)
              for k, _, _ in SIAF_ROWS}

    proj = ccrs.NorthPolarStereo(central_longitude=-45) if HAS_CARTOPY else None
    kw   = {'projection': proj} if HAS_CARTOPY else {}

    fig, axes = plt.subplots(3, 2, figsize=(13, 19), subplot_kw=kw)
    fig.patch.set_facecolor('white')

    fig.suptitle(
        f'CMIP6 Multi-Model Mean  ·  Spatial SIAF  ·  p < 0.05  ·  '
        f'Apr–Sep  ·  {n_models} models\n'
        r'IS = $-$d$\alpha$/dT  (direct $\alpha$~T OLS, 1982–2014)  ·  '
        f'dots = ≥{int(AGREE_THRESH * 100)}% model agreement on sign',
        fontsize=10.5, fontweight='bold', y=0.997)

    # Column titles on top row only
    axes[0, 0].set_title(f'CMIP6 MMM\n(model RS,  {n_models} models)',
                          fontsize=11, fontweight='bold', pad=8)
    axes[0, 1].set_title(r'2$\sigma$ inter-model spread',
                          fontsize=11, fontweight='bold', pad=8)

    pcm_mmm = pcm_sig = None
    for ax_mmm, ax_sig, (key, label, color) in zip(
            axes[:, 0], axes[:, 1], SIAF_ROWS):

        # MMM column – stippled for sign agreement
        pcm_mmm = _siaf_panel(ax_mmm, mmm[key], stk3d[key], 0., 30., 'YlOrRd')

        # Spread column – no stippling (sign agreement is not meaningful on spread)
        pcm_sig = _siaf_panel(ax_sig, sigma2[key], None, 0., 20., 'Blues')

        # Row label to the left of the MMM column
        ax_mmm.text(-0.08, 0.5, label,
                    transform=ax_mmm.transAxes,
                    fontsize=13, fontweight='bold', color=color,
                    ha='center', va='center', rotation=90)

    # One shared colorbar per column
    cb1 = fig.colorbar(pcm_mmm, ax=axes[:, 0].tolist(),
                       orientation='vertical', shrink=0.85,
                       fraction=0.03, pad=0.03, aspect=35)
    cb1.set_label(r'W m$^{-2}$ K$^{-1}$', fontsize=10)
    cb1.ax.tick_params(labelsize=8)

    cb2 = fig.colorbar(pcm_sig, ax=axes[:, 1].tolist(),
                       orientation='vertical', shrink=0.85,
                       fraction=0.03, pad=0.03, aspect=35)
    cb2.set_label(r'W m$^{-2}$ K$^{-1}$', fontsize=10)
    cb2.ax.tick_params(labelsize=8)

    fig.subplots_adjust(hspace=0.04, wspace=0.08,
                        top=0.965, bottom=0.02,
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
    print('CMIP6 SIAF SPATIAL MMM  —  IS = −dα/dT  (direct OLS, p < 0.05)')
    print(f'  Target grid : {TGT_SHAPE[0]} rows × {TGT_SHAPE[1]} cols  '
          f'(1°×1°, 56–90 N)')
    print(f'  N_WORKERS   : {N_WORKERS}')
    print(f'  CACHE_DIR   : {CACHE_DIR}')
    print(f'  FORCE       : {FORCE_RECOMPUTE}')
    print(f'  cartopy     : {HAS_CARTOPY}')
    print('=' * 70)

    # ── Phase 1: per-model computation (parallel) ──────────────────────────
    print('\nPhase 1  –  per-model native → 1°×1° regrid')
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_model_spatial, m): m
                   for m in MODEL_INFO}
        for fut in as_completed(futures):
            mdl = futures[fut]
            try:
                _, ok = fut.result()
                print(f'  {"v" if ok else "x"} {mdl}', flush=True)
            except Exception as exc:
                print(f'  x {mdl}  ({exc})', flush=True)

    # ── Phase 2: aggregate MMM ─────────────────────────────────────────────
    print('\nPhase 2  –  computing MMM')
    mmm, stk3d, models = load_and_stack()

    for key in ('is_tot', 'is_ext', 'is_drk', 'siaf'):
        flat = mmm[key][np.isfinite(mmm[key])]
        if flat.size:
            print(f'  {key:8s}  mean={flat.mean():.5f}  '
                  f'p2={np.percentile(flat, 2):.5f}  '
                  f'p98={np.percentile(flat, 98):.5f}')

    # ── Phase 3: plot ──────────────────────────────────────────────────────
    print('\nPhase 3  –  plotting')
    out_png = os.path.join(PLOT_DIR, 'cmip6_siaf_spatial_mmm.png')
    plot_siaf_mmm(mmm, stk3d, len(models), out_png)

    print(f'\nDone  ->  {out_png}')
