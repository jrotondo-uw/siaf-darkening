#!/usr/bin/env python
"""
cmip6_siaf_spatial_cesm2_range.py
───────────────────────────────────
Spatial SIAF for the CESM2 ensemble showing its full range:

  Col 1: ensemble mean – 2σ  (lower bound)
  Col 2: ensemble mean        (central estimate)
  Col 3: ensemble mean + 2σ  (upper bound)

Rows: SIAF_ext / SIAF_drk / SIAF_tot  →  3 rows × 3 columns

σ is the per-pixel inter-member std across all CESM2 members.

Two-phase execution
───────────────────
  Phase 1 – per-member spatial computation, cached to
             CACHE_DIR/spatial_cesm2_{member}.npz
  Phase 2 – load cache, compute pixel-wise mean ± 2σ, plot.

Output: PLOT_DIR/cmip6_siaf_spatial_cesm2_range.png
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
TARGET_MODEL = 'CESM2'   # must match key in MODEL_INFO

PLOT_DIR  = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/'
CSV_PATH  = os.path.join(PLOT_DIR, 'siaf_cmip6_pistone_summary.csv')
CACHE_DIR = os.path.join(PLOT_DIR, 'spatial_cache_cesm2')
os.makedirs(CACHE_DIR, exist_ok=True)

OUT_PATH  = os.path.join(PLOT_DIR, 'cmip6_siaf_spatial_cesm2_range.png')

N_WORKERS       = 4
FORCE_RECOMPUTE = False
AGREE_THRESH    = 0.70

# ── Common target grid ────────────────────────────────────────────────────────
_LATS = np.arange(56.5, 90., 1.0)
_LONS = np.arange(-179.5, 180., 1.0)
TGT_LON2D, TGT_LAT2D = np.meshgrid(_LONS, _LATS)
TGT_SHAPE = TGT_LAT2D.shape      # (33, 360)
TGT_NTOT  = int(np.prod(TGT_SHAPE))

_TGT_XYZ  = latlon_to_xyz(TGT_LAT2D.ravel(),
                            TGT_LON2D.ravel()).astype(np.float32)
_TGT_TREE = cKDTree(_TGT_XYZ)

# ── PS raster constants ───────────────────────────────────────────────────────
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


def _smooth_std(field_2d, sigma_km=80., lat_1d=None):
    """
    NaN-aware Gaussian smooth of a std/spread field, with the longitude-
    direction kernel width scaled by 1/cos(lat).

    Near the pole, 1 degree of longitude shrinks to near-zero physical
    width, so a constant-degree smoothing kernel is far too narrow there —
    leftover grid-scale noise in the longitude direction gets geometrically
    stretched into long radial spokes once rasterized to polar stereo.
    Scaling the kernel by 1/cos(lat) keeps the smoothing scale ~constant
    in physical km at every latitude.
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
    """Fill small NaN gaps from coarse model grids by local averaging."""
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


def _smooth_ps_pixels(ps_2d, sigma_px):
    """
    NaN-aware Gaussian smooth directly in PS-pixel space.
    Smoothing in lat-lon space before rasterizing can't fully kill pole-
    region spokes: near the projection's centre, adjacent PS pixels can
    map to widely-separated longitude indices in the (33,360) grid (since
    longitude wraps almost instantly close to the pole), so the bilinear
    resample step itself reintroduces sharp jumps there. PS-pixel space
    has no such singularity, so smoothing here is the robust fix for
    whatever the lat-lon-space smoothing leaves behind.
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
    v      = _PS_VALID
    lat_c  = np.clip(_PS_LAT_F[v], 0., 32.).astype(np.float64)
    lon_c  = (_PS_LON_F[v] + PAD).astype(np.float64)
    v_interp = map_coordinates(fp, [lat_c, lon_c], order=1, mode='nearest')
    m_interp = map_coordinates(mp, [lat_c, lon_c], order=1, mode='nearest')
    out = np.full((n, n), np.nan, np.float32)
    ok  = v.copy()
    ok[v] = m_interp > 0.5
    out[ok] = (v_interp / np.maximum(m_interp, 1e-6))[m_interp > 0.5].astype(
        np.float32)
    if smooth_px > 0:
        out = _smooth_ps_pixels(out, smooth_px)
    return out


# ============================================================
# REGRID UTILITY
# ============================================================

def _scatter_avg(lat2d, lon2d, field):
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
    valid = [a for a in arrs if a is not None]
    if not valid:
        return None
    return np.nanmean(np.stack(valid, 0), axis=0).astype(np.float32)


# ============================================================
# PHASE 0 — FIND CESM2 MEMBERS
# ============================================================

def find_cesm2_members(csv_path, model_key):
    df  = pd.read_csv(csv_path)
    df  = df[df['model'] == model_key]
    members = sorted(df['member'].unique().tolist())
    print(f'\nPhase 0  –  {model_key} members in CSV: {len(members)}')
    for m in members:
        val = float(df[df['member'] == m]['siaf_global'].mean())
        print(f'    {m:<15}  siaf_global = {val:.4f} W m⁻² K⁻¹')
    return members


# ============================================================
# PHASE 1 — PER-MEMBER WORKER
# ============================================================

def _safe_name(s):
    return s.replace('/', '_').replace(' ', '_')


def process_cesm2_member(member):
    """
    Compute seasonal-mean IS/RS/SIAF for one CESM2 member on its native
    grid, scatter-avg to 1°×1°, and cache to
    CACHE_DIR/spatial_cesm2_{member}.npz.
    Returns (member, success_bool).
    """
    tag   = _safe_name(member)
    cache = os.path.join(CACHE_DIR, f'spatial_cesm2_{tag}.npz')
    model = TARGET_MODEL

    if not FORCE_RECOMPUTE and os.path.exists(cache):
        print(f'  [{model}/{member}] cache hit', flush=True)
        return member, True

    prim = MODEL_INFO[model]['member']
    T    = compute_T_annual_global(model, member, TREND_YEARS)
    if T is None or len(T) < 10:
        T = compute_T_annual_global(model, prim, TREND_YEARS)
    if T is None or len(T) < 10:
        print(f'  [{model}/{member}] insufficient T_annual – skip', flush=True)
        return member, False

    try:
        IS_m, ctx = compute_member_IS_native(model, member, T)
    except Exception as exc:
        print(f'  [{model}/{member}] IS error: {exc}', flush=True)
        return member, False
    if IS_m is None:
        return member, False

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
        return member, False

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
    print(f'  [{model}/{member}] saved', flush=True)
    del IS_m, ctx, RS_m
    gc.collect()
    return member, True


# ============================================================
# PHASE 2 — LOAD + STATISTICS
# ============================================================

FIELDS = ('is_tot', 'is_ext', 'is_drk', 'rs', 'siaf')


def load_and_stats(members):
    """
    Load per-member .npz files, inpaint gaps, derive SIAF components,
    and return pixel-wise mean, mean-2σ, mean+2σ for each SIAF field.
    """
    stacks = {f: [] for f in FIELDS}
    loaded = []

    for member in members:
        tag  = _safe_name(member)
        path = os.path.join(CACHE_DIR, f'spatial_cesm2_{tag}.npz')
        if not os.path.exists(path):
            print(f'  [load] missing: {member}')
            continue
        with np.load(path) as d:
            for f in FIELDS:
                arr = (d[f].astype(np.float32) if f in d
                       else np.full(TGT_SHAPE, np.nan, np.float32))
                stacks[f].append(_inpaint_gaps(arr))
        loaded.append(member)

    if not loaded:
        raise RuntimeError(f'No .npz files found in {CACHE_DIR}')
    print(f'  Loaded {len(loaded)} member(s): {loaded}')

    stk3d = {f: np.stack(stacks[f], 0) for f in FIELDS}   # (N, H, W)

    # Derive SIAF components from IS × RS
    for comp in ('ext', 'drk'):
        k = f'siaf_{comp}'
        stk3d[k] = stk3d[f'is_{comp}'] * stk3d['rs']
    stk3d['siaf_tot'] = stk3d['siaf']

    # Pixel-wise statistics
    stats = {}
    for k in ('siaf_ext', 'siaf_drk', 'siaf_tot'):
        mean_  = np.nanmean(stk3d[k], axis=0).astype(np.float32)
        sig_   = np.nanstd(stk3d[k],  axis=0).astype(np.float32)
        sig_sm = _smooth_std(sig_)
        stats[k] = {
            'mean':  mean_,
            'lo':    (mean_ - 2. * sig_sm).astype(np.float32),
            'hi':    (mean_ + 2. * sig_sm).astype(np.float32),
            'stk':   stk3d[k],
        }
    return stats, loaded


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


def _panel(ax, field_2d, stk, vmin, vmax, cmap='YlOrRd', smooth_px=0.):
    """Draw one panel using PS-rasterized imshow. stk=None skips stippling."""
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


def plot_cesm2_range(stats, loaded, out_path):
    """
    3-row × 3-column figure:
      Rows : SIAF_ext / SIAF_drk / SIAF_tot
      Cols : Mean – 2σ  /  Mean  /  Mean + 2σ
    All panels share the same YlOrRd 0–30 colorbar for direct comparison.
    Mean column also has agreement stippling.
    """
    n = len(loaded)
    proj = ccrs.NorthPolarStereo(central_longitude=-45) if HAS_CARTOPY else None
    kw   = {'projection': proj} if HAS_CARTOPY else {}

    fig, axes = plt.subplots(3, 3, figsize=(18, 19), subplot_kw=kw)
    fig.patch.set_facecolor('white')

    members_str = ',  '.join(loaded)
    fig.suptitle(
        f'CESM2 Ensemble  ·  Spatial SIAF  ·  p < 0.05  ·  Apr–Sep  ·  '
        f'{n} members\n'
        f'σ = per-pixel inter-member std  ·  '
        f'members: {members_str}',
        fontsize=9.5, fontweight='bold', y=0.999)

    COL_TITLES = [r'Mean $-$ 2$\sigma$', 'Mean', r'Mean $+$ 2$\sigma$']
    for j, title in enumerate(COL_TITLES):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold', pad=8)

    vmin, vmax = 0., 30.
    pcm_last = None
    for i, (key, label, color) in enumerate(SIAF_ROWS):
        s = stats[key]
        # lo / mean / hi — clamp lo at 0 (SIAF is non-negative)
        fields = [
            np.where(s['lo']   < 0, 0., s['lo']).astype(np.float32),
            s['mean'],
            s['hi'],
        ]
        stks  = [None, s['stk'], None]      # stipple only on mean column
        smpx  = [5., 0., 5.]                # smooth lo/hi (σ-derived), not mean
        for j, (field, stk, sp) in enumerate(zip(fields, stks, smpx)):
            pcm = _panel(axes[i, j], field, stk, vmin, vmax, 'YlOrRd',
                        smooth_px=sp)
            pcm_last = pcm

        # Row label on the left of column 0
        axes[i, 0].text(-0.10, 0.5, label,
                         transform=axes[i, 0].transAxes,
                         fontsize=13, fontweight='bold', color=color,
                         ha='center', va='center', rotation=90)

    # Single shared colorbar on the right
    cb = fig.colorbar(pcm_last, ax=axes[:, :].ravel().tolist(),
                      orientation='vertical', shrink=0.80,
                      fraction=0.015, pad=0.02, aspect=40)
    cb.set_label(r'W m$^{-2}$ K$^{-1}$', fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.subplots_adjust(hspace=0.04, wspace=0.06,
                        top=0.963, bottom=0.02,
                        left=0.09, right=0.90)
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  -> {out_path}')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    if TARGET_MODEL not in MODEL_INFO:
        raise SystemExit(f'ERROR: {TARGET_MODEL} not found in MODEL_INFO. '
                         f'Available: {sorted(MODEL_INFO.keys())}')

    print('=' * 70)
    print(f'CESM2 SPATIAL SIAF RANGE  —  mean ± 2σ  (p < 0.05, Apr–Sep)')
    print(f'  Target grid : {TGT_SHAPE[0]} × {TGT_SHAPE[1]}  (1°×1°, 56–90 N)')
    print(f'  N_WORKERS   : {N_WORKERS}')
    print(f'  CACHE_DIR   : {CACHE_DIR}')
    print(f'  CSV         : {CSV_PATH}')
    print('=' * 70)

    # ── Phase 0: identify members ─────────────────────────────────────────
    members = find_cesm2_members(CSV_PATH, TARGET_MODEL)
    if not members:
        raise SystemExit(f'No {TARGET_MODEL} members found in {CSV_PATH}')

    # ── Phase 1: per-member computation ──────────────────────────────────
    print(f'\nPhase 1  –  computing spatial fields for {len(members)} members')
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_cesm2_member, m): m for m in members}
        for fut in as_completed(futures):
            member = futures[fut]
            try:
                _, ok = fut.result()
                print(f'  {"✓" if ok else "✗"} {TARGET_MODEL}/{member}',
                      flush=True)
            except Exception as exc:
                print(f'  ✗ {TARGET_MODEL}/{member}  ({exc})', flush=True)

    # ── Phase 2: statistics ────────────────────────────────────────────────
    print('\nPhase 2  –  computing pixel-wise mean ± 2σ')
    stats, loaded = load_and_stats(members)

    for k in ('siaf_ext', 'siaf_drk', 'siaf_tot'):
        m_ = stats[k]['mean']
        s_ = np.nanstd(stats[k]['stk'], axis=0)
        print(f'  {k:10s}  Arctic mean = {np.nanmean(m_):.4f}  '
              f'Arctic 2σ = {np.nanmean(2*s_):.4f}')

    # ── Phase 3: plot ──────────────────────────────────────────────────────
    print('\nPhase 3  –  plotting')
    plot_cesm2_range(stats, loaded, OUT_PATH)

    print(f'\nDone  ->  {OUT_PATH}')
