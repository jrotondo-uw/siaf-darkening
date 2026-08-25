"""
compute_alpha_eff_sh_v13.py — daily-pair preprocessing for APP-x

Reproduces Siddiqui & Horvat (2026) Eq. 4 effective albedo decomposition.

Critical fix vs v11
-------------------
APP-X DAILY-MEAN CONSTRUCTION matches Ali's Resampling_Observations.ipynb
(lines 376-403):

  1. APP-x has 2 satellite passes per day (AM ascending + PM descending)
  2. Keep ONLY days with exactly 2 passes (drop single-pass days)
  3. For those days, keep ONLY pixels where BOTH passes have finite data
     (intersection mask via coarsen + .all())
  4. Daily mean = average of the 2 valid passes
  5. Monthly mean = mean of daily means
  6. Apply value-range QC ([0,1] for albedo)

v11's approach was simpler: sum/count over all daily files for the month,
which included single-pass-only pixels (potentially biased values from
e.g. low-sun-angle passes). The difference matters for marginal pixels
where only one of the two daily passes is valid — concentrated in the
marginal ice zone where Q'sw and α'i are most variable.

If filenames don't expose a 2-pass-per-day structure (single file per
day), the loader falls back to the v11 method automatically.

Retained from v11
-----------------
* Arctic OCEAN mask (lat≥66 AND finite SIC, matching paper line 144-146)
* wtmean_paper: NaN→0 in numerator, full ocean area in denominator
* Ice-masked αi (SIC > 0.15)
* Eq. 4 pure terms (no Fig 2c,f bundling)
* Linear polyfit + secant slope (paper line 815: degree=1 for obs)
* 3-year running mean smoothing before fit
"""

import glob, os, re, warnings
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as NC4Dataset
from scipy.spatial import cKDTree
from pyproj import Transformer

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── directories ───────────────────────────────────────────────────────────────
DIR_APPX  = '/glade/derecho/scratch/jrotondo/APP-x_observations/'
DIR_SIC   = '/glade/derecho/scratch/jrotondo/SIC_data/'
DIR_CLARA = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'

# ── constants ─────────────────────────────────────────────────────────────────
LAT_MIN    = 66.0
ALPHA_W    = 0.07
SIC_THRESH = 0.15
REF_START  = 1982
REF_END    = 2002
MONTHS     = [5, 6, 7, 8]
OCEAN_MASK_YEAR  = 2002
OCEAN_MASK_MONTH = 6
POLY_DEGREE = 1   # paper line 815: degree = 1 for obs

PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# ── helpers ───────────────────────────────────────────────────────────────────
def latlon_to_xyz(lat, lon):
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    return np.column_stack([
        np.cos(phi)*np.cos(lam),
        np.cos(phi)*np.sin(lam),
        np.sin(phi),
    ])


def wtmean_paper(arr, cos_wt, ocean_mask):
    w = cos_wt
    num_mask = ocean_mask & np.isfinite(arr) & np.isfinite(w)
    num = np.sum(arr[num_mask] * w[num_mask])
    den_mask = ocean_mask & np.isfinite(w)
    den = np.sum(w[den_mask])
    return float(num / den) if den > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0 — grids
# ══════════════════════════════════════════════════════════════════════════════

sample_appx = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[0]
with NC4Dataset(sample_appx, 'r') as ds:
    appx_lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
    appx_lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
if appx_lat_raw.ndim == 1:
    APPX_LON2D, APPX_LAT2D = np.meshgrid(appx_lon_raw, appx_lat_raw)
else:
    APPX_LAT2D, APPX_LON2D = appx_lat_raw, appx_lon_raw
APPX_SHAPE  = APPX_LAT2D.shape
N_APPX      = APPX_LAT2D.size
APPX_COS    = np.maximum(np.cos(np.deg2rad(APPX_LAT2D)).ravel(), 0.)
ARCTIC_APPX = APPX_LAT2D.ravel() >= LAT_MIN

sample_sic = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_*.nc'))[0]
with xr.open_dataset(sample_sic) as ds:
    sic_x = ds['x'].values
    sic_y = ds['y'].values
sic_xx, sic_yy = np.meshgrid(sic_x, sic_y)
_tfm = Transformer.from_crs(PSN25_PROJ4, 'EPSG:4326', always_xy=True)
PSN_LON2D, PSN_LAT2D = _tfm.transform(sic_xx, sic_yy)
PSN_SHAPE = PSN_LAT2D.shape
PSN_DX = float(np.abs(np.diff(sic_x).mean()))
PSN_DY = float(np.abs(np.diff(sic_y).mean()))
PSN_CELL_AREA_M2 = PSN_DX * PSN_DY
print(f"PSN25 cell area: {PSN_CELL_AREA_M2:.3e} m²")

PSN_TREE = cKDTree(latlon_to_xyz(PSN_LAT2D.ravel(), PSN_LON2D.ravel()))
_, IDX_PSN_TO_APPX = PSN_TREE.query(
    latlon_to_xyz(APPX_LAT2D.ravel(), APPX_LON2D.ravel()), workers=-1)
def regrid_psn_appx(arr2d):
    return arr2d.ravel()[IDX_PSN_TO_APPX]

sample_clara = sorted(glob.glob(f'{DIR_CLARA}/*.nc'))[0]
with xr.open_dataset(sample_clara) as ds:
    cl_lat = ds['lat'].values
    cl_lon = ds['lon'].values
if cl_lat.ndim == 1:
    CLARA_LON2D, CLARA_LAT2D = np.meshgrid(cl_lon, cl_lat)
else:
    CLARA_LAT2D, CLARA_LON2D = cl_lat, cl_lon
CLARA_SHAPE  = CLARA_LAT2D.shape
N_CLARA      = CLARA_LAT2D.size
CLARA_COS    = np.maximum(np.cos(np.deg2rad(CLARA_LAT2D)).ravel(), 0.)
ARCTIC_CLARA = CLARA_LAT2D.ravel() >= LAT_MIN
_, IDX_PSN_TO_CLARA = PSN_TREE.query(
    latlon_to_xyz(CLARA_LAT2D.ravel(), CLARA_LON2D.ravel()), workers=-1)
def regrid_psn_clara(arr2d):
    return arr2d.ravel()[IDX_PSN_TO_CLARA]
APPX_TREE = cKDTree(latlon_to_xyz(APPX_LAT2D.ravel(), APPX_LON2D.ravel()))
_, IDX_APPX_TO_CLARA = APPX_TREE.query(
    latlon_to_xyz(CLARA_LAT2D.ravel(), CLARA_LON2D.ravel()), workers=-1)
def regrid_appx_clara(arr1d):
    return arr1d[IDX_APPX_TO_CLARA]

print(f"APP-x grid : {APPX_SHAPE}  Arctic pixels: {ARCTIC_APPX.sum()}")
print(f"CLARA grid : {CLARA_SHAPE}  Arctic pixels: {ARCTIC_CLARA.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 0b — Arctic OCEAN masks
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Arctic ocean masks from reference SIC field …")

def _load_ref_sic_for_mask():
    flist = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{OCEAN_MASK_YEAR}*.nc'))
    for f in flist:
        try:
            with xr.open_dataset(f) as ds:
                t = pd.Timestamp(ds['time'].values[0])
                if t.month != OCEAN_MASK_MONTH:
                    continue
                raw = ds['cdr_seaice_conc'].squeeze().values.astype(float).ravel()
                if np.nanmax(raw) > 1.5:
                    raw /= 100.
                return raw
        except Exception:
            continue
    return None

_ref_sic = _load_ref_sic_for_mask()
if _ref_sic is None:
    raise RuntimeError("Could not load reference SIC for ocean mask")
_ocean_psn   = np.isfinite(_ref_sic)
_ocean_appx  = regrid_psn_appx (_ocean_psn.astype(np.float32)).astype(bool)
_ocean_clara = regrid_psn_clara(_ocean_psn.astype(np.float32)).astype(bool)
OCEAN_APPX  = ARCTIC_APPX  & _ocean_appx
OCEAN_CLARA = ARCTIC_CLARA & _ocean_clara
print(f"  APP-x Arctic-ocean pixels: {OCEAN_APPX.sum()}")
print(f"  CLARA Arctic-ocean pixels: {OCEAN_CLARA.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — APP-x loaders (daily-pair + fallback)
# ══════════════════════════════════════════════════════════════════════════════

VAR_SWDOWN = 'cdr_surface_downwelling_shortwave_flux'

# Polar-APP-X v02r00 filename anatomy (verified on jrotondo's tree):
#     Polar-APP-X_v02r00_Nhem_0400_d20220102_c20220105.nc
#                               ^^^^         ^^^^^^^^
#                               pass-HHMM    creation date
# The pass-HHMM (4 digits) appears right BEFORE _d<YYYYMMDD>.
# Some older / local-mirror filenames may put the time AFTER the date as
# _t<HHMM>; we support both.
_FNAME_TIME_BEFORE_DATE = re.compile(r'_(\d{4})_d(\d{8})')
_FNAME_TIME_AFTER_DATE  = re.compile(r'_d(\d{8})_t(\d{4})')
_FNAME_DATE_ONLY        = re.compile(r'_d(\d{8})')

def _parse_appx_time(path):
    base = os.path.basename(path)
    m = _FNAME_TIME_BEFORE_DATE.search(base)
    if m:
        hour_s, date_s = m.group(1), m.group(2)
        return date_s, int(hour_s[:2])
    m = _FNAME_TIME_AFTER_DATE.search(base)
    if m:
        date_s, hour_s = m.group(1), m.group(2)
        return date_s, int(hour_s[:2])
    m = _FNAME_DATE_ONLY.search(base)
    if m:
        return m.group(1), None
    return None, None


def load_appx_monthly_daily_pair(year, mn):
    """
    Ali-style: keep only days with 2 passes, intersect-mask, daily mean, monthly mean.
    """
    search = [
        f'{DIR_APPX}/{year}/Polar-APP-X*d{year}{mn:02d}*.nc',
        f'{DIR_APPX}/Polar-APP-X*d{year}{mn:02d}*.nc',
    ]
    flist = sorted(set(f for p in search for f in glob.glob(p)))
    if not flist:
        return None, None

    # Bucket files by date
    by_date = {}
    for f in flist:
        date_s, hr = _parse_appx_time(f)
        if date_s is None:
            continue
        by_date.setdefault(date_s, []).append((hr if hr is not None else 0, f))
    if not by_date:
        return None, None

    # Choose albedo variable name from first available file
    alb_var = None
    with NC4Dataset(flist[0], 'r') as ds:
        for v in ds.variables:
            if 'albedo' in v.lower() and ds.variables[v].ndim >= 2:
                alb_var = v
                break

    sw_day_sum  = np.zeros(N_APPX, np.float64)
    sw_day_cnt  = np.zeros(N_APPX, np.float64)
    alb_day_sum = np.zeros(N_APPX, np.float64)
    alb_day_cnt = np.zeros(N_APPX, np.float64)

    for date_s, items in by_date.items():
        if len(items) != 2:
            continue
        items.sort(key=lambda x: x[0])
        f_am, f_pm = items[0][1], items[1][1]
        try:
            with NC4Dataset(f_am, 'r') as d_am, NC4Dataset(f_pm, 'r') as d_pm:
                if VAR_SWDOWN in d_am.variables and VAR_SWDOWN in d_pm.variables:
                    sw_a = np.ma.filled(d_am.variables[VAR_SWDOWN][:],
                                        np.nan).ravel().astype(np.float64)
                    sw_p = np.ma.filled(d_pm.variables[VAR_SWDOWN][:],
                                        np.nan).ravel().astype(np.float64)
                    sw_ok = (np.isfinite(sw_a) & (sw_a >= 0.) &
                             np.isfinite(sw_p) & (sw_p >= 0.))
                    sw_day = np.where(sw_ok, 0.5*(sw_a + sw_p), np.nan)
                    finite = np.isfinite(sw_day)
                    sw_day_sum[finite] += sw_day[finite]
                    sw_day_cnt[finite] += 1.

                if alb_var and alb_var in d_am.variables and alb_var in d_pm.variables:
                    fv_a = float(getattr(d_am.variables[alb_var],
                                          '_FillValue', 9999.))
                    fv_p = float(getattr(d_pm.variables[alb_var],
                                          '_FillValue', 9999.))
                    a_a = np.ma.filled(d_am.variables[alb_var][:],
                                       np.nan).ravel().astype(np.float64)
                    a_p = np.ma.filled(d_pm.variables[alb_var][:],
                                       np.nan).ravel().astype(np.float64)
                    a_a[a_a == fv_a] = np.nan
                    a_p[a_p == fv_p] = np.nan
                    a_ok = (np.isfinite(a_a) & (a_a >= 0.) & (a_a <= 1.) &
                            np.isfinite(a_p) & (a_p >= 0.) & (a_p <= 1.))
                    a_day = np.where(a_ok, 0.5*(a_a + a_p), np.nan)
                    finite = np.isfinite(a_day)
                    alb_day_sum[finite] += a_day[finite]
                    alb_day_cnt[finite] += 1.
        except Exception:
            continue

    swdn = np.where(sw_day_cnt  > 0, sw_day_sum  / sw_day_cnt,  np.nan).astype(np.float32)
    alb  = np.where(alb_day_cnt > 0, alb_day_sum / alb_day_cnt, np.nan).astype(np.float32)
    if not np.any(np.isfinite(swdn)):
        return None, None
    return swdn, alb


def load_appx_monthly_fallback(year, mn):
    """v11-style: simple sum/count over all daily files in the month."""
    search = [
        f'{DIR_APPX}/{year}/Polar-APP-X*d{year}{mn:02d}*.nc',
        f'{DIR_APPX}/Polar-APP-X*d{year}{mn:02d}*.nc',
    ]
    flist = sorted(set(f for p in search for f in glob.glob(p)))
    if not flist:
        return None, None

    swdn_sum = np.zeros(N_APPX, np.float64)
    sw_cnt   = np.zeros(N_APPX, np.float64)
    alb_sum  = np.zeros(N_APPX, np.float64)
    alb_cnt  = np.zeros(N_APPX, np.float64)

    for f in flist:
        try:
            with NC4Dataset(f, 'r') as ds:
                if VAR_SWDOWN in ds.variables:
                    sw = (np.ma.filled(ds.variables[VAR_SWDOWN][:], np.nan)
                          .ravel().astype(np.float64))
                    ok = np.isfinite(sw) & (sw >= 0.)
                    swdn_sum[ok] += sw[ok]
                    sw_cnt[ok]   += 1.
                alb_var = next(
                    (v for v in ds.variables
                     if 'albedo' in v.lower() and ds.variables[v].ndim >= 2),
                    None)
                if alb_var:
                    fv  = float(getattr(ds.variables[alb_var], '_FillValue', 9999.))
                    alb = (np.ma.filled(ds.variables[alb_var][:], np.nan)
                           .ravel().astype(np.float64))
                    alb[alb == fv] = np.nan
                    ok = np.isfinite(alb) & (alb >= 0.) & (alb <= 1.)
                    alb_sum[ok] += alb[ok]
                    alb_cnt[ok] += 1.
        except Exception:
            continue

    swdn = np.where(sw_cnt  > 0, swdn_sum / sw_cnt,  np.nan).astype(np.float32)
    alb  = np.where(alb_cnt > 0, alb_sum  / alb_cnt, np.nan).astype(np.float32)
    return (None, None) if not np.any(np.isfinite(swdn)) else (swdn, alb)


def _appx_files_have_pass_structure():
    """If APP-x filenames usually have 2 passes per day, use the pair method."""
    sample = sorted(glob.glob(f'{DIR_APPX}/**/*.nc', recursive=True))[:200]
    by_date = {}
    for f in sample:
        ds_str, hr = _parse_appx_time(f)
        if ds_str is None:
            continue
        by_date.setdefault(ds_str, []).append(hr)
    dates_with_pairs = sum(
        1 for hrs in by_date.values()
        if len(hrs) == 2 and hrs[0] is not None and hrs[1] is not None
           and hrs[0] != hrs[1]
    )
    return len(by_date) > 0 and dates_with_pairs / len(by_date) > 0.5


USE_DAILY_PAIR = _appx_files_have_pass_structure()
print(f"\nAPP-x file structure: "
      f"{'2 passes/day detected — using daily-pair method (Ali-style)' if USE_DAILY_PAIR else 'single file/day — using simple monthly mean (v11-style)'}")

def load_appx_monthly(year, mn):
    if USE_DAILY_PAIR:
        return load_appx_monthly_daily_pair(year, mn)
    return load_appx_monthly_fallback(year, mn)


def load_sic_monthly(year, mn):
    flist = sorted(glob.glob(f'{DIR_SIC}/sic_psn25_{year}*.nc'))
    if not flist:
        return None, None
    n = PSN_LAT2D.size
    sic_sum = np.zeros(n, np.float64)
    cnt     = np.zeros(n, np.float64)
    for f in flist:
        try:
            with xr.open_dataset(f) as ds:
                if pd.Timestamp(ds['time'].values[0]).month != mn:
                    continue
                raw = ds['cdr_seaice_conc'].squeeze().values.astype(float).ravel()
                if np.nanmax(raw) > 1.5:
                    raw /= 100.
                ok = np.isfinite(raw) & (raw >= 0.) & (raw <= 1.)
                sic_sum[ok] += raw[ok]
                cnt[ok]     += 1.
        except Exception:
            continue
    if cnt.sum() == 0:
        return None, None
    sic_psn = np.where(cnt > 0, sic_sum / cnt, np.nan).astype(np.float32)
    sia_mkm2 = float(np.nansum(sic_psn * PSN_CELL_AREA_M2) / 1e12)
    return sic_psn, sia_mkm2


clara_files = sorted(glob.glob(f'{DIR_CLARA}/*.nc'))
clara_year  = {}
for _f in clara_files:
    with xr.open_dataset(_f) as _ds:
        for _y in np.unique(_ds['time'].dt.year.values):
            clara_year.setdefault(int(_y), []).append(_f)

def load_clara_monthly(year, mn):
    flist = clara_year.get(year, [])
    if not flist:
        return None
    try:
        with xr.open_mfdataset(flist, combine='by_coords') as ds:
            snow = ds['black_sky_albedo_snow_mean'] / 100.
            gen  = ds['black_sky_albedo_mean']      / 100.
            alb  = xr.where(np.isfinite(snow), snow, gen)
            sel  = alb.sel(time=(alb['time.month'] == mn))
            if sel.time.size == 0:
                return None
            arr = sel.mean(dim='time').values.astype(np.float32).ravel()
            return np.where((arr >= 0.) & (arr <= 1.), arr, np.nan)
    except Exception:
        return None


print("\nPass 1: loading all monthly fields …")
years = list(range(REF_START, 2023))
fields = {}

for year in years:
    fields[year] = {}
    for mn in MONTHS:
        swdn, alb_composite = load_appx_monthly(year, mn)
        sic_psn, sia_mkm2   = load_sic_monthly(year, mn)
        alb_clara           = load_clara_monthly(year, mn)

        if swdn is None or sic_psn is None:
            continue

        sic_appx  = regrid_psn_appx(sic_psn)
        sic_clara = regrid_psn_clara(sic_psn)
        alb_ice_appx = np.where(sic_appx > SIC_THRESH, alb_composite, np.nan).astype(np.float32)
        alb_ice_clara = None
        if alb_clara is not None:
            alb_ice_clara = np.where(sic_clara > SIC_THRESH, alb_clara, np.nan).astype(np.float32)

        fields[year][mn] = dict(
            swdn=swdn, alb_composite=alb_composite,
            alb_ice_appx=alb_ice_appx, sic_appx=sic_appx,
            alb_ice_clara=alb_ice_clara, sic_clara=sic_clara,
            sia_mkm2=sia_mkm2,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — climatology
# ══════════════════════════════════════════════════════════════════════════════
print("\nPass 2: building per-pixel reference climatologies …")

ref = {}
for mn in MONTHS:
    ref_yrs = [yr for yr in range(REF_START, REF_END + 1)
               if mn in fields.get(yr, {})]
    if not ref_yrs:
        ref[mn] = None
        continue

    def _stack_mean(key, n):
        arrs = [fields[yr][mn][key] for yr in ref_yrs
                if fields[yr][mn].get(key) is not None]
        return (np.nanmean(np.stack(arrs, 0), 0)
                if arrs else np.full(n, np.nan, np.float32))

    ref[mn] = dict(
        qsw          = _stack_mean('swdn',          N_APPX),
        C            = _stack_mean('sic_appx',      N_APPX),
        ai_composite = _stack_mean('alb_composite', N_APPX),
        ai_ice       = _stack_mean('alb_ice_appx',  N_APPX),
        C_cl         = _stack_mean('sic_clara',     N_CLARA),
        ai_ice_cl    = _stack_mean('alb_ice_clara', N_CLARA),
    )
    r = ref[mn]
    r['qsw_scalar']    = wtmean_paper(r['qsw'],    APPX_COS, OCEAN_APPX)
    r['C_scalar']      = wtmean_paper(r['C'],      APPX_COS, OCEAN_APPX)
    r['ai_ice_scalar'] = wtmean_paper(r['ai_ice'], APPX_COS, OCEAN_APPX)
    print(f"  mn={mn}  Qsw_ref={r['qsw_scalar']:.2f}  "
          f"C_ref={r['C_scalar']:.4f}  αi_ice_ref={r['ai_ice_scalar']:.4f}")

ref_sia_vals = [fields[yr][mn]['sia_mkm2']
                for yr in range(REF_START, REF_END+1)
                for mn in MONTHS
                if mn in fields.get(yr, {})
                and np.isfinite(fields[yr][mn].get('sia_mkm2', np.nan))]
SIA_REF_MKM2 = float(np.mean(ref_sia_vals))
print(f"\n  SIA_ref (MJJA 1982-2002) = {SIA_REF_MKM2:.3f}  million km²")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — pure Eq. 4 terms
# ══════════════════════════════════════════════════════════════════════════════
print("\nPass 3: computing pixel-level Eq. 4 terms …")

def _pixel_terms(C, C_r, ai_ice, ai_ice_r, ai_comp, ai_comp_r, qsw, qsw_r):
    dC   = C - C_r
    dai  = ai_ice - ai_ice_r
    ai_ice_r_filled = np.where(np.isfinite(ai_ice_r), ai_ice_r, ALPHA_W)
    aeff_r = C_r * ai_ice_r_filled + (1. - C_r) * ALPHA_W
    dQsw = qsw - qsw_r
    T1  =  dC  * qsw_r * (ALPHA_W - ai_ice_r)
    T2  = -dai * qsw_r * C_r
    T45 =  dQsw * (1. - aeff_r)
    dQ  = qsw * (1. - ai_comp) - qsw_r * (1. - ai_comp_r)
    return T1, T2, T45, dQ


def terms_appx(year, mn):
    r = ref.get(mn); d = fields.get(year, {}).get(mn)
    if r is None or d is None: return None
    T1, T2, T45, dQ = _pixel_terms(
        d['sic_appx'], r['C'], d['alb_ice_appx'], r['ai_ice'],
        d['alb_composite'], r['ai_composite'], d['swdn'], r['qsw'])
    ocean_mask = (OCEAN_APPX
                  & np.isfinite(r['C']) & np.isfinite(r['qsw'])
                  & np.isfinite(d['sic_appx']) & np.isfinite(d['swdn'])
                  & (d['swdn'] > 5.))
    if ocean_mask.sum() == 0: return None
    am = lambda a: wtmean_paper(a, APPX_COS, ocean_mask)
    return dict(T1=am(T1), T2=am(T2), T45=am(T45), dQ=am(dQ),
                sia_mkm2=d['sia_mkm2'])


def terms_clara(year, mn):
    r = ref.get(mn); d = fields.get(year, {}).get(mn)
    if r is None or d is None: return None
    if d.get('alb_ice_clara') is None: return None
    qsw   = regrid_appx_clara(d['swdn'])
    qsw_r = regrid_appx_clara(r['qsw'])
    ai_ice  = d['alb_ice_clara']; ai_ice_r = r['ai_ice_cl']
    ai_comp   = np.where(np.isfinite(ai_ice),   ai_ice,   ALPHA_W)
    ai_comp_r = np.where(np.isfinite(ai_ice_r), ai_ice_r, ALPHA_W)
    T1, T2, T45, dQ = _pixel_terms(
        d['sic_clara'], r['C_cl'], ai_ice, ai_ice_r,
        ai_comp, ai_comp_r, qsw, qsw_r)
    ocean_mask = (OCEAN_CLARA
                  & np.isfinite(r['C_cl']) & np.isfinite(qsw_r)
                  & np.isfinite(d['sic_clara']) & np.isfinite(qsw)
                  & (qsw > 5.))
    if ocean_mask.sum() == 0: return None
    am = lambda a: wtmean_paper(a, CLARA_COS, ocean_mask)
    return dict(T1=am(T1), T2=am(T2), T45=am(T45), dQ=am(dQ),
                sia_mkm2=d['sia_mkm2'])


monthly_appx  = {}
monthly_clara = {}
for year in years:
    for mn in MONTHS:
        ta = terms_appx(year, mn)
        tc = terms_clara(year, mn)
        if ta is not None: monthly_appx[(year, mn)]  = ta
        if tc is not None: monthly_clara[(year, mn)] = tc
print(f"  APP-x : {len(monthly_appx)} records")
print(f"  CLARA : {len(monthly_clara)} records")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — paper-exact effective albedo
# ══════════════════════════════════════════════════════════════════════════════

def smooth_3yr(arr):
    arr = np.asarray(arr, dtype=float)
    return np.convolve(arr, np.ones(3)/3., mode='same')


def mean_poly_slope(x, y, sia_ref, qsw_ref, degree=POLY_DEGREE):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < degree + 2: return np.nan, np.nan
    x_ok, y_ok = x[ok], y[ok]
    y_s = smooth_3yr(y_ok)
    valid = np.isfinite(y_s)
    x_s, y_s = x_ok[valid], y_s[valid]
    if x_s.size < degree + 2: return np.nan, np.nan
    coeffs = np.polyfit(x_s, y_s, deg=degree)
    poly   = np.poly1d(coeffs)
    xmin, xmax = x_s.min(), x_s.max()
    slope = (poly(xmax) - poly(xmin)) / (xmax - xmin)
    alpha = -(sia_ref / qsw_ref) * slope
    return alpha, slope


def annual_mjja(monthly_dict):
    rows = []
    for yr in sorted({k[0] for k in monthly_dict}):
        recs = [monthly_dict[(yr, mn)] for mn in MONTHS if (yr, mn) in monthly_dict]
        if not recs: continue
        row = {k: np.nanmean([r[k] for r in recs])
               for k in ('T1','T2','T45','dQ','sia_mkm2')}
        row['year'] = yr
        rows.append(row)
    return pd.DataFrame(rows)


def run_decomposition(monthly_dict, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    df = annual_mjja(monthly_dict)
    if df.empty:
        print("  No data — skipping"); return None

    df_recent = df[(df['year'] >= 1982) & (df['year'] <= 2020)]
    print(f"\n  Mean anomalies 1982-2020 (compare to paper Fig 2c W/m²):")
    print(f"    Term I   (ice area)        = {df_recent['T1'].mean():+7.3f}  W/m²   (paper: 2.6)")
    print(f"    Term II  (ice albedo)      = {df_recent['T2'].mean():+7.3f}  W/m²   (paper: 3.7)")
    print(f"    Term IV+V (shortwave)      = {df_recent['T45'].mean():+7.3f}  W/m²   (paper: 1.0)")
    sum_w = (df_recent['T1']+df_recent['T2']+df_recent['T45']).mean()
    print(f"    Sum (I + II + IV+V)        = {sum_w:+7.3f}  W/m²   (paper: 7.3)")
    print(f"    Total dQ (absorbed-SW)     = {df_recent['dQ'].mean():+7.3f}  W/m²")

    qsw_ref = np.nanmean([ref[mn]['qsw_scalar'] for mn in MONTHS if ref.get(mn)])
    print(f"\n  Effective albedos: 3-yr smooth → linear polyfit → secant slope")
    print(f"  Qsw_ref = {qsw_ref:.2f}  SIA_ref = {SIA_REF_MKM2:.3f}  degree = {POLY_DEGREE}\n")

    x_sia = df['sia_mkm2'].values
    results = {}
    for col, name, paper_val in [
        ('T1',  'Ice area  ', 0.13),
        ('T2',  'Ice albedo', 0.19),
        ('T45', 'Shortwave ', 0.01),
        ('dQ',  'Sum (dQ)  ', 0.33),
    ]:
        a, slope = mean_poly_slope(x_sia, df[col].values,
                                    SIA_REF_MKM2, qsw_ref, POLY_DEGREE)
        results[col] = a
        print(f"    α_eff {name} = {a:+.4f}   "
              f"(slope = {slope:+.4f} W/m²/mkm²)   paper: {paper_val}")

    sum_alpha = results['T1'] + results['T2'] + results['T45']
    print(f"\n    Sum I + II + IV+V        = {sum_alpha:+.4f}   (paper: 0.33)")

    return dict(label=label,
                a_I=results['T1'], a_II=results['T2'],
                a_IVV=results['T45'], a_tot=results['dQ'])


res_appx  = run_decomposition(monthly_appx,  'APP-x')
res_clara = run_decomposition(monthly_clara, 'CLARA-A3 (CLARA albedo + APP-x SW)')

# ══════════════════════════════════════════════════════════════════════════════
print("\n\n=== FINAL SUMMARY — Fig 2c,f effective albedos ===")
hdr = f"{'Dataset':<40} {'α_I':>8} {'α_II':>8} {'α_IV+V':>9} {'α_tot':>8}"
print(hdr); print('-' * len(hdr))
for res in [res_appx, res_clara]:
    if res is None: continue
    print(f"{res['label']:<40} "
          f"{res['a_I']:8.4f} {res['a_II']:8.4f}"
          f" {res['a_IVV']:9.4f} {res['a_tot']:8.4f}")
print(f"{'Paper Fig 2c,f':<40} {'0.13':>8} {'0.19':>8}"
      f" {'0.01':>9} {'0.33':>8}")