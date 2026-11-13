"""
Combined 16-Panel Validation Figure: In-Situ Albedo vs CLARA-A3 and APP-x
=========================================================================
4 rows × 4 columns.  Rows alternate product:

  Row 1 (A–D)  CLARA-A3:  Tower  |  Transects  |  Helicopter  |  Summit Greenland
  Row 2 (E–H)  APP-x:     Tower  |  Transects  |  Helicopter  |  Summit Greenland
  Row 3 (I–L)  CLARA-A3:  Drone  |  ARISE 2014 |  ARCSIX 2024 |  Raven Greenland
  Row 4 (M–P)  APP-x:     Drone  |  ARISE 2014 |  ARCSIX 2024 |  Raven Greenland

CLARA-A3 panels use the EXACT same logic as the 8-panel script (code 1):
  A) SZA-matched × SIC-blended tower vs CLARA-A3  (color = DOY)
  B) |ΔSZA| ≤ 3° filtered transects               (color = surface type)
  C) linear SZA correction for helicopter          (color = dist to sat pixel)
  D) Summit pentad mean, Cartesian IDW, no SIC     (color = DOY)
  I) |ΔSZA| ≤ 3°, altitude-weighted drone         (color = alt-weighted AGL)
  J) |ΔSZA| ≤ 3°, ARISE 2014, nobs_snow > 0       (color = dist to sat pixel)
  K) |ΔSZA| ≤ 3°, ARCSIX 2024                     (color = dist to sat pixel)
  L) Raven pentad mean, Cartesian IDW, no SIC      (color = DOY)

APP-x panels (E–H, M–P) mirror exactly, using closest 04/14 UTC slot.
SZA < 70° enforced everywhere.
"""

import xarray as xr
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path
from datetime import timedelta
from scipy import stats
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from shapely.geometry import Point
from cartopy.io import shapereader
from netCDF4 import Dataset as NC4Dataset
from pyproj import Transformer
import pvlib

plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.1,
    'grid.alpha': 0.3,
})

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
base_dir          = "/glade/derecho/scratch/jrotondo/albedo_satellite_validation/mosiac_surface_flux"
dir_sial          = "/glade/derecho/scratch/jrotondo/clara_a3_obs/"
dir_appx          = "/glade/derecho/scratch/jrotondo/APP-x_observations/"
dir_sic           = "/glade/derecho/scratch/jrotondo/SIC_data/"
transect_dir      = "/glade/derecho/scratch/jrotondo/mosiac_transects/"
drone_dir         = "/glade/derecho/scratch/jrotondo/albedo_satellite_validation/drone_flux"
arise_data_dir    = "/glade/derecho/scratch/jrotondo/arise_data"
heli_data_dir     = "/glade/derecho/scratch/jrotondo/helicopter_data/"
arcsix_data_dir   = Path("/glade/derecho/scratch/jrotondo/ARCSIX_obs")
summit_tower_file = "/glade/derecho/scratch/jrotondo/summit_greenland/summit_30min_jan2011tojun2014_seb_20160926.cdf"
raven_sleigh_dir  = "/glade/derecho/scratch/jrotondo/summit_greenland/sleigh_level2"
sat_dir           = dir_sial   # alias used by CLARA-A3 helpers
output_dir        = "/glade/work/jrotondo/combined_validation/"
os.makedirs(output_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
ROTATION_ANGLE    = -45.0
MAX_SZA           = 70.0
MAX_DELTA_SZA     = 3.0

SUMMIT_LAT,  SUMMIT_LON =  72.58, -38.48
RAVEN_LAT,   RAVEN_LON  =  66.48, -46.30
GREENLAND_MIN_DAYS       = 3

# APP-x
APPX_UTC_SLOTS  = [4, 14]
APPX_FILL_VAL   = 9999.0
MAX_DAY_OFFSET  = 2
TEMPORAL_DECAY  = 0.40
PSN25_PROJ4 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
               '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# Distance / count thresholds (matching code-1 values)
MAX_DIST_HELI    = 12.5
MIN_PTS_HELI     = 3
MIN_DOWN_SW_HELI = 1.0
ALT_MIN_HELI_M   = 10.0
MAX_DIST_ARISE   = 12.5
MIN_PTS_ARISE    = 3
MIN_DOWN_SW_ARISE= 50.0
MAX_DIST_ARCSIX  = 12.5
MIN_PTS_ARCSIX   = 3
ALT_WEIGHT_FLOOR_M  = 2.0
ALT_MIN_VALID_M     = 1.0
ALT_MAX_VALID_M     = 500.0
MAX_PITCH_ROLL_DEG  = 10.0
MIN_DOWN_SW_DRONE   = 50.0
MIN_VALID_PTS_DRONE = 10

SURFACE_KEYWORDS = {
    'lead': [
        'on lead', 'lead edge', 'next to lead', 'flat nilas', 'nilas',
        'grease ice', 'new crack', 'refrozen crack', 'frost flowers',
        'new ice frost flowers', 'gray ice', 'white crack', 'rafted ice',
        'rafted surface', 'crack with ice', 'crack with 2',
    ],
    'melt pond': [
        'melt pond', 'blue pond', 'dark pond', 'gray pond', 'green/algae',
        'pond edge', 'pond (edge)', 'frozen pond', 'frozen over pond',
        'over frozen pond', 'refrozen pond', 'frozen mush', 'drained pond',
        'proto-pond', 'now ponded', 'thin pond', 'narrow pond', 'deep pond',
        'starting of pond', 'small pond', 'turquoise pond', 'night light blue',
        'medium blue', 'blue/green', 'holey slush/pond', 'slush/pond',
        'slushy pond', 'ice lid', 'snow/pond', 'pond/', ' pond', 'mp edge',
        'mp!', 'mp =', 'mp by', 'mp depth', '1 cm lid', 'ice lid on pond',
        'thin ice lid', '1.5cm ice lid', '2cm ice lid', '2.5cm ice lid',
        '3cm ice lid', '4cm ice lid',
    ],
    'snow': [
        'snow drift', 'snow dune', 'fresh snow', 'thick snow', 'thin snow',
        'medium snow', 'scant snow', 'wet snow', 'porous snow', 'crumbly snow',
        'crunchy snow', 'holey snow', 'saturated snow', 'slushy snow',
        'pillar snow', 'homogeneous snow', 'snowy', 'snow!', 'snow,',
        'snow next', 'snow free', 'snow over', 'drift', 'dune', 'ssl',
        'crust', 'glazing', 'melt forms', 'melt grains', 'melt rounded',
        'rounded grains', 'large grains', 'needle snow', 'fine grains',
        'wet rounded', 'porous surface', 'porous/wet', 'porous bumpy', 'crunch',
    ],
    'bare ice': [
        'bare ice', 'bare-ish', 'level clean ice', 'white ice', 'bright blue',
        'syi snow', 'sediment', 'sedimented', 'dirty ice', 'dirty ridge',
        'dirty snow', 'dark patches', 'algae', 'rubble', 'ridge', 'mound',
        'blocks', 'bump', 'iceberg', '1 cm ice', 'wet', 'dirty hole',
        'chl and optics', 'dark', 'light',
    ],
    'slush': [
        'slush', 'slushy', 'mush', 'saturated', 'wet/water',
        'water below', 'dirty dirty slush', 'frozen mush',
        'holey slush', 'some slush',
    ],
}
SURFACE_PRIORITY    = ['lead', 'melt pond', 'slush', 'snow', 'bare ice', 'mixed']
SURFACE_COLORS      = {
    'lead':      '#1f77b4',
    'melt pond': '#17becf',
    'snow':      '#9467bd',
    'bare ice':  '#2ca02c',
    'slush':     '#ff7f0e',
    'mixed':     '#e377c2',
    'unknown':   '#7f7f7f',
}
DOMINANCE_THRESHOLD = 0.75

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY FUNCTIONS  (identical to code-1)
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371.0


def calc_stats(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return dict(r=np.nan, bias=np.nan, rmse=np.nan, n=0)
    xv, yv = x[valid], y[valid]
    return dict(r=float(np.corrcoef(xv, yv)[0, 1]),
                bias=float((yv - xv).mean()),
                rmse=float(np.sqrt(((yv - xv)**2).mean())),
                n=int(valid.sum()))


def _stats_text(s):
    return (f"r = {s['r']:.3f}\nbias = {s['bias']:+.3f}\n"
            f"RMSE = {s['rmse']:.3f}\nn = {s['n']}")


def ocean_albedo_payne(sza_deg):
    if not np.isfinite(sza_deg):
        return 0.07
    cos_sza = np.cos(np.deg2rad(np.clip(float(sza_deg), 0.0, 89.9)))
    return 0.037 / (1.1 * cos_sza**1.4 + 0.15)


def spatial_interp_idw(lat, lon, lats_2d, lons_2d, data_2d,
                        min_valid=2, max_dist_km=25.0):
    fl = lats_2d.flatten(); fo = lons_2d.flatten(); fd = data_2d.flatten()
    tree = cKDTree(np.column_stack([fl, fo]))
    _, idxs = tree.query([lat, lon], k=4)
    nbr_d = fd[idxs]; nbr_l = fl[idxs]; nbr_o = fo[idxs]
    valid = np.isfinite(nbr_d) & (nbr_d >= 0) & (nbr_d <= 1)
    if valid.sum() < min_valid:
        return np.nan
    dists = haversine_km(lon, lat, nbr_o[valid], nbr_l[valid])
    if dists.min() > max_dist_km:
        return np.nan
    w = 1.0 / (dists + 0.01)
    return float(np.sum(w * nbr_d[valid]) / w.sum())


def spatial_interp_idw_greenland(lat, lon, lats_2d, lons_2d, data_2d, k=4):
    lats_f = lats_2d.flatten(); lons_f = lons_2d.flatten()
    data_f = data_2d.flatten()
    lat_r  = np.deg2rad(lats_f); lon_r  = np.deg2rad(lons_f)
    tlat_r = np.deg2rad(lat);    tlon_r = np.deg2rad(lon)
    xs = np.cos(lat_r) * np.cos(lon_r)
    ys = np.cos(lat_r) * np.sin(lon_r)
    zs = np.sin(lat_r)
    tx = np.cos(tlat_r) * np.cos(tlon_r)
    ty = np.cos(tlat_r) * np.sin(tlon_r)
    tz = np.sin(tlat_r)
    tree = cKDTree(np.column_stack([xs, ys, zs]))
    dists, idxs = tree.query([tx, ty, tz], k=k)
    vals  = data_f[idxs]
    valid = np.isfinite(vals) & (vals >= 0) & (vals <= 1)
    if not np.any(valid):
        return np.nan
    dists = np.where(dists < 1e-10, 1e-10, dists)
    w     = (1.0 / dists) * valid
    if w.sum() == 0:
        return np.nan
    return float(np.sum(vals * w) / w.sum())


def create_land_mask(lats, lons):
    land_shp   = shapereader.natural_earth(resolution='10m', category='physical', name='land')
    land_geoms = list(shapereader.Reader(land_shp).geometries())
    mask = np.zeros(len(lats), dtype=bool)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        if np.isnan(la) or np.isnan(lo):
            continue
        mask[i] = not any(g.contains(Point(lo, la)) for g in land_geoms)
    return mask


def remove_outliers(df_, sat_col, ref_col):
    if len(df_) == 0:
        return df_
    res = df_[sat_col] - df_[ref_col]
    bad = (df_[sat_col] < 0.01) | (np.abs(res) > 3*res.std()) | (res < -0.5)
    return df_[~bad].copy()


def sza_filter(df_, delta_col):
    return df_[df_[delta_col].notna() &
               (df_[delta_col].abs() <= MAX_DELTA_SZA)].copy()


def apply_linear_sza_correction(df_, raw_col, sat_col, delta_col, corr_col):
    df_fit = df_[df_[delta_col].notna() & df_[raw_col].notna()].copy()
    bias   = df_fit[sat_col] - df_fit[raw_col]
    if len(df_fit) >= 3:
        slope, intercept, _, _, _ = stats.linregress(df_fit[delta_col], bias)
        df_[corr_col] = df_[raw_col] + slope * df_[delta_col]
        print(f"    SZA correction: slope={slope:.5f}/°  "
              f"intercept={intercept:+.4f}  n={len(df_fit)}")
    else:
        df_[corr_col] = np.nan
        print("    SZA correction: insufficient data")
    return df_


def parse_start_time(tv):
    try:
        t = int(float(str(tv).strip()))
        return t // 100, t % 100
    except:
        return None, None


def compute_transect_sza(lat, lon, date_str, hhmm_val):
    hh, mm = parse_start_time(hhmm_val)
    if hh is None:
        return np.nan
    try:
        dt  = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                           f"{hh:02d}:{mm:02d}:00", tz='UTC')
        loc = pvlib.location.Location(latitude=lat, longitude=lon)
        sp  = loc.get_solarposition(pd.DatetimeIndex([dt]))
        return float(sp['apparent_zenith'].iloc[0])
    except:
        return np.nan


def classify_surface(notes_series):
    if notes_series is None or len(notes_series) == 0:
        return 'unknown'
    counts = {k: 0 for k in SURFACE_KEYWORDS}
    for note in notes_series.dropna():
        nl = str(note).lower().strip()
        for stype, kws in SURFACE_KEYWORDS.items():
            if any(kw in nl for kw in kws):
                counts[stype] += 1
                break
    total = sum(counts.values())
    if total == 0:
        return 'unknown'
    mx = max(counts.values())
    if mx / total < DOMINANCE_THRESHOLD:
        return 'mixed'
    cands = [s for s, c in counts.items() if c == mx]
    for p in SURFACE_PRIORITY:
        if p in cands:
            return p
    return cands[0]


def classify_surface_from_filename(fname, surface_type):
    if surface_type != 'unknown':
        return surface_type
    if '_SYI_' in fname:
        return 'bare ice'
    m = re.search(r'mosaic_(\d{8})', fname)
    if '_ROV3_' in fname and m and int(m.group(1)) < 20200601:
        return 'snow'
    return 'unknown'


# ══════════════════════════════════════════════════════════════════════════════
# CLARA-A3 FILE UTILITIES  (identical to code-1)
# ══════════════════════════════════════════════════════════════════════════════

def find_pentad_file_for_date(flight_date, sat_dir_):
    for f in sorted(glob.glob(f"{sat_dir_}/SALpm{flight_date.year}*.nc")):
        ps = pd.to_datetime(os.path.basename(f)[5:13])
        if ps <= flight_date < ps + timedelta(days=5):
            return f
    return None


def find_bracketing_pentads(mdate, sat_dir_):
    clara_files = sorted(glob.glob(f"{sat_dir_}/SALpm{mdate.year}*.nc"))
    if not clara_files:
        return None, None, 0.5, 0.5
    pentads = []
    for f in clara_files:
        try:
            ps = pd.to_datetime(os.path.basename(f)[5:13])
            pentads.append({'file': f, 'start': ps,
                            'end': ps + timedelta(days=5),
                            'center': ps + timedelta(days=2.5)})
        except:
            continue
    pentads.sort(key=lambda p: p['start'])
    pb = pa = None
    for i, p in enumerate(pentads):
        if p['start'] <= mdate < p['end']:
            pb = p; pa = pentads[i+1] if i+1 < len(pentads) else p; break
        elif mdate < p['start']:
            pb = pentads[i-1] if i > 0 else p; pa = p; break
    if pb is None and pa is None:
        pb = pa = pentads[-1] if pentads else None
    pb = pb or pa; pa = pa or pb
    if pb is None:
        return None, None, 0.5, 0.5
    span = (pa['center'] - pb['center']).total_seconds() / 86400.0
    if span > 0:
        wb = (pa['center'] - mdate).total_seconds() / 86400.0 / span
        wa = (mdate - pb['center']).total_seconds() / 86400.0 / span
        t  = wb + wa
        wb, wa = (wb/t, wa/t) if t > 0 else (0.5, 0.5)
    else:
        wb = wa = 0.5
    return pb, pa, wb, wa


def get_sat_sza(lat, lon, pentad_file):
    try:
        with xr.open_dataset(pentad_file) as ds:
            if 'sun_zenith_angle_mean' not in ds.variables:
                return np.nan
            sza_2d  = ds['sun_zenith_angle_mean'].squeeze().values.astype(float)
            lats_2d = ds['lat'].values
            lons_2d = ds['lon'].values
        tree = cKDTree(np.column_stack([lats_2d.flatten(), lons_2d.flatten()]))
        _, idx = tree.query([lat, lon])
        val = sza_2d.flatten()[idx]
        return float(val) if np.isfinite(val) else np.nan
    except:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# APP-x FILE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_appx_files_for_date(date, appx_dir=dir_appx):
    date_str = pd.Timestamp(date).strftime('%Y%m%d')
    year     = pd.Timestamp(date).year
    result   = {}
    for slot in APPX_UTC_SLOTS:
        for sdir in [os.path.join(appx_dir, str(year)), appx_dir]:
            if not os.path.isdir(sdir):
                continue
            for pat in [f"*_{slot:04d}_*d{date_str}*.nc",
                        f"*_{slot:02d}00_*d{date_str}*.nc",
                        f"*_{slot:04d}*{date_str}*.nc"]:
                matches = sorted(glob.glob(os.path.join(sdir, pat)))
                if matches:
                    result[slot] = matches[0]; break
            if slot in result:
                break
    if not result:
        for sdir in [os.path.join(appx_dir, str(year)), appx_dir]:
            if not os.path.isdir(sdir):
                continue
            matches = sorted(glob.glob(f"{sdir}/*d{date_str}*.nc"))
            if matches:
                result[14] = matches[0]; break
    return result


def best_appx_slot(insitu_utc_hour, available_slots):
    if not available_slots:
        return None, None
    best = min(available_slots.keys(), key=lambda h: abs(h - insitu_utc_hour))
    return best, available_slots[best]


def load_appx_albedo(appx_file):
    try:
        with NC4Dataset(appx_file, 'r') as ds:
            lat_raw = np.ma.filled(np.ma.asarray(ds.variables['latitude'][:]),  np.nan).astype(float)
            lon_raw = np.ma.filled(np.ma.asarray(ds.variables['longitude'][:]), np.nan).astype(float)
            alb     = np.ma.filled(np.ma.asarray(ds.variables['cdr_surface_albedo'][0]), np.nan).astype(float)
            alb[alb >= APPX_FILL_VAL] = np.nan
            alb[(alb < 0) | (alb > 1)] = np.nan
        if lat_raw.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_raw, lat_raw)
        else:
            lat2d, lon2d = lat_raw, lon_raw
        return lat2d, lon2d, alb
    except Exception as e:
        print(f"    APP-x load error {os.path.basename(appx_file)}: {e}")
        return None, None, None


def get_appx_sza(lat, lon, date, utc_hour):
    try:
        dt = pd.Timestamp(
            f"{pd.Timestamp(date).strftime('%Y-%m-%d')} {int(utc_hour):02d}:00:00",
            tz='UTC')
        loc = pvlib.location.Location(latitude=float(lat), longitude=float(lon))
        return float(loc.get_solarposition(pd.DatetimeIndex([dt]))['apparent_zenith'].iloc[0])
    except:
        return np.nan


def get_appx_albedo_composite(lat, lon, date, preferred_utc_hour,
                               max_dist_km=25.0, min_valid=2):
    slots0 = find_appx_files_for_date(date)
    slot0, fpath0 = best_appx_slot(preferred_utc_hour, slots0)
    if fpath0 is not None:
        la, lo, al = load_appx_albedo(fpath0)
        if la is not None:
            v = spatial_interp_idw(lat, lon, la, lo, al,
                                   min_valid=min_valid, max_dist_km=max_dist_km)
            if np.isfinite(v):
                return v, slot0, 0
    wsum = 0.0; wtot = 0.0; best_slot = slot0 or 14
    for off in range(-MAX_DAY_OFFSET, MAX_DAY_OFFSET + 1):
        if off == 0:
            continue
        slots_c = find_appx_files_for_date(pd.Timestamp(date) + pd.Timedelta(days=off))
        slot_c, fpath_c = best_appx_slot(preferred_utc_hour, slots_c)
        if fpath_c is None:
            continue
        la, lo, al = load_appx_albedo(fpath_c)
        if la is None:
            continue
        v = spatial_interp_idw(lat, lon, la, lo, al,
                               min_valid=min_valid, max_dist_km=max_dist_km)
        if not np.isfinite(v):
            continue
        w = np.exp(-TEMPORAL_DECAY * abs(off))
        wsum += w * v; wtot += w
    if wtot > 0:
        return wsum / wtot, best_slot, None
    return np.nan, None, None


def get_appx_albedo_greenland(lat, lon, date, utc_hour=14,
                               max_dist_km=50.0, min_valid=2):
    """Same composite logic but wider spatial cap for fixed Greenland sites."""
    return get_appx_albedo_composite(lat, lon, date, utc_hour,
                                     max_dist_km=max_dist_km, min_valid=min_valid)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — MOSAiC Tower  (CLARA-A3, code-1 exact)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading Tower data")


def _rotate_coords(x, y, angle_deg):
    a = np.deg2rad(angle_deg)
    return x*np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a)


def load_tower_with_sza(directory, file_pattern):
    data_list = []
    files = sorted([f for f in os.listdir(directory)
                    if f.startswith(file_pattern) and f.endswith(".nc")])
    for f in files:
        try:
            ds  = xr.open_dataset(os.path.join(directory, f))
            req = ['up_short_hemisp', 'down_short_hemisp',
                   'up_short_hemisp_qc', 'down_short_hemisp_qc']
            if not all(v in ds.variables for v in req):
                ds.close(); continue
            lat_var  = 'lat_tower' if 'lat_tower' in ds else 'lat'
            lon_var  = 'lon_tower' if 'lon_tower' in ds else 'lon'
            valid_qc = ((ds['up_short_hemisp_qc']   == 0) &
                        (ds['down_short_hemisp_qc'] == 0) &
                        (ds['down_short_hemisp']    > 10.0))
            albedo   = (ds['up_short_hemisp'] / ds['down_short_hemisp']).where(valid_qc)
            lat      = ds[lat_var].where(valid_qc)
            lon      = ds[lon_var].where(valid_qc)
            sza_vals = np.full(len(ds['time']), np.nan)
            if 'zenith_true' in ds.variables:
                raw      = ds['zenith_true'].values.flatten().astype(float)
                sza_vals = np.where(valid_qc.values.flatten(), raw, np.nan)
            tmp = pd.DataFrame({'time':   ds['time'].values,
                                'albedo': albedo.values.flatten(),
                                'lat':    lat.values.flatten(),
                                'lon':    lon.values.flatten(),
                                'sza':    sza_vals,
                                }).dropna(subset=['albedo', 'lat', 'lon']).set_index('time')
            tmp = tmp[(tmp['albedo'] >= 0) & (tmp['albedo'] <= 1) &
                      (tmp['sza'] < MAX_SZA)]
            data_list.append(tmp)
            ds.close()
        except Exception as e:
            print(f"  {f}: {e}")
    return pd.concat(data_list).sort_index() if data_list else pd.DataFrame()


def load_sic_pentad():
    sic_list = []
    for sic_file in sorted(glob.glob(f"{dir_sic}/sic_psn25_2020*.nc")):
        try:
            with xr.open_dataset(sic_file) as ds:
                sic = ds['cdr_seaice_conc'].squeeze()
                if float(sic.max()) > 1.5:
                    sic = sic / 100.0
                if 'time' in ds:
                    tval = pd.Timestamp(ds['time'].values.flat[0]).floor('D')
                else:
                    tval = pd.Timestamp(os.path.basename(sic_file).split('_')[2])
                if 'time' not in sic.dims:
                    sic = sic.expand_dims(time=[tval])
                sic_list.append(sic)
        except Exception as e:
            print(f"  SIC: {e}")
    if sic_list:
        return (xr.concat(sic_list, dim='time').sortby('time')
                .resample(time='5D', label='left').mean())
    return None


all_sza_dfs = []
for key, (path, pat) in {
    'met':    (os.path.join(base_dir, "central_observatory"), "mosmet"),
    'asfs30': (os.path.join(base_dir, "station_30"),          "mosmet.asfs30"),
    'asfs50': (os.path.join(base_dir, "station_50"),          "mosmet.asfs50"),
}.items():
    df_tmp = load_tower_with_sza(path, pat)
    if not df_tmp.empty:
        all_sza_dfs.append(df_tmp)
        print(f"  {key}: {len(df_tmp)} obs")

tower_sza_df = pd.concat(all_sza_dfs).sort_index() if all_sza_dfs else pd.DataFrame()
sic_pentad   = load_sic_pentad()
transformer  = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
sial_files   = sorted(glob.glob(f"{dir_sial}/SALpm2020*.nc"))


def match_tower_clara(tower_df, sial_files_, sic_pentad_):
    """CLARA-A3 tower match — code-1 exact."""
    results = []
    for sial_path in sorted(sial_files_):
        try:
            cdate  = pd.to_datetime(os.path.basename(sial_path)[5:13])
            mask   = ((tower_df.index >= cdate) &
                      (tower_df.index <  cdate + pd.Timedelta(days=5)))
            subset = tower_df[mask]
            if len(subset) < 24:
                continue
            avg_lat = subset['lat'].mean()
            avg_lon = subset['lon'].mean()
            tx, ty  = transformer.transform(avg_lon, avg_lat)
            sx, sy  = _rotate_coords(tx, ty, -ROTATION_ANGLE)

            with xr.open_dataset(sial_path) as ds:
                def _ipt(var, scale=1.0):
                    if var not in ds.variables:
                        return np.nan
                    try:
                        return float(ds[var].squeeze()
                                     .interp(x=tx, y=ty, method='linear').item()) / scale
                    except:
                        return np.nan
                sat_sza     = _ipt('sun_zenith_angle_mean')
                sat_sza_std = _ipt('sun_zenith_angle_std')
                bal_sat     = _ipt('blue_sky_albedo_all_mean', scale=100.0)

            if not np.isfinite(bal_sat):
                continue
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue

            tower_sza_matched = np.nan
            if np.isfinite(sat_sza) and np.isfinite(sat_sza_std):
                win    = max(3.0, sat_sza_std)
                in_win = subset[np.abs(subset['sza'] - sat_sza) <= win]
                if len(in_win) >= 6:
                    tower_sza_matched = in_win['albedo'].mean()

            sic_val = np.nan
            if sic_pentad_ is not None:
                try:
                    sl = sic_pentad_.sel(time=cdate, method='nearest',
                                         tolerance=pd.Timedelta(days=2))
                    sic_val = float(sl.interp(x=sx, y=sy, method='linear').item())
                except:
                    pass

            if np.isfinite(sic_val) and sic_val < 0:
                continue

            alpha_ocn = ocean_albedo_payne(sat_sza)

            def blend(alb):
                return (sic_val * alb + (1.0 - sic_val) * alpha_ocn
                        if np.isfinite(sic_val) and np.isfinite(alb) else np.nan)

            results.append({
                'date':              cdate,
                'tower_sza_matched': blend(tower_sza_matched),
                'tower_scaled':      blend(subset['albedo'].mean()),
                'sial_blue':         bal_sat,
            })
        except Exception as e:
            print(f"  {os.path.basename(sial_path)}: {e}")
    return (pd.DataFrame(results).set_index('date').sort_index()
            if results else pd.DataFrame())


def match_tower_appx(tower_df, sic_pentad_):
    """APP-x tower match — daily, best UTC slot, SIC-blended."""
    results = []
    for tdate in sorted(tower_df.index.normalize().unique()):
        tdate    = pd.Timestamp(tdate)
        day_mask = tower_df.index.normalize() == tdate.normalize()
        day_data = tower_df[day_mask]
        if len(day_data) < 6:
            continue
        avg_lat    = float(day_data['lat'].mean())
        avg_lon    = float(day_data['lon'].mean())
        insitu_utc_h = float(np.mean(day_data.index.hour) +
                             np.mean(day_data.index.minute) / 60.0)
        sat_alb, slot_used, _ = get_appx_albedo_composite(
            avg_lat, avg_lon, tdate, insitu_utc_h)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(avg_lat, avg_lon, tdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        # SIC blend
        tx, ty = transformer.transform(avg_lon, avg_lat)
        sx, sy = _rotate_coords(tx, ty, -ROTATION_ANGLE)
        sic_val = np.nan
        if sic_pentad_ is not None:
            try:
                sl      = sic_pentad_.sel(time=tdate, method='nearest',
                                          tolerance=pd.Timedelta(days=3))
                sic_val = float(sl.interp(x=sx, y=sy, method='linear').item())
            except:
                pass
        if not np.isfinite(sic_val) or sic_val < 0:
            continue
        win    = MAX_DELTA_SZA * 2
        in_win = day_data[np.abs(day_data['sza'] - sat_sza) <= win]
        tower_alb_raw = (float(in_win['albedo'].mean())
                         if len(in_win) >= 6 else float(day_data['albedo'].mean()))
        alpha_ocn    = ocean_albedo_payne(sat_sza)
        tower_blend  = sic_val * tower_alb_raw + (1 - sic_val) * alpha_ocn
        results.append({'date':         tdate,
                        'tower_sza_matched': tower_blend,
                        'tower_scaled':      tower_blend,
                        'sial_blue':         sat_alb})
    return (pd.DataFrame(results).set_index('date').sort_index()
            if results else pd.DataFrame())


print("  Matching tower to CLARA-A3...")
df_tower_c = match_tower_clara(tower_sza_df, sial_files, sic_pentad)
print(f"    {len(df_tower_c)} valid pentads (CLARA-A3)")

print("  Matching tower to APP-x...")
df_tower_a = match_tower_appx(tower_sza_df, sic_pentad)
print(f"    {len(df_tower_a)} valid days (APP-x)")

# Build x/y arrays for both products (identical helper)
def _tower_xy(df_t):
    if df_t.empty:
        return np.array([]), np.array([]), np.array([])
    series = (df_t['tower_sza_matched']
              .where(df_t['tower_sza_matched'].notna(), df_t['tower_scaled']))
    valid  = series.notna() & df_t['sial_blue'].notna()
    return (series[valid].values,
            df_t.loc[valid, 'sial_blue'].values,
            series[valid].index.dayofyear)

x_tower_c, y_tower_c, doy_tower_c = _tower_xy(df_tower_c)
x_tower_a, y_tower_a, doy_tower_a = _tower_xy(df_tower_a)
stats_tower_c = calc_stats(x_tower_c, y_tower_c)
stats_tower_a = calc_stats(x_tower_a, y_tower_a)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B/F — MOSAiC Transects  (|ΔSZA| ≤ MAX_DELTA_SZA, code-1 exact)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading Transects")

csv_files       = sorted(glob.glob(os.path.join(transect_dir, "broadbandalbedo_mosaic_*.csv")))
transect_records_raw = []
for f in csv_files:
    fname = os.path.basename(f)
    m     = re.search(r"mosaic_(\d{8})", fname)
    if not m:
        continue
    measurement_date = m.group(1)
    try:
        df_t = pd.read_csv(f, skiprows=5)
        with open(f, 'r') as fh:
            lines = fh.readlines()
        start_time_raw = lines[0].split(',')[1].strip()
        lat = float(lines[3].split(',')[1].strip())
        lon = float(lines[4].split(',')[1].strip())
        df_t = df_t[(df_t['Albedo'] >= 0) & (df_t['Albedo'] <= 1) & df_t['Albedo'].notna()]
        if len(df_t) < 2:
            continue
        transect_albedo = df_t['Albedo'].mean()
        surface_type    = classify_surface(df_t.get('Notes'))
        surface_type    = classify_surface_from_filename(fname, surface_type)
        hh, mm          = parse_start_time(start_time_raw)
        if hh is None:
            continue
        transect_sza = compute_transect_sza(lat, lon, measurement_date, start_time_raw)
        if not np.isfinite(transect_sza) or transect_sza >= MAX_SZA:
            continue
        transect_records_raw.append({
            'date':            measurement_date,
            'lat':             lat,
            'lon':             lon,
            'transect_albedo': transect_albedo,
            'surface_type':    surface_type,
            'transect_sza':    transect_sza,
            'utc_hour':        hh + mm / 60.0,
        })
    except Exception as e:
        print(f"  {fname}: {e}")

print(f"  {len(transect_records_raw)} raw transect records")


def match_transects_clara(records):
    rows = []
    for rec in records:
        mdate          = pd.to_datetime(rec['date'], format='%Y%m%d')
        lat, lon       = rec['lat'], rec['lon']
        pb, pa, wb, wa = find_bracketing_pentads(mdate, sat_dir)
        if pb is None:
            continue
        try:
            with xr.open_dataset(pb['file']) as ds_b:
                if 'blue_sky_albedo_all_mean' not in ds_b.variables:
                    continue
                alb_b   = ds_b['blue_sky_albedo_all_mean'].squeeze().values / 100.0
                lats_2d = ds_b['lat'].values; lons_2d = ds_b['lon'].values
            with xr.open_dataset(pa['file']) as ds_a:
                alb_a = ds_a['blue_sky_albedo_all_mean'].squeeze().values / 100.0
            sab_b = spatial_interp_idw(lat, lon, lats_2d, lons_2d, alb_b)
            sab_a = spatial_interp_idw(lat, lon, lats_2d, lons_2d, alb_a)
            if not (np.isfinite(sab_b) and np.isfinite(sab_a)):
                continue
            sat_albedo = wb * sab_b + wa * sab_a
            sat_sza_b  = get_sat_sza(lat, lon, pb['file'])
            sat_sza_a  = get_sat_sza(lat, lon, pa['file'])
            if   np.isfinite(sat_sza_b) and np.isfinite(sat_sza_a):
                sat_sza = wb * sat_sza_b + wa * sat_sza_a
            elif np.isfinite(sat_sza_b): sat_sza = sat_sza_b
            elif np.isfinite(sat_sza_a): sat_sza = sat_sza_a
            else: sat_sza = np.nan
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue
            rows.append({**rec,
                         'sat_albedo': sat_albedo,
                         'delta_sza':  rec['transect_sza'] - sat_sza})
        except:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'transect_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


def match_transects_appx(records):
    rows = []
    for rec in records:
        mdate    = pd.to_datetime(rec['date'], format='%Y%m%d')
        lat, lon = rec['lat'], rec['lon']
        utc_h    = rec['utc_hour']
        sat_alb, slot_used, _ = get_appx_albedo_composite(lat, lon, mdate, utc_h)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(lat, lon, mdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        rows.append({**rec,
                     'sat_albedo': sat_alb,
                     'delta_sza':  rec['transect_sza'] - sat_sza})
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'transect_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


df_transect_c = match_transects_clara(transect_records_raw)
df_transect_a = match_transects_appx(transect_records_raw)
print(f"  {len(df_transect_c)} transects (CLARA-A3)  |  {len(df_transect_a)} (APP-x)")
stats_transect_c = calc_stats(df_transect_c['transect_albedo'].values,
                               df_transect_c['sat_albedo'].values)
stats_transect_a = calc_stats(df_transect_a['transect_albedo'].values,
                               df_transect_a['sat_albedo'].values)

valid_surf   = [s for s in SURFACE_PRIORITY + ['unknown']
                if s in df_transect_c.get('surface_type', pd.Series([], dtype=str)).values]
surf_patches = [mpatches.Patch(color=SURFACE_COLORS.get(s, '#7f7f7f'), label=s.title())
                for s in valid_surf]

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C/G — MOSAiC Helicopter + HELiPOD  (linear SZA correction, code-1 exact)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading Helicopter / HELiPOD")


def extract_date_heli(filename):
    m = re.search(r"(\d{4}_\d{2}_\d{2})", filename)
    return m.group(1).replace('_', '') if m else None


def load_tab_file(fpath):
    header_lineno = None
    with open(fpath, 'r', errors='replace') as fh:
        for i, line in enumerate(fh):
            if line.strip() == '*/':
                header_lineno = i + 1
                break
    if header_lineno is None:
        return None, False
    try:
        df_ = pd.read_csv(fpath, sep='\t', skiprows=header_lineno,
                          parse_dates=['Date/Time'], low_memory=False,
                          on_bad_lines='warn')
    except Exception as e:
        print(f"    Read error: {e}"); return None, False
    df_ = df_.rename(columns={'Date/Time': 'timestamp'})
    required = ['timestamp', 'Latitude', 'Longitude', 'Altitude [m]',
                'Eu [W/m**2] (corrected)', 'Ed [W/m**2] (corrected)']
    if any(c not in df_.columns for c in required):
        return None, False
    has_sza_col = 'SZA [deg]' in df_.columns
    df_['Albedo'] = df_['Eu [W/m**2] (corrected)'] / df_['Ed [W/m**2] (corrected)']
    df_ = df_[df_['Ed [W/m**2] (corrected)'] > MIN_DOWN_SW_HELI]
    df_ = df_[(df_['Albedo'] >= 0) & (df_['Albedo'] <= 1)]
    df_ = df_[df_['Altitude [m]'] > ALT_MIN_HELI_M]
    df_ = df_.dropna(subset=['timestamp', 'Latitude', 'Longitude'])
    return (df_ if len(df_) > 0 else None), has_sza_col


# Build raw pixel-grouped records (shared between CLARA-A3 and APP-x)
heli_pixel_records = []
for fpath in sorted(glob.glob(os.path.join(heli_data_dir, "*.tab"))):
    fname      = os.path.basename(fpath)
    date_str_h = extract_date_heli(fname)
    if not date_str_h:
        continue
    try:
        df_h, has_sza = load_tab_file(fpath)
        if df_h is None:
            continue
        flight_date_h = pd.to_datetime(date_str_h, format='%Y%m%d')
        instrument    = 'HELiPOD' if has_sza else 'helicopter'

        # Build a satellite pixel grid from the first available pentad for
        # grouping purposes; we just need approximate lat/lon pixel indices.
        pb, pa, wb, wa = find_bracketing_pentads(flight_date_h, sat_dir)
        if pb is None:
            continue
        with xr.open_dataset(pb['file']) as ds_b:
            if 'blue_sky_albedo_all_mean' not in ds_b.variables:
                continue
            alb_b   = ds_b['blue_sky_albedo_all_mean'].squeeze().values / 100.0
            lats_2d = ds_b['lat'].values; lons_2d = ds_b['lon'].values
            ice_mask = (ds_b['sea_ice_extent'].squeeze().values > 0
                        if 'sea_ice_extent' in ds_b.variables
                        else lats_2d >= 66.0)

        valid_pix = ice_mask & np.isfinite(alb_b)
        vi        = np.where(valid_pix)
        v_lats    = lats_2d[vi]; v_lons = lons_2d[vi]
        if len(v_lats) == 0:
            continue

        tree = cKDTree(np.column_stack([v_lats, v_lons]))
        _, indices = tree.query(np.column_stack([df_h['Latitude'].values,
                                                  df_h['Longitude'].values]))
        dist_km = np.array([haversine_km(df_h['Longitude'].iloc[i],
                                          df_h['Latitude'].iloc[i],
                                          v_lons[indices[i]], v_lats[indices[i]])
                             for i in range(len(df_h))])
        df_h = df_h.copy()
        df_h['pixel_idx'] = indices; df_h['dist_km'] = dist_km
        df_near = df_h[df_h['dist_km'] <= MAX_DIST_HELI]
        if len(df_near) == 0:
            continue

        for pix_idx, grp in df_near.groupby('pixel_idx'):
            if len(grp) < MIN_PTS_HELI:
                continue
            pix_lat     = float(grp['Latitude'].mean())
            pix_lon     = float(grp['Longitude'].mean())
            heli_albedo = float(grp['Albedo'].mean())
            mean_dist   = float(grp['dist_km'].mean())

            if has_sza and 'SZA [deg]' in grp.columns:
                heli_sza = float(grp['SZA [deg]'].mean())
            else:
                heli_sza = compute_transect_sza(
                    pix_lat, pix_lon,
                    flight_date_h.strftime('%Y%m%d'),
                    grp['timestamp'].iloc[len(grp)//2].strftime('%H%M')
                    if hasattr(grp['timestamp'].iloc[0], 'strftime')
                    else str(int(grp['timestamp'].iloc[len(grp)//2].hour * 100 +
                                 grp['timestamp'].iloc[len(grp)//2].minute)))

            if not np.isfinite(heli_sza) or heli_sza >= MAX_SZA:
                continue

            mid_ts   = grp['timestamp'].iloc[len(grp)//2]
            utc_hour = (mid_ts.hour + mid_ts.minute / 60.0
                        if hasattr(mid_ts, 'hour') else 14.0)

            heli_pixel_records.append({
                'date':         flight_date_h,
                'instrument':   instrument,
                'lat':          pix_lat,
                'lon':          pix_lon,
                'heli_albedo':  heli_albedo,
                'heli_sza':     heli_sza,
                'utc_hour':     utc_hour,
                'mean_dist_km': mean_dist,
                'pb': pb, 'pa': pa, 'wb': wb, 'wa': wa,
            })
    except Exception as e:
        print(f"  Helicopter {fname}: {e}")

print(f"  {len(heli_pixel_records)} helicopter pixel records")


def match_heli_clara(pixel_records):
    rows = []
    for rec in pixel_records:
        pb, pa, wb, wa = rec['pb'], rec['pa'], rec['wb'], rec['wa']
        pix_lat, pix_lon = rec['lat'], rec['lon']
        try:
            with xr.open_dataset(pb['file']) as ds_b:
                alb_b   = ds_b['blue_sky_albedo_all_mean'].squeeze().values / 100.0
                lats_2d = ds_b['lat'].values; lons_2d = ds_b['lon'].values
            with xr.open_dataset(pa['file']) as ds_a:
                alb_a = ds_a['blue_sky_albedo_all_mean'].squeeze().values / 100.0
            alb_int = wb * spatial_interp_idw(pix_lat, pix_lon, lats_2d, lons_2d, alb_b) + \
                      wa * spatial_interp_idw(pix_lat, pix_lon, lats_2d, lons_2d, alb_a)
            if not np.isfinite(alb_int):
                continue
            sat_sza_b = get_sat_sza(pix_lat, pix_lon, pb['file'])
            sat_sza_a = get_sat_sza(pix_lat, pix_lon, pa['file'])
            if   np.isfinite(sat_sza_b) and np.isfinite(sat_sza_a):
                sat_sza = wb * sat_sza_b + wa * sat_sza_a
            elif np.isfinite(sat_sza_b): sat_sza = sat_sza_b
            elif np.isfinite(sat_sza_a): sat_sza = sat_sza_a
            else: sat_sza = np.nan
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue
            rows.append({**{k: v for k, v in rec.items()
                            if k not in ('pb','pa','wb','wa')},
                         'sat_albedo': alb_int,
                         'delta_sza':  rec['heli_sza'] - sat_sza,
                         'mean_dist_km': rec['mean_dist_km']})
        except:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'heli_albedo')
        df = apply_linear_sza_correction(df, 'heli_albedo', 'sat_albedo',
                                          'delta_sza', 'heli_albedo_corr')
    return df


def match_heli_appx(pixel_records):
    rows = []
    for rec in pixel_records:
        pix_lat, pix_lon = rec['lat'], rec['lon']
        utc_h            = rec['utc_hour']
        mdate            = pd.Timestamp(rec['date'])
        sat_alb, slot_used, _ = get_appx_albedo_composite(pix_lat, pix_lon, mdate, utc_h)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(pix_lat, pix_lon, mdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        rows.append({**{k: v for k, v in rec.items()
                        if k not in ('pb','pa','wb','wa')},
                     'sat_albedo': sat_alb,
                     'delta_sza':  rec['heli_sza'] - sat_sza,
                     'mean_dist_km': rec['mean_dist_km']})
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'heli_albedo')
        df = apply_linear_sza_correction(df, 'heli_albedo', 'sat_albedo',
                                          'delta_sza', 'heli_albedo_corr')
    return df


df_heli_c = match_heli_clara(heli_pixel_records)
df_heli_a = match_heli_appx(heli_pixel_records)
print(f"  {len(df_heli_c)} heli records (CLARA-A3)  |  {len(df_heli_a)} (APP-x)")
stats_heli_c = calc_stats(
    df_heli_c['heli_albedo_corr'].values if len(df_heli_c) and 'heli_albedo_corr' in df_heli_c else [],
    df_heli_c['sat_albedo'].values       if len(df_heli_c) else [])
stats_heli_a = calc_stats(
    df_heli_a['heli_albedo_corr'].values if len(df_heli_a) and 'heli_albedo_corr' in df_heli_a else [],
    df_heli_a['sat_albedo'].values       if len(df_heli_a) else [])

# ══════════════════════════════════════════════════════════════════════════════
# PANELS D/H — Summit Greenland  |  PANELS L/P — Raven Greenland
# (pentad-mean, Cartesian IDW, no SIC blend — code-1 exact for CLARA-A3)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading Greenland towers")


def load_summit_tower(tower_file_path):
    try:
        ds        = xr.open_dataset(tower_file_path)
        ds        = ds.where(ds != -999)
        base_time = pd.to_datetime(
            ds["base_time"].values.item(), unit="ns", origin="unix")
        time      = base_time + pd.to_timedelta(ds["time_offset"].values, unit="s")
        ds        = ds.assign_coords(time=("time", time))
        albedo    = ds["swup"] / ds["swdn"]
        albedo    = albedo.where(ds["swdn"] > 10)
        albedo    = albedo.where((albedo >= 0) & (albedo <= 1))
        loc       = pvlib.location.Location(latitude=SUMMIT_LAT, longitude=SUMMIT_LON)
        ts_idx    = pd.DatetimeIndex(time).tz_localize('UTC')
        sza_arr   = loc.get_solarposition(ts_idx)['apparent_zenith'].values
        albedo    = albedo.where(sza_arr < MAX_SZA)
        df_s      = albedo.to_dataframe(name='albedo').reset_index()
        df_s      = df_s.dropna().set_index('time')
        df_s      = df_s.loc['2012-01-01':'2013-12-31']
        ds.close()
        return df_s.resample('D').mean().dropna()
    except Exception as e:
        print(f"  Summit tower load error: {e}")
        return pd.DataFrame()


def load_raven_tower(sleigh_dir_path):
    sleigh_files = sorted(glob.glob(os.path.join(sleigh_dir_path, "sleighsfsseb*.nc")))
    if not sleigh_files:
        print(f"  No SLEIGH files found in {sleigh_dir_path}")
        return pd.DataFrame()
    print(f"  Found {len(sleigh_files)} SLEIGH files")
    try:
        ds     = xr.open_mfdataset(sleigh_files, combine="by_coords")
        albedo = ds["up_short_hemisp"] / ds["down_short_hemisp"]
        albedo = albedo.where(ds["down_short_hemisp"] > 10)
        albedo = albedo.where((albedo >= 0) & (albedo <= 1))
        loc    = pvlib.location.Location(latitude=RAVEN_LAT, longitude=RAVEN_LON)
        ts_idx = pd.DatetimeIndex(ds.time.values).tz_localize('UTC')
        sza_arr = loc.get_solarposition(ts_idx)['apparent_zenith'].values
        albedo  = albedo.where(sza_arr < MAX_SZA)
        df_r    = albedo.to_dataframe(name='albedo').reset_index()
        df_r    = df_r.dropna().set_index('time')
        df_r    = df_r.loc['2024-01-01':'2024-12-31']
        ds.close()
        return df_r.resample('D').mean().dropna()
    except Exception as e:
        print(f"  Raven tower load error: {e}")
        return pd.DataFrame()


def match_greenland_clara(tower_daily, site_lat, site_lon, sial_pattern, site_label):
    """Identical to code-1 match_greenland_tower."""
    records = []
    for sial_path in sorted(glob.glob(sial_pattern)):
        try:
            fname  = os.path.basename(sial_path)
            cdate  = pd.to_datetime(fname[5:13])
            pend   = cdate + pd.Timedelta(days=5)
            mask   = (tower_daily.index >= cdate) & (tower_daily.index < pend)
            subset = tower_daily[mask]
            if len(subset) < GREENLAND_MIN_DAYS:
                continue
            avg_albedo = float(subset['albedo'].mean())
            if not np.isfinite(avg_albedo):
                continue
            with xr.open_dataset(sial_path) as ds:
                if 'blue_sky_albedo_all_mean' not in ds.variables:
                    continue
                alb_2d  = ds['blue_sky_albedo_all_mean'].squeeze().values / 100.0
                lats_2d = ds['lat'].values
                lons_2d = ds['lon'].values
                sat_sza = get_sat_sza(site_lat, site_lon, sial_path)
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue
            sat_alb = spatial_interp_idw_greenland(
                site_lat, site_lon, lats_2d, lons_2d, alb_2d)
            if not np.isfinite(sat_alb) or sat_alb <= 0:
                continue
            records.append({'date':         cdate,
                            'tower_albedo': avg_albedo,
                            'sat_albedo':   sat_alb,
                            'sat_sza':      sat_sza,
                            'n_days':       int(len(subset)),
                            'doy':          int(cdate.dayofyear)})
        except Exception as e:
            print(f"  {site_label} {os.path.basename(sial_path)}: {e}")
    df_g = (pd.DataFrame(records).set_index('date').sort_index()
            if records else pd.DataFrame())
    if len(df_g) > 0:
        df_g = remove_outliers(df_g, 'sat_albedo', 'tower_albedo')
    return df_g


def match_greenland_appx(tower_daily, site_lat, site_lon, years, site_label):
    """Pentad-mean tower vs APP-x."""
    all_dates      = tower_daily.index
    pentad_starts  = pd.date_range(all_dates.min().floor('5D'),
                                   all_dates.max().ceil('5D'), freq='5D')
    records = []
    for cdate in pentad_starts:
        pend   = cdate + pd.Timedelta(days=5)
        mask   = (tower_daily.index >= cdate) & (tower_daily.index < pend)
        subset = tower_daily[mask]
        if len(subset) < GREENLAND_MIN_DAYS:
            continue
        avg_albedo = float(subset['albedo'].mean())
        if not np.isfinite(avg_albedo):
            continue
        sat_alb, slot_used, _ = get_appx_albedo_greenland(
            site_lat, site_lon, cdate, utc_hour=14)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(site_lat, site_lon, cdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        records.append({'date':         cdate,
                        'tower_albedo': avg_albedo,
                        'sat_albedo':   sat_alb,
                        'doy':          int(cdate.dayofyear)})
    df_g = (pd.DataFrame(records).set_index('date').sort_index()
            if records else pd.DataFrame())
    if len(df_g) > 0:
        df_g = remove_outliers(df_g, 'sat_albedo', 'tower_albedo')
    return df_g


summit_daily = load_summit_tower(summit_tower_file)
raven_daily  = load_raven_tower(raven_sleigh_dir)
print(f"  Summit: {len(summit_daily)} valid days  |  Raven: {len(raven_daily)} valid days")

df_summit_c = match_greenland_clara(summit_daily, SUMMIT_LAT, SUMMIT_LON,
                                     f"{dir_sial}/SALpm201[23]*.nc", 'Summit')
df_raven_c  = match_greenland_clara(raven_daily, RAVEN_LAT, RAVEN_LON,
                                     f"{dir_sial}/SALpm2024*.nc", 'Raven')
df_summit_a = match_greenland_appx(summit_daily, SUMMIT_LAT, SUMMIT_LON, [2012, 2013], 'Summit')
df_raven_a  = match_greenland_appx(raven_daily, RAVEN_LAT, RAVEN_LON, [2024], 'Raven')
print(f"  Summit CLARA={len(df_summit_c)}  APP-x={len(df_summit_a)}")
print(f"  Raven  CLARA={len(df_raven_c)}   APP-x={len(df_raven_a)}")

stats_summit_c = calc_stats(df_summit_c['tower_albedo'].values if len(df_summit_c) else [],
                              df_summit_c['sat_albedo'].values   if len(df_summit_c) else [])
stats_raven_c  = calc_stats(df_raven_c['tower_albedo'].values  if len(df_raven_c)  else [],
                              df_raven_c['sat_albedo'].values    if len(df_raven_c)  else [])
stats_summit_a = calc_stats(df_summit_a['tower_albedo'].values if len(df_summit_a) else [],
                              df_summit_a['sat_albedo'].values   if len(df_summit_a) else [])
stats_raven_a  = calc_stats(df_raven_a['tower_albedo'].values  if len(df_raven_a)  else [],
                              df_raven_a['sat_albedo'].values    if len(df_raven_a)  else [])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL E/M — MOSAiC HELiX Drone  (|ΔSZA| ≤ MAX_DELTA_SZA, code-1 exact)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading HELiX Drone")

drone_records_raw = []
for fpath in sorted(glob.glob(os.path.join(drone_dir, "MOSAiC_CU_HELIX_*.nc"))):
    try:
        ds         = xr.open_dataset(fpath)
        base_time  = pd.Timestamp(float(ds['base_time'].values), unit='s', tz='UTC')
        time_off   = ds['time_offset'].values.astype(float)
        timestamps = pd.to_datetime(
            [base_time + pd.Timedelta(seconds=float(t)) for t in time_off])
        mdate = base_time.floor('D').tz_localize(None)

        lat     = ds['latitude'].values.astype(float)
        lon     = ds['longitude'].values.astype(float)
        alt     = ds['relative_alt'].values.astype(float)
        up      = ds['upwelling_shortwave_flux_in_air'].values.astype(float)
        down    = ds['downwelling_shortwave_flux_in_air'].values.astype(float)
        qc_flag = ds['qc_flag_shortwave_flux_in_air'].values.astype(float)
        pitch   = ds['platform_pitch'].values.astype(float)
        roll    = ds['platform_roll'].values.astype(float)
        ds.close()

        alb_raw = np.where(down > 0, up / down, np.nan)
        qc_mask = ((qc_flag == 0) & (down > MIN_DOWN_SW_DRONE) &
                   np.isfinite(alb_raw) & (alb_raw >= 0.03) & (alb_raw <= 1.0) &
                   (alt >= ALT_MIN_VALID_M) & (alt <= ALT_MAX_VALID_M) &
                   (np.abs(pitch) <= MAX_PITCH_ROLL_DEG) &
                   (np.abs(roll)  <= MAX_PITCH_ROLL_DEG) &
                   np.isfinite(lat) & np.isfinite(lon))
        if int(qc_mask.sum()) < MIN_VALID_PTS_DRONE:
            continue

        alb_qc = alb_raw[qc_mask]; alt_qc = alt[qc_mask]
        lat_qc = lat[qc_mask];     lon_qc = lon[qc_mask]
        ts_qc  = timestamps[qc_mask]

        alt_eff = np.maximum(alt_qc, ALT_WEIGHT_FLOOR_M)
        weights = alt_eff**2; weights /= weights.sum()

        drone_albedo = float(np.sum(weights * alb_qc))
        avg_lat      = float(np.average(lat_qc, weights=weights))
        avg_lon      = float(np.average(lon_qc, weights=weights))
        alt_mean_w   = float(np.average(alt_qc, weights=weights))

        loc = pvlib.location.Location(latitude=avg_lat, longitude=avg_lon)
        sza_vals = []
        for ts in ts_qc:
            try:
                sp = loc.get_solarposition(pd.DatetimeIndex([ts]))
                sza_vals.append(float(sp['apparent_zenith'].iloc[0]))
            except:
                sza_vals.append(np.nan)
        sza_arr   = np.array(sza_vals); fin = np.isfinite(sza_arr)
        drone_sza = (float(np.sum((weights[fin] / weights[fin].sum()) * sza_arr[fin]))
                     if fin.sum() > 0 else np.nan)
        if not np.isfinite(drone_sza) or drone_sza >= MAX_SZA:
            continue

        mid_ts   = ts_qc[len(ts_qc) // 2]
        utc_hour = mid_ts.hour + mid_ts.minute / 60.0 if hasattr(mid_ts, 'hour') else 14.0

        drone_records_raw.append({'date':        mdate,
                                   'lat':         avg_lat,
                                   'lon':         avg_lon,
                                   'drone_albedo':drone_albedo,
                                   'drone_sza':   drone_sza,
                                   'utc_hour':    utc_hour,
                                   'alt_mean_w':  alt_mean_w})
    except Exception as e:
        print(f"  {os.path.basename(fpath)}: {e}")

print(f"  {len(drone_records_raw)} drone flights (before satellite match)")


def match_drone_clara(records):
    rows = []
    for rec in records:
        mdate          = pd.Timestamp(rec['date'])
        lat, lon       = rec['lat'], rec['lon']
        pb, pa, wb, wa = find_bracketing_pentads(mdate, sat_dir)
        if pb is None:
            continue
        try:
            with xr.open_dataset(pb['file']) as ds_b:
                if 'blue_sky_albedo_all_mean' not in ds_b.variables:
                    continue
                alb_b   = ds_b['blue_sky_albedo_all_mean'].squeeze().values / 100.0
                lats_2d = ds_b['lat'].values; lons_2d = ds_b['lon'].values
            with xr.open_dataset(pa['file']) as ds_a:
                alb_a = ds_a['blue_sky_albedo_all_mean'].squeeze().values / 100.0
            sab_b = spatial_interp_idw(lat, lon, lats_2d, lons_2d, alb_b)
            sab_a = spatial_interp_idw(lat, lon, lats_2d, lons_2d, alb_a)
            if not (np.isfinite(sab_b) and np.isfinite(sab_a)):
                continue
            sat_albedo = wb * sab_b + wa * sab_a
            sat_sza_b  = get_sat_sza(lat, lon, pb['file'])
            sat_sza_a  = get_sat_sza(lat, lon, pa['file'])
            if   np.isfinite(sat_sza_b) and np.isfinite(sat_sza_a):
                sat_sza = wb * sat_sza_b + wa * sat_sza_a
            elif np.isfinite(sat_sza_b): sat_sza = sat_sza_b
            elif np.isfinite(sat_sza_a): sat_sza = sat_sza_a
            else: sat_sza = np.nan
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue
            rows.append({**rec,
                         'sat_albedo': sat_albedo,
                         'delta_sza':  rec['drone_sza'] - sat_sza})
        except:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'drone_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


def match_drone_appx(records):
    rows = []
    for rec in records:
        mdate    = pd.Timestamp(rec['date'])
        lat, lon = rec['lat'], rec['lon']
        utc_h    = rec['utc_hour']
        sat_alb, slot_used, _ = get_appx_albedo_composite(lat, lon, mdate, utc_h)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(lat, lon, mdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        rows.append({**rec,
                     'sat_albedo': sat_alb,
                     'delta_sza':  rec['drone_sza'] - sat_sza})
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'drone_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


df_drone_c = match_drone_clara(drone_records_raw)
df_drone_a = match_drone_appx(drone_records_raw)
print(f"  {len(df_drone_c)} drone flights (CLARA-A3)  |  {len(df_drone_a)} (APP-x)")
stats_drone_c = calc_stats(df_drone_c['drone_albedo'].values if len(df_drone_c) else [],
                            df_drone_c['sat_albedo'].values   if len(df_drone_c) else [])
stats_drone_a = calc_stats(df_drone_a['drone_albedo'].values if len(df_drone_a) else [],
                            df_drone_a['sat_albedo'].values   if len(df_drone_a) else [])

# ══════════════════════════════════════════════════════════════════════════════
# PANELS F/N — NASA ARISE 2014  (|ΔSZA| ≤ MAX_DELTA_SZA, nobs_snow>0, code-1 exact)
# PANELS G/O — NASA ARCSIX 2024  (|ΔSZA| ≤ MAX_DELTA_SZA, code-1 exact)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading Aircraft (ARISE + ARCSIX)")


def read_bbr_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_hdr = int(lines[0].strip().split(',')[0]); data = []
    for line in lines[n_hdr:]:
        if line.strip():
            try: data.append([float(x.strip()) for x in line.split(',')])
            except: continue
    data = np.array(data)
    utc = data[:, 0]; dn = data[:, 3]; up = data[:, 4]
    dn[dn == -9999.0] = np.nan; up[up == -9999.0] = np.nan
    albedo = np.full_like(dn, np.nan)
    ok = (dn > MIN_DOWN_SW_ARISE) & np.isfinite(dn) & np.isfinite(up)
    albedo[ok] = np.clip(up[ok] / dn[ok], 0, 1)
    return utc, albedo


def read_4star_nav(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_hdr = int(lines[0].strip().split(',')[0]); data = []
    for line in lines[n_hdr:]:
        if line.strip():
            try: data.append([float(x.strip()) for x in line.split(',')])
            except: continue
    data = np.array(data)
    utc = data[:, 0]; alt = data[:, 2]; c3 = data[:, 3]; c4 = data[:, 4]
    def _bad(arr): return float(np.sum(np.abs(arr) > 90)) / len(arr) if len(arr) else 0.0
    c3v = c3[(c3 != -9999.0) & np.isfinite(c3)]
    c4v = c4[(c4 != -9999.0) & np.isfinite(c4)]
    lat, lon = (c4, c3) if _bad(c3v) > 0.01 and _bad(c4v) <= 0.01 else (c3, c4)
    ok = ((lat != -9999.0) & (lon != -9999.0) & (alt != -9999.0) &
          np.isfinite(lat) & np.isfinite(lon))
    return utc[ok], lat[ok], lon[ok], alt[ok]


def compute_sza_pvlib_arr(lats, lons, timestamps_utc):
    loc = pvlib.location.Location(latitude=float(np.nanmean(lats)),
                                  longitude=float(np.nanmean(lons)))
    try:
        idx = pd.DatetimeIndex(timestamps_utc)
        if idx.tz is None:
            idx = idx.tz_localize('UTC')
        return float(loc.get_solarposition(idx)['apparent_zenith'].mean())
    except:
        return np.nan


def read_ict(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_hdr     = int(lines[0].split(',')[0])
    var_names = [v.strip() for v in lines[n_hdr - 1].split(',')]
    return pd.read_csv(filename, skiprows=n_hdr, header=None,
                       names=var_names, na_values=[-9999, -99999, -8888])


# ── build raw aircraft pixel records (shared) ────────────────────────────────
arise_pixel_records  = []
arcsix_pixel_records = []

# ARISE
for bbr_file in sorted(glob.glob(f"{arise_data_dir}/ARISE-BBR_C130_*.ict")):
    date_str  = bbr_file.split('_')[-2].replace('.ict', '')
    star_file = f"{arise_data_dir}/ARISE-4STAR-ZENRAD_C130_{date_str}_R0.ict"
    if not os.path.exists(star_file):
        continue
    try:
        utc_bbr, albedo                    = read_bbr_file(bbr_file)
        utc_nav, lat_nav, lon_nav, alt_nav = read_4star_nav(star_file)
        if len(utc_nav) == 0:
            continue
        lat_m = interp1d(utc_nav, lat_nav, bounds_error=False, fill_value=np.nan)(utc_bbr)
        lon_m = interp1d(utc_nav, lon_nav, bounds_error=False, fill_value=np.nan)(utc_bbr)
        flight_date  = pd.to_datetime(date_str, format='%Y%m%d')
        midnight_utc = pd.Timestamp(flight_date.date()).tz_localize('UTC')
        ts_utc       = midnight_utc + pd.to_timedelta(utc_bbr, unit='s')
        loc_arise = pvlib.location.Location(
            latitude=float(np.nanmean(lat_m[np.isfinite(lat_m)])),
            longitude=float(np.nanmean(lon_m[np.isfinite(lon_m)])))
        sza_arise = loc_arise.get_solarposition(
            ts_utc.tz_localize('UTC') if ts_utc.tz is None else ts_utc
        )['apparent_zenith'].values
        not_land  = create_land_mask(lat_m, lon_m)
        valid     = (np.isfinite(albedo) & np.isfinite(lat_m) &
                     np.isfinite(lon_m) & (lat_m >= 66.0) & not_land &
                     (sza_arise < MAX_SZA))
        if not np.any(valid):
            continue
        ac_lats = lat_m[valid]; ac_lons = lon_m[valid]
        ac_alb  = albedo[valid]; ac_sza  = sza_arise[valid]
        ac_ts   = ts_utc[valid]

        # Group by 0.25° cell
        cell_df = pd.DataFrame({
            'alb': ac_alb, 'sza': ac_sza, 'lat': ac_lats, 'lon': ac_lons,
            'cy': (ac_lats / 0.25).astype(int), 'cx': (ac_lons / 0.25).astype(int),
            'utc_h': [t.hour + t.minute / 60.0 if hasattr(t, 'hour') else 14.0
                      for t in ac_ts],
            'ts': list(ac_ts),
        })
        for (_, __), grp in cell_df.groupby(['cy', 'cx']):
            if len(grp) < MIN_PTS_ARISE:
                continue
            arise_pixel_records.append({
                'date':           flight_date,
                'lat':            float(grp['lat'].mean()),
                'lon':            float(grp['lon'].mean()),
                'aircraft_albedo':float(grp['alb'].mean()),
                'aircraft_sza':   float(grp['sza'].mean()),
                'utc_hour':       float(grp['utc_h'].mean()),
                'mean_dist_km':   float(grp[['lat', 'lon']].std().mean() * 111.0),
                'ts_list':        list(grp['ts']),
            })
    except Exception as e:
        print(f"  ARISE {date_str}: {e}")

# ARCSIX
for rad_file in sorted(arcsix_data_dir.glob("ARCSIX-BBR_P3B_*.ict")):
    date_str = rad_file.stem.split("_")[-2]
    nav_file = arcsix_data_dir / f"ARCSIX-MetNav_P3B_{date_str}_R0.ict"
    if not nav_file.exists():
        continue
    try:
        rad  = read_ict(rad_file); nav = read_ict(nav_file)
        data = pd.merge(rad, nav, on="Time_Start", how="inner")
        data["albedo"] = data["UP_CM22_SOLAR_Irrad"] / data["DN_CM22_SOLAR_Irrad"]
        data = data[(data["albedo"] >= 0) & (data["albedo"] <= 1) &
                    (data["Solar_Zenith_Angle"] < MAX_SZA)]
        if data.empty:
            continue
        not_land = create_land_mask(data["Latitude"].values, data["Longitude"].values)
        valid    = (data["Latitude"] >= 66.0).values & not_land
        if not np.any(valid):
            continue
        flight_date = pd.to_datetime(date_str, format='%Y%m%d')
        midnight    = pd.Timestamp(flight_date.date()).tz_localize('UTC')
        ac_lats     = data.loc[valid, "Latitude"].values
        ac_lons     = data.loc[valid, "Longitude"].values
        ac_alb      = data.loc[valid, "albedo"].values
        ac_sza      = data.loc[valid, "Solar_Zenith_Angle"].values
        ac_utc      = (midnight + pd.to_timedelta(data.loc[valid, "Time_Start"].values, unit='s'))
        ac_utch     = np.array([t.hour + t.minute / 60.0 for t in ac_utc])
        cell_df = pd.DataFrame({
            'alb': ac_alb, 'sza': ac_sza, 'lat': ac_lats, 'lon': ac_lons,
            'cy': (ac_lats / 0.25).astype(int), 'cx': (ac_lons / 0.25).astype(int),
            'utc_h': ac_utch,
        })
        for (_, __), grp in cell_df.groupby(['cy', 'cx']):
            if len(grp) < MIN_PTS_ARCSIX:
                continue
            arcsix_pixel_records.append({
                'date':           flight_date,
                'lat':            float(grp['lat'].mean()),
                'lon':            float(grp['lon'].mean()),
                'aircraft_albedo':float(grp['alb'].mean()),
                'aircraft_sza':   float(grp['sza'].mean()),
                'utc_hour':       float(grp['utc_h'].mean()),
                'mean_dist_km':   float(grp[['lat', 'lon']].std().mean() * 111.0),
            })
    except Exception as e:
        print(f"  ARCSIX {date_str}: {e}")

print(f"  ARISE pixel records: {len(arise_pixel_records)}")
print(f"  ARCSIX pixel records: {len(arcsix_pixel_records)}")


def _match_aircraft_clara(pixel_records, nobs_snow_filter=False):
    rows = []
    for rec in pixel_records:
        mdate = pd.Timestamp(rec['date'])
        lat, lon = rec['lat'], rec['lon']
        pentad_file = find_pentad_file_for_date(mdate, sat_dir)
        if pentad_file is None:
            continue
        try:
            with xr.open_dataset(pentad_file) as ds:
                if 'blue_sky_albedo_all_mean' not in ds.variables:
                    continue
                all_alb_2d   = ds['blue_sky_albedo_all_mean'].squeeze().values / 100.0
                lats_2d      = ds['lat'].values; lons_2d = ds['lon'].values
                nobs_snow_2d = (ds['nobs_snow'].squeeze().values
                                if 'nobs_snow' in ds.variables
                                else np.ones_like(all_alb_2d))
            if nobs_snow_filter:
                mask = (np.isfinite(all_alb_2d) & (all_alb_2d >= 0) &
                        (all_alb_2d <= 1) & (nobs_snow_2d > 0))
            else:
                mask = np.isfinite(all_alb_2d) & (all_alb_2d >= 0) & (all_alb_2d <= 1)
            if not np.any(mask):
                continue
            sat_alb = spatial_interp_idw(lat, lon, lats_2d, lons_2d,
                                          np.where(mask, all_alb_2d, np.nan))
            if not np.isfinite(sat_alb):
                continue
            sat_sza = get_sat_sza(lat, lon, pentad_file)
            if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
                continue
            delta_sza = rec['aircraft_sza'] - sat_sza
            rows.append({**rec, 'sat_albedo': sat_alb, 'delta_sza': delta_sza})
        except:
            pass
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'aircraft_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


def _match_aircraft_appx(pixel_records):
    rows = []
    for rec in pixel_records:
        mdate    = pd.Timestamp(rec['date'])
        lat, lon = rec['lat'], rec['lon']
        utc_h    = rec['utc_hour']
        sat_alb, slot_used, _ = get_appx_albedo_composite(lat, lon, mdate, utc_h)
        if not np.isfinite(sat_alb):
            continue
        sat_sza = get_appx_sza(lat, lon, mdate, slot_used or 14)
        if not np.isfinite(sat_sza) or sat_sza >= MAX_SZA:
            continue
        delta_sza = rec['aircraft_sza'] - sat_sza
        rows.append({**rec, 'sat_albedo': sat_alb, 'delta_sza': delta_sza})
    df = pd.DataFrame(rows)
    if len(df):
        df = remove_outliers(df, 'sat_albedo', 'aircraft_albedo')
        df = sza_filter(df, 'delta_sza')
    return df


df_arise_c  = _match_aircraft_clara(arise_pixel_records,  nobs_snow_filter=True)
df_arise_a  = _match_aircraft_appx(arise_pixel_records)
df_arcsix_c = _match_aircraft_clara(arcsix_pixel_records, nobs_snow_filter=False)
df_arcsix_a = _match_aircraft_appx(arcsix_pixel_records)
print(f"  ARISE  CLARA={len(df_arise_c)}  APP-x={len(df_arise_a)}")
print(f"  ARCSIX CLARA={len(df_arcsix_c)} APP-x={len(df_arcsix_a)}")

stats_arise_c  = calc_stats(df_arise_c['aircraft_albedo'].values  if len(df_arise_c)  else [],
                             df_arise_c['sat_albedo'].values       if len(df_arise_c)  else [])
stats_arise_a  = calc_stats(df_arise_a['aircraft_albedo'].values  if len(df_arise_a)  else [],
                             df_arise_a['sat_albedo'].values       if len(df_arise_a)  else [])
stats_arcsix_c = calc_stats(df_arcsix_c['aircraft_albedo'].values if len(df_arcsix_c) else [],
                             df_arcsix_c['sat_albedo'].values      if len(df_arcsix_c) else [])
stats_arcsix_a = calc_stats(df_arcsix_a['aircraft_albedo'].values if len(df_arcsix_a) else [],
                             df_arcsix_a['sat_albedo'].values      if len(df_arcsix_a) else [])

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS  (code-1 exact, ylabel now a parameter)
# ══════════════════════════════════════════════════════════════════════════════

def _add_trend(ax, x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan
    z  = np.polyfit(x[valid], y[valid], 1)
    xf = np.linspace(float(np.nanmin(x[valid])), float(np.nanmax(x[valid])), 200)
    ax.plot(xf, np.polyval(z, xf), 'r-', lw=1.8, zorder=4,
            label=f'y={z[0]:.2f}x{z[1]:+.2f}')
    return z[0], z[1]


def _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel,
              xlim=(0, 1), ylim=(0, 1)):
    lo = max(xlim[0], ylim[0]); hi = min(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.45, zorder=2, label='1:1')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=8.5)
    ax.text(0.03, 0.97, _stats_text(s_d), transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, lw=0.5))
    ax.set_title(f'{panel_label}  {title}', loc='left', fontsize=9, pad=4)


def draw_panel(ax, x, y, c_vals, cmap, norm, s_d,
               panel_label, title, xlabel, ylabel,
               colorbar_label=None, xlim=(0, 1), ylim=(0, 1)):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sc = ax.scatter(x, y, c=c_vals, cmap=cmap, norm=norm,
                    s=38, edgecolors='k', linewidth=0.4, alpha=0.75, zorder=3)
    _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.8, handlelength=1.4)
    if colorbar_label is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label(colorbar_label, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    return sc


def draw_surface_panel(ax, x, y, surface_col, surf_patches_,
                        s_d, panel_label, title, xlabel, ylabel):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    c_vals = [SURFACE_COLORS.get(s, '#7f7f7f') for s in surface_col]
    ax.scatter(x, y, c=c_vals, s=38, edgecolors='k',
               linewidth=0.4, alpha=0.75, zorder=3)
    z0, z1 = _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel)
    handles = surf_patches_[:]
    if np.isfinite(z0):
        handles = [Line2D([0], [0], color='r', lw=1.8,
                          label=f'y={z0:.2f}x{z1:+.2f}')] + handles
    handles += [Line2D([0], [0], color='k', ls='--', lw=1.2,
                       alpha=0.45, label='1:1')]
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              framealpha=0.8, ncol=2, handlelength=1.1)


def draw_greenland_panel(ax, x, y, doy, s_d,
                          panel_label, title, xlabel, ylabel, site_label=''):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() > 0:
        pad  = 0.04
        dmin = max(0.0, min(x[valid].min(), y[valid].min()) - pad)
        dmax = min(1.0, max(x[valid].max(), y[valid].max()) + pad)
    else:
        dmin, dmax = 0.5, 1.0
    sc = ax.scatter(x, y, c=doy, cmap='viridis',
                    norm=plt.Normalize(vmin=1, vmax=366),
                    s=38, edgecolors='k', linewidth=0.4, alpha=0.8, zorder=3)
    _add_trend(ax, x, y)
    _frame_ax(ax, s_d, panel_label, title, xlabel, ylabel,
              xlim=(dmin, dmax), ylim=(dmin, dmax))
    if site_label:
        ax.text(0.03, 0.03, f'{site_label}\nPentad mean match\n(no SIC blend)',
                transform=ax.transAxes, fontsize=7, va='bottom', ha='left',
                style='italic', color='#444444')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.8, handlelength=1.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75, aspect=16)
    cb.set_label('Day of Year', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)


def no_data_panel(ax, panel_label, title, reason='No data available'):
    ax.text(0.5, 0.5, reason, ha='center', va='center',
            transform=ax.transAxes, fontsize=9, color='gray', style='italic')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{panel_label}  {title}', loc='left', fontsize=9, pad=4)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE  4 rows × 4 columns
# Row 0 (A–D): CLARA-A3  Tower | Transects | Heli | Summit
# Row 1 (E–H): APP-x     Tower | Transects | Heli | Summit
# Row 2 (I–L): CLARA-A3  Drone | ARISE     | ARCSIX | Raven
# Row 3 (M–P): APP-x     Drone | ARISE     | ARCSIX | Raven
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Building 16-panel figure (4 rows × 4 columns)...")

fig, axes = plt.subplots(4, 4, figsize=(26, 26),
                          gridspec_kw={'hspace': 0.52, 'wspace': 0.36})

CLARA_Y = 'CLARA-A3 Albedo'
APPX_Y  = 'APP-x Albedo'

# ─────────────────────────────────────────────────────────────────────────────
# Row 0 — CLARA-A3
# ─────────────────────────────────────────────────────────────────────────────

# A) Tower CLARA-A3
ax = axes[0, 0]
if len(x_tower_c) > 0:
    draw_panel(ax, x_tower_c, y_tower_c, doy_tower_c,
               'viridis', plt.Normalize(1, 366), stats_tower_c,
               'A)', 'MOSAiC Tower (CLARA-A3)',
               'Tower Albedo\n(SZA-matched × SIC blend)', CLARA_Y,
               colorbar_label='Day of Year')
else:
    no_data_panel(ax, 'A)', 'MOSAiC Tower (CLARA-A3)')

# B) Transects CLARA-A3
ax = axes[0, 1]
if len(df_transect_c) > 0:
    draw_surface_panel(ax,
        df_transect_c['transect_albedo'].values,
        df_transect_c['sat_albedo'].values,
        df_transect_c['surface_type'].values,
        surf_patches, stats_transect_c,
        'B)', f'MOSAiC Transects (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Transect Albedo', CLARA_Y)
else:
    no_data_panel(ax, 'B)', f'MOSAiC Transects (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# C) Helicopter CLARA-A3
ax = axes[0, 2]
if len(df_heli_c) > 0 and 'heli_albedo_corr' in df_heli_c.columns:
    df_ch   = df_heli_c[df_heli_c['heli_albedo_corr'].notna()].copy()
    norm_ch = plt.Normalize(vmin=0, vmax=MAX_DIST_HELI)
    sc_ch   = None
    for instr, mkr in {'helicopter': 'o', 'HELiPOD': '^'}.items():
        sub = df_ch[df_ch['instrument'] == instr]
        if sub.empty: continue
        sc_ch = ax.scatter(sub['heli_albedo_corr'], sub['sat_albedo'],
                           c=sub['mean_dist_km'], cmap='cividis', norm=norm_ch,
                           s=50, marker=mkr, edgecolors='k', linewidth=0.4,
                           alpha=0.75, zorder=3,
                           label=f'{instr} (n={len(sub)})')
    _add_trend(ax, df_ch['heli_albedo_corr'].values, df_ch['sat_albedo'].values)
    _frame_ax(ax, stats_heli_c, 'C)',
              'MOSAiC Heli+HELiPOD\n(CLARA-A3, linear SZA corr.)',
              'Helicopter/HELiPOD Albedo\n(SZA-corrected)', CLARA_Y)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    if sc_ch is not None:
        cb = fig.colorbar(sc_ch, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label('Dist. to sat. pixel (km)', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
else:
    no_data_panel(ax, 'C)', 'MOSAiC Heli+HELiPOD (CLARA-A3)')

# D) Summit CLARA-A3
ax = axes[0, 3]
if len(df_summit_c) > 0:
    draw_greenland_panel(ax,
        df_summit_c['tower_albedo'].values,
        df_summit_c['sat_albedo'].values,
        df_summit_c['doy'].values,
        stats_summit_c,
        'D)', 'Summit Greenland (CLARA-A3, 2012–2013)',
        'Tower Albedo (pentad mean)', CLARA_Y,
        site_label=f'Summit ({SUMMIT_LAT:.2f}°N, {abs(SUMMIT_LON):.2f}°W)')
else:
    no_data_panel(ax, 'D)', 'Summit Greenland (CLARA-A3, 2012–2013)',
                  reason='No Summit data\n(check SALpm2012/13 files)')

# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — APP-x
# ─────────────────────────────────────────────────────────────────────────────

# E) Tower APP-x
ax = axes[1, 0]
if len(x_tower_a) > 0:
    draw_panel(ax, x_tower_a, y_tower_a, doy_tower_a,
               'viridis', plt.Normalize(1, 366), stats_tower_a,
               'E)', 'MOSAiC Tower (APP-x)',
               'Tower Albedo\n(SZA-matched × SIC blend)', APPX_Y,
               colorbar_label='Day of Year')
else:
    no_data_panel(ax, 'E)', 'MOSAiC Tower (APP-x)')

# F) Transects APP-x
ax = axes[1, 1]
if len(df_transect_a) > 0:
    surf_patches_a = [mpatches.Patch(color=SURFACE_COLORS.get(s, '#7f7f7f'),
                                      label=s.title())
                      for s in SURFACE_PRIORITY + ['unknown']
                      if s in df_transect_a['surface_type'].values]
    draw_surface_panel(ax,
        df_transect_a['transect_albedo'].values,
        df_transect_a['sat_albedo'].values,
        df_transect_a['surface_type'].values,
        surf_patches_a, stats_transect_a,
        'F)', f'MOSAiC Transects (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Transect Albedo', APPX_Y)
else:
    no_data_panel(ax, 'F)', f'MOSAiC Transects (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# G) Helicopter APP-x
ax = axes[1, 2]
if len(df_heli_a) > 0 and 'heli_albedo_corr' in df_heli_a.columns:
    df_ah   = df_heli_a[df_heli_a['heli_albedo_corr'].notna()].copy()
    norm_ah = plt.Normalize(vmin=0, vmax=MAX_DIST_HELI)
    sc_ah   = None
    for instr, mkr in {'helicopter': 'o', 'HELiPOD': '^'}.items():
        sub = df_ah[df_ah['instrument'] == instr]
        if sub.empty: continue
        sc_ah = ax.scatter(sub['heli_albedo_corr'], sub['sat_albedo'],
                           c=sub['mean_dist_km'], cmap='cividis', norm=norm_ah,
                           s=50, marker=mkr, edgecolors='k', linewidth=0.4,
                           alpha=0.75, zorder=3,
                           label=f'{instr} (n={len(sub)})')
    _add_trend(ax, df_ah['heli_albedo_corr'].values, df_ah['sat_albedo'].values)
    _frame_ax(ax, stats_heli_a, 'G)',
              'MOSAiC Heli+HELiPOD\n(APP-x, linear SZA corr.)',
              'Helicopter/HELiPOD Albedo\n(SZA-corrected)', APPX_Y)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    if sc_ah is not None:
        cb = fig.colorbar(sc_ah, ax=ax, pad=0.02, shrink=0.75, aspect=16)
        cb.set_label('Dist. to sat. pixel (km)', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
else:
    no_data_panel(ax, 'G)', 'MOSAiC Heli+HELiPOD (APP-x)')

# H) Summit APP-x
ax = axes[1, 3]
if len(df_summit_a) > 0:
    draw_greenland_panel(ax,
        df_summit_a['tower_albedo'].values,
        df_summit_a['sat_albedo'].values,
        df_summit_a['doy'].values,
        stats_summit_a,
        'H)', 'Summit Greenland (APP-x, 2012–2013)',
        'Tower Albedo (pentad mean)', APPX_Y,
        site_label=f'Summit ({SUMMIT_LAT:.2f}°N, {abs(SUMMIT_LON):.2f}°W)')
else:
    no_data_panel(ax, 'H)', 'Summit Greenland (APP-x, 2012–2013)',
                  reason='No Summit data or\nno APP-x files for 2012–13')

# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — CLARA-A3
# ─────────────────────────────────────────────────────────────────────────────

# I) Drone CLARA-A3
ax = axes[2, 0]
if len(df_drone_c) > 0:
    alt_vals = df_drone_c['alt_mean_w'].values
    draw_panel(ax,
        df_drone_c['drone_albedo'].values, df_drone_c['sat_albedo'].values,
        alt_vals, 'plasma',
        plt.Normalize(vmin=0, vmax=np.nanpercentile(alt_vals, 95)),
        stats_drone_c,
        'I)', f'MOSAiC Drone (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Drone Albedo\n(altitude-weighted)', CLARA_Y,
        colorbar_label='Alt-weighted mean AGL (m)')
else:
    no_data_panel(ax, 'I)', f'MOSAiC Drone (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# J) ARISE CLARA-A3
ax = axes[2, 1]
if len(df_arise_c) > 0:
    draw_panel(ax,
        df_arise_c['aircraft_albedo'].values, df_arise_c['sat_albedo'].values,
        df_arise_c['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARISE), stats_arise_c,
        'J)', f'NASA ARISE 2014 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', CLARA_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'J)', f'NASA ARISE 2014 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# K) ARCSIX CLARA-A3
ax = axes[2, 2]
if len(df_arcsix_c) > 0:
    draw_panel(ax,
        df_arcsix_c['aircraft_albedo'].values, df_arcsix_c['sat_albedo'].values,
        df_arcsix_c['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARCSIX), stats_arcsix_c,
        'K)', f'NASA ARCSIX 2024 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', CLARA_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'K)', f'NASA ARCSIX 2024 (CLARA-A3, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# L) Raven CLARA-A3
ax = axes[2, 3]
if len(df_raven_c) > 0:
    draw_greenland_panel(ax,
        df_raven_c['tower_albedo'].values,
        df_raven_c['sat_albedo'].values,
        df_raven_c['doy'].values,
        stats_raven_c,
        'L)', 'Raven / SLEIGH (CLARA-A3, 2024)',
        'SLEIGH Tower Albedo (pentad mean)', CLARA_Y,
        site_label=f'Raven ({RAVEN_LAT:.2f}°N, {abs(RAVEN_LON):.2f}°W)')
else:
    no_data_panel(ax, 'L)', 'Raven / SLEIGH (CLARA-A3, 2024)',
                  reason='No Raven data\n(check SLEIGH path\nor SALpm2024 files)')

# ─────────────────────────────────────────────────────────────────────────────
# Row 3 — APP-x
# ─────────────────────────────────────────────────────────────────────────────

# M) Drone APP-x
ax = axes[3, 0]
if len(df_drone_a) > 0:
    alt_vals_a = df_drone_a['alt_mean_w'].values
    draw_panel(ax,
        df_drone_a['drone_albedo'].values, df_drone_a['sat_albedo'].values,
        alt_vals_a, 'plasma',
        plt.Normalize(vmin=0, vmax=np.nanpercentile(alt_vals_a, 95)),
        stats_drone_a,
        'M)', f'MOSAiC Drone (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Drone Albedo\n(altitude-weighted)', APPX_Y,
        colorbar_label='Alt-weighted mean AGL (m)')
else:
    no_data_panel(ax, 'M)', f'MOSAiC Drone (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# N) ARISE APP-x
ax = axes[3, 1]
if len(df_arise_a) > 0:
    draw_panel(ax,
        df_arise_a['aircraft_albedo'].values, df_arise_a['sat_albedo'].values,
        df_arise_a['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARISE), stats_arise_a,
        'N)', f'NASA ARISE 2014 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', APPX_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'N)', f'NASA ARISE 2014 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# O) ARCSIX APP-x
ax = axes[3, 2]
if len(df_arcsix_a) > 0:
    draw_panel(ax,
        df_arcsix_a['aircraft_albedo'].values, df_arcsix_a['sat_albedo'].values,
        df_arcsix_a['mean_dist_km'].values,
        'cividis', plt.Normalize(vmin=0, vmax=MAX_DIST_ARCSIX), stats_arcsix_a,
        'O)', f'NASA ARCSIX 2024 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)',
        'Aircraft Albedo', APPX_Y,
        colorbar_label='Dist. to sat. pixel (km)')
else:
    no_data_panel(ax, 'O)', f'NASA ARCSIX 2024 (APP-x, |ΔSZA|≤{MAX_DELTA_SZA}°)')

# P) Raven APP-x
ax = axes[3, 3]
if len(df_raven_a) > 0:
    draw_greenland_panel(ax,
        df_raven_a['tower_albedo'].values,
        df_raven_a['sat_albedo'].values,
        df_raven_a['doy'].values,
        stats_raven_a,
        'P)', 'Raven / SLEIGH (APP-x, 2024)',
        'SLEIGH Tower Albedo (pentad mean)', APPX_Y,
        site_label=f'Raven ({RAVEN_LAT:.2f}°N, {abs(RAVEN_LON):.2f}°W)')
else:
    no_data_panel(ax, 'P)', 'Raven / SLEIGH (APP-x, 2024)',
                  reason='No Raven data or\nno APP-x files for 2024')

# ── Row stripe labels ─────────────────────────────────────────────────────────
'''stripe_info = [
    (0.99, '#2563a8', 'Rows A–D  |  CLARA-A3  (pentad, ~25 km)'),
    (0.745,'#b35a00', 'Rows E–H  |  APP-x  (daily, 04 or 14 UTC slot)'),
    (0.498,'#2563a8', 'Rows I–L  |  CLARA-A3  (pentad, ~25 km)'),
    (0.248,'#b35a00', 'Rows M–P  |  APP-x  (daily, 04 or 14 UTC slot)'),
]
for y_pos, color, label in stripe_info:
    fig.text(0.005, y_pos, label, fontsize=9, fontweight='bold',
             va='top', ha='left', color='white',
             bbox=dict(boxstyle='round,pad=0.35', fc=color, alpha=0.90, lw=0))'''

# ── Super-title ───────────────────────────────────────────────────────────────
'''fig.suptitle(
    f'Arctic & Greenland In-Situ Albedo vs CLARA-A3 and APP-x\n'
    f'All panels: SZA < {MAX_SZA:.0f}°  │  '
    f'Panels B,F,I,J,K,M,N,O: |ΔSZA| ≤ {MAX_DELTA_SZA:.0f}°  │  '
    f'Panels C,G: linear SZA correction  │  '
    f'Panels D,H,L,P: pentad-mean tower match (no SIC blend)',
    fontsize=11, fontweight='bold', y=1.012)'''

outpath = os.path.join(output_dir, 'combined_16panel_validation.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {outpath}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 88)
print(f"{'Pan':<4}  {'Dataset':<40}  {'Product':<10}  "
      f"{'r':>7}  {'bias':>8}  {'RMSE':>8}  {'n':>6}")
print("-" * 88)
all_stats_list = [
    ('A', 'MOSAiC Tower (SZA-matched, SIC blend)',                'CLARA-A3', stats_tower_c),
    ('B', f'MOSAiC Transects (|ΔSZA|≤{MAX_DELTA_SZA}°)',         'CLARA-A3', stats_transect_c),
    ('C', 'MOSAiC Helicopter+HELiPOD (linear SZA corr.)',         'CLARA-A3', stats_heli_c),
    ('D', 'Summit Greenland 2012–13 (pentad mean)',                'CLARA-A3', stats_summit_c),
    ('E', 'MOSAiC Tower (SZA-matched, SIC blend)',                'APP-x',    stats_tower_a),
    ('F', f'MOSAiC Transects (|ΔSZA|≤{MAX_DELTA_SZA}°)',         'APP-x',    stats_transect_a),
    ('G', 'MOSAiC Helicopter+HELiPOD (linear SZA corr.)',         'APP-x',    stats_heli_a),
    ('H', 'Summit Greenland 2012–13 (pentad mean)',                'APP-x',    stats_summit_a),
    ('I', f'HELiX Drone (|ΔSZA|≤{MAX_DELTA_SZA}°)',              'CLARA-A3', stats_drone_c),
    ('J', f'ARISE 2014 nobs_snow>0 (|ΔSZA|≤{MAX_DELTA_SZA}°)',  'CLARA-A3', stats_arise_c),
    ('K', f'ARCSIX 2024 (|ΔSZA|≤{MAX_DELTA_SZA}°)',              'CLARA-A3', stats_arcsix_c),
    ('L', 'Raven Greenland 2024 SLEIGH (pentad mean)',             'CLARA-A3', stats_raven_c),
    ('M', f'HELiX Drone (|ΔSZA|≤{MAX_DELTA_SZA}°)',              'APP-x',    stats_drone_a),
    ('N', f'ARISE 2014 (|ΔSZA|≤{MAX_DELTA_SZA}°)',               'APP-x',    stats_arise_a),
    ('O', f'ARCSIX 2024 (|ΔSZA|≤{MAX_DELTA_SZA}°)',              'APP-x',    stats_arcsix_a),
    ('P', 'Raven Greenland 2024 SLEIGH (pentad mean)',             'APP-x',    stats_raven_a),
]
for label, name, product, s in all_stats_list:
    r_str    = f"{s['r']:7.3f}"     if np.isfinite(s['r'])    else '    nan'
    bias_str = f"{s['bias']:+8.4f}" if np.isfinite(s['bias']) else '     nan'
    rmse_str = f"{s['rmse']:8.4f}"  if np.isfinite(s['rmse']) else '     nan'
    print(f"  {label:<3}  {name:<40}  {product:<10}  "
          f"{r_str}  {bias_str}  {rmse_str}  {s['n']:>6d}")
print("=" * 88)