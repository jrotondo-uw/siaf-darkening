#!/usr/bin/env python
"""Perovich broadband survey-line means vs CLARA-A3 and APP-x.
Uses existing SHEBA spatial samplers: linear CLARA; nearest APP-x grid point.
CLARA reference = mean of available spatial survey means within its pentad.
APP-x reference = spatial survey mean on the survey date. No interpolation in time,
no SIC blend, no spectral-to-broadband conversion. Camp geolocation is a proxy.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

RAD = "/glade/work/jrotondo/sheba_tara_nird/Joey/data/radiation"
DIR_SIAL = "/glade/derecho/scratch/jrotondo/clara_a3_obs"
DIR_APPX = "/glade/derecho/scratch/jrotondo/APP-x_observations"
DIR_SIC = "/glade/derecho/scratch/jrotondo/SIC_data"
OUT = "/glade/work/jrotondo/albedo_satellite_validation/sheba_perovich_transects"

ALPHA_WATER = 0.06          # Jin et al. 2004 open-ocean broadband albedo
ROTATION_ANGLE = -45.0      # NSIDC PSN25 grid only
PENTAD_DAYS = 5


def rotate(x, y, deg):
    a = np.deg2rad(deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


# Spatial survey means, with existing daily camp positions as location proxies.
s = pd.read_csv(f"{OUT}/perovich_transect_daily.csv", parse_dates=['date'])
print('Surveys:', len(s), 'missing positions:', s.lat.isna().sum())
s = s.dropna(subset=['lat','lon','alb']).copy()
tf = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
s['x'], s['y'] = tf.transform(s.lon.values, s.lat.values)
# Transect broadband measurements already sample the ice surface mosaic.
# Report unblended albedo, as for the existing MOSAiC transect validation.
s['fsic'] = 1.0
s['alb_blend'] = s.alb

# ---------------------------------------------------------------- CLARA-A3
rows = []
for f in sorted(glob.glob(f"{DIR_SIAL}/SALpm*.nc")):
    m = re.search(r"SALpm(\d{8})", os.path.basename(f))
    t0 = pd.Timestamp(m.group(1))
    t1 = t0 + pd.Timedelta(days=PENTAD_DAYS)
    sub = s[(s.date >= t0) & (s.date < t1)]
    if len(sub) == 0:
        continue
    ds = xr.open_dataset(f)
    xm, ym = float(sub.x.mean()), float(sub.y.mean())
    try:
        sat = float(ds.blue_sky_albedo_all_mean.isel(time=0)
                    .interp(x=xm, y=ym, method="linear").values) / 100.0
        sza = float(ds.sun_zenith_angle_mean.isel(time=0)
                    .interp(x=xm, y=ym, method="linear").values)
        nob = float(ds.nobs_total.isel(time=0)
                    .interp(x=xm, y=ym, method="nearest").values)
    except Exception as e:
        ds.close()
        continue
    ds.close()
    if not np.isfinite(sat) or sat <= 0 or sat > 1.05:
        continue
    rows.append(dict(pentad=t0, n_days=len(sub),
                     lat=sub.lat.mean(), lon=sub.lon.mean(),
                     insitu=sub.alb_blend.mean(), insitu_raw=sub.alb.mean(),
                     fsic=sub.fsic.mean(), sat=sat, sza=sza, nobs=nob))

cl = pd.DataFrame(rows)
print(f"\nCLARA-A3 pentad matchups: {len(cl)}")


# ---------------------------------------------------------------- APP-x
rows = []
for _, r in s.iterrows():
    stamp = r.date.strftime("%Y%m%d")
    cand = sorted(glob.glob(f"{DIR_APPX}/{r.date.year}/*_d{stamp}_*.nc"))
    if not cand:
        continue
    vals = []
    for f in cand:
        ds = xr.open_dataset(f)
        v = ds.get("cdr_surface_albedo")
        if v is None:
            ds.close()
            continue
        v = v.isel(Time=0) if "Time" in v.dims else v
        # APP-x carries 2-D latitude/longitude; nearest-neighbour by great circle
        la = ds.latitude.values
        lo = ds.longitude.values
        dist = (np.radians(la - r.lat) ** 2
                + (np.radians(lo - r.lon) * np.cos(np.radians(r.lat))) ** 2)
        j, k = np.unravel_index(np.nanargmin(dist), dist.shape)
        a = float(np.asarray(v)[j, k])
        ds.close()
        if np.isfinite(a) and 0 < a <= 1.05:
            vals.append(a)
    if vals:
        rows.append(dict(date=r.date, lat=r.lat, lon=r.lon,
                         insitu=r.alb_blend, insitu_raw=r.alb,
                         fsic=r.fsic, sat=float(np.mean(vals)), n_ov=len(vals)))

ax = pd.DataFrame(rows)
print(f"APP-x daily matchups: {len(ax)}")


# ---------------------------------------------------------------- stats
def stats(df, label):
    if len(df) < 3:
        print(f"{label}: too few matchups ({len(df)})")
        return None
    b = df.sat - df.insitu
    r = np.corrcoef(df.insitu, df.sat)[0, 1]
    out = dict(platform="SHEBA Perovich transects 1998", product=label, n=len(df),
               insitu_mean=df.insitu.mean(), sat_mean=df.sat.mean(),
               bias=b.mean(), mean_abs_diff=b.abs().mean(),
               rmse=float(np.sqrt((b ** 2).mean())), r=r,
               insitu_sd=df.insitu.std(), sat_sd=df.sat.std())
    print(f"\n{label}:  n={out['n']}  in-situ {out['insitu_mean']:.3f}  "
          f"sat {out['sat_mean']:.3f}")
    print(f"   bias {out['bias']:+.3f}   mean|diff| {out['mean_abs_diff']:.3f}   "
          f"RMSE {out['rmse']:.3f}   r {out['r']:+.3f}")
    return out


res = [x for x in (stats(cl, "CLARA-A3"), stats(ax, "APP-x")) if x]

os.makedirs(OUT, exist_ok=True)
cl.to_csv(f"{OUT}/perovich_clara_matchups.csv", index=False)
ax.to_csv(f"{OUT}/perovich_appx_matchups.csv", index=False)
pd.DataFrame(res).to_csv(f"{OUT}/perovich_validation_stats.csv", index=False)
print(f"\nwrote matchup tables and stats to {OUT}")

# monthly breakdown, to see where the melt season sits
if len(cl):
    print("\nCLARA-A3 by month:")
    g = cl.assign(m=cl.pentad.dt.to_period("M")).groupby("m").agg(
        n=("sat", "size"), insitu=("insitu", "mean"), sat=("sat", "mean"))
    g["bias"] = g.sat - g.insitu
    print(g.round(3).to_string())
