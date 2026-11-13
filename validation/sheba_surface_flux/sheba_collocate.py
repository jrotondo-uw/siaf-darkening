#!/usr/bin/env python
"""
SHEBA (1997-98) vs CLARA-A3 and APP-x surface albedo collocation.

Follows the method already used in this project for daily-mean platforms
(the Greenland-tower route in the Methods): pentad-mean to pentad-mean for
CLARA-A3, day-to-day for APP-x, with no instantaneous SZA filtering, because
the SHEBA record in hand is daily means and carries no per-minute SZA.

Grid conventions taken from mosiac_sfc_flux.ipynb:
  - CLARA-A3 SAL x/y are polar-stereographic compatible with EPSG:3413,
    so interp(x=, y=) works on raw EPSG:3413 coordinates.
  - NSIDC PSN25 SIC is rotated ~45 deg relative to EPSG:3413 true-east,
    so SIC lookups rotate (x, y) by -45 deg first.
  - In-situ albedo over sea ice is blended with an open-water prior
    alpha_ocn = 0.06 weighted by SIC, to represent the sub-pixel open water
    a point instrument cannot sample.
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
OUT = "/glade/work/jrotondo/albedo_satellite_validation/sheba_surface_flux"

ALPHA_WATER = 0.06          # Jin et al. 2004 open-ocean broadband albedo
ROTATION_ANGLE = -45.0      # NSIDC PSN25 grid only
PENTAD_DAYS = 5


def rotate(x, y, deg):
    a = np.deg2rad(deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


# ---------------------------------------------------------------- in situ
d = pd.read_csv(f"{RAD}/onice_daily_qc.csv")
s = d[(d.camp == "SHEBA 1997-98") & d.alb.notna()].copy()
s["date"] = pd.to_datetime(s["date"])
s = s[["date", "lat", "lon", "alb", "sic", "n_sw"]].sort_values("date").reset_index(drop=True)
print(f"SHEBA in-situ days with albedo: {len(s)}  "
      f"{s.date.min().date()} -> {s.date.max().date()}")
print(f"  lat {s.lat.min():.2f}-{s.lat.max():.2f}  "
      f"alb {s.alb.min():.3f}-{s.alb.max():.3f}  sic mean {s.sic.mean():.1f}")

tf = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
s["x"], s["y"] = tf.transform(s.lon.values, s.lat.values)

# SIC from the file, as a fraction; fall back to the NSIDC field if absent
s["fsic"] = np.clip(s.sic.values / 100.0, 0.0, 1.0)
s["alb_blend"] = s.fsic * s.alb + (1.0 - s.fsic) * ALPHA_WATER


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
    out = dict(platform="SHEBA 1997-98", product=label, n=len(df),
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
cl.to_csv(f"{OUT}/sheba_clara_matchups.csv", index=False)
ax.to_csv(f"{OUT}/sheba_appx_matchups.csv", index=False)
pd.DataFrame(res).to_csv(f"{OUT}/sheba_validation_stats.csv", index=False)
print(f"\nwrote matchup tables and stats to {OUT}")

# monthly breakdown, to see where the melt season sits
if len(cl):
    print("\nCLARA-A3 by month:")
    g = cl.assign(m=cl.pentad.dt.to_period("M")).groupby("m").agg(
        n=("sat", "size"), insitu=("insitu", "mean"), sat=("sat", "mean"))
    g["bias"] = g.sat - g.insitu
    print(g.round(3).to_string())
