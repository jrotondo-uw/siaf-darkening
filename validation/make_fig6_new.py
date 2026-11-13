#!/usr/bin/env python
"""
New validation panels for Fig. 6: SHEBA 1997-98 and Tara 2007, each against
CLARA-A3 and APP-x. Panel style follows the existing Fig. 6 convention:
satellite albedo against in-situ albedo, 1:1 dashed reference, OLS fit in red
where |r| > 0.5, and r / bias / RMSE / n in the upper left.

Written as a separate file, NOT over f06.png: the full Fig. 6 is a 16-panel
grid over eight platforms, and the other eight platforms' matchups exist only
inside the per-platform notebooks, so they cannot be redrawn here.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

SHEBA = "/glade/work/jrotondo/albedo_satellite_validation/sheba_surface_flux"
TARA = "/glade/work/jrotondo/albedo_satellite_validation/tara_surface_flux"
OUT = "/glade/work/jrotondo/albedo_satellite_validation"

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.linewidth": 1.0, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 11, "savefig.dpi": 300, "figure.dpi": 300,
})

PANELS = [
    ("sheba_clara_matchups.csv", SHEBA, "pentad", "CLARA-A3", "SHEBA 1997-98", False),
    ("sheba_appx_matchups.csv", SHEBA, "date", "APP-x", "SHEBA 1997-98", False),
    ("tara_clara_matchups.csv", TARA, "pentad", "CLARA-A3", "Tara 2007", True),
    ("tara_appx_matchups.csv", TARA, "date", "APP-x", "Tara 2007", True),
]

fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
letters = "abcd"

for i, (fn, d, tcol, prod, plat, is_band) in enumerate(PANELS):
    ax = axes.flat[i]
    df = pd.read_csv(os.path.join(d, fn), parse_dates=[tcol])
    doy = df[tcol].dt.dayofyear
    sc = ax.scatter(df.insitu, df.sat, c=doy, cmap="viridis", s=34,
                    edgecolor="k", linewidth=0.4, zorder=3)
    lo, hi = 0.30, 1.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, zorder=2)

    b = (df.sat - df.insitu)
    r = np.corrcoef(df.insitu, df.sat)[0, 1]
    rmse = float(np.sqrt((b ** 2).mean()))
    if abs(r) > 0.5 and len(df) > 2:
        sl, ic = sps.linregress(df.insitu, df.sat)[:2]
        xs = np.array([lo, hi])
        ax.plot(xs, sl * xs + ic, "r-", lw=1.4, zorder=4)

    txt = (f"$r$ = {r:+.2f}\nbias = {b.mean():+.3f}\n"
           f"RMSE = {rmse:.3f}\n$n$ = {len(df)}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=0.35))

    ttl = f"({letters[i]}) {plat} vs {prod}"
    if is_band:
        ttl += "\n320–950 nm band albedo"
        for s in ax.spines.values():
            s.set_color("#B00020")
            s.set_linewidth(1.6)
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, zorder=1)
    if i // 2 == 1:
        ax.set_xlabel("in-situ albedo")
    if i % 2 == 0:
        ax.set_ylabel("satellite albedo")

cb = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.035, pad=0.03)
cb.set_label("day of year", fontsize=10)

fig.suptitle("Validation of CLARA-A3 and APP-x against the two pre-2012 drifting ice stations",
             fontsize=12, y=0.98)
fig.text(0.5, 0.005,
         "Red-framed panels use a 320–950 nm band albedo and are not comparable "
         "to the broadband platforms.",
         ha="center", fontsize=8.5, color="#B00020")

for p in (f"{OUT}/f06_new_platforms.png", f"{OUT}/f06_new_platforms.pdf"):
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)
