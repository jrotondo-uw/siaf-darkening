#!/usr/bin/env python
"""
Figure 6 - validation of CLARA-A3 and APP-x against all ten in-situ and airborne
platforms, including SHEBA (1997-98) and Tara (2007).

4 rows x 5 columns = 20 panels, preserving the original convention:
rows 1 and 3 are CLARA-A3, rows 2 and 4 are APP-x.
Panels lettered A-T.

The eight original platforms come from the pickle written by save_validation_data.py.
SHEBA and Tara come from the collocations in albedo_satellite_validation/.
"""
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

PKL   = "/glade/work/jrotondo/combined_validation/validation_data.pkl"
BASE  = "/glade/work/jrotondo/albedo_satellite_validation"
OUTD  = "/glade/work/jrotondo/ncc_paper/figures_final"
os.makedirs(OUTD, exist_ok=True)

D = pickle.load(open(PKL, "rb"))
sheba_c = pd.read_csv(f"{BASE}/sheba_surface_flux/sheba_clara_matchups.csv")
sheba_a = pd.read_csv(f"{BASE}/sheba_surface_flux/sheba_appx_matchups.csv")
tara_c  = pd.read_csv(f"{BASE}/tara_surface_flux/tara_clara_matchups.csv")
tara_a  = pd.read_csv(f"{BASE}/tara_surface_flux/tara_appx_matchups.csv")

# in-situ column, satellite column, per platform key
COLS = {"tower":    ("tower_scaled",   "sial_blue"),
        "transect": ("transect_albedo", "sat_albedo"),
        "heli":     ("heli_albedo_corr","sat_albedo"),
        "drone":    ("drone_albedo",    "sat_albedo"),
        "arise":    ("aircraft_albedo", "sat_albedo"),
        "arcsix":   ("aircraft_albedo", "sat_albedo"),
        "summit":   ("tower_albedo",    "sat_albedo"),
        "raven":    ("tower_albedo",    "sat_albedo")}

NAMES = {"tower": "MOSAiC tower", "transect": "MOSAiC transects",
         "heli": "MOSAiC heli/HELiPOD", "drone": "MOSAiC HELiX drone",
         "arise": "ARISE 2014", "arcsix": "ARCSIX 2024",
         "summit": "Summit", "raven": "Raven/SLEIGH",
         "sheba": "SHEBA 1997-98", "tara": "Tara 2007"}

ORDER = ["tower", "transect", "heli", "drone", "arise",
         "arcsix", "summit", "raven", "sheba", "tara"]


def series(key, prod):
    """Return (x_insitu, y_sat) for one platform and product, or (None, None)."""
    if key == "sheba":
        df = sheba_c if prod == "c" else sheba_a
        return df["insitu"].values, df["sat"].values
    if key == "tara":
        df = tara_c if prod == "c" else tara_a
        return df["insitu"].values, df["sat"].values
    df = D.get(f"df_{key}_{prod}")
    if df is None or len(df) == 0:
        return None, None
    xc, yc = COLS[key]
    if xc not in df.columns or yc not in df.columns:
        return None, None
    s = df[[xc, yc]].dropna()
    return s[xc].values, s[yc].values


plt.rcParams.update({"font.family": "sans-serif", "axes.linewidth": 1.0,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

fig, axes = plt.subplots(4, 5, figsize=(17.5, 14.4))
letters = "ABCDEFGHIJKLMNOPQRST"
LO, HI = 0.0, 1.05

# row/col placement: rows 0,2 = CLARA-A3 ; rows 1,3 = APP-x
plan = []
for blk in (0, 1):                       # two blocks of five platforms
    for pi, key in enumerate(ORDER[blk * 5:(blk + 1) * 5]):
        plan.append((blk * 2 + 0, pi, key, "c"))
        plan.append((blk * 2 + 1, pi, key, "a"))

for r, c, key, prod in plan:
    ax = axes[r, c]
    idx = r * 5 + c
    x, y = series(key, prod)
    prod_name = "CLARA-A3" if prod == "c" else "APP-x"
    band = (key == "tara")

    if x is None or len(x) == 0:
        ax.text(0.5, 0.5, "no valid\nmatchups", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.scatter(x, y, s=16, c="#2E86AB" if prod == "c" else "#A23B72",
                   edgecolor="k", linewidth=0.3, alpha=0.75, zorder=3)
        ax.plot([LO, HI], [LO, HI], "k--", lw=1.0, zorder=2)
        b = y - x
        r_ = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
        rmse = float(np.sqrt((b ** 2).mean()))
        if len(x) > 2 and abs(r_) > 0.5:
            sl, ic = sps.linregress(x, y)[:2]
            xs = np.array([LO, HI]); ax.plot(xs, sl * xs + ic, "r-", lw=1.3, zorder=4)
        ax.text(0.04, 0.96,
                f"$r$ = {r_:+.2f}\nbias = {b.mean():+.3f}\nRMSE = {rmse:.3f}\n$n$ = {len(x)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
                bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=0.3))
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal")
        ax.grid(alpha=0.22, zorder=1)

    ttl = f"({letters[idx]}) {NAMES[key]}\n{prod_name}"
    if band:
        ttl += "  — 320–950 nm band"
        for sp in ax.spines.values():
            sp.set_color("#B00020"); sp.set_linewidth(1.7)
    ax.set_title(ttl, fontsize=9.2)
    if r == 3:
        ax.set_xlabel("in-situ albedo", fontsize=10)
    if c == 0:
        ax.set_ylabel("satellite albedo", fontsize=10)

fig.suptitle("Validation of CLARA-A3 and APP-x surface albedo against ten in-situ "
             "and airborne datasets", fontsize=14, y=0.995)
fig.text(0.5, 0.005,
         "Rows 1 and 3: CLARA-A3.  Rows 2 and 4: APP-x.  "
         "Red-framed panels (O,T) use a 320–950 nm band albedo and are not "
         "comparable with the broadband platforms.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout(rect=[0, 0.015, 1, 0.985])

for ext in ("png", "pdf"):
    p = f"{OUTD}/f06.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("wrote", p)

# panel letter map, for the caption
print("\npanel map:")
for r, c, key, prod in plan:
    print(f"  ({letters[r*5+c]}) {NAMES[key]:22s} {'CLARA-A3' if prod=='c' else 'APP-x'}")
