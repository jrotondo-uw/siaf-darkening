#!/usr/bin/env python
"""How much of each footprint does the SIC-variability filter remove (31 models)?

ols_bivariate (bivariate_diag.py) refuses the bivariate fit where the SIC
sigma over the record is below MIN_SIC_SD = 0.005 (constant SIC included), and
member_bivariate then drops those pixels from every term of the region means.
The Methods claim, written for the 24-model run, is that this touches only
BCC-CSM2-MR (2.8% of its area-loss and 1.6% of its darkening footprint).

Here the fit is wrapped so that the refused pixels are KEPT (b = D = 0, so the
region selection still takes them) and flagged; each row then carries the
pixel count and cos-latitude area with and without them. Same members, same
grid_patch, same masks as bivariate_model_31.py. Writes sd_filter_check.csv
next to the main run and cross-checks n_pixels - n_filtered against it.
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028")
import bivariate_model_31 as B31     # noqa: E402  (applies grid_patch on import)

C, BM = B31.C, B31.BM
_orig = BM.ols_bivariate
BM.FILTERED = [None]


def _keep_all(y, x1, x2, *a, **k):
    b, d, n, r, uy, ux = _orig(y, x1, x2, *a, **k)
    fill = np.isfinite(uy) & np.isfinite(ux) & ~(np.isfinite(b) & np.isfinite(d))
    b, d = b.copy(), d.copy()
    b[fill], d[fill] = 0.0, 0.0
    BM.FILTERED[0] = fill
    return b, d, n, r, uy, ux


BM.ols_bivariate = _keep_all

# add area and filtered counts to each row (source patch, asserted once)
import inspect   # noqa: E402
_src = inspect.getsource(BM.member_bivariate)
_old = "region=reg, n_pixels=int(sel.sum()), n_years=len(yrs),"
assert _src.count(_old) == 1, "row dict not found"
_src = _src.replace(_old, (
    "region=reg, n_pixels=int(sel.sum()), n_years=len(yrs), "
    "area=float(coslat[sel].sum()), "
    "n_filtered=int((sel & FILTERED[0]).sum()), "
    "area_filtered=float(coslat[sel & FILTERED[0]].sum()),"))
exec(compile(_src, BM.__file__, "exec"), BM.__dict__)

OUT = "/glade/derecho/scratch/jrotondo/fig5_rerun31_patched_20261123"


def one(job):
    model, retained = job
    t0 = time.time()
    try:
        inst, mem = B31.pick(model, retained)
        C.MODEL_INFO[model] = dict(institution=inst, member=mem)
        T = C.compute_T_annual_global(model, mem, C.TREND_YEARS)
        rows = BM.member_bivariate(model, mem, T)
        return model, rows, None, time.time() - t0
    except Exception as e:
        return model, [], f"{type(e).__name__}: {e} || " + traceback.format_exc()[-400:], time.time() - t0


def main():
    import multiprocessing as mp
    jobs = sorted(B31.ensemble().items())
    rows = []
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "14")), maxtasksperchild=1) as pool:
        for k, (model, r, err, dt) in enumerate(pool.imap_unordered(one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {model}: {'FAILED ' + err if err else len(r)} ({dt:.0f}s)", flush=True)
            rows.extend(r)
    cols = ["model", "member", "month", "region", "n_pixels", "n_filtered", "area", "area_filtered"]
    a = pd.DataFrame(rows)[cols]
    if os.environ.get("SAVE_FULL") == "1":
        # I and L are plain univariate slopes, so with the refused pixels kept
        # they ARE the unfiltered values; b, bL and D are not (filled with 0)
        # and are deliberately not written.
        pd.DataFrame(rows)[["model", "member", "month", "region", "n_pixels", "n_years", "I", "L"]].to_csv(
            os.path.join(OUT, "model_bivariate_by_month_nofilter.csv"), index=False)
        print("wrote model_bivariate_by_month_nofilter.csv")
    base = pd.read_csv(os.path.join(OUT, "model_bivariate_by_month.csv"))
    m = a.merge(base[["model", "month", "region", "n_pixels"]].rename(columns={"n_pixels": "n_main"}),
                on=["model", "month", "region"], how="outer", indicator=True)
    m.to_csv(os.path.join(OUT, "sd_filter_check.csv"), index=False)

    both = m[m._merge == "both"]
    bad = both[both.n_pixels - both.n_filtered != both.n_main]
    print(f"\ncross-check vs main run: {len(both)} rows matched, {len(bad)} mismatched")
    if len(bad):
        print(bad.to_string(index=False))
    print("rows only in the main run:", m[m._merge == "right_only"][["model", "month", "region"]].values.tolist())
    print("rows only WITHOUT the filter (region < 20 px with it):")
    print(m[m._merge == "left_only"].to_string(index=False))

    g = a.groupby(["model", "region"])[["n_pixels", "n_filtered", "area", "area_filtered"]].sum()
    g["px_pct"] = 100 * g.n_filtered / g.n_pixels
    g["area_pct"] = 100 * g.area_filtered / g.area
    print("\nfiltered share, all six months pooled (models with any):")
    print(g[g.n_filtered > 0].round(2).to_string())
    mm = a[a.n_filtered > 0].assign(px_pct=lambda x: 100 * x.n_filtered / x.n_pixels,
                                    area_pct=lambda x: 100 * x.area_filtered / x.area)
    print("\nby month:")
    print(mm.round(2).to_string(index=False))
    tot = a.groupby("region")[["n_pixels", "n_filtered", "area", "area_filtered"]].sum()
    print("\nwhole ensemble:", (100 * tot.area_filtered / tot.area).round(3).to_dict())


if __name__ == "__main__":
    main()
