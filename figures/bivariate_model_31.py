#!/usr/bin/env python
"""Direct-slope decomposition rerun on the 31-model headline ensemble.

The original run (bivariate_model.py, 11 Nov 2026) covered the 25 models of
MODEL_INFO: five of them are not in the headline ensemble, and twelve of the
ensemble's models were never run. This driver uses exactly the headline
ensemble -- every model with at least three retained members in
siaf_cmip6_summary_final.csv (31 models, 378 members) -- and for each model
the first RETAINED member in lexical order, the rule of the four-way
diagnostic test. The regression itself is bivariate_model.member_bivariate,
unchanged, so masks, thresholds and the classification are identical.
"""
import glob
import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, "/glade/work/jrotondo/siaf_drk_paper_final/python")
sys.path.insert(0, "/glade/work/jrotondo/siaf_drk_paper_final/notebooks/revision_20261028")
import cmip6_siaf_calculation as C   # noqa: E402
import bivariate_model as BM         # noqa: E402

# The headline SIAF ensemble (recompute_shard.py) runs with grid_patch, whose
# root-merge fix completes members whose NCAR-collection series is gappy from
# the August ESGF staging (FGOALS-g3 r1-r3, NorESM2-LM r1). Without it those
# members load truncated. Applied at import so spawn workers get it too; it
# leaves members whose files already span 1982-2014 bit-for-bit unchanged.
if os.environ.get("GRID_PATCH", "1") == "1":
    sys.path.insert(0, "/glade/derecho/scratch/jrotondo")
    import grid_patch                # noqa: E402
    grid_patch.apply(C)

OUT = os.environ.get("OUT_DIR", "/glade/derecho/scratch/jrotondo/fig5_rerun31_patched_20261123")
SUMMARY = "/glade/derecho/scratch/jrotondo/siaf_cmip6_summary_final.csv"
ROOTS = [C.CMIP6_ROOT, C.STAGE_ROOT]


def ensemble():
    t = pd.read_csv(SUMMARY)
    ok = t[["siaf_global", "siaf_ext_global", "siaf_drk_global"]].notna().all(axis=1)
    t = t[ok]
    cnt = t.groupby("model").size()
    return {m: sorted(t[t.model == m].member) for m in sorted(cnt[cnt >= 3].index)}


def pick(model, retained):
    insts = sorted({p.split("/")[-3] for r in ROOTS
                    for p in glob.glob(f"{r}/*/{model}/historical")})
    for mem in retained:
        for inst in insts:
            if all(C.find_var_files(inst, model, mem, v)
                   for v in ("rsus", "rsds", "siconc", "tas")):
                return inst, mem
    return None, None


def one(job):
    model, retained = job
    t0 = time.time()
    inst = mem = None
    try:
        inst, mem = pick(model, retained)
        if mem is None:
            return (model, inst, mem, [],
                    "no retained member with rsus/rsds/siconc/tas", time.time() - t0)
        C.MODEL_INFO[model] = dict(institution=inst, member=mem)
        T = C.compute_T_annual_global(model, mem, C.TREND_YEARS)
        if not T:
            return model, inst, mem, [], "no tas", time.time() - t0
        rows = BM.member_bivariate(model, mem, T)
        if not rows:
            return model, inst, mem, [], "no rows", time.time() - t0
        return model, inst, mem, rows, None, time.time() - t0
    except Exception as e:
        tb = traceback.format_exc()[-800:].replace("\n", " | ")
        msg = f"{type(e).__name__}: {e}"[:300] + " || " + tb
        return model, inst, mem, [], msg, time.time() - t0


def main():
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    ens = ensemble()
    nproc = int(os.environ.get("NPROC", "14"))
    nmem = sum(len(v) for v in ens.values())
    print(f"{len(ens)} models, {nmem} retained members, nproc={nproc}", flush=True)
    jobs = sorted(ens.items())
    rows, status = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc, maxtasksperchild=1) as pool:
        for k, res in enumerate(pool.imap_unordered(one, jobs), 1):
            model, inst, mem, r, err, dt = res
            status.append(dict(model=model, institution=inst, member=mem,
                               n_rows=len(r), error=err or "", seconds=round(dt)))
            state = ("FAILED " + err) if err else ("OK %d rows" % len(r))
            print(f"[{k}/{len(jobs)}] {model} {mem}: {state} ({dt:.0f}s)", flush=True)
            rows.extend(r)
            pd.DataFrame(rows).to_csv(
                os.path.join(OUT, "model_bivariate_by_month.partial.csv"), index=False)
            pd.DataFrame(status).to_csv(os.path.join(OUT, "run_status.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "model_bivariate_by_month.csv"), index=False)
    per_model = df.groupby(["model", "region"])[["I", "b", "L", "bL", "D"]].mean()
    per_model.reset_index().to_csv(
        os.path.join(OUT, "model_bivariate_models.csv"), index=False)
    print(f"models with rows: {df.model.nunique()}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
