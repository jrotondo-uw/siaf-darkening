"""IS analogue of uncertainty_joint_20261023/extract.py (24 Nov 2026).

Per member: the annual, area-weighted albedo-anomaly series with NO radiative
weighting, Q_IS(t) = <-[alpha(t) - alpha_bar]>, built on exactly the masks, zero
fill and Arctic-ocean domain averaging that give is_tot_arctic
(compute_spatial_temporal_mean_is), so its regression on temperature is that
member's IS. Temperatures are the ones saved by the joint extraction (identical
to compute_T_annual_global output). No observations enter.
usage: extract_is.py <worker k> <n workers> <max attempts per member>
"""
import sys, traceback, inspect, gc, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/glade/derecho/scratch/jrotondo/uncertainty_is_20261124')
(ROOT/'members').mkdir(parents=True, exist_ok=True)
JOINT = Path('/glade/derecho/scratch/jrotondo/uncertainty_joint_20261023')
sys.path.insert(0, '/glade/derecho/scratch/jrotondo/uncertainty_mmm_20261023/deps')
from global_land_mask import globe  # noqa: F401  (same dependency set as the joint run)
MAX_ATTEMPTS = 2
sys.path.insert(0, '/glade/u/home/jrotondo/work/siaf_drk_paper_final/python')
sys.path.insert(0, '/glade/derecho/scratch/jrotondo')


LATN = ('lat', 'latitude', 'nav_lat', 'lat_2d', 'TLAT')


def _north_subset(da, ds, lim=45.):
    """Lazy spatial subset to the rows that reach `lim` N before loading (login-node memory).
    Every SIC point north of `lim` is kept, so the nearest-neighbour SIC for any albedo
    point >= 60N, and therefore every quantity this script saves, is unchanged; each
    member's point IS is re-checked against is_tot_arctic afterwards."""
    lat = next((da.coords[n] for n in LATN if n in da.coords), None)
    if lat is None:
        return da
    lv = np.asarray(lat.values, dtype=float)
    lv = np.where(np.abs(lv) <= 90., lv, np.nan)
    if lat.ndim == 1 and lat.dims[0] in da.dims:
        idx = np.where(lv >= lim)[0]
        return da.isel({lat.dims[0]: idx}) if len(idx) else da
    if lat.ndim == 2 and all(dd in da.dims for dd in lat.dims):
        with np.errstate(all='ignore'):
            r = np.where(np.nanmax(lv, axis=1) >= lim)[0]
            c = np.where(np.nanmax(lv, axis=0) >= lim)[0]
        if 0 < len(r) < lv.shape[0]:
            return da.isel({lat.dims[0]: slice(r.min(), r.max() + 1)})
        if 0 < len(c) < lv.shape[1]:
            return da.isel({lat.dims[1]: slice(c.min(), c.max() + 1)})
    return da


def prep(m):
    import grid_patch
    if getattr(m, '_is_ready', False):
        return
    import dask
    dask.config.set(scheduler='synchronous')   # one chunk at a time: bounded memory
    grid_patch.apply(m)
    orig = m.load_filtered

    def load_filtered_lean(files, variable, years, months):
        if variable != 'siconc':
            return orig(files, variable, years, months)
        try:   # identical to load_filtered except for the lazy northern subset
            ds = m._open_robust(files, variable)
            da = ds[variable]
            tvals = da.time.values
            _, first_occ = np.unique(tvals, return_index=True)
            if len(first_occ) < len(tvals):
                da = da.isel(time=np.sort(first_occ))
            ys = set(years); ms = set(months)
            mask = np.array([int(t.dt.year) in ys and int(t.dt.month) in ms
                             for t in da.time], dtype=bool)
            da = _north_subset(da.isel(time=mask), ds).load(); ds.close()
            da, _, _, _ = m._std_latlon(da)
            return da
        except Exception as e:
            print(f'  load_filtered_lean({variable}): {e}', flush=True)
            return None
    m.load_filtered = load_filtered_lean
    src = inspect.getsource(m.compute_member_IS_native)
    src = src.replace('        result[mn] = {',
                      '        _is_capture[mn] = (np.asarray(valid_years), stk.astype(np.float32), '
                      '(changing & np.isfinite(dalpha_dT)).copy(), (stable & np.isfinite(dalpha_dT)).copy())\n'
                      '        result[mn] = {')
    assert src.count('_is_capture[mn]') == 1
    exec(compile(src, 'captured_original_IS', 'exec'), m.__dict__)
    m._is_ready = True


def task(model, member):
    out = ROOT/'members'/f'{model}_{member}.npz'
    if out.exists():
        return
    att = ROOT/'members'/f'{model}_{member}.attempts'
    n_att = int(att.read_text()) if att.exists() else 0
    if n_att >= MAX_ATTEMPTS:
        return
    att.write_text(str(n_att + 1))   # survives an OOM kill, so a retry pass can skip repeat offenders
    t0 = time.time()
    try:
        import cmip6_siaf_calculation as m
        prep(m)
        m._is_capture = {}
        z = np.load(JOINT/'members'/f'{model}_{member}.npz')
        temps = {int(y): float(v) for y, v in zip(z['years'], z['temperature'])}
        if len(temps) != 33:
            raise ValueError('Need all 33 annual model temperatures')
        fields, ctx = m.compute_member_IS_native(model, member, temps)
        if fields is None or any(fields.get(mn) is None for mn in m.MONTH_NUMS):
            raise ValueError('Incomplete albedo months')
        years = np.arange(1982, 2015); comps = ['tot', 'ext', 'drk']
        qa = np.full((6, 33, 3), np.nan); point = np.full((6, 3), np.nan)
        for mi, mn in enumerate(m.MONTH_NUMS):
            ay, alpha, changing, stable = m._is_capture[mn]
            annual_alpha = {int(y): a for y, a in zip(ay, alpha)}
            alpha_mean = np.nanmean(alpha, axis=0)
            support = [changing | stable, changing, stable]
            support = [s & np.isfinite(alpha_mean) for s in support]
            for ci, c in enumerate(comps):
                point[mi, ci] = m.arctic_ocean_mean_native(fields[mn][c], ctx)
            for yi, year in enumerate(years):
                if int(year) not in annual_alpha:
                    continue
                # fixed spatial support; a missing annual albedo is a zero anomaly (as in the joint run)
                da = np.nan_to_num(annual_alpha[int(year)] - alpha_mean, nan=0.)
                for ci in range(3):
                    af = m.fill_zero_domain(np.where(support[ci], -da, np.nan), ctx['lat2d'])
                    qa[mi, yi, ci] = m.arctic_ocean_mean_native(af, ctx)
        valid = np.isfinite(qa).all(axis=(0, 2))
        if valid.sum() < 25:
            raise ValueError(f'Only {valid.sum()} complete annual records')
        q = qa[:, valid, :].mean(axis=0)
        assert np.allclose(q[:, 0], q[:, 1] + q[:, 2], atol=1e-7)
        tmp = out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, model=model, member=member, years=years[valid],
                            temperature=np.array([temps[int(y)] for y in years[valid]]),
                            q_is=q, original_point=point.mean(axis=0), n_months=6)
        tmp.rename(out)
        att.unlink(missing_ok=True)
        print(f'DONE {model} {member} {int(valid.sum())} {time.time()-t0:.0f}s', flush=True)
    except Exception:
        (ROOT/'members'/f'{model}_{member}.error').write_text(traceback.format_exc())
        print(f'FAIL {model} {member}', flush=True)
    finally:
        try:
            m._is_capture = {}
        except Exception:
            pass
        gc.collect()


if __name__ == '__main__':
    k, K, MAX_ATTEMPTS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    d = pd.read_csv(JOINT/'requested_members.csv')
    todo = list(d.itertuples(index=False, name=None))[k::K]
    print(f'worker {k}/{K}: {len(todo)} members', flush=True)
    for model, member in todo:
        task(model, member)
    print('WORKER FINISHED', k, flush=True)
