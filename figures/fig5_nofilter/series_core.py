"""Direct-slope block bootstrap on one region-mean series pair.

Line for line the L and I parts of fig5_uncertainty_20261112/bootstrap_core.refits
for a single column: residuals about each series' own mean and the common MMM
calendar trend, circular 3-year blocks over calendar positions 1982-2014, the
common trend restored at target positions, missing whole years left missing.
The joint (partial-slope) refit is not computed.
"""
import numpy as np


def block_indices(B=4000, n=33, L=3, seed=77):
    rng = np.random.default_rng(seed); start = rng.integers(0, n, (B, (n + L - 1) // L))
    return ((start[:, :, None] + np.arange(L)) % n).reshape(B, -1)[:, :n]


def refits_direct(a, c, t, years, slopes, indices):
    years = np.asarray(years, int); n = 33; B = len(indices)
    x = np.arange(1982, 2015, dtype=float); x -= x.mean()
    av = np.zeros(n); av[years - 1982] = 1
    count = np.zeros((B, n)); xcount = count.copy(); x2count = count.copy()
    bi = np.arange(B)[:, None]
    np.add.at(count, (bi, indices), 1); np.add.at(xcount, (bi, indices), x[None, :])
    np.add.at(x2count, (bi, indices), x[None, :] ** 2)
    N = count @ av; X = xcount @ av; X2 = x2count @ av
    assert np.min(N) >= 10
    ba, bc, bt = slopes
    rt = np.zeros(n); rt[years - 1982] = t - np.mean(t) - bt * x[years - 1982]
    st = count @ rt + bt * X; stt = count @ (rt * rt) + 2 * bt * (xcount @ rt) + bt * bt * X2; stt -= st * st / N
    assert np.all(stt > 1e-12)
    targetrt = xcount @ rt
    ra = np.zeros(n); rc = np.zeros(n)
    ra[years - 1982] = a - a.mean() - ba * x[years - 1982]
    rc[years - 1982] = c - c.mean() - bc * x[years - 1982]
    sa = count @ ra + ba * X; sc = count @ rc + bc * X
    xa = xcount @ ra; xc = xcount @ rc
    tc = count @ (rc * rt) + bt * xc + bc * targetrt + bc * bt * X2 - sc * st / N
    ta = count @ (ra * rt) + bt * xa + ba * targetrt + ba * bt * X2 - sa * st / N
    return -tc / stt, -ta / stt
