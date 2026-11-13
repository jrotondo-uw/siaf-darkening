import numpy as np
src = open("bivariate_diag.py").read().split("# ------------------------------------------------------------------ main ---")[0]
ns = {}; exec(compile(src, "bd", "exec"), ns)
ols_bivariate, ols_univariate, detrend_time = ns["ols_bivariate"], ns["ols_univariate"], ns["detrend_time"]

rng = np.random.default_rng(0); T, N = 40, 500
Tvec = np.linspace(-0.4, 0.9, T) + rng.normal(0, .08, T)
b_true = rng.uniform(0.3, 0.9, N); d_true = rng.uniform(-0.08, -0.01, N)
c = (0.8 - 0.25*Tvec[:,None] + rng.normal(0, .05, (T,N))).clip(0,1)
a = 0.2 + b_true*c + d_true*Tvec[:,None] + rng.normal(0, .004, (T,N))
a[rng.random((T,N)) < 0.12] = np.nan
c[rng.random((T,N)) < 0.05] = np.nan

b, d, n, r, Iu, Lu = ols_bivariate(a, c, Tvec)
I, L, D = -Iu, -Lu, -d
ident = I - (b*L + D); fin = np.isfinite(ident)
ok = lambda cond: "PASS" if cond else "*** FAIL ***"

print(f"1 identity I=b*L+D  max|resid| {np.nanmax(np.abs(ident[fin])):.2e}  "
      f"{ok(np.nanmax(np.abs(ident[fin])) < 1e-12)}")
print(f"2 b recovered       corr {np.corrcoef(b[fin], b_true[fin])[0,1]:.5f}  "
      f"{ok(np.corrcoef(b[fin], b_true[fin])[0,1] > 0.99)}")
print(f"3 D recovered       corr {np.corrcoef(D[fin], -d_true[fin])[0,1]:.5f}  "
      f"{ok(np.corrcoef(D[fin], -d_true[fin])[0,1] > 0.95)}")

# support mismatch: albedo-only support must NOT equal matched support
uni_raw, _ = ols_univariate(a, Tvec)
mism = np.nanmax(np.abs(uni_raw[fin] + I[fin]))
print(f"4 support mismatch  max {mism:.2e}  "
      f"{ok(mism > 1e-6)}  (expected nonzero - this is the bias being measured)")

# on the MATCHED subset the two must agree exactly
a_m = np.where(np.isfinite(c), a, np.nan)
uni_m, _ = ols_univariate(a_m, Tvec)
agree = np.nanmax(np.abs(uni_m[fin] + I[fin]))
print(f"5 matched support   max {agree:.2e}  {ok(agree < 1e-12)}")

print(f"6 detrend pure ramp max {np.nanmax(np.abs(detrend_time(np.tile(np.arange(T,dtype=float).reshape(-1,1),(1,5))*3.+7.))):.2e}  "
      f"{ok(np.nanmax(np.abs(detrend_time(np.tile(np.arange(T,dtype=float).reshape(-1,1),(1,5))*3.+7.))) < 1e-10)}")

a2 = a.copy(); a2[:,0] = np.nan; a2[:5,0] = 1.0
b2, *_ = ols_bivariate(a2, c, Tvec)
print(f"7 min_n enforced    {ok(np.isnan(b2[0]))}")
print(f"\n   mean |r(c,T)| = {np.nanmean(np.abs(r)):.3f}  (synthetic data is "
      f"deliberately collinear; b still recovers)")
