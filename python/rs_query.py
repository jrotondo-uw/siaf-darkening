import pandas as pd, numpy as np, os
obs="/glade/work/jrotondo/ncc_paper/plots/sensitivity/siaf_pixelwise_summary.csv"
d=pd.read_csv(obs)
print("=== OBS RS (CMIP6 period, p=0.05) ===")
for m in ["CERES","Kernel"]:
    r=d[(d.rs_method==m)&(d.period=="CMIP6")&(np.isclose(d.pval,0.05))]
    if len(r):
        r=r.iloc[0]
        print(f"{m:7s}  RS_arctic(full)={r['RS_arctic']:.2f}  "
              f"RS_ext_pa={r['RS_ext_per_area']:.2f}  RS_drk_pa={r['RS_drk_per_area']:.2f}  "
              f"IS_tot_arctic={r['IS_tot_arctic']:.4f}")
# CMIP6 CSV columns
cm="/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/siaf_cmip6_summary.csv"
c=pd.read_csv(cm)
print("\n=== CMIP6 CSV columns (RS/IS) ===")
print([col for col in c.columns if 'rs' in col.lower() or 'is_' in col.lower()])
print("n rows:",len(c),"models:",c['model'].nunique() if 'model' in c else '?')
# sample one member
if 'rs_arctic' in c.columns:
    print("\nsample rs_arctic range:", c['rs_arctic'].min(), "-", c['rs_arctic'].max(),
          "mean:", c['rs_arctic'].mean())
for col in ['rs_ext_per_area','rs_drk_per_area','rs_tot_per_area']:
    if col in c.columns:
        print(f"  has {col}: mean={c[col].mean():.2f}")
    else:
        print(f"  MISSING {col}")
# spatial RS field availability
print("\n=== spatial RS field caches? ===")
import glob
for pat in ['/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/*rs*',
            '/glade/work/jrotondo/ncc_paper/sensitivity_cache_obs/*RS*',
            '/glade/work/jrotondo/ncc_paper/plots/sensitivity/*rs*npy',
            '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/spatial_cache*/*.npz']:
    g=glob.glob(pat)
    print(f"{pat}: {len(g)} files", g[:3])
