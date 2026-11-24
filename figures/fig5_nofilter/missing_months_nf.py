import pandas as pd, numpy as np
M = ['April', 'May', 'June', 'July', 'August', 'September']
d = pd.read_csv('/glade/derecho/scratch/jrotondo/fig5_rerun31_patched_20261123/model_bivariate_by_month_nofilter.csv')
o = pd.read_csv('/glade/derecho/scratch/jrotondo/sdfilter_test_20261123/obs_nofilter/stage3_bivariate.csv').set_index(['region', 'month'])
allm = sorted(d.model.unique())
for reg in ('area_loss', 'darkening'):
    for m in M:
        have = set(d[(d.region == reg) & (d.month == m)].model)
        miss = [x for x in allm if x not in have]
        if miss:
            print(reg, m, 'missing:', miss)
for reg, var in (('area_loss', 'L'), ('darkening', 'I')):
    p = d[d.region == reg].pivot_table(index='model', columns='month', values=var)[M]
    for mdl in p.index[p.isna().any(axis=1)]:
        av = [m for m in M if np.isfinite(p.loc[mdl, m])]
        mm = p.loc[mdl, av].mean(); ob = np.mean([o.loc[(reg, m), var] for m in av])
        print(f'{reg} {mdl}: {len(av)} months, model {mm:.4f} vs obs {ob:.4f} ->', 'below' if mm < ob else 'ABOVE')
    for m in M:
        v = d[(d.region == reg) & (d.month == m)].set_index('model')[var]
        ab = v[v >= o.loc[(reg, m), var]]
        print(f'  {reg} {m}: above obs {len(ab)}: {sorted(ab.index)}')
