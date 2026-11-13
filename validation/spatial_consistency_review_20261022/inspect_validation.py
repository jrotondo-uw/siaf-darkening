import pickle,pandas as pd
p='/glade/work/jrotondo/combined_validation/validation_data.pkl'
with open(p,'rb') as f:d=pickle.load(f)
for k,v in d.items():
 if k.startswith('df_'):
  print('\n',k,len(v),list(v.columns))
  cols=[c for c in v.columns if any(w in c.lower() for w in ['date','time','pentad','doy'])]
  print(v[cols].head(3).to_string(index=False))
 elif k.startswith(('x_tower','y_tower','doy_tower')):print(k,len(v))
