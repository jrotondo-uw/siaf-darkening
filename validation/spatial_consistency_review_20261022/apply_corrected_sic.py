from pathlib import Path
import pickle,pandas as pd,numpy as np
R=Path(__file__).resolve().parent
with (R/'validation_IDW_only_controlled.pkl').open('rb') as f:D=pickle.load(f)
a=pd.read_csv(R/'tower_blending_audit.csv',parse_dates=['date'])
for prod in ['c','a']:
 d=D['df_tower_'+prod].copy(); q=a[a['product']==prod].set_index('date').reindex(d.index)
 assert q.reference_corrected_sic.notna().all()
 np.testing.assert_allclose(d.sial_blue,q.satellite_idw,equal_nan=True)
 np.testing.assert_allclose(d.tower_sza_matched.fillna(d.tower_scaled),q.reference_stored)
 d['tower_reference_before_sic_fix']=d.tower_sza_matched.fillna(d.tower_scaled)
 d['tower_sza_matched']=q.reference_corrected_sic
 d['sic_corrected']=q.sic_corrected
 D['df_tower_'+prod]=d
 x=d.tower_sza_matched;y=d.sial_blue;ok=np.isfinite(x)&np.isfinite(y);e=y[ok]-x[ok]
 D['x_tower_'+prod]=x.to_numpy();D['y_tower_'+prod]=y.to_numpy();D['doy_tower_'+prod]=d.index.dayofyear.to_numpy()
 D['stats_tower_'+prod]=dict(n=int(ok.sum()),bias=float(e.mean()),rmse=float(np.sqrt((e**2).mean())),r=float(x[ok].corr(y[ok])))
 print(prod,D['stats_tower_'+prod])
D['sic_correction_provenance']='MOSAiC tower SIC lookup uses SIC file declared CRS (EPSG:3411), no extra 45-degree rotation. Uses reference_corrected_sic from tower_blending_audit.csv. Satellite IDW-only values, prior selected observations, SZA matching and temporal selection unchanged. Audit can be regenerated with audit_tower_blend.py.'
with (R/'validation_IDW_corrected_SIC.pkl').open('wb') as f:pickle.dump(D,f)
s=(R/'plot_figure9_no_fallback.py').read_text().replace('validation_IDW_only_controlled.pkl','validation_IDW_corrected_SIC.pkl').replace('Figure_9_IDW_only','Figure_9_corrected_SIC').replace('Figure 9 — IDW only, no nearest-neighbor fallback','Figure 9 — In-situ albedo validation').replace('Spatial-only review: prior retained observations and temporal matching; four-neighbor IDW only for both products; no nearest-neighbor fallback','Four-neighbor IDW for both products; corrected MOSAiC tower SIC coordinates; prior observation selection and temporal matching retained')
(R/'plot_figure9_corrected_sic.py').write_text(s)
