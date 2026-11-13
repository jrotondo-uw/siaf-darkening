from pathlib import Path
import pickle,pandas as pd,numpy as np
R=Path(__file__).resolve().parent
with (R/'validation_IDW_fallback.pkl').open('rb') as f:D=pickle.load(f)
a=pd.read_csv(R/'tower_blending_audit.csv',parse_dates=['date'])
for prod in ['c','a']:
 d=D['df_tower_'+prod].copy();q=a[a['product']==prod].set_index('date').reindex(d.index)
 assert q.reference_corrected_sic.notna().all()
 np.testing.assert_allclose(d.tower_sza_matched.fillna(d.tower_scaled),q.reference_stored)
 d['tower_reference_before_sic_fix']=d.tower_sza_matched.fillna(d.tower_scaled)
 d['tower_sza_matched']=q.reference_corrected_sic;d['sic_corrected']=q.sic_corrected
 D['df_tower_'+prod]=d
 x=d.tower_sza_matched;y=d.sial_blue;ok=np.isfinite(x)&np.isfinite(y);e=y[ok]-x[ok]
 D['x_tower_'+prod]=x.to_numpy();D['y_tower_'+prod]=y.to_numpy();D['doy_tower_'+prod]=d.index.dayofyear.to_numpy()
 D['stats_tower_'+prod]=dict(n=int(ok.sum()),bias=float(e.mean()),rmse=float(np.sqrt((e**2).mean())),r=float(x[ok].corr(y[ok])))
 print(prod,D['stats_tower_'+prod],flush=True)
D['sic_correction_provenance']='Correct SIC-file CRS without extra rotation; reference_corrected_sic from tower_blending_audit.csv; original selected observations and SZA/temporal matching retained. Both satellites IDW first with valid nearest-cell spatial fallback within existing distance limits.'
with (R/'validation_fallback_corrected_SIC.pkl').open('wb') as f:pickle.dump(D,f)
s=(R/'plot_figure9_no_fallback.py').read_text().replace('validation_IDW_only_controlled.pkl','validation_fallback_corrected_SIC.pkl').replace('/no_fallback','/fallback').replace('Figure_9_IDW_only','Figure_9_chronological')
# Each pair occupies adjacent columns: CLARA-A3 left and APP-x right.
layout=[((4,0),(4,1),'SHEBA surface flux (1997–98)'),((5,0),(5,1),'SHEBA transects (1998)'),((4,2),(4,3),'Tara (2007)'),((0,3),(1,3),'Summit (2012–13)'),((2,1),(3,1),'ARISE (2014)'),((0,0),(1,0),'MOSAiC tower (2020)'),((0,1),(1,1),'MOSAiC transects (2020)'),((0,2),(1,2),'MOSAiC helicopter / HELiPOD (2020)'),((2,0),(3,0),'MOSAiC drone (2020)'),((2,3),(3,3),'Raven / SLEIGH (2024)'),((2,2),(3,2),'ARCSIX (2024)')]
insert='\n_chrono_axes = axes.copy()\n'
for i,(c,a,title) in enumerate(layout):
 for j,orig in enumerate([c,a]):insert+=f'axes[{orig[0]}, {orig[1]}] = _chrono_axes[{i//2}, {(i%2)*2+j}]\n'
insert+='axes[5,2] = _chrono_axes[5,2]\naxes[5,3] = _chrono_axes[5,3]\n'
s=s.replace("CLARA_Y = 'CLARA-A3 Albedo'",insert+"\nCLARA_Y = 'CLARA-A3 Albedo'")
start=s.index("axes[5,2].text(0, .95, 'Perovich broadband")
end=s.index('\noutpath =',start)
ending=''
for i,(_,_,title) in enumerate(layout):
 for j,prod in enumerate(['CLARA-A3','APP-x']):
  label=chr(65+2*i+j)
  ending+=f"_chrono_axes[{i//2}, {(i%2)*2+j}].set_title({(label+') '+title+chr(10)+prod)!r},loc='left',fontsize=FS_TITLE,pad=4)\n"
ending+="fig.suptitle('Figure 9 — In-situ albedo validation', fontsize=24, y=.94)\n"
s=s[:start]+ending+s[end:]
s+='\nall_stats_list.extend([(\"C\",\"SHEBA transects\",\"CLARA-A3\",_stats(_pc)),(\"D\",\"SHEBA transects\",\"APP-x\",_stats(_pa))])\n_pd.DataFrame([dict(dataset=name,product=product,**st) for _,name,product,st in all_stats_list]).to_csv("'+str(R/'Figure_9_chronological_statistics.csv')+'",index=False)\n'
(R/'plot_figure9_chronological.py').write_text(s)
compile(s,'plot_figure9_chronological.py','exec')
