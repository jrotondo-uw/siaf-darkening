from pathlib import Path
import pickle,numpy as np,pandas as pd,datetime,shutil,json
R=Path(__file__).resolve().parent
with (R/'validation_IDW_fallback.pkl').open('rb') as f:old=pickle.load(f)
with (R/'validation_fallback_corrected_SIC.pkl').open('rb') as f:new=pickle.load(f)
for k,v in old.items():
 if 'tower_' in k:continue
 if isinstance(v,pd.DataFrame):pd.testing.assert_frame_equal(v,new[k])
 elif isinstance(v,np.ndarray):np.testing.assert_equal(v,new[k])
for p in ['c','a']:
 np.testing.assert_equal(old['y_tower_'+p],new['y_tower_'+p])
 assert new['stats_tower_'+p]['n']==(20 if p=='c' else 86)
for kind in ['tower','transect','heli','drone','raven','arcsix']:
 d=new['df_'+kind+'_c']
 print(kind,'dates:',str(d.index.min()),str(d.index.max()),'datecolumn:',d['date'].min() if 'date' in d else '')
for ext in ['png','pdf']:assert (R/('Figure_9_chronological.'+ext)).stat().st_size>10000
B=R/'backups'/('before_chronological_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));B.mkdir(parents=True)
main=Path('/glade/work/jrotondo/albedo_satellite_validation/make_fig6_20panel.py');nb=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb');F=Path('/glade/work/jrotondo/ncc_paper/figures_final')
for p in [main,nb]+[F/(stem+'.'+ext) for stem in ['f06','f09'] for ext in ['png','pdf']]:
 if p.exists():shutil.copy2(p,B/p.name)
s=main.read_text().replace('IDW-only sampling','IDW with nearest-neighbor backup').replace('apply_corrected_sic.py','build_chronological.py').replace('plot_figure9_corrected_sic.py','plot_figure9_chronological.py').replace('Figure_9_corrected_SIC','Figure_9_chronological')
main.write_text(s)
for stem in ['f09','f06']:
 for ext in ['png','pdf']:shutil.copy2(R/('Figure_9_chronological.'+ext),F/(stem+'.'+ext))
n=json.loads(nb.read_text())
for c in n['cells']:
 s=''.join(c['source'])
 if c['cell_type']=='markdown' and s.startswith('## Figure 9\n'):
  c['source']=['## Figure 9\n\n','Main version: chronological campaign order, paired CLARA-A3 / APP-x panels. Both products use four-neighbor IDW first, with valid nearest-cell spatial fallback within existing distance limits. Corrected MOSAiC tower SIC coordinates are retained; prior observation selection, SZA matching and temporal matching are unchanged. SHEBA daily radiometer observations are labeled surface flux (tower provenance not established); the separate 1998 surveys are labeled SHEBA transects. Coastal Barrow transects remain excluded.\n\n','Data: `'+str(R/'validation_fallback_corrected_SIC.pkl')+'`. Rebuild with `'+str(R/'build_chronological.py')+'`. Earlier variants are retained below.\n']
 if c['cell_type']=='code' and s.strip()=='%run '+str(main):c['outputs']=[];c['execution_count']=None
nb.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
print('Verified; promoted main figure. Backup:',B)
