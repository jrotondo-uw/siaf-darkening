from pathlib import Path
import shutil,json,datetime,pickle,numpy as np,pandas as pd
R=Path(__file__).resolve().parent
with (R/'validation_IDW_only_controlled.pkl').open('rb') as f:old=pickle.load(f)
with (R/'validation_IDW_corrected_SIC.pkl').open('rb') as f:new=pickle.load(f)
for k,v in old.items():
 if 'tower_' in k:continue
 if isinstance(v,pd.DataFrame):pd.testing.assert_frame_equal(v,new[k])
 elif isinstance(v,np.ndarray):np.testing.assert_equal(v,new[k])
for p,n,rmse in [('c',19,.099795),('a',86,.355464)]:
 assert new['stats_tower_'+p]['n']==n
 assert abs(new['stats_tower_'+p]['rmse']-rmse)<1e-6
 np.testing.assert_equal(old['y_tower_'+p],new['y_tower_'+p])
for ext in ['png','pdf']:assert (R/('Figure_9_corrected_SIC.'+ext)).stat().st_size>10000
B=R/'backups'/('before_main_corrected_SIC_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));B.mkdir(parents=True)
main=Path('/glade/work/jrotondo/albedo_satellite_validation/make_fig6_20panel.py')
nb=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb')
F=Path('/glade/work/jrotondo/ncc_paper/figures_final')
for p in [main,nb]+[F/(stem+'.'+ext) for stem in ['f06','f09'] for ext in ['png','pdf']]:
 if p.exists():shutil.copy2(p,B/p.name)
# Preserve historical plotting scripts; make the original notebook entry point
# render the accepted IDW-only / corrected-SIC version on every future run.
main.write_text('''# Main Figure 9: IDW-only sampling, corrected MOSAiC tower SIC coordinates.
# Rebuild data with apply_corrected_sic.py in the review directory.
import runpy, shutil
from pathlib import Path
R = Path("'''+str(R)+'''")
runpy.run_path(str(R / "plot_figure9_corrected_sic.py"), run_name="__main__")
F = Path("'''+str(F)+'''")
for stem in ["f09", "f06"]:
    for ext in ["png", "pdf"]:
        shutil.copy2(R / ("Figure_9_corrected_SIC." + ext), F / (stem + "." + ext))
print("Updated main Figure 9 (f09 and legacy f06).")
''')
for stem in ['f09','f06']:
 for ext in ['png','pdf']:shutil.copy2(R/('Figure_9_corrected_SIC.'+ext),F/(stem+'.'+ext))
n=json.loads(nb.read_text())
for i,c in enumerate(n['cells']):
 s=''.join(c['source'])
 if c['cell_type']=='markdown' and s.strip()=='## Figure 9':
  c['source']=['## Figure 9\n','\n','Main version: four-neighbor IDW for both products, without nearest-neighbor spatial fallback. MOSAiC tower references use SIC at the correct location (the SIC file’s declared CRS, without the erroneous extra 45° rotation). Prior observation selection, SZA matching and temporal matching are retained. Includes SHEBA Perovich transects; coastal Barrow transects remain excluded.\n','\n','Data: `'+str(R/'validation_IDW_corrected_SIC.pkl')+'`. Rebuild corrected references with `'+str(R/'apply_corrected_sic.py')+'`; the underlying audit is in `audit_tower_blend.py`. Earlier sensitivity figures are retained below.\n']
 if c['cell_type']=='code' and s.strip()=='%run '+str(main):c['outputs']=[];c['execution_count']=None
nb.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
print('Validated unchanged satellite values and other platform data. Backup:',B)
print('Promoted corrected SIC figure to main notebook entry point and f09/f06 PNG/PDF.')
