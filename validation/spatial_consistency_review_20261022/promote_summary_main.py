from pathlib import Path
import json,shutil,datetime,runpy
R=Path(__file__).resolve().parent
F=Path('/glade/work/jrotondo/ncc_paper/figures_final')
main=Path('/glade/work/jrotondo/albedo_satellite_validation/make_fig6_20panel.py')
nb=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb')
B=R/'backups'/('before_summary_main_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));B.mkdir(parents=True)
for p in [main,nb,R/'plot_validation_summary_ccc.py']+[F/(stem+'.'+ext) for stem in ['f06','f09'] for ext in ['png','pdf']]:
 if p.exists():shutil.copy2(p,B/p.name)
# The uploaded revision is kept separate until backup is complete.
shutil.copy2(R/'plot_validation_summary_ccc_revised.py',R/'plot_validation_summary_ccc.py')
s=(R/'plot_figure9_chronological.py').read_text().replace('Figure 9 — In-situ albedo validation','Supplementary figure — In-situ albedo matchups').replace('Figure_9_chronological.png','Supplementary_validation_scatterplots.png').replace('Figure_9_chronological.pdf','Supplementary_validation_scatterplots.pdf')
(R/'plot_supplementary_validation.py').write_text(s)
main.write_text('''# Main Figure 9: bias, RMSE and CCC; detailed matchups are supplementary.
from pathlib import Path
import runpy, shutil
R=Path("'''+str(R)+'''")
F=Path("'''+str(F)+'''")
runpy.run_path(str(R/'compute_ccc.py'),run_name='__main__')
runpy.run_path(str(R/'plot_validation_summary_ccc.py'),run_name='__main__')
for ext in ['png','pdf']:
    for stem in ['f09','f06']:
        shutil.copy2(R/('Validation_summary_bias_RMSE_CCC.'+ext),F/(stem+'.'+ext))
print('Main Figure 9 summary exported to f09 and legacy f06.')
''')
runpy.run_path(str(main),run_name='__main__')
runpy.run_path(str(R/'plot_supplementary_validation.py'),run_name='__main__')
for ext in ['png','pdf']:shutil.copy2(R/('Supplementary_validation_scatterplots.'+ext),F/('supp_validation_scatterplots.'+ext))
n=json.loads(nb.read_text())
for c in n['cells']:
 s=''.join(c['source'])
 if c['cell_type']=='markdown' and s.startswith('## Figure 9\n'):
  c['source']=['## Figure 9\n\n','Main paper figure: bias, RMSE and Lin’s concordance correlation coefficient (CCC), with chronological dataset rows and product-specific matched-pair counts. Uses corrected MOSAiC tower SIC and IDW with valid nearest-neighbor backup for both products. Prior observation selection and temporal matching are unchanged. Detailed scatterplots are now a supplementary figure below.\n\n','Caption: `'+str(R/'Validation_summary_CCC_caption.txt')+'`.\n']
 if c['cell_type']=='code' and s.strip()=='%run '+str(main):
  c.update(source=['%run '+str(main)+'\n','from IPython.display import Image, display\n','display(Image(filename="'+str(F/'f09.png')+'"))\n'],outputs=[],execution_count=None)
 if c['cell_type']=='markdown' and s.startswith('## Candidate supplementary figure: validation bias, RMSE and CCC'):
  c['source']=['## Supplementary figure: detailed validation scatterplots\n\n','Full chronological scatterplot comparisons underlying the main Figure 9 summary. Separate CLARA-A3 and APP-x panels retain the matched observations, regression diagnostics and platform-specific information. Exported as `supp_validation_scatterplots.png/pdf`; final supplementary numbering is not assigned here.\n']
 if c['cell_type']=='code' and 'plot_validation_summary_ccc.py' in s:
  c.update(source=['%run '+str(R/'plot_supplementary_validation.py')+'\n','from IPython.display import Image, display\n','display(Image(filename="'+str(R/'Supplementary_validation_scatterplots.png')+'"))\n'],outputs=[],execution_count=None)
nb.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
print('Updated main Figure 9 and supplementary scatterplot section. Backup:',B)
