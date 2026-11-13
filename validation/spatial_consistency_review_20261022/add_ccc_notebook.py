from pathlib import Path
import json,shutil,datetime
R=Path(__file__).resolve().parent
p=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb')
n=json.loads(p.read_text());tag='## Candidate supplementary figure: validation bias, RMSE and CCC'
if not any(tag in ''.join(c['source']) for c in n['cells']):
 b=R/'backups'/('before_CCC_summary_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));b.mkdir(parents=True);shutil.copy2(p,b/p.name)
 n['cells'].extend([dict(cell_type='markdown',metadata={},source=[tag+'\n\n','Separate three-panel summary of the current Figure 9 statistics, ordered chronologically by campaign. Each dataset remains separate. Sample counts are retained matched pairs, not independent sample sizes; product dates and counts may differ. Full Figure 9 remains unchanged.\n\n','Draft caption: `'+str(R/'Validation_summary_CCC_caption.txt')+'`.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run '+str(R/'compute_ccc.py')+'\n','%run '+str(R/'plot_validation_summary_ccc.py')+'\n','from IPython.display import Image, display\n','display(Image(filename="'+str(R/'Validation_summary_bias_RMSE_CCC.png')+'"))\n'])])
 p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
print('Summary section available; main Figure 9 unchanged.')
