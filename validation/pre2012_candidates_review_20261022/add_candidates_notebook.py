from pathlib import Path
import json,shutil
p=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb')
n=json.loads(p.read_text());tag='Pre-2012 candidates: separate review figure';original=n['cells'].copy()
if not any(tag in ''.join(c['source']) for c in n['cells']):
 backup=p.with_name(p.stem+'.before_pre2012_candidates_20261022.ipynb');assert not backup.exists();shutil.copy2(p,backup)
 root='/glade/work/jrotondo/albedo_satellite_validation/pre2012_candidates_review_20261022'
 n['cells'].extend([dict(cell_type='markdown',metadata={},source=['## '+tag+'\n','\n','Barrow broadband transects: 2008, 2009 North/South, 2010. ICESCAPE 2010/2011 spectra are shown separately, not treated as broadband validation. Barrow 2000–2002 remains pending raw data.\n','Coastal spatial representativeness and source QA caveats are documented in `'+root+'/README.md`. Existing figures and matching methods are preserved.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run '+root+'/plot_candidates.py\n','from IPython.display import display, Image\n','display(Image(filename="'+root+'/figures/Pre2012_candidates_review.png"))\n'])])
 assert n['cells'][:len(original)]==original
 p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n');print('Appended candidate review section; existing cells unchanged.')
else:print('Section already exists.')
