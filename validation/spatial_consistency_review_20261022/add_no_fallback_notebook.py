from pathlib import Path
import json,shutil
p=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb');r='/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022'
n=json.loads(p.read_text());tag='Figure 9 sensitivity: IDW only, no nearest-neighbor fallback'
if not any(tag in ''.join(c['source']) for c in n['cells']):
 b=p.with_name(p.stem+'.before_IDW_only_sensitivity_20261022.ipynb')
 if not b.exists():shutil.copy2(p,b)
 n['cells'].extend([dict(cell_type='markdown',metadata={},source=['## '+tag+'\n','\n','Separate controlled review with nearest-neighbor spatial fallback disabled for both products. The IDW implementation, previously retained observation pool, in-situ references, and temporal matching rules are held fixed. Existing adjacent-day temporal fallback rules remain where already used; this change disables only the nearest-neighbor spatial fallback. Barrow remains excluded; previous figures are preserved.\n','Comparison statistics: `'+r+'/IDW_only_vs_fallback.csv`.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run '+r+'/plot_figure9_no_fallback.py\n'])])
 p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n');print('Added separate IDW-only sensitivity section; existing sections preserved.')
