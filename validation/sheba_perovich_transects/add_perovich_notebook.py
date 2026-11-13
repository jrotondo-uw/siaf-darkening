from pathlib import Path
import json,shutil
p=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb')
n=json.loads(p.read_text());tag='Figure 9 variant: SHEBA Perovich transects'
if not any(tag in ''.join(c['source']) for c in n['cells']):
 backup=p.with_name(p.stem+'.before_perovich_20261022.ipynb');assert not backup.exists();shutil.copy2(p,backup)
 n['cells'].extend([
  dict(cell_type='markdown',metadata={},source=['## '+tag+'\n','\n','Adds two broadband survey-line panels to the baseline Figure 9. Existing Figure 9 and the date/pentad sensitivity versions are preserved.\n','Source: UCAR/EOL 13.826, DOI 10.5065/D6SB444G; NOAA TOTALB mirror.\n','Raw data, spatial means, satellite matchups, QC decisions and README are in `/glade/work/jrotondo/albedo_satellite_validation/sheba_perovich_transects/`.\n','These are spatial averages of available surveys, not a continuous daily series. Four dates lack camp coordinates and two files have ambiguous date headers; exclusions and the two May year corrections are documented.\n']),
  dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run /glade/work/jrotondo/albedo_satellite_validation/sheba_perovich_transects/make_fig9_with_perovich.py\n'])])
 p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n');print('Added separate Perovich figure section; original cells preserved')
else:print('Section already present')
