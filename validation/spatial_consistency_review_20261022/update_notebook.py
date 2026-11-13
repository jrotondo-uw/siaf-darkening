from pathlib import Path
import json,shutil
P=Path('/glade/u/home/jrotondo/work/siaf_drk_paper_final/notebooks/siaf_figures_revised.ipynb');B=Path('/glade/work/jrotondo/albedo_satellite_validation');R=B/'spatial_consistency_review_20261022'
n=json.loads(P.read_text());backup=P.with_name(P.stem+'.before_IDW_and_Barrow_exclusion_20261022.ipynb')
if not backup.exists():shutil.copy2(P,backup)
for i,c in enumerate(n['cells']):
 if c['cell_type']=='markdown' and '## Pre-2012 candidates: separate review figure' in ''.join(c['source']):
  c['source']=['## Excluded candidate: coastal Barrow/Utqiaġvik transects\n','\n','Excluded from validation at the user’s request: approximately 200 m landfast-ice transects about 1 km offshore are not spatially representative of the 25 km satellite grids. This is a representativeness exclusion, not a threshold based on agreement.\n','The exploratory figure and raw downloads remain archived in `/glade/work/jrotondo/albedo_satellite_validation/pre2012_candidates_review_20261022/`. ICESCAPE spectra remain exploratory and are not broadband validation data. Barrow 2000–2002 was never added. SHEBA Perovich transects remain included.\n']
  if i+1<len(n['cells']) and n['cells'][i+1]['cell_type']=='code' and 'pre2012_candidates_review_20261022/plot_candidates.py' in ''.join(n['cells'][i+1]['source']):
   n['cells'][i+1]={'cell_type':'markdown','metadata':{},'source':['The candidate plot is archived; its execution cell was retired following the exclusion decision.\n']}
tag='Figure 9: IDW with nearest-neighbor fallback for both products'
if not any(tag in ''.join(c['source']) for c in n['cells']):
 n['cells'].extend([dict(cell_type='markdown',metadata={},source=['## '+tag+'\n','\n','Current spatial-consistency review: identical four-neighbor inverse-distance weighting with nearest-grid-cell fallback for CLARA-A3 and APP-x across all panels. Includes SHEBA Perovich; excludes coastal Barrow transects. Prior observations, in-situ reference values, temporal matching and screening selections are retained; pairs without either valid IDW support or a valid nearest cell within the distance limit are dropped. Earlier versions remain historical comparisons.\n','Methods: `'+str(R/'README.md')+'`.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=['%run '+str(R/'plot_figure9_fallback.py')+'\n'])])
P.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
old=B/'pre2012_candidates_review_20261022/README.md';s=old.read_text()
notice='> STATUS: Barrow coastal transects excluded from validation by user decision on 2026-10-22 because of coastal spatial representativeness. This directory is an archived exploratory review.\n\n'
if notice not in s:old.write_text(notice+s)
print('Notebook: Barrow retired; IDW section appended; backup preserved.')

# Keep excluded exploratory calculations outside the current figure directory.
archive=B/"pre2012_candidates_review_20261022"/"archived_IDW_diagnostic"
archive.mkdir(exist_ok=True)
if (R/"barrow").exists() and not (archive/"barrow").exists():shutil.move(str(R/"barrow"),str(archive/"barrow"))
if (R/"spatial_consistency_stats.csv").exists():
 import pandas as pd
 z=pd.read_csv(R/"spatial_consistency_stats.csv")
 if z.platform.str.startswith("Barrow").any():
  z[z.platform.str.startswith("Barrow")].to_csv(archive/"excluded_Barrow_stats.csv",index=False)
  z[~z.platform.str.startswith("Barrow")].to_csv(R/"spatial_consistency_stats.csv",index=False)
