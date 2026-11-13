# Main Figure 9: bias, RMSE and CCC; detailed matchups are supplementary.
from pathlib import Path
import runpy, shutil
R=Path("/glade/work/jrotondo/albedo_satellite_validation/spatial_consistency_review_20261022")
F=Path("/glade/work/jrotondo/ncc_paper/figures_final")
runpy.run_path(str(R/'compute_ccc.py'),run_name='__main__')
runpy.run_path(str(R/'plot_validation_summary_ccc.py'),run_name='__main__')
for ext in ['png','pdf']:
    for stem in ['f09','f06']:
        shutil.copy2(R/('Validation_summary_bias_RMSE_CCC.'+ext),F/(stem+'.'+ext))
print('Main Figure 9 summary exported to f09 and legacy f06.')
