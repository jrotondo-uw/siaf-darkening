#!/bin/bash
cd /glade/derecho/scratch/jrotondo/fig5_uncertainty_nofilter_20261123 || exit 1
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH -u PROJ_LIB PROJ_DATA=/glade/u/apps/opt/conda/envs/npl/share/proj OMP_NUM_THREADS=4 /glade/u/apps/opt/conda/envs/npl/bin/python3 -u"
echo "start $(date) on $(hostname)"
$PY validate_series.py && $PY boot_nf.py
echo "rc=$? end $(date)"
