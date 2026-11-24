#!/bin/bash
# usage: launch_worker.sh WORKER  -- one detached extraction worker on this node (8 in all)
cd /glade/derecho/scratch/jrotondo/fig5_uncertainty_nofilter_20261123 || exit 1
if pgrep -u jrotondo -f "[p]ython3 -u extract_nf.py" > /dev/null; then echo "already running here"; exit 1; fi
setsid -f env -u PYTHONPATH -u LD_LIBRARY_PATH -u PROJ_LIB PROJ_DATA=/glade/u/apps/opt/conda/envs/npl/share/proj OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE PAIRS_FILE=/glade/derecho/scratch/jrotondo/fig5_uncertainty_nofilter_20261123/remaining.csv WORKER=$1 WORKERS=8 /glade/u/apps/opt/conda/envs/npl/bin/python3 -u extract_nf.py > extract_nf_r2_$1.log 2>&1 < /dev/null
echo "launched worker $1 on $(hostname)"
