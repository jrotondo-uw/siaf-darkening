#!/bin/bash
# usage: launch_worker2.sh WORKER WORKERS PAIRS_FILE TAG -- one detached worker on this node
cd /glade/derecho/scratch/jrotondo/fig5_uncertainty_nofilter_20261123 || exit 1
if pgrep -u jrotondo -f "[p]ython3 -u extract_nf.py" > /dev/null; then echo "already running on $(hostname)"; exit 1; fi
setsid -f env -u PYTHONPATH -u LD_LIBRARY_PATH -u PROJ_LIB PROJ_DATA=/glade/u/apps/opt/conda/envs/npl/share/proj OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE PAIRS_FILE=$3 WORKER=$1 WORKERS=$2 /glade/u/apps/opt/conda/envs/npl/bin/python3 -u extract_nf.py > extract_nf_$4_$1.log 2>&1 < /dev/null
echo "launched worker $1/$2 on $(hostname)"
