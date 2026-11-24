#!/bin/bash
# Login-node run (per-user cgroup: 8 cores, 10 GiB). Two workers; a member whose worker is
# killed is retried in the next pass, and anything killed twice is finally run alone.
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /glade/derecho/scratch/jrotondo/uncertainty_is_20261124
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH /glade/u/apps/opt/conda/envs/npl/bin/python3 -u"
for pass in 1 2 3; do
  for k in 0 1; do nice -n 10 $PY extract_is.py $k 2 2 > log_p${pass}_w$k.txt 2>&1 & done
  wait
done
nice -n 10 $PY extract_is.py 0 1 4 > log_solo.txt 2>&1
echo ALL_DONE > all_done.flag
