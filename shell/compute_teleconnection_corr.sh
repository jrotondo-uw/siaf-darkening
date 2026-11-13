#!/bin/bash -l
#PBS -N compute_teleconnection_corr
#PBS -A UOKL0045
#PBS -l select=1:ncpus=3:mem=128GB
#PBS -l walltime=6:00:00
#PBS -q develop
#PBS -j oe
#PBS -m ae
#PBS -M jrotondo@uw.edu

###############################################################################
# Script to calculate SIAL-SIC spatial correlation maps
###############################################################################

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

# Source conda properly for PBS
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate cice-scm-da

echo "====================================================================="
echo ">>> STARTING SIAL-SIC CORRELATION ANALYSIS <<<"
echo "Job ID : ${PBS_JOBID}"
echo "Host : $(hostname)"
echo "Start Time : $(date)"
echo "====================================================================="

# Set matplotlib to use non-interactive backend
export MPLBACKEND=Agg

# Navigate to work directory (adjust path as needed)
cd /glade/work/jrotondo/siaf_drk_paper_final/python/

# Run the correlation analysis
python compute_teleconnection_corr.py

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "====================================================================="
    echo ">>> SUCCESS: Correlation analysis completed at $(date)"
    echo "====================================================================="
else
    echo "====================================================================="
    echo ">>> FAILURE: Script exited with status $STATUS at $(date)"
    echo "====================================================================="
fi

exit $STATUS