#!/bin/bash
#PBS -N siaf_spatial_mmm
#PBS -A UOKL0045
#PBS -q cpudev
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs/spatial_mmm_cpudev.log

# ── Scratch tmp (avoids /tmp filling on login node) ──────────────────────────
export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs

# ── Python environment ────────────────────────────────────────────────────────
# Option A: NCAR conda module (most common on Derecho)
module load conda
# Initialize conda for non-interactive shell
eval "$(conda shell.bash hook)"
conda activate cice-scm-da   # e.g. npl, npl-2024a, or your custom env

# Option B: if you use a full path instead, comment out Option A and use:
# /path/to/conda/envs/myenv/bin/python

# ── Verify environment ────────────────────────────────────────────────────────
echo "Python: $(which python)"
echo "Start:  $(date)"

# ── Run ───────────────────────────────────────────────────────────────────────
cd /glade/u/home/jrotondo/work/siaf_drk_paper_final/python

python cmip6_siaf_spatial_mmm.py

echo "Finish: $(date)"
