#!/bin/bash
#PBS -N siaf_cesm2_spatial
#PBS -A UOKL0045
#PBS -q develop
#PBS -l select=1:ncpus=8:mem=128GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs/siaf_cesm2_spatial.log

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs

module load conda
eval "$(conda shell.bash hook)"
conda activate cice-scm-da

cd /glade/u/home/jrotondo/work/siaf_drk_paper_final/python
python cmip6_siaf_spatial_cesm2_range.py
