#!/bin/bash
#PBS -N cesm2_gdex_siaf
#PBS -A UOKL0045
#PBS -q develop
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs/cesm2_gdex_siaf.log

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /glade/u/home/jrotondo/work/siaf_drk_paper_final/logs

module load conda
eval "$(conda shell.bash hook)"
conda activate cice-scm-da

cd /glade/u/home/jrotondo/work/siaf_drk_paper_final/python
python cesm2_gdex_siaf.py
