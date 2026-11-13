#!/usr/bin/env bash
#PBS -N download_mosmet_nc
#PBS -A UOKL0045
#PBS -l select=1:ncpus=1:mem=8GB
#PBS -l walltime=1:00:00
#PBS -l job_priority=economy
#PBS -q main
#PBS -j oe
#PBS -m ae
#PBS -M jrotondo@uw.edu

cd /glade/derecho/scratch/jrotondo/albedo_satellite_validation/mosiac_surface_flux

set -e

# Step 1: Download the RDF metadata for the dataset
curl -s -L "https://arcticdata.io/metacat/d1/mn/v2/object/resource_map_doi:10.18739/A2PV6B83F" -o resourcemap.rdf

# Step 2: Extract unique UUIDs from the RDF
grep -oP 'urn:uuid:[a-f0-9-]+' resourcemap.rdf | sort -u > uuid_list.txt

# Step 3: Filter UUIDs based on known .nc UUIDs or download all blindly
# In this case, we'll just keep them all and try downloading — skip mime guessing
mkdir -p central_observatory
cd central_observatory

echo "Downloading files from UUIDs..."
while read -r uuid; do
  url="https://arcticdata.io/metacat/d1/mn/v2/object/$uuid"
  echo "→ Downloading: $url"
  wget --content-disposition "$url"
done < ../uuid_list.txt