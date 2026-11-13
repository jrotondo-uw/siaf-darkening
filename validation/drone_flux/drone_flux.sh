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

cd /glade/derecho/scratch/jrotondo/albedo_satellite_validation/drone_flux

set -e

# Step 1: Download the RDF metadata for the dataset
curl -s -L "https://arcticdata.io/metacat/d1/mn/v2/object/resource_map_doi:10.18739/A22J6857H" -o resourcemap.rdf

# Step 2: Extract unique UUIDs from the RDF
grep -oP 'urn:uuid:[a-f0-9-]+' resourcemap.rdf | sort -u > uuid_list.txt

# Step 3: Download files, skipping any that fail (e.g., 401 errors)
echo "Downloading files from UUIDs..."
while read -r uuid; do
  url="https://arcticdata.io/metacat/d1/mn/v2/object/$uuid"
  echo "→ Downloading: $url"
  
  # Try to download and skip on error
  wget --content-disposition "$url" || echo "✗ Skipped due to download error: $uuid"
done < uuid_list.txt