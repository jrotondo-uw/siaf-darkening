# SHEBA Perovich transects

Official dataset: UCAR/EOL 13.826, Wavelength-integrated Albedos.
DOI: https://doi.org/10.5065/D6SB444G
Downloaded source: https://psl.noaa.gov/arctic/sheba/netcdf/datasets/Ice_snow_data/ICEDATA/OPTICS/TOTALB/
Instrument: Kipp and Zonen albedometer; metadata reports uncertainty +/-0.01.
Raw files are preserved unchanged in raw/. source_and_qc.json records URLs and SHA-256 hashes.

64 dated surveys and 3,982 unique position/date measurements were recovered.
Two May header years were corrected from 1999 to 1998 using the explicit file title and metadata; May 8 also has an identical individual file dated 1998. Duplicate May 8 data were counted once.
ALBT0405.CSV conflicts with its April 1 header, and ALBT1011.CSV has a malformed year; both are retained but excluded pending clarification.
Survey lengths vary. Spatial means are trapezoidal integrals divided by the observed line length, with no extrapolation. Point means and spatial variability are retained alongside them.
Four survey dates lack positions in the existing daily camp track and are excluded from satellite matching: 1998-06-05, 1998-07-13, 1998-07-21, 1998-08-24.
Camp coordinates are proxies for transect location. CLARA uses the existing linear spatial interpolation and means available surveys within its pentad. APP-x uses the existing nearest-grid sampling and matches only survey dates. Neither temporal gap filling nor an SIC blend is applied to the transect albedo.

The new figure adds a distinct pair of Perovich panels to the baseline broadband Figure 9; it does not replace the original radiation-series panels or overwrite the earlier date/pentad variants. Both SHEBA records represent the same campaign, not independent temporal coverage.

Current original SHEBA series provenance:
Figure 9 reads SHEBA 1997-98 rows from /glade/work/jrotondo/sheba_tara_nird/Joey/data/radiation/onice_daily_qc.csv.
These albedos exactly match sheba_daily.csv and are the ratio of daily mean upward/downward shortwave flux (verified to floating-point precision). The copied subset contains 185 rows. The original external raw archive/instrument citation was not recovered from these copied files/scripts; do not label it as Perovich transects or assert a specific tower dataset without further provenance.
