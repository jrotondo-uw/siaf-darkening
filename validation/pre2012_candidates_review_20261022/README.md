> STATUS: Barrow coastal transects excluded from validation by user decision on 2026-10-22 because of coastal spatial representativeness. This directory is an archived exploratory review.

# Pre-2012 candidates: separate exploratory review

Created 2026-10-22. Existing Figure 9, date-matched and pentad-matched versions are unchanged.

## Included observations
- Barrow 2008: 21 survey dates (April 8–June 17).
- Barrow 2009 North: 26 survey dates (April 14–June 23).
- Barrow 2009 South: 21 survey dates (May 21–June 25).
- Barrow 2010: 18 survey dates (June 2–21).
- ICESCAPE 2010: 10 source spectral workbooks downloaded.
- ICESCAPE 2011: 8 of 9 linked source spectral workbooks downloaded; station 7 / 0718.xls returns HTTP 404. Station 8 has the same filename but is a different link; station identifiers are preserved to prevent overwriting.
- Barrow 2000–2002: publication verified, but raw dated observations have not been located. No invented, digitized or substitute data are used. This candidate remains pending.

## Sources
Barrow author pages: https://chrispolashenski.com/2008.php, https://chrispolashenski.com/2009N.php, https://chrispolashenski.com/2009S.php, https://chrispolashenski.com/2010.php.
Official integrated-albedo archive: https://doi.org/10.5065/D66T0JRJ (EOL 106.422, now Arctic Data Center). The author site supplies 2010 dates later than the archive catalog's June 8 endpoint; actual source files were used.
Cruise source pages: https://chrispolashenski.com/cruise_data2010.php and https://chrispolashenski.com/cruise_data2011.php.
ICESCAPE methods: https://doi.org/10.1002/2015JC011163.
Barrow 2000–2002 methods: https://doi.org/10.1029/2003JC001866.

Download manifests retain URLs, success status and hashes. The author site's certificate is expired; only public data files were downloaded with certificate verification disabled. No downloaded executable was run.

## Broadband reference and QA
Use author-provided daily spatial-average files: 2008_Kipps_Avg.csv, Average_North.csv, Average_South.csv and Average.csv. Blank dates are omitted, not interpolated. Add April 8, 2008 using the arithmetic average of its 41 equally spaced raw measurements, which is absent from the daily-average summary. Instrument: Kipp & Zonen CM14 broadband albedometer, 300–3000 nm. Site-specific metadata says 5 m spacing in 2008, 2.5 m in 2009 and 2010, along approximately 200 m lines.

Raw-file spatial means were independently compared with author summaries; see raw_mean_audit.csv. Published summaries are retained, rather than replacing them with inconsistent raw headers: kipps_0616.csv has a June 15 header but agrees with June 16 summary; June 14 has multiple survey times, with a daily aggregate in the summary. Minor differences also occur (e.g. June 10, 2008 ~0.0015). These are preliminary author-processed observations, not a new fully adjudicated raw-data product. Raw files remain unchanged.

Locations: 2008 uses the instrument metadata location (71.36568 N, 156.54417 W), not the archive's combined bounding box. 2009 uses midpoints of the documented UTM zone 4N line endpoints, transformed assuming WGS84. 2010 uses the midpoint of the documented endpoints (71.37096 N, 156.50975 W). The 2010 PDF labels longitudes E, evidently a hemisphere typo: the official archive has negative longitudes, and the documented Barrow location is west. This correction is explicit. Survey calendar dates are used as supplied; local-to-UTC shifts are not inferred.

## Satellite matching
Separate output directory: /glade/work/jrotondo/albedo_satellite_validation/pre2012_candidates_review_20261022.
CLARA: existing EPSG:3413 linear interpolation of blue_sky_albedo_all_mean/100 at each site; no nearest fallback. Reference is the equal-day mean of available survey spatial means within [pentad start, start+5 days). These are sampled-day pentad means, not necessarily five complete observation days. 27 valid site-pentad pairs.
APP-x: existing nearest grid point by latitude/longitude distance, mean of valid same-day overpasses; no adjacent-day fallback. 82 valid site-day pairs. Finite satellite albedo >0 and <=1.05, matching existing acceptance rules.
No sea-ice-concentration blending; no temporal gap-filling; no common-date intersection; no change to established figures or match methods. This follows the original direct-transect comparison, not the optional common-date/pentad sensitivity variants.

All four Barrow sites select the same APP-x grid point (j=105,k=148; 71.42 N,156.75 W). Nearby sites and repeated days are not independent satellite samples. Their coastal 200 m transects do not necessarily represent the full satellite footprint. No land-fraction correction was made. Treat all statistics as exploratory; do not pool the site counts as independent evidence. Bias = satellite minus in situ. CLARA and APP-x counts represent different temporal units.

## ICESCAPE display
Use the workbook's supplied Ave Albedo column, with dates read from workbook headers, not inferred station labels. Some workbook station labels conflict with website numbering; both are preserved in icescape_inventory.csv. The figure displays 400–1300 nm only; the extracted CSV preserves 350–2500 nm. Values outside [0,1] are masked in the display only. Source spectra are unsmoothed, with no inter-detector correction; a conspicuous near-1000-nm feature in one 2011 curve needs instrument QA. Means of targeted sampled surfaces are not area-weighted floe means. No broadband albedo or satellite performance statistic is inferred from these spectra. Adding ICESCAPE to broadband validation would require a justified irradiance-weighted integration, spectral quality control and spatial representativeness assessment.

## Remaining candidate
The 2000–2002 Barrow study includes tundra and freshwater ice as well as Chukchi sea ice and Elson Lagoon. Only appropriate marine sites should enter sea-ice validation once raw dates, locations and albedos are obtained. The publication and public archive/author searches did not yield those files. This review therefore includes its status explicitly, rather than claiming the campaign has been validated.
