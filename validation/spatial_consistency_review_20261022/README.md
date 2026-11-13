# Figure 9: IDW first, nearest-grid-cell fallback

Current separate review: Figure_9_IDW_nearest_fallback. Created 2026-10-22 at the user's request. Both CLARA-A3 and APP-x use the same spatial rule. Includes the SHEBA Perovich transects; excludes Barrow coastal transects. Historical figures and raw inputs are retained.

## Spatial rule
1. Try the existing shared Figure 9 four-neighbor inverse-distance-weighting routine. It selects four centers using a latitude/longitude-coordinate KD-tree, keeps finite albedos in [0,1], requires at least two valid neighbors, and uses weights 1/(haversine distance in km + 0.01 km).
2. If IDW fails, try the geographically nearest grid center, found using a spherical-coordinate KD-tree. Its albedo must be finite and within [0,1]. Do not search outward for a farther valid-valued cell if the nearest cell is missing.
3. Apply the same distance limit to both products: 25 km, or the pre-existing Greenland setting of 50 km for Summit/Raven. The IDW check applies to the nearest valid member of its four-center neighborhood; the fallback check applies to the nearest grid center. Keep existing downstream positive-albedo conditions where applicable.
4. If neither estimator succeeds, leave the match missing. There is no unconditional filling or extrapolation.

The IDW rule deliberately preserves the shared historical routine (including its coordinate-space neighbor selection) rather than introducing an additional change to its neighborhood definition. This is not a footprint average or land-mask-aware interpolation.

For APP-x composite routines, the spatial fallback is tried on the selected date/overpass before the pre-existing temporal fallback to adjacent days. For SHEBA/Tara/Perovich, the original same-day-only overpass averaging remains; individual overpasses may now use IDW or nearest, recorded in the output.

## Scope of this figure
This is a spatial-method review on the previously retained observation pool, not a fresh end-to-end re-selection of all raw measurements. It holds in-situ reference values, original candidate dates/pentads, temporal weighting rules, prior outlier and solar-angle selections fixed. Previously rejected observation dates are not newly admitted. The existing CLARA aircraft/transect/drone records already succeeded with the common IDW routine and remain unchanged. All retained APP-x records are re-evaluated so the new spatial fallback precedes inherited temporal fallback. MOSAiC tower, Summit/Raven CLARA, and both products for SHEBA/Tara/Perovich are re-evaluated.

A full raw-data regeneration would be a distinct analysis because new spatial sampling can change solar-angle/reference matching and which previously rejected observations qualify. Do not interpret changes in statistics as an independently controlled product-performance comparison when dates or valid overpass counts differ.

## Found in the audit
Original spatial methods were inconsistent:
- MOSAiC tower: CLARA linear native x/y; APP-x common IDW.
- MOSAiC transects, helicopter, HELiX drone, ARISE, ARCSIX: common IDW for both.
- Summit/Raven: CLARA used a different spherical IDW implementation; APP-x common IDW.
- SHEBA daily radiometers, Tara broadband and Perovich transects: CLARA linear native x/y; APP-x nearest cell.

The old linear CLARA path also assumed its x/y coordinates were EPSG:3413. They are not compatible with that assumption. At grid (j=104,k=146) in SALpm20080605000000321AVPOS01NP.nc, stored x/y are (-837500,1887500) m, while projecting that cell's own latitude/longitude into EPSG:3413 gives approximately (-1893599,729644) m. The new samplers use the supplied latitude/longitude arrays directly and avoid this mismatch. Earlier linear-coordinate results should be treated as superseded. This review does not retroactively redo ancillary solar-angle/reference selections that were held fixed from the previous retained observation pool.

## Barrow exclusion
Barrow/Utqiaġvik 2008, 2009 North/South and 2010 are excluded by explicit user decision: approximately 200 m landfast-ice transects about 1 km offshore do not resolve the 25 km grid-scale coastal mixture. This is a spatial-representativeness exclusion, not an agreement threshold. Raw files and exploratory plots remain archived. Barrow 2000–2002 was never added; ICESCAPE spectra were never broadband validation observations. SHEBA Perovich remains included.
Location: https://doi.org/10.1029/2011JC007231. Grid resolution is verified from native satellite file metadata in grid_audit.json.

## Reproduction and diagnostics
Run rebuild_fallback.py in npl with the previously installed pvlib directory on PYTHONPATH, then verify_fallback_outputs.py and plot_figure9_fallback.py. Rebuild outputs go only to this review directory. Existing source functions are loaded without executing their top-level workflow.

validation_IDW_fallback.pkl contains the main panels. fallback/{sheba_surface_flux,tara_surface_flux,sheba_perovich_transects}/ holds the added station/transect CSVs. spatial_method distinguishes idw, nearest, and idw+nearest; n_nearest_samples counts contributing sampled fields/overpasses rather than independent observation dates. Legacy ancillary and overpass-count fields are explicitly prefixed legacy_. IDW_fallback_stats.csv reports fallback usage among retained rows. Tests cover IDW precedence, nearest fallback, invalid nearest rejection, and distance bounds; output checks verify retained dates, references and unaffected CLARA panels.
