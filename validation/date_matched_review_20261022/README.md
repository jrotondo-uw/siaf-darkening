# Figure 9: shared-date sensitivity check
Separate review variant; the main notebook, source pickle, and f06/f09 figures are unchanged.

MOSAiC tower, SHEBA, Tara: retain only CLARA pentads having all five APP-x daily matches; retain APP-x records on exactly those five-day windows. Native daily/pentad averaging and original spatial/SZA/QC methods remain unchanged. Existing CLARA in-situ means are not recalculated.
Flights, drone and transects: intersect observation calendar dates and retain each product's existing valid spatial records; these are not necessarily identical pixels or instruments.
Summit/Raven: existing five-day anchors differ between products. Resample APP-x on CLARA start dates with the original APP-x sampling, SZA filter, temporal fallback, and outlier filter; use the same tower means as CLARA, then intersect successful dates. APP-x remains the original representative-date estimate, not a new five-day satellite mean.
ARISE: no APP-x data, so neither product has shared dates.

This is a date-availability sensitivity check, not a controlled solar-zenith-angle experiment. Product-specific pixels, SZA corrections, temporal compositing/fallback, and averaging can still differ. Source matching code and current plotting code are copied alongside the variant for provenance.
