# Tara broadband input for revised Figure 9

Source: Aku Riihela, supplied via Funet FileSender on 2026-10-22.
Original files: Tara_radiomet_Apr-Oct_clean.xls and Tara_coord_23Mar_19Sep.txt.

Rebuild daily data with `python prepare_tara_broadband.py`, then matchups with
`python tara_collocate.py` in the npl environment. Run the notebook's Figure 9
cell (`make_fig6_20panel.py`) to regenerate f09.png/pdf and legacy f06.png/pdf.
The small `_vendor/xlrd` reader supports the original legacy Excel format.

Method: daily broadband albedo is sum(SW Out)/sum(SW In) for accepted minute
samples, not an integral of the spectral instrument. Keep finite fluxes with
SW In >=20 W/m2 and ratios in [0,1]. Exclude all repeated timestamps. Positions
are interpolated in EPSG:3413 within track coverage, with gaps <=2 hours.
Timestamps are assumed UTC because the supplied files do not state a timezone.
No minimum daily sample count is imposed; `n` is retained for coverage review.
CLARA-A3 retains the existing equal-day pentad averaging and spatial sampling;
APP-x retains the existing daily sampling. Satellite SIC files were unavailable
for this run, so the previous 100% SIC fallback applies to all 162 days.
These processing choices are explicit assumptions, not provider-supplied QC.

Original spectral data and matchup tables are retained. New daily data and
matchup tables use the `tara_broadband_` prefix. The old figure, plotting code,
collocation script, and notebook are backed up in
`../backups/tara_broadband_20261022/`.

## Data provenance and credit

The Tara drifting ice station radiation measurements were collected within the
DAMOCLES research project (https://www.damocles-eu.org/). They are redistributed
here in QA-treated form with the permission of the DAMOCLES investigators, via
Erko Jakobson (University of Tartu) and Timo Vihma (FMI); there is no official
Tara Arctic archive. Please credit DAMOCLES for the measurements in any reuse.

Deposited here (QA-treated, `tara_broadband_` prefix):

| file | contents |
|---|---|
| `tara_broadband_daily.csv` | daily broadband albedo, 1 Apr - 20 Sep 2007 |
| `tara_broadband_clara_matchups.csv` | CLARA-A3 collocated pairs |
| `tara_broadband_appx_matchups.csv` | APP-x collocated pairs |
| `tara_broadband_validation_stats.csv` | bias, RMSE, correlation, CCC |

The screening in `tara_broadband_processing.json` is our own, documented above;
it is not provider-supplied QC. The original radiometer workbook
(`Tara_radiomet_Apr-Oct_clean.xls`, 34 MB) is not redistributed here and is
available from FMI.
