# Sea ice darkening of surviving Arctic sea ice

Analysis code for *"Darkening of surviving Arctic sea ice contributes one third of the sea ice albedo feedback"* (Rotondo et al., submitted to *Communications Earth & Environment*).

The sea ice albedo feedback (SIAF) is decomposed into two spatially distinct components — loss of ice **area**, and **darkening** of ice that persists — and compared against 378 CMIP6 historical ensemble members. SIAF is computed as the product of an ice sensitivity (IS, the change in surface albedo per degree of global warming) and a radiative sensitivity (RS, the top-of-atmosphere flux response per unit albedo change).

See [the November 25 revision notes](REVISION_20261125.md) for the corrected observational weighting, primary residual uncertainty, source-data coverage, and local reproduction commands.

## Layout

```
python/            core analysis, decomposition, and model evaluation routines
shell/             PBS submission scripts for Derecho execution
notebooks/         interactive workflows and figure development (final_figures.ipynb, siaf_figures_revised.ipynb)
figures/           standalone publication figure generation scripts (f01–f09, s01–s10; see Figure Generation)
source_data/       numerical source data for every figure, also deposited on Zenodo
validation/        satellite–in-situ validation collocation routines (SHEBA, Tara, MOSAiC, airborne campaigns)
sensitivity/       atmospheric, kernel, cloud/aerosol, and decadal sensitivity checks
sanity_checks/     ocean-masking and grid-regridding verification tests
cmip_6_tests/      diagnostic tests for low-ice CMIP6 ensemble members
esgf_wget_scripts/ ESGF acquisition scripts for CMIP6 model radiative fields
misc/              solar zenith angle utilities for tower and transect validation
logs/              run and execution logs
```

## Principal Analysis Modules

| Script | Purpose |
|---|---|
| `python/siaf_obs_analysis.py` | Observational SIAF decomposition from CLARA-A3/APP-x and NSIDC SIC |
| `python/cmip6_siaf_calculation.py` | 378-member CMIP6 ensemble SIAF via the APRP method |
| `python/build_albedo_cache.py` | Preprocessing and harmonization of CLARA-A3 / APP-x albedo fields |
| `python/compute_sic_trend_maps.py` | Per-pixel SIC trends defining the area-loss and darkening geographic masks |
| `python/compute_siaf_covariance.py` | Two-sided classification leakage and residual darkening in retreat zones |
| `python/albedo_sic_discrimination_compute.py` | ROC / marginal ice zone ice-edge discrimination and AUC analysis |
| `python/save_validation_data.py` | Satellite–in-situ collocation for surface albedo validation |
| `python/compute_dalpha_dt_evolution.py` | Decadal evolution of the albedo–temperature sensitivity |
| `python/cesm2_gdex_siaf.py` | Large-ensemble CESM2 GDEX radiative evaluation |
| `python/compute_teleconnection_corr.py` | Arctic Amplification and atmospheric teleconnection analysis |

## Figure Generation (`figures/`)

Script names follow the paper's figure numbers, except Supplementary Figure 11. The numbers behind each figure are in `source_data/`, which is also deposited on Zenodo (doi:10.5281/zenodo.24204444).

| Figure | Script | Source data (`source_data/`) |
|---|---|---|
| Figure 1 | `f01.py` | `SourceData_Figure1.csv` |
| Figure 2 | `f02.py` | `SourceData_Figure2.csv` |
| Figure 3 | `f03.py` | `SourceData_Figure3.csv` |
| Figure 4 | `f04.py` | `SourceData_Figure4_maps.csv` (the mapped values), `SourceData_Figure4.csv` (pan-Arctic summaries) |
| Figure 5 | `f05.py` | `SourceData_Figure5.csv` (a–b: family means and the observed point), `SourceData_Figure5_cd.csv` (c–d: observations and model means), `SourceData_Figure5_cd_models.csv` (c–d: individual models) |
| Figure 6 | `f06.py` | `SourceData_Figure6.csv` |
| Figure 7 | `f07.py` | `SourceData_Figure7.csv`, `SourceData_Figure7_Observed.csv` |
| Figure 8 | `f08.py` | `SourceData_Figure8.csv` |
| Figure 9 | `f09.py` | `SourceData_Figure9.csv` |
| Supplementary Figure 1 | `s01.py` | `SourceData_Supplementary_Figure1.csv` |
| Supplementary Figure 2 | `s02.py` | `SourceData_Supplementary_Figure2.csv` |
| Supplementary Figure 3 | `s03.py` | `SourceData_Supplementary_Figure3.csv` |
| Supplementary Figure 4 | `s04.py` | `SourceData_Supplementary_Figure4.csv`, `SourceData_Supplementary_Figure4_Observed.csv` |
| Supplementary Figure 5 | `s05.py` | `SourceData_Supplementary_Figure5.csv` |
| Supplementary Figure 6 | `s06.py` | `SourceData_Supplementary_Figure6.csv` |
| Supplementary Figures 7–9 | `s07.py`, `s08.py`, `s09.py`: April–May, June–July and August–September, otherwise identical | `SourceData_Supplementary_Figure7_maps.csv`, `SourceData_Supplementary_Figure8_maps.csv`, `SourceData_Supplementary_Figure9_maps.csv` (the classified cells); `SourceData_Supplementary_Figure7.csv`, `SourceData_Supplementary_Figure8.csv`, `SourceData_Supplementary_Figure9.csv` (the share printed on each panel) |
| Supplementary Figure 10 | `s10.py`: a supplied image, which the script only checks for | `SourceData_Supplementary_Figure10.csv` (a description of the panels; the overpass geometry behind the image is not in this repository) |
| Supplementary Figure 11 | `s09_validation.py`, written by `make_s09.py` | `SourceData_Supplementary_Figure11.csv` |

`source_data/SourceData_Supplementary_DiagnosticConsistency.csv` holds the per-model ice sensitivities of the four-way test in the Supplementary Information's Diagnostic consistency section (all-sky or clear-sky albedo regressed on 2 m air or blended temperature, 25 models). That test has no figure.

Supporting scripts:
- `run_one.py`: runs one figure script and saves its figure as PNG and PDF at 300 dpi.
- `regen_figs.sh`: Master batch execution script.
- `bivariate_model.py`, `bivariate_diag.py`: Joint BMA and Bayesian likelihood weighting models.
- `make_s09.py`: writes `s09_validation.py`, the generator for the in-situ albedo validation figure (Supplementary Figure 11).
- `bivariate_model_31.py`, `bivariate_sdcheck_31.py`, `restage_f05_31.py`, `restage_f05_nofilter.py`, `fig5_nofilter/`: Figure 5c–d on the 31-model headline ensemble (one retained member per model), without the filter that dropped pixels whose SIC σ over the record is below 0.005; `f05.py` draws panels c–d from the staged `f05_cd_*.csv` tables.
- `compute_leakage_budget.py`, `compute_leakage_flux.py`, `siaf_obs_core.py` (written by `make_core.py`: a header plus the verbatim first 588 lines of `python/siaf_obs_analysis.py`): Supplementary Figure 3, the two-sided classification leakage in W m⁻², converted with the published radiative sensitivity, masks and area weights; `s03.py` draws it from `leakage_flux.csv`.
- `uncertainty_is/` (`extract_is.py`, `run_is.sh`, `is_boot.py`, `is_seeds.py`): the model-based 2σ on the observed ice sensitivity, i.e. the SIAF model-based procedure (common family-weighted multi-model-mean trend subtraction, paired circular 3-year blocks, 4,000 replicates) applied to the albedo series without radiative weighting. It writes `uncertainty_is.csv`, which `f07.py` reads for the gray band in Figure 7; `uncertainty.csv` holds the SIAF model-based uncertainty read by `f06.py`, `f08.py` and `s05.py`.
- `f07_observed_source_data.py`: writes `source_data/SourceData_Figure7_Observed.csv`, the two observational points, the ice-sensitivity band and the family-weighted multi-model mean drawn in Figure 7, with their intervals.
- `s05_source_data.py`: writes `source_data/SourceData_Supplementary_Figure5.csv`, the subset models, their weights and the observed target drawn in Supplementary Figure 5, with the selection and weighting of `s05.py`.
- `f04_source_data.py`, `f05_ab_source_data.py`, `f05_cd_source_data.py`, `s07_09_source_data.py`: write `SourceData_Figure4_maps.csv`, `SourceData_Figure5.csv`, `SourceData_Figure5_cd_models.csv` and `SourceData_Supplementary_Figure{7,8,9}_maps.csv`, the values drawn in Figure 4, Figure 5 and Supplementary Figures 7–9. Each reads the same inputs as its figure script; the docstrings define the columns.
- `*_superseded.py`: earlier versions kept for the record; none of them draws a figure in the paper.

## Validation Routines (`validation/`)

Contains collocation algorithms, nearest-neighbor fallback logic, and Lin's Concordance Correlation Coefficient (CCC) routines for ground-truth field campaigns:
- `sheba_surface_flux/` & `sheba_perovich_transects/`: SHEBA tower and 200-m albedo line matchups.
- `tara_surface_flux/`: Tara polar drift radiometer matchups.
- `drone_flux/`: MOSAiC uncrewed aircraft drone flux collocations.
- `spatial_consistency_review_20261022/`: Multi-platform spatial consistency and fallback audits.
- `date_matched_review_20261022/`: Product-specific exact-date matching versus pentad matching.

## Sensitivity Analysis (`sensitivity/`)

- Water vapor and aerosol checks: `check_era5_tcwv.py`, `check_merra2_aod.py`, `download_airs_tpw.py`, `download_calipso_l3.py`, `download_modis_tpw.py`.
- Regridding and kernel sensitivity: `diagnose_regridding.py`, `test_rs.py`, `uncertainty_of_siaf.py`.
- Decadal robustness: `compare_decadal_siafs.py`, `compare_to_viirs.py`, `compare_to_vpp_x.py`.

## Data Inputs

The satellite and model inputs are available from the archives listed below. Individual Tara and Raven/SLEIGH observations, coordinates and matchup pairs are not redistributed; access is available upon request to the data providers, subject to permission. Public figure-level validation summary statistics are retained. Tara data provenance and access are described in `validation/tara_surface_flux/README_broadband.md`.
- **CLARA-A3 SAL**: EUMETSAT CM SAF (DOI: 10.5676/EUM_SAF_CM/CLARA_AVHRR/V003)
- **APP-x**: NOAA NCEI (CDR Version 2.0)
- **Sea ice concentration**: NOAA/NSIDC Climate Data Record Version 6 (G02202)
- **CERES EBAF Ed4.2**: NASA Langley ASDC (DOI: 10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1)
- **ERA5 shortwave radiative kernels**: Huang et al. (2023), Mendeley Data, Version 4
- **CMIP6 Historical Simulations**: Earth System Grid Federation (ESGF)
- **In-situ observations**: PANGAEA and NSF Arctic Data Center (MOSAiC, SHEBA, ARISE, ARCSIX)

## Software Environment

Execution was performed in Python 3.13.1 on NCAR Derecho with the following core packages:
- `numpy==2.2.2`
- `scipy==1.15.1`
- `pandas==2.2.3`
- `xarray==2025.1.1`
- `dask==2025.4.0`
- `netcdf4==1.7.2`
- `statsmodels==0.14.4`
- `scikit-learn==1.6.1`
- `cartopy==0.24.0`
- `shapely==2.1.0`
- `pyproj==3.7.1`
- `esmf==8.8.0`
- `matplotlib==3.10.0`
- `global-land-mask` (essential for Arctic land/ocean boundaries)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
