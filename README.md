# Sea ice darkening of surviving Arctic sea ice

Analysis code for *"Sea Ice Darkening of Surviving Arctic Sea Ice Accelerates Radiative Forcing"* (Rotondo et al., Nature Climate Change).

The sea ice albedo feedback (SIAF) is decomposed into two spatially distinct components — loss of ice **area**, and **darkening** of ice that persists — and compared against 378 CMIP6 historical ensemble members. SIAF is computed as the product of an ice sensitivity (IS, the change in surface albedo per degree of global warming) and a radiative sensitivity (RS, the top-of-atmosphere flux response per unit albedo change).

## Layout

```
python/            core analysis, decomposition, and model evaluation routines
shell/             PBS submission scripts for Derecho execution
notebooks/         interactive workflows and figure development (final_figures.ipynb, siaf_figures_revised.ipynb)
figures/           standalone publication figure generation scripts (f01–f09, s01–s09, s11)
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

The directory `figures/` provides modular, standalone plotting scripts:
- `f01.py`–`f09.py`: Main Text Figures 1 through 9.
- `s01.py`–`s09.py`, `s11.py`: Supplementary Figures 1 through 9.
- `bivariate_model.py`, `bivariate_diag.py`: Joint BMA and Bayesian likelihood weighting models.
- `regen_figs.sh`: Master batch execution script.

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

All observational and model datasets are publicly available from their respective archives:
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
