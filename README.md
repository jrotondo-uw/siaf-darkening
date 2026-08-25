# Sea ice darkening of surviving Arctic sea ice

Analysis code for *"Sea Ice Darkening of Surviving Arctic Sea Ice Accelerates
Radiative Forcing"* (Rotondo et al.).

The sea ice albedo feedback (SIAF) is decomposed into two spatially distinct
components — loss of ice **area**, and **darkening** of ice that persists — and
compared against 378 CMIP6 historical ensemble members. SIAF is computed as the
product of an ice sensitivity (IS, the change in surface albedo per degree of
global warming) and a radiative sensitivity (RS, the top-of-atmosphere flux
response per unit albedo change).

## Layout

```
python/      analysis and computation
shell/       PBS submission scripts for the above
notebooks/   final_figures.ipynb — main text and Extended Data figures
misc/        solar zenith angle utilities for tower validation
logs/        run logs
```

## Principal scripts

| script | purpose |
|---|---|
| `python/siaf_obs_analysis.py` | observational SIAF decomposition |
| `python/cmip6_siaf_calculation.py` | CMIP6 ensemble SIAF via the APRP method |
| `python/build_albedo_cache.py` | preprocessing of CLARA-A3 / APP-x albedo fields |
| `python/compute_sic_trend_maps.py` | per-pixel SIC trends defining the area-loss and darkening masks |
| `python/compute_siaf_covariance.py` | residual darkening within area-loss pixels |
| `python/albedo_sic_discrimination_compute.py` | ROC / marginal ice zone ice-edge discrimination |
| `python/save_validation_data.py` | satellite–in-situ collocation for albedo validation |
| `python/compute_dalpha_dt_evolution.py` | temporal evolution of the albedo–temperature sensitivity |
| `notebooks/final_figures.ipynb` | figure generation |

Shell scripts in `shell/` submit the corresponding Python jobs to PBS on
NCAR Derecho and encode the resource requests used for the published runs.

## Data

No data are included. All inputs are publicly available:

- **CLARA-A3 SAL** — EUMETSAT CM SAF
- **APP-x** — NOAA NCEI
- **Sea ice concentration** — NSIDC Climate Data Record (G02202)
- **CERES EBAF Ed4.2** — NASA Langley ASDC
- **ERA5 radiative kernels** — Huang et al. (2023)
- **CMIP6** — Earth System Grid Federation

Paths in the scripts point at NCAR GLADE and will need adjusting for other
systems.

## Environment

Python with `numpy`, `scipy`, `pandas`, `xarray`, `netCDF4`, `matplotlib`,
`cartopy`, `scikit-learn`, and `global-land-mask`.

> `global_land_mask` is required. Without it the ocean mask silently falls back
> to all-ocean, which admits Greenland and Siberia into the Arctic domain and
> roughly halves every domain-averaged result.

## Citation

Citation details will be added on publication.
