#!/usr/bin/env python
"""
compute_tower_sza_by_pentad.py
──────────────────────────────
Computes the mean Sun Zenith Angle (SZA) of the MOSAiC tower observations
for each CLARA-A3 pentad window — exactly replicating the QC and matching
logic from save_validation_data.py.

For each pentad the SZA reported is the mean of all tower observations
that fell inside the CLARA SZA window [sat_sza ± max(3°, sat_sza_std)],
i.e. exactly the observations whose albedo was averaged to form the
pentad-mean tower albedo used in the validation figure.

Output:
  • console table (copy-paste to email)
  • CSV at OUT_CSV
"""

import os, glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ── CONFIG — mirrors save_validation_data.py exactly ─────────────────────────
BASE_DIR   = '/glade/derecho/scratch/jrotondo/albedo_satellite_validation/mosiac_surface_flux'
DIR_SIAL   = '/glade/derecho/scratch/jrotondo/clara_a3_obs/'
OUT_CSV    = '/glade/work/jrotondo/ncc_paper/plots/validation/mosaic_tower_sza_by_pentad.csv'

# Three tower stations (same as save_validation_data.py)
TOWER_STATIONS = {
    'met':    (os.path.join(BASE_DIR, 'central_observatory'), 'mosmet'),
    'asfs30': (os.path.join(BASE_DIR, 'station_30'),          'mosmet.asfs30'),
    'asfs50': (os.path.join(BASE_DIR, 'station_50'),          'mosmet.asfs50'),
}

MAX_SZA         = 70.0   # hard SZA cut (matches save_validation_data.py)
MIN_OBS_PENTAD  = 24     # minimum tower obs per pentad window (same)
MIN_WIN_OBS     = 6      # minimum obs inside CLARA SZA window to report matched SZA


# ── 1. LOAD TOWER DATA ────────────────────────────────────────────────────────
def load_tower_with_sza(directory, file_pattern):
    """
    Mirrors load_tower_with_sza() from save_validation_data.py exactly:
    reads up_short_hemisp / down_short_hemisp with their QC flags and
    extracts zenith_true → sza column.
    """
    data_list = []
    files = sorted([f for f in os.listdir(directory)
                    if f.startswith(file_pattern) and f.endswith('.nc')])
    for f in files:
        try:
            ds  = xr.open_dataset(os.path.join(directory, f))
            req = ['up_short_hemisp', 'down_short_hemisp',
                   'up_short_hemisp_qc', 'down_short_hemisp_qc']
            if not all(v in ds.variables for v in req):
                ds.close(); continue

            lat_var = 'lat_tower' if 'lat_tower' in ds else 'lat'
            lon_var = 'lon_tower' if 'lon_tower' in ds else 'lon'

            valid_qc = ((ds['up_short_hemisp_qc']   == 0) &
                        (ds['down_short_hemisp_qc'] == 0) &
                        (ds['down_short_hemisp']    > 10.0))
            albedo   = (ds['up_short_hemisp'] / ds['down_short_hemisp']).where(valid_qc)
            lat      = ds[lat_var].where(valid_qc)
            lon      = ds[lon_var].where(valid_qc)

            sza_vals = np.full(len(ds['time']), np.nan)
            if 'zenith_true' in ds.variables:
                raw      = ds['zenith_true'].values.flatten().astype(float)
                sza_vals = np.where(valid_qc.values.flatten(), raw, np.nan)

            tmp = pd.DataFrame({
                'time':   ds['time'].values,
                'albedo': albedo.values.flatten(),
                'lat':    lat.values.flatten(),
                'lon':    lon.values.flatten(),
                'sza':    sza_vals,
            }).dropna(subset=['albedo', 'lat', 'lon']).set_index('time')

            tmp = tmp[(tmp['albedo'] >= 0) & (tmp['albedo'] <= 1) &
                      (tmp['sza'] < MAX_SZA)]
            data_list.append(tmp)
            ds.close()
        except Exception as e:
            print(f'  WARNING {f}: {e}')

    return pd.concat(data_list).sort_index() if data_list else pd.DataFrame()


# ── 2. LOAD ALL STATIONS ──────────────────────────────────────────────────────
print('Loading MOSAiC tower data ...')
all_frames = []
for key, (path, pat) in TOWER_STATIONS.items():
    df_tmp = load_tower_with_sza(path, pat)
    if not df_tmp.empty:
        all_frames.append(df_tmp)
        print(f'  {key}: {len(df_tmp):,} obs')
    else:
        print(f'  {key}: no data')

if not all_frames:
    raise RuntimeError('No tower data loaded from any station.')

tower_df = pd.concat(all_frames).sort_index()
tower_df = tower_df[~tower_df.index.duplicated(keep='first')]
print(f'  Total combined: {len(tower_df):,} obs  '
      f'({tower_df.index[0].date()} → {tower_df.index[-1].date()})')


# ── 3. FIND CLARA PENTAD FILES (2020 only — MOSAiC period) ───────────────────
print('\nFinding CLARA-A3 pentad files ...')
sial_files = sorted(glob.glob(f'{DIR_SIAL}/SALpm2020*.nc'))
if not sial_files:
    raise FileNotFoundError(f'No CLARA files found in {DIR_SIAL} for 2020')
print(f'  Found {len(sial_files)} pentad files')


# ── 4. MATCH — mirror match_tower_clara() logic ───────────────────────────────
def get_sat_sza(lat, lon, pentad_file):
    """Read mean and std of CLARA SZA at nearest grid point."""
    try:
        with xr.open_dataset(pentad_file) as ds:
            if 'sun_zenith_angle_mean' not in ds.variables:
                return np.nan, np.nan
            sza_mean = ds['sun_zenith_angle_mean'].squeeze().values.astype(float)
            sza_std  = (ds['sun_zenith_angle_std'].squeeze().values.astype(float)
                        if 'sun_zenith_angle_std' in ds.variables
                        else np.full_like(sza_mean, np.nan))
            lats_2d  = ds['lat'].values
            lons_2d  = ds['lon'].values
        tree  = cKDTree(np.column_stack([lats_2d.flatten(), lons_2d.flatten()]))
        _, idx = tree.query([lat, lon])
        mu  = float(sza_mean.flatten()[idx])
        sig = float(sza_std.flatten()[idx])
        return (mu if np.isfinite(mu) else np.nan,
                sig if np.isfinite(sig) else np.nan)
    except Exception as e:
        return np.nan, np.nan


print('\nMatching tower observations to CLARA pentads ...\n')

records = []
for sial_path in sial_files:
    fname  = os.path.basename(sial_path)
    cdate  = pd.to_datetime(fname[5:13])           # e.g. SALpm20200501.nc → 2020-05-01
    pend   = cdate + pd.Timedelta(days=5)

    # Subset tower data to this pentad window
    mask   = (tower_df.index >= cdate) & (tower_df.index < pend)
    subset = tower_df[mask]

    if len(subset) < MIN_OBS_PENTAD:
        print(f'  {fname[5:13]}: only {len(subset)} obs — skip (need {MIN_OBS_PENTAD})')
        continue

    avg_lat = float(subset['lat'].mean())
    avg_lon = float(subset['lon'].mean())

    # Read CLARA SZA at tower location
    sat_sza_mu, sat_sza_std = get_sat_sza(avg_lat, avg_lon, sial_path)

    if not np.isfinite(sat_sza_mu) or sat_sza_mu >= MAX_SZA:
        print(f'  {fname[5:13]}: CLARA SZA invalid or ≥{MAX_SZA}° — skip')
        continue

    # SZA-matched window (same formula as save_validation_data.py)
    win    = max(3.0, sat_sza_std if np.isfinite(sat_sza_std) else 3.0)
    in_win = subset[np.abs(subset['sza'] - sat_sza_mu) <= win]

    n_win        = len(in_win)
    mean_sza_win = float(in_win['sza'].mean())   if n_win >= MIN_WIN_OBS else np.nan
    mean_alb_win = float(in_win['albedo'].mean()) if n_win >= MIN_WIN_OBS else np.nan

    # Also report the unwindowed (all-obs) mean SZA for context
    mean_sza_all = float(subset['sza'].mean())
    mean_alb_all = float(subset['albedo'].mean())

    records.append({
        'pentad':              cdate.strftime('%Y-%m-%d'),
        'date_end':            (pend - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'n_obs_total':         len(subset),
        'n_obs_sza_window':    n_win,
        'clara_sza_mean':      round(sat_sza_mu, 2),
        'clara_sza_window':    round(win, 2),
        # ── SZA-window matched (used for validation figure) ──────────────────
        'mean_sza_matched':    round(mean_sza_win, 2) if np.isfinite(mean_sza_win) else np.nan,
        'mean_albedo_matched': round(mean_alb_win, 4) if np.isfinite(mean_alb_win) else np.nan,
        # ── All-obs mean (full pentad) ───────────────────────────────────────
        'mean_sza_all_obs':    round(mean_sza_all, 2),
        'mean_albedo_all_obs': round(mean_alb_all, 4),
    })

    status = (f'N_win={n_win}  SZA_matched={mean_sza_win:.1f}°'
              if np.isfinite(mean_sza_win) else f'N_win={n_win} (< {MIN_WIN_OBS}, no match)')
    print(f'  {cdate.date()} – {(pend-pd.Timedelta(days=1)).date()}  '
          f'N={len(subset):4d}  CLARA_SZA={sat_sza_mu:.1f}±{win:.1f}°  {status}')

# ── 5. OUTPUT ─────────────────────────────────────────────────────────────────
result = pd.DataFrame(records)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
result.to_csv(OUT_CSV, index=False, float_format='%.2f')
print(f'\nSaved → {OUT_CSV}')

# ── Pretty table ──────────────────────────────────────────────────────────────
print('\n' + '=' * 80)
print(f'MOSAiC tower mean SZA by CLARA-A3 pentad  (SZA < {MAX_SZA}°, '
      f'min {MIN_WIN_OBS} obs in window)')
print('SZA "matched" = obs within CLARA_SZA ± window — these are the exact obs')
print('used to compute the pentad-mean tower albedo in the validation figure.')
print('=' * 80)
hdr = (f"{'Pentad':<12}  {'Date range':<23}  "
       f"{'N_win':>5}  {'CLARA SZA':>10}  {'Win':>5}  "
       f"{'Mean SZA (matched)':>19}  {'Mean albedo':>12}")
print(hdr)
print('-' * 80)

for _, row in result.iterrows():
    dr = f"{row['pentad']} – {row['date_end']}"
    sza_str = (f"{row['mean_sza_matched']:>19.2f}°"
               if np.isfinite(row['mean_sza_matched'])
               else f"{'(< min obs)':>19}")
    alb_str = (f"{row['mean_albedo_matched']:>12.4f}"
               if np.isfinite(row['mean_albedo_matched'])
               else f"{'—':>12}")
    print(f"{row['pentad']:<12}  {dr:<23}  "
          f"{int(row['n_obs_sza_window']):>5}  "
          f"{row['clara_sza_mean']:>10.2f}°  "
          f"{row['clara_sza_window']:>4.1f}°  "
          f"{sza_str}  {alb_str}")

print('=' * 80)

valid = result[result['mean_sza_matched'].notna()]
print(f'\n{len(valid)} pentads with SZA-matched data  '
      f'(out of {len(result)} pentads with ≥{MIN_OBS_PENTAD} tower obs)')
if len(valid):
    print(f'Overall mean SZA (matched obs): {valid["mean_sza_matched"].mean():.2f}°')
    print(f'SZA range across pentads:       '
          f'{valid["mean_sza_matched"].min():.2f}° – '
          f'{valid["mean_sza_matched"].max():.2f}°')
