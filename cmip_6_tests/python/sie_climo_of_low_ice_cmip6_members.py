import numpy as np
import xarray as xr
import pandas as pd
import glob, os, re, warnings
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

CMIP6_ROOT   = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT   = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'
NOAA_CDR_DIR = '/glade/derecho/scratch/jrotondo/SIC_data/'
OUT_CSV      = '/glade/work/jrotondo/ncc_paper/plots/cmip6_pistone_only/sie_native_diagnostic_v3.csv'

TARGET_YEAR = 1985
MONTHS      = [3, 9]
MONTH_NAMES = {3: 'March', 9: 'September'}
SIC_THRESH  = 0.15
N_WORKERS   = 4

MODEL_INFO = {
    'CESM2'          : dict(institution='NCAR',                member='r1i1p1f1'),
    'E3SM-1-0'       : dict(institution='E3SM-Project',        member='r1i1p1f1'),
    'MPI-ESM1-2-HR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'MPI-ESM1-2-LR'  : dict(institution='MPI-M',               member='r1i1p1f1'),
    'IPSL-CM6A-LR'   : dict(institution='IPSL',                member='r1i1p1f1'),
    'MIROC6'         : dict(institution='MIROC',               member='r1i1p1f1'),
    'MIROC-ES2L'     : dict(institution='MIROC',               member='r1i1p1f2'),
    'CanESM5'        : dict(institution='CCCma',               member='r1i1p1f1'),
    'ACCESS-ESM1-5'  : dict(institution='CSIRO',               member='r1i1p1f1'),
    'GFDL-ESM4'      : dict(institution='NOAA-GFDL',           member='r1i1p1f1'),
    'UKESM1-0-LL'    : dict(institution='MOHC',                member='r10i1p1f2'),
    'HadGEM3-GC31-LL': dict(institution='MOHC',                member='r1i1p1f3'),
    'CNRM-CM6-1'     : dict(institution='CNRM-CERFACS',        member='r10i1p1f2'),
    'BCC-CSM2-MR'    : dict(institution='BCC',                 member='r1i1p1f1'),
    'ACCESS-CM2'     : dict(institution='CSIRO-ARCCSS',        member='r1i1p1f1'),
    'CNRM-ESM2-1'    : dict(institution='CNRM-CERFACS',        member='r1i1p1f2'),
    'EC-Earth3'      : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'EC-Earth3-Veg'  : dict(institution='EC-Earth-Consortium', member='r1i1p1f1'),
    'GFDL-CM4'       : dict(institution='NOAA-GFDL',           member='r1i1p1f2'),
    'GISS-E2-1-G'    : dict(institution='NASA-GISS',           member='r1i1p1f1'),
    'GISS-E2-1-H'    : dict(institution='NASA-GISS',           member='r1i1p1f1'),
    'INM-CM4-8'      : dict(institution='INM',                 member='r1i1p1f1'),
    'INM-CM5-0'      : dict(institution='INM',                 member='r1i1p1f1'),
    'MRI-ESM2-0'     : dict(institution='MRI',                 member='r1i1p1f1'),
    'NorESM2-LM'     : dict(institution='NCC',                 member='r1i1p1f1'),
    'NorESM2-MM'     : dict(institution='NCC',                 member='r1i1p1f1'),
}

FILL_THRESHOLD = 1e10   # anything larger is a fill value, not a lat

# ── HELPERS ───────────────────────────────────────────────────────────────────
def find_siconc_file(inst, model, member):
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in ['SImon', 'SIday']:
            d = os.path.join(root, inst, model, 'historical',
                             member, table, 'siconc')
            if not os.path.isdir(d): continue
            files = sorted(glob.glob(
                os.path.join(d, '**', 'siconc_*.nc'), recursive=True))
            if not files: continue
            bns = [os.path.basename(f) for f in files]
            gr  = [f for f, b in zip(files, bns) if '_gr_' in b]
            if gr: files = gr
            for f in files:
                m = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', os.path.basename(f))
                if m:
                    s, e = int(m.group(1)[:4]), int(m.group(2)[:4])
                    if s <= TARGET_YEAR <= e:
                        return f
    return None


def find_true_lat2d(f, target_shape):
    """
    Scan all variables/coords for a usable 2D geographic latitude array.

    Handles three cases:
      1. Exact 2D match with fill values (CESM2, NorESM2-LM) — mask fills
      2. 1D lat matching first spatial dim (E3SM, GISS, INM) — broadcast
      3. Standard 2D lat without fills (most models)
    """
    try:
        with xr.open_dataset(f, use_cftime=True) as ds:
            all_names = list(ds.data_vars) + list(ds.coords)

            # Prioritise lat-named variables
            priority = [n for n in all_names if 'lat' in n.lower()]
            rest     = [n for n in all_names if n not in priority]

            for name in priority + rest:
                try:
                    arr = np.asarray(ds[name].values, dtype=float)
                except Exception:
                    continue

                # ── Case 1: exact 2D shape match ─────────────────────
                if arr.shape == target_shape:
                    # Mask fill values before range check
                    clean = np.where(np.abs(arr) < FILL_THRESHOLD, arr, np.nan)
                    flat  = clean.ravel(); flat = flat[np.isfinite(flat)]
                    if len(flat) > 0 and flat.min() >= -90.1 and flat.max() <= 90.1:
                        return clean, name   # NaN where fill values were

                # ── Case 2: 1D matching first spatial dim → broadcast ─
                elif arr.ndim == 1 and arr.shape[0] == target_shape[0]:
                    clean = np.where(np.abs(arr) < FILL_THRESHOLD, arr, np.nan)
                    flat  = clean[np.isfinite(clean)]
                    if len(flat) > 0 and flat.min() >= -90.1 and flat.max() <= 90.1:
                        lat2d = np.tile(clean[:, np.newaxis],
                                        (1, target_shape[1]))
                        return lat2d, f'{name}(1D→2D)'

    except Exception:
        pass
    return None, None


def load_areacello(inst, model, member, target_shape):
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for mem in list(dict.fromkeys([member, 'r1i1p1f1', 'r1i1p1f2'])):
            for expt in ['historical', 'piControl', 'amip']:
                for table in ['Ofx', 'fx', 'SImon', 'Amon']:
                    for var in ['areacello', 'areacella']:
                        d = os.path.join(root, inst, model, expt,
                                         mem, table, var)
                        if not os.path.isdir(d): continue
                        files = sorted(glob.glob(
                            os.path.join(d, '**', f'{var}_*.nc'),
                            recursive=True))
                        if not files: continue
                        try:
                            with xr.open_dataset(files[0]) as ds:
                                area = ds[var].values.astype(float)
                            if area.shape == target_shape:
                                return area, var
                        except Exception:
                            continue
    return None, None


def obs_sie_both():
    PSN25 = ('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 '
             '+x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
    xs = -3850e3 + np.arange(304) * 25e3
    ys = -5350e3 + np.arange(448) * 25e3
    xx, yy = np.meshgrid(xs, ys)
    tfm = Transformer.from_crs(PSN25, 'EPSG:4326', always_xy=True)
    _, lat2d = tfm.transform(xx, yy)
    arctic = lat2d >= 60.0
    SIC_VARS = ['cdr_seaice_conc', 'cdr_seaice_conc_monthly',
                'goddard_merged_seaice_conc']
    result = {}
    for f in sorted(glob.glob(os.path.join(NOAA_CDR_DIR, 'sic_psn25_*.nc'))):
        try:
            with xr.open_dataset(f, mask_and_scale=True) as ds:
                t = pd.Timestamp(ds['time'].values[0])
                if t.year != TARGET_YEAR or t.month not in MONTHS: continue
                for v in SIC_VARS:
                    if v in ds:
                        raw = ds[v].squeeze().values.astype(float)
                        if np.nanmax(raw) > 1.5: raw /= 100.
                        raw = np.where((raw >= 0) & (raw <= 1), raw, np.nan)
                        result[t.month] = float(
                            np.sum(arctic & np.isfinite(raw) &
                                   (raw >= SIC_THRESH)) * 625.0 / 1e6)
                        break
        except Exception: continue
    return result.get(3, np.nan), result.get(9, np.nan)


# ── PER-MODEL WORKER ──────────────────────────────────────────────────────────
def process_model_sie(model):
    inst   = MODEL_INFO[model]['institution']
    member = MODEL_INFO[model]['member']
    row    = dict(model=model, member=member,
                  sie_mar=np.nan, sie_sep=np.nan,
                  bias_mar=np.nan, bias_sep=np.nan,
                  area_source='none', lat_source='none',
                  siconc_file='none',
                  ice_cells_mar=0, ice_cells_sep=0,
                  notes='')
    notes = []

    f = find_siconc_file(inst, model, member)
    if f is None:
        row['notes'] = 'no siconc file — data not staged'
        return row
    row['siconc_file'] = os.path.basename(f)

    results    = {}
    ice_counts = {}
    area_src   = 'none'
    lat_src    = 'none'

    for month in MONTHS:
        try:
            ds = xr.open_dataset(f, use_cftime=True)
            da = ds['siconc']

            yr_arr = np.array([int(t.dt.year)  for t in da.time])
            mn_arr = np.array([int(t.dt.month) for t in da.time])
            idx    = np.where((yr_arr == TARGET_YEAR) & (mn_arr == month))[0]
            if len(idx) == 0:
                ds.close()
                notes.append(f'no {MONTH_NAMES[month]} {TARGET_YEAR}')
                continue

            sic = da.isel(time=idx[0]).values.astype(float)
            ds.close()

            if np.nanmax(sic) > 1.5: sic /= 100.
            sic = np.where((sic >= 0) & (sic <= 1), sic, np.nan)

            # ── True 2D lat (handles fills + 1D broadcast) ───────────
            lat2d, lat_name = find_true_lat2d(f, sic.shape)
            if lat2d is None:
                notes.append(f'no lat2d ({MONTH_NAMES[month]})')
                continue
            lat_src = lat_name

            # ── Cell area ─────────────────────────────────────────────
            area_m2, area_name = load_areacello(inst, model, member, sic.shape)
            if area_m2 is not None:
                area_src = area_name
            else:
                dlat = float(np.nanmean(np.abs(np.diff(lat2d, axis=0)))) \
                       if lat2d.ndim == 2 else 1.0
                if not np.isfinite(dlat) or dlat == 0:
                    lat_valid = lat2d[np.isfinite(lat2d)]
                    dlat = float((lat_valid.max() - lat_valid.min()) /
                                 sic.shape[0])
                dlon = 360.0 / sic.shape[1]
                area_m2 = np.maximum(
                    np.cos(np.deg2rad(lat2d)) * dlat * dlon *
                    (111_320.0 ** 2), 0.0)
                area_src = f'cos-lat ({dlat:.3f}°x{dlon:.3f}°)'

            # ── SIE ───────────────────────────────────────────────────
            arctic = lat2d >= 60.0
            ice    = (arctic & np.isfinite(sic) & (sic >= SIC_THRESH) &
                      np.isfinite(area_m2) & (area_m2 > 0))
            sie    = float(np.nansum(area_m2[ice])) / 1e12
            results[month]    = sie
            ice_counts[month] = int(ice.sum())

        except Exception as e:
            notes.append(f'{MONTH_NAMES[month]}: {str(e)[:60]}')

    row['sie_mar']       = results.get(3, np.nan)
    row['sie_sep']       = results.get(9, np.nan)
    row['ice_cells_mar'] = ice_counts.get(3, 0)
    row['ice_cells_sep'] = ice_counts.get(9, 0)
    row['area_source']   = area_src
    row['lat_source']    = lat_src
    row['notes']         = ' | '.join(notes) if notes else 'ok'
    return row


# ── RUN ───────────────────────────────────────────────────────────────────────
obs_mar, obs_sep = obs_sie_both()
print(f"\n{'='*80}")
print(f"  SIE DIAGNOSTIC v3 — ALL CMIP6 MODELS — {TARGET_YEAR}")
print(f"{'='*80}")
print(f"  OBS: March={obs_mar:.3f}  September={obs_sep:.3f}  Mkm²\n")

all_rows = []
with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(process_model_sie, m): m for m in MODEL_INFO}
    for fut in as_completed(futures):
        model = futures[fut]
        try:
            row = fut.result()
            row['bias_mar'] = row['sie_mar'] - obs_mar \
                              if np.isfinite(row['sie_mar']) else np.nan
            row['bias_sep'] = row['sie_sep'] - obs_sep \
                              if np.isfinite(row['sie_sep']) else np.nan
            all_rows.append(row)
        except Exception as e:
            all_rows.append(dict(
                model=model, member=MODEL_INFO[model]['member'],
                sie_mar=np.nan, sie_sep=np.nan,
                bias_mar=np.nan, bias_sep=np.nan,
                area_source='ERROR', lat_source='ERROR',
                siconc_file='ERROR',
                ice_cells_mar=0, ice_cells_sep=0,
                notes=str(e)[:80]))

df = pd.DataFrame(all_rows).sort_values('bias_sep')
df.to_csv(OUT_CSV, index=False)

# ── PRINT ─────────────────────────────────────────────────────────────────────
print(f"  {'Model':<22} {'Mar':>7} {'Sep':>7} {'ΔMar':>7} {'ΔSep':>7} "
      f"{'IceCells(Sep)':>14}  {'Area src':<26} {'Lat src':<18} Notes")
print(f"  {'-'*120}")

for _, r in df.iterrows():
    mar  = f"{r.sie_mar:.2f}"   if np.isfinite(r.sie_mar)  else '  NaN'
    sep  = f"{r.sie_sep:.2f}"   if np.isfinite(r.sie_sep)  else '  NaN'
    bmar = f"{r.bias_mar:+.2f}" if np.isfinite(r.bias_mar) else '  NaN'
    bsep = f"{r.bias_sep:+.2f}" if np.isfinite(r.bias_sep) else '  NaN'
    flag = '⚠️ ' if (np.isfinite(r.bias_sep) and abs(r.bias_sep) > 2) \
           else '❌ ' if not np.isfinite(r.sie_sep) else '   '
    print(f"  {flag}{r.model:<20} {mar:>7} {sep:>7} {bmar:>7} {bsep:>7} "
          f"{r.ice_cells_sep:>14,}  {str(r.area_source):<26} "
          f"{str(r.lat_source):<18} {r.notes}")

# ── SUMMARY WITH FIXED OPERATOR PRECEDENCE ────────────────────────────────────
ok   = df[(df['bias_sep'].abs() <= 2) & df['bias_sep'].notna()]
low  = df[(df['bias_sep'] < -2)       & df['bias_sep'].notna()]
high = df[(df['bias_sep'] >  2)       & df['bias_sep'].notna()]
fail = df[df['sie_sep'].isna()]

print(f"\n  Saved → {OUT_CSV}")
print(f"\n  ── CATEGORIES ──────────────────────────────────────────────────────")
print(f"  ✅ Close to OBS (|ΔSep| ≤ 2):  {', '.join(ok['model'].tolist()) or 'none'}")
print(f"  🔵 Too little ice (ΔSep < -2): {', '.join(low['model'].tolist()) or 'none'}")
print(f"  🔴 Too much ice   (ΔSep > +2): {', '.join(high['model'].tolist()) or 'none'}")
print(f"  ❌ Failed / no data:            {', '.join(fail['model'].tolist()) or 'none'}")