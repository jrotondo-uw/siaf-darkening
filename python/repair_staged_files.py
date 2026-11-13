#!/usr/bin/env python
"""
repair_staged_files.py
──────────────────────
Scans STAGE_ROOT for corrupt or unreadable NetCDF files, then
re-downloads clean replacements from ESGF.

Two failure modes handled:
  1. HDF / IO backend errors  — file exists but is unreadable
     (MPI-ESM1-2-LR r1 rsdscs, r21 rsdt; MIROC6 staged rsdt chunks)
  2. concat_failed remnants   — directory contains only raw _chunk_*.nc
     or _tmp_*.nc files with no merged output
     (NorESM2-LM r1 rsdt/rsutcs; NorESM2-MM r1 all vars)

Workflow:
  Phase 1  – walk STAGE_ROOT, probe every .nc file, collect bad ones
  Phase 2  – search ESGF for fresh copies of bad (model, member, variable)
  Phase 3  – download replacements (ThreadPoolExecutor, N_DL_WORKERS=2)
             old corrupt files are deleted before writing new ones

Usage:
  python repair_staged_files.py [--dry-run] [--models MODEL [MODEL ...]]

  --dry-run   Print what would be done without deleting or downloading
  --models    Limit scan to specific models (e.g. NorESM2-LM MPI-ESM1-2-LR)

Set ESGF_NODE and STAGE_ROOT below to match your environment.
"""

import os
import re
import gc
import sys
import glob
import time
import shutil
import hashlib
import argparse
import requests
import threading
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIG — edit to match your environment
# ============================================================
STAGE_ROOT     = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'
ESGF_NODE      = 'https://esgf-data.dkrz.de/esg-search'
N_DL_WORKERS   = 2
TRIM_START_YEAR = 1982

# Home-node overrides (same as audit script)
HOME_NODES = {
    'ACCESS-CM2'   : 'https://esgf.nci.org.au/esg-search',
    'ACCESS-ESM1-5': 'https://esgf.nci.org.au/esg-search',
    'CanESM5'      : 'https://esgf-node.llnl.gov/esg-search',
    'HadGEM3-GC31-LL': 'https://esgf-index1.ceda.ac.uk/esg-search',
    'UKESM1-0-LL'  : 'https://esgf-index1.ceda.ac.uk/esg-search',
    'MIROC6'       : 'https://esgf-node.llnl.gov/esg-search',
    'NorESM2-LM'   : 'https://esgf-data.dkrz.de/esg-search',
    'NorESM2-MM'   : 'https://esgf-data.dkrz.de/esg-search',
}

BLOCKED_HOSTS = {'esg1.umr-cnrm.fr', 'esgf.ichec.ie'}

# Variable → CMIP6 table
VAR_TABLE_MAP = {
    'tas'    : ['Amon'],
    'rsus'   : ['Amon'],
    'rsds'   : ['Amon'],
    'rsdt'   : ['Amon'],
    'rsut'   : ['Amon'],
    'rsutcs' : ['Amon', 'CFmon'],
    'rsdscs' : ['Amon', 'CFmon'],
    'rsuscs' : ['Amon', 'CFmon'],
    'siconc' : ['SImon'],
}

# ============================================================
# HELPERS
# ============================================================

def ts():
    return datetime.now().strftime('%H:%M:%S')


def _is_chunk_or_tmp(path):
    """True if this looks like a leftover concat chunk/tmp file."""
    bn = os.path.basename(path)
    return '_chunk_' in bn or bn.startswith('_tmp_') or bn.endswith('.tmp')


def _open_dataset_compat(path):
    """
    Open a single NetCDF file compatibly across xarray versions.
    Newer xarray (>=2024) deprecates use_cftime as a kwarg;
    use CFDatetimeCoder if available, otherwise fall back.
    """
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(path, decode_times=time_coder)
    except AttributeError:
        # xarray < 2024 — old API still works
        return xr.open_dataset(path, use_cftime=True)


def _probe_file(path):
    """
    Try to open a single .nc file and read one value.
    Returns (ok: bool, error_msg: str).
    """
    try:
        ds = _open_dataset_compat(path)
        varnames = [v for v in ds.data_vars]
        if varnames:
            _ = ds[varnames[0]].isel(
                {d: 0 for d in ds[varnames[0]].dims}
            ).values
        ds.close()
        return True, ''
    except Exception as e:
        return False, str(e)


def _parse_staged_path(filepath):
    """
    Extract (institution, model, member, table, variable) from a
    STAGE_ROOT path like:
      STAGE_ROOT/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/rsdscs/xxx.nc
    Returns None if path doesn't match.
    """
    rel = os.path.relpath(filepath, STAGE_ROOT)
    parts = rel.split(os.sep)
    # Expected: inst / model / 'historical' / member / table / variable / file
    if len(parts) < 7:
        return None
    inst, model, exp, member, table, variable = parts[:6]
    if exp != 'historical':
        return None
    return dict(institution=inst, model=model, member=member,
                table=table, variable=variable, filepath=filepath)


# ============================================================
# PHASE 1  –  SCAN FOR CORRUPT FILES
# ============================================================

def scan_stage_root(models_filter=None):
    """
    Walk STAGE_ROOT and probe every .nc file.
    Returns a list of dicts describing bad files/directories.
    """
    print(f'\n[{ts()}] ── PHASE 1: scanning {STAGE_ROOT} ──', flush=True)

    bad_files   = []   # {institution, model, member, table, variable, filepath, reason}
    concat_dirs = []   # dirs that contain only chunks/tmp with no merged file

    for root, dirs, files in os.walk(STAGE_ROOT):
        nc_files = [f for f in files if f.endswith('.nc')]
        if not nc_files:
            continue

        # Check for concat_failed: all files are chunk/tmp, no real nc
        real_nc    = [f for f in nc_files if not _is_chunk_or_tmp(f)]
        chunk_only = [f for f in nc_files if _is_chunk_or_tmp(f)]

        if chunk_only and not real_nc:
            # Directory has only leftover chunks — concat never finished
            rel = os.path.relpath(root, STAGE_ROOT)
            parts = rel.split(os.sep)
            if len(parts) >= 6:
                inst, model, exp, member, table, variable = parts[:6]
                if exp == 'historical':
                    if models_filter and model not in models_filter:
                        continue
                    print(f'  [concat_failed] {model}/{member}/{variable} '
                          f'({len(chunk_only)} chunk(s), no merged file)',
                          flush=True)
                    concat_dirs.append(dict(
                        institution=inst, model=model, member=member,
                        table=table, variable=variable,
                        dirpath=root,
                        chunk_files=[os.path.join(root, f) for f in chunk_only],
                        reason='concat_failed',
                    ))
            continue

        # Probe real nc files for readability
        for fname in real_nc:
            fpath = os.path.join(root, fname)
            info = _parse_staged_path(fpath)
            if info is None:
                continue
            if models_filter and info['model'] not in models_filter:
                continue

            ok, errmsg = _probe_file(fpath)
            if not ok:
                print(f'  [corrupt] {info["model"]}/{info["member"]}/'
                      f'{info["variable"]}/{fname}', flush=True)
                print(f'    reason: {errmsg[:120]}', flush=True)
                info['reason'] = errmsg
                bad_files.append(info)

    print(f'\n  Found {len(bad_files)} corrupt file(s) and '
          f'{len(concat_dirs)} concat_failed variable dir(s).', flush=True)
    return bad_files, concat_dirs


# ============================================================
# PHASE 2  –  ESGF SEARCH
# ============================================================

def esgf_search(node, model, member, variable, table):
    """Search ESGF for files for a single (model, member, variable, table)."""
    params = dict(
        type='File',
        project='CMIP6',
        experiment_id='historical',
        source_id=model,
        variable_id=variable,
        variant_label=member,
        table_id=table,
        format='application/solr+json',
        fields='url,title,checksum,checksum_type,size',
        limit=1000,
        distrib='true',
        latest='true',
    )
    try:
        r = requests.get(f'{node}/search', params=params, timeout=60)
        r.raise_for_status()
        docs = r.json()['response']['docs']
        results = []
        for doc in docs:
            for url_entry in doc.get('url', []):
                if '|HTTPServer' in url_entry:
                    url = url_entry.split('|')[0]
                    host = url.split('/')[2]
                    if host not in BLOCKED_HOSTS:
                        results.append(dict(
                            url=url,
                            title=doc.get('title', ''),
                            size=doc.get('size', 0),
                            checksum=doc.get('checksum', [''])[0],
                            checksum_type=doc.get('checksum_type', [''])[0],
                        ))
        return results
    except Exception as e:
        print(f'    ESGF search error ({node}): {e}', flush=True)
        return []


def find_esgf_replacements(repair_items):
    """
    For each (model, member, variable) that needs repair, search ESGF.
    Returns list of {model, member, variable, table, url, title, size, ...}.
    """
    print(f'\n[{ts()}] ── PHASE 2: searching ESGF ──', flush=True)

    # Deduplicate by (model, member, variable)
    seen = set()
    unique = []
    for item in repair_items:
        key = (item['model'], item['member'], item['variable'])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    downloads = []
    for item in unique:
        model    = item['model']
        member   = item['member']
        variable = item['variable']
        node     = HOME_NODES.get(model, ESGF_NODE)
        tables   = VAR_TABLE_MAP.get(variable, ['Amon'])

        found = []
        for table in tables:
            results = esgf_search(node, model, member, variable, table)
            if not results and node != ESGF_NODE:
                results = esgf_search(ESGF_NODE, model, member, variable, table)
            for r in results:
                r['model']    = model
                r['member']   = member
                r['variable'] = variable
                r['table']    = table
                found.append(r)
            if found:
                break

        if found:
            print(f'  {model}/{member}/{variable}: {len(found)} file(s) on ESGF',
                  flush=True)
            downloads.extend(found)
        else:
            print(f'  {model}/{member}/{variable}: NOT FOUND on ESGF', flush=True)

    return downloads


# ============================================================
# TRIM HELPER
# ============================================================

def _trim_file_inplace(filepath, start_year=TRIM_START_YEAR):
    """Trim a NetCDF file to start_year in-place using CDO."""
    tmp = filepath + '.trim_tmp.nc'
    try:
        result = subprocess.run(
            ['cdo', f'selyear,{start_year}/2100', filepath, tmp],
            capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, filepath)
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ============================================================
# PHASE 3  –  DOWNLOAD
# ============================================================

_dl_lock = threading.Lock()

def _download_one(dl_item, dry_run=False):
    """
    Download one file entry from ESGF.
    Saves to the correct STAGE_ROOT subdirectory, then trims to TRIM_START_YEAR.
    """
    url      = dl_item['url']
    model    = dl_item['model']
    member   = dl_item['member']
    variable = dl_item['variable']
    table    = dl_item['table']
    title    = dl_item['title']

    # Reconstruct the institution from the path pattern in STAGE_ROOT
    # Look for existing dir or guess from URL
    inst_guess = url.split('/')[2].split('.')[0]   # rough fallback
    # Better: find existing staged dir for this model
    for d in glob.glob(os.path.join(STAGE_ROOT, '*', model)):
        inst_guess = os.path.basename(os.path.dirname(d))
        break

    dest_dir = os.path.join(STAGE_ROOT, inst_guess, model,
                            'historical', member, table, variable)
    dest_path = os.path.join(dest_dir, title)

    if dry_run:
        print(f'  [DRY-RUN] would download {title} -> {dest_dir}', flush=True)
        return 'dry_run'

    os.makedirs(dest_dir, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp_path = dest_path + '.download_tmp'
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp_path, dest_path)
        _trim_file_inplace(dest_path)
        print(f'  [OK] {model}/{member}/{variable}/{title}', flush=True)
        return 'ok'
    except Exception as e:
        print(f'  [FAIL] {model}/{member}/{variable}/{title}: {e}', flush=True)
        if os.path.exists(dest_path + '.download_tmp'):
            os.remove(dest_path + '.download_tmp')
        return 'fail'


def delete_bad_files(bad_files, concat_dirs, dry_run=False):
    """
    Remove corrupt files and concat_failed chunk directories so
    fresh downloads land cleanly.
    """
    print(f'\n[{ts()}] ── Removing corrupt/chunk files ──', flush=True)
    for item in bad_files:
        fp = item['filepath']
        if dry_run:
            print(f'  [DRY-RUN] would delete {fp}', flush=True)
        else:
            try:
                os.remove(fp)
                print(f'  deleted {fp}', flush=True)
            except Exception as e:
                print(f'  could not delete {fp}: {e}', flush=True)

    for item in concat_dirs:
        for fp in item['chunk_files']:
            if dry_run:
                print(f'  [DRY-RUN] would delete chunk {os.path.basename(fp)}',
                      flush=True)
            else:
                try:
                    os.remove(fp)
                except Exception as e:
                    print(f'  could not delete chunk {fp}: {e}', flush=True)


def run_downloads(downloads, dry_run=False):
    print(f'\n[{ts()}] ── PHASE 3: downloading {len(downloads)} file(s) '
          f'({N_DL_WORKERS} workers) ──', flush=True)

    results = {'ok': 0, 'fail': 0, 'dry_run': 0}
    with ThreadPoolExecutor(max_workers=N_DL_WORKERS) as pool:
        futures = {pool.submit(_download_one, dl, dry_run): dl
                   for dl in downloads}
        for fut in as_completed(futures):
            status = fut.result()
            results[status] = results.get(status, 0) + 1

    print(f'\n  Download summary: {results}', flush=True)
    return results


# ============================================================
# CONCAT REPAIR  (for concat_failed dirs after fresh download)
# ============================================================

def attempt_concat_repair(concat_dirs, dry_run=False):
    """
    After fresh files are downloaded, check if a previously concat_failed
    dir now has real files and try to verify them.
    """
    if not concat_dirs:
        return
    print(f'\n[{ts()}] ── Checking concat_failed dirs after download ──',
          flush=True)
    for item in concat_dirs:
        dirpath  = item['dirpath']
        model    = item['model']
        member   = item['member']
        variable = item['variable']
        nc_files = sorted(glob.glob(os.path.join(dirpath, '*.nc')))
        real     = [f for f in nc_files if not _is_chunk_or_tmp(f)]
        if real:
            print(f'  {model}/{member}/{variable}: {len(real)} file(s) now present',
                  flush=True)
            for f in real:
                ok, err = _probe_file(f)
                if ok:
                    print(f'    OK  {os.path.basename(f)}', flush=True)
                else:
                    print(f'    BAD {os.path.basename(f)}: {err[:80]}', flush=True)
        else:
            print(f'  {model}/{member}/{variable}: still empty after download',
                  flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Repair corrupt CMIP6 staged files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print actions without deleting or downloading')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Limit to specific models (e.g. NorESM2-LM MPI-ESM1-2-LR)')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('CMIP6 STAGED FILE REPAIR', flush=True)
    print(f'  STAGE_ROOT  : {STAGE_ROOT}', flush=True)
    print(f'  ESGF node   : {ESGF_NODE}', flush=True)
    print(f'  DL workers  : {N_DL_WORKERS}', flush=True)
    print(f'  Dry run     : {args.dry_run}', flush=True)
    if args.models:
        print(f'  Models      : {args.models}', flush=True)
    print('=' * 70, flush=True)

    # ── Phase 1: scan ──────────────────────────────────────────────────
    bad_files, concat_dirs = scan_stage_root(models_filter=args.models)

    if not bad_files and not concat_dirs:
        print('\nNothing to repair. All staged files look clean.', flush=True)
        return

    # ── Summarise what needs repair ────────────────────────────────────
    print('\nREPAIR NEEDED:', flush=True)
    print(f'  corrupt files    : {len(bad_files)}', flush=True)
    print(f'  concat_failed dirs: {len(concat_dirs)}', flush=True)

    # Build combined repair list for ESGF search
    # For corrupt files: need the specific file's variable
    # For concat_failed: need to re-fetch the whole variable's date range
    repair_items = []
    for item in bad_files:
        repair_items.append({
            'model'    : item['model'],
            'member'   : item['member'],
            'variable' : item['variable'],
            'table'    : item['table'],
        })
    for item in concat_dirs:
        repair_items.append({
            'model'    : item['model'],
            'member'   : item['member'],
            'variable' : item['variable'],
            'table'    : item['table'],
        })

    # ── Phase 2: ESGF search ───────────────────────────────────────────
    downloads = find_esgf_replacements(repair_items)

    if not downloads:
        print('\nNo ESGF replacements found. Nothing to download.', flush=True)
        return

    # ── Delete corrupt/chunk files before downloading ──────────────────
    delete_bad_files(bad_files, concat_dirs, dry_run=args.dry_run)

    # ── Phase 3: download ──────────────────────────────────────────────
    run_downloads(downloads, dry_run=args.dry_run)

    # ── Post-download concat check ─────────────────────────────────────
    attempt_concat_repair(concat_dirs, dry_run=args.dry_run)

    print(f'\n[{ts()}] DONE', flush=True)


if __name__ == '__main__':
    main()