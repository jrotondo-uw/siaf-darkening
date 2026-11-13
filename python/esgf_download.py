#!/usr/bin/env python
"""
audit_and_download_cmip6.py
───────────────────────────
1. Audits both CMIP6_ROOT and STAGE_ROOT for every model × member × variable
   required by the SIAF Pistone B2 pipeline.
2. Prints a colour-coded availability table.
3. For anything missing, searches ESGF and either:
     a) downloads files directly into STAGE_ROOT (parallel, trimmed to
        TRIM_START_YEAR), or
     b) writes a wget script per variable if --dry-run is passed.
   Files with many chunks (e.g. EC-Earth3 monthly files) are downloaded
   into a temp directory and concatenated into a single NetCDF (also trimmed).

Key fixes vs. previous version
  - SKIP_HOSTS blacklist: any URL whose host is in this set is dropped at
    search time, so the French node (esg1.umr-cnrm.fr) is never tried even
    if distrib=true returns it.
  - Per-file replica list: search now keeps ALL non-blacklisted URLs for each
    file, ranked by PREFERRED_HOSTS.  download_file() walks the list so a
    timeout on one host automatically falls through to the next replica.
  - Shorter per-attempt timeout (300 s instead of 600 s).

Parallelism
  Downloads are I/O-bound, so ThreadPoolExecutor is used (not processes).
  N_DOWNLOAD_WORKERS controls concurrency.

Date trimming
  CMIP6 historical files often start in 1850.  After a successful download
  (or during concat), any data before TRIM_START_YEAR is sliced out and the
  file is overwritten in-place via a safe rename from a .tmp file.

--from-priority-csv MODE
  Pass the esgf_stage_priority.csv produced by esgf_upgrade_scan.py to
  restrict downloads to only the (model, member, variable) triples that
  can reach full_T07 or approx_T07.

Requirements:
    pip install pyesgf requests xarray
"""

import os, sys, glob, json, time, argparse, re, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import xarray as xr

os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = '1'

try:
    from pyesgf.search import SearchConnection
    HAS_PYESGF = True
except ImportError:
    HAS_PYESGF = False
    print("WARNING: pyesgf not installed. Falling back to direct ESGF REST API.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CMIP6_ROOT = '/glade/collections/cmip/CMIP6/CMIP'
STAGE_ROOT = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

# Earliest year to keep after download.  Data before this is trimmed.
TRIM_START_YEAR = 1982

# Parallel download workers
N_DOWNLOAD_WORKERS = 2

ESGF_NODES = [
    'https://esgf-data.dkrz.de/esg-search',
    'https://esgf.ceda.ac.uk/esg-search',
    'https://esgf-node.ipsl.upmc.fr/esg-search',
    'https://esgf-node.llnl.gov/esg-search',
]
ESGF_NODE = ESGF_NODES[0]

# Preferred download hosts in order (first = most preferred).
PREFERRED_HOSTS = [
    'esgf-data.dkrz.de',
    'esgf.ceda.ac.uk',
    'esgf.nci.org.au',
    'esgf-node.ipsl.upmc.fr',
    'aims3.llnl.gov',
    'esgf.ichec.ie',
    'esg1.umr-cnrm.fr',       # kept as last-resort fallback below SKIP_HOSTS
]

# Hosts to NEVER download from — URLs from these are dropped at search time.
# When a host comes back online, remove it from this set.
SKIP_HOSTS = {
    'esg1.umr-cnrm.fr',   # French CNRM node — persistent ConnectTimeout
    'esgf3.dkrz.de',      # occasional DNS failures
}

ESGF_LIMIT       = 500
CONCAT_THRESHOLD = 20

REQUIRED_VARS = ['tas', 'siconc', 'rsus', 'rsds']
OPTIONAL_VARS = ['rsdt', 'rsut', 'rsutcs', 'rsdscs', 'rsuscs']
ALL_VARS      = REQUIRED_VARS + OPTIONAL_VARS

VAR_TABLE_MAP = {
    'tas'    : ['Amon'],
    'rsus'   : ['Amon'],
    'rsds'   : ['Amon'],
    'rsdt'   : ['Amon'],
    'rsut'   : ['Amon'],
    'rsutcs' : ['Amon', 'CFmon'],
    'rsdscs' : ['Amon', 'CFmon'],
    'rsuscs' : ['Amon', 'CFmon'],
    'siconc' : ['SImon', 'SIday'],
}

MODEL_INFO = {
    'ACCESS-CM2'      : dict(institution='CSIRO-ARCCSS',       primary='r1i1p1f1'),
    'AWI-CM-1-1-MR'   : dict(institution='AWI',                primary='r1i1p1f1'),
    'CAMS-CSM1-0'     : dict(institution='CAMS',               primary='r1i1p1f1'),
    'CAS-ESM2-0'      : dict(institution='CAS',                primary='r1i1p1f1'),
    'CIESM'           : dict(institution='THU',                primary='r1i1p1f1'),
    'CMCC-CM2-SR5'    : dict(institution='CMCC',               primary='r1i1p1f1'),
    'CNRM-ESM2-1'     : dict(institution='CNRM-CERFACS',       primary='r1i1p1f2'),
    'EC-Earth3'       : dict(institution='EC-Earth-Consortium',primary='r1i1p1f1'),
    'EC-Earth3-Veg'   : dict(institution='EC-Earth-Consortium',primary='r1i1p1f1'),
    'FGOALS-g3'       : dict(institution='CAS',                primary='r1i1p1f1'),
    'FIO-ESM-2-0'     : dict(institution='FIO-QLNM',           primary='r1i1p1f1'),
    'GFDL-CM4'        : dict(institution='NOAA-GFDL',          primary='r1i1p1f2'),
    'GISS-E2-1-G'     : dict(institution='NASA-GISS',          primary='r1i1p1f1'),
    'GISS-E2-1-H'     : dict(institution='NASA-GISS',          primary='r1i1p1f1'),
    'INM-CM4-8'       : dict(institution='INM',                primary='r1i1p1f1'),
    'INM-CM5-0'       : dict(institution='INM',                primary='r1i1p1f1'),
    'KACE-1-0-G'      : dict(institution='NIMS-KMA',           primary='r1i1p1f1'),
    'MCM-UA-1-0'      : dict(institution='UA',                 primary='r1i1p1f1'),
    'MIROC-ES2L'      : dict(institution='MIROC',              primary='r1i1p1f2'),
    'MRI-ESM2-0'      : dict(institution='MRI',                primary='r1i1p1f1'),
    'CESM2'           : dict(institution='NCAR',               primary='r1i1p1f1'),
    'E3SM-1-0'        : dict(institution='E3SM-Project',       primary='r1i1p1f1'),
    'MPI-ESM1-2-HR'   : dict(institution='MPI-M',              primary='r1i1p1f1'),
    'MPI-ESM1-2-LR'   : dict(institution='MPI-M',              primary='r1i1p1f1'),
    'IPSL-CM6A-LR'    : dict(institution='IPSL',               primary='r1i1p1f1'),
    'MIROC6'          : dict(institution='MIROC',              primary='r1i1p1f1'),
    'CanESM5'         : dict(institution='CCCma',              primary='r1i1p1f1'),
    'ACCESS-ESM1-5'   : dict(institution='CSIRO',              primary='r1i1p1f1'),
    'GFDL-ESM4'       : dict(institution='NOAA-GFDL',          primary='r1i1p1f1'),
    'UKESM1-0-LL'     : dict(institution='MOHC',               primary='r10i1p1f2'),
    'HadGEM3-GC31-LL' : dict(institution='MOHC',               primary='r1i1p1f3'),
    'CNRM-CM6-1'      : dict(institution='CNRM-CERFACS',       primary='r10i1p1f2'),
    'BCC-CSM2-MR'     : dict(institution='BCC',                primary='r1i1p1f1'),
    'NorESM2-LM'      : dict(institution='NCC',                primary='r1i1p1f1'),
    'NorESM2-MM'      : dict(institution='NCC',                primary='r1i1p1f1'),
}

RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RESET  = '\033[0m'


# ─────────────────────────────────────────────────────────────────────────────
# URL RANKING & FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def _rank_url(url):
    """Return sort key for a URL. Returns 999 if host is blacklisted."""
    for host in SKIP_HOSTS:
        if host in url:
            return 999
    for i, host in enumerate(PREFERRED_HOSTS):
        if host in url:
            return i
    return 98


def _is_usable(url):
    return _rank_url(url) < 999


# ─────────────────────────────────────────────────────────────────────────────
# ESGF NODE PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _probe_esgf_nodes():
    global ESGF_NODE
    for node in ESGF_NODES:
        try:
            r = requests.get(f'{node}/search',
                             params={'type': 'Dataset', 'limit': 1,
                                     'format': 'application/solr+json'},
                             timeout=10)
            if r.status_code == 200:
                ESGF_NODE = node
                print(f"  Using ESGF node: {node}")
                return
        except Exception:
            continue
    print("WARNING: No ESGF node responded — searches will fail.")


# ─────────────────────────────────────────────────────────────────────────────
# MEMBER FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def _skip_member(model, member):
    if model in ('EC-Earth3', 'EC-Earth3-Veg'):
        m = re.match(r'r(\d+)i', member)
        if m and int(m.group(1)) > 300:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prefer_regular_grid(files):
    if not files:
        return files
    bns    = [os.path.basename(f) for f in files]
    is_gr  = lambda bn: ('_gr_' in bn) or ('_gr1_' in bn) or ('_gr2_' in bn)
    is_gn  = lambda bn: '_gn_' in bn
    has_gr = any(is_gr(bn) for bn in bns)
    has_gn = any(is_gn(bn) for bn in bns)
    if has_gr and has_gn:
        return [f for f, bn in zip(files, bns) if is_gr(bn)]
    return files


def find_var_files(inst, model, member, variable):
    """Search both CMIP6_ROOT and STAGE_ROOT. Returns (files, source_tag)."""
    tables = VAR_TABLE_MAP.get(variable, ['Amon'])
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        for table in tables:
            d = os.path.join(root, inst, model, 'historical', member, table, variable)
            if os.path.isdir(d):
                files = sorted(glob.glob(
                    os.path.join(d, '**', f'{variable}_*.nc'), recursive=True))
                if files:
                    tag = 'CMIP6_ROOT' if root == CMIP6_ROOT else 'STAGE'
                    return _prefer_regular_grid(files), tag
    return [], None


def discover_local_members(inst, model):
    members = set()
    for root in [CMIP6_ROOT, STAGE_ROOT]:
        base = os.path.join(root, inst, model, 'historical')
        if not os.path.isdir(base):
            continue
        for mem in sorted(os.listdir(base)):
            for v in ALL_VARS:
                for table in VAR_TABLE_MAP[v]:
                    p = os.path.join(base, mem, table, v)
                    if os.path.isdir(p):
                        members.add(mem)
                        break
    return sorted(members)


def discover_esgf_members(model):
    params = dict(
        type='Dataset', project='CMIP6', activity_id='CMIP',
        experiment_id='historical', source_id=model,
        distrib='true', latest='true', limit=500,
        format='application/solr+json', fields='member_id', facets='member_id',
    )
    try:
        r = requests.get(f'{ESGF_NODE}/search', params=params, timeout=30)
        r.raise_for_status()
        counts = r.json()['facet_counts']['facet_fields'].get('member_id', [])
        return sorted(counts[::2])
    except Exception as e:
        print(f"    ESGF member discovery failed for {model}: {e}")
        return []


def audit_model(model):
    inst    = MODEL_INFO[model]['institution']
    primary = MODEL_INFO[model]['primary']

    local_members = discover_local_members(inst, model)
    esgf_members  = discover_esgf_members(model)
    all_members   = sorted(
        m for m in set(local_members) | set(esgf_members) | {primary}
        if not _skip_member(model, m)
    )

    audit = {}
    for mem in all_members:
        row = {}
        for v in ALL_VARS:
            files, src = find_var_files(inst, model, mem, v)
            if files:
                row[v] = ('ok', src)
            else:
                status = 'missing' if v in REQUIRED_VARS else 'optional_missing'
                row[v] = (status, None)
        audit[mem] = row
    return audit


# ─────────────────────────────────────────────────────────────────────────────
# ESGF SEARCH  — returns list of (filename, [url, url, ...])
# ─────────────────────────────────────────────────────────────────────────────

def _parse_docs_to_replica_map(docs):
    """
    Parse ESGF search docs into {filename: [ranked_url, ...]} keeping all
    non-blacklisted replicas so download_file() can fall through on failure.
    """
    file_urls = defaultdict(list)
    for doc in docs:
        title = doc.get('title', '')
        for u in doc.get('url', []):
            parts = u.split('|')
            if len(parts) == 3 and parts[2] == 'HTTPServer':
                url = parts[0]
                if _is_usable(url) and url not in file_urls[title]:
                    file_urls[title].append(url)
    # Sort each file's list by host preference
    for title in file_urls:
        file_urls[title].sort(key=_rank_url)
    return file_urls


def esgf_search_rest(model, member, variable, table):
    HOME_NODES = {
        'MIROC6'          : 'https://esgf-node.ipsl.upmc.fr/esg-search',
        'CanESM5'         : 'https://esgf.ceda.ac.uk/esg-search',
        'ACCESS-ESM1-5'   : 'https://esgf.nci.org.au/esg-search',
        'ACCESS-CM2'      : 'https://esgf.nci.org.au/esg-search',
        'NorESM2-LM'      : 'https://esgf-data.dkrz.de/esg-search',
        'NorESM2-MM'      : 'https://esgf-data.dkrz.de/esg-search',
        'HadGEM3-GC31-LL' : 'https://esgf.ceda.ac.uk/esg-search',
        'UKESM1-0-LL'     : 'https://esgf.ceda.ac.uk/esg-search',
        # Route CNRM to DKRZ — French node (esg1.umr-cnrm.fr) is in SKIP_HOSTS
        'CNRM-CM6-1'      : 'https://esgf-data.dkrz.de/esg-search',
        'CNRM-ESM2-1'     : 'https://esgf-data.dkrz.de/esg-search',
    }
    node = HOME_NODES.get(model, ESGF_NODE)
    params = dict(
        type='File', project='CMIP6', activity_id='CMIP',
        experiment_id='historical', source_id=model, member_id=member,
        variable_id=variable, table_id=table, distrib='true', latest='true',
        limit=ESGF_LIMIT, format='application/solr+json', fields='title,url',
    )

    search_nodes = [node]
    if node != ESGF_NODE:
        search_nodes.append(ESGF_NODE)
    # Also try CEDA as a last-resort for CNRM data
    ceda = 'https://esgf.ceda.ac.uk/esg-search'
    if ceda not in search_nodes:
        search_nodes.append(ceda)

    for search_node in search_nodes:
        try:
            r = requests.get(f'{search_node}/search', params=params, timeout=30)
            r.raise_for_status()
            docs = r.json()['response']['docs']
            file_urls = _parse_docs_to_replica_map(docs)
            if file_urls:
                # Return sorted by best available URL for first replica
                return sorted(file_urls.items(),
                              key=lambda x: _rank_url(x[1][0]))
        except Exception as e:
            print(f"    ESGF search failed ({search_node}): {e}")
    return []


def esgf_search_pyesgf(model, member, variable, table):
    if not HAS_PYESGF:
        return esgf_search_rest(model, member, variable, table)
    try:
        conn = SearchConnection(ESGF_NODE, distrib=True)
        ctx  = conn.new_context(
            facets='project,experiment_id,source_id,member_id,variable_id,table_id',
            project='CMIP6', activity_id='CMIP', experiment_id='historical',
            source_id=model, member_id=member, variable_id=variable,
            table_id=table, latest=True,
        )
        file_urls = defaultdict(list)
        for ds in ctx.search():
            for f in ds.file_context().search():
                url = f.download_url
                if url and _is_usable(url):
                    fname = f.filename
                    if url not in file_urls[fname]:
                        file_urls[fname].append(url)
        for fname in file_urls:
            file_urls[fname].sort(key=_rank_url)
        if file_urls:
            return sorted(file_urls.items(), key=lambda x: _rank_url(x[1][0]))
        # pyesgf returned nothing usable — fall back to REST
        return esgf_search_rest(model, member, variable, table)
    except Exception as e:
        print(f"    pyesgf search failed: {e}; falling back to REST")
        return esgf_search_rest(model, member, variable, table)


def search_esgf(model, member, variable):
    for table in VAR_TABLE_MAP.get(variable, ['Amon']):
        hits = esgf_search_pyesgf(model, member, variable, table)
        if hits:
            return table, hits
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# DATE TRIMMING
# ─────────────────────────────────────────────────────────────────────────────

def _trim_to_start_year(path, variable, start_year=TRIM_START_YEAR):
    """
    If the NetCDF at `path` has data before `start_year`, slice it to
    [start_year, end] and overwrite the file in-place via atomic rename.
    """
    try:
        ds = xr.open_dataset(path, use_cftime=True)
        if 'time' not in ds.dims:
            ds.close()
            return
        first_year = int(ds.time.dt.year.values[0])
        if first_year >= start_year:
            ds.close()
            return

        print(f"      trimming {os.path.basename(path)}: "
              f"{first_year} → {start_year}+", flush=True)
        ds_trim = ds.sel(time=ds.time.dt.year >= start_year).load()
        ds.close()

        if ds_trim.sizes.get('time', 0) == 0:
            ds_trim.close()
            os.remove(path)
            print(f"      removed (entirely pre-{start_year}): "
                  f"{os.path.basename(path)}", flush=True)
            return

        tmp = path + '.trim_tmp'
        ds_trim.to_netcdf(tmp)
        ds_trim.close()
        os.replace(tmp, path)

    except Exception as e:
        print(f"      trim failed for {os.path.basename(path)}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD  — accepts a list of fallback URLs per file
# ─────────────────────────────────────────────────────────────────────────────

def download_file(urls, dest_path):
    """
    Try each URL in `urls` (ranked best-first) with up to 3 attempts each.
    Falls through to the next URL on persistent failure.
    `urls` may be a plain string or a list.
    Returns True on success.
    """
    if isinstance(urls, str):
        urls = [urls]

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fname = os.path.basename(dest_path)

    for url in urls:
        host = url.split('/')[2] if '/' in url else url
        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                size = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                mb = os.path.getsize(dest_path) / 1e6
                sz = f'{size/1e6:.1f} MB' if size else f'{mb:.1f} MB'
                print(f'      ✓ {fname}  ({sz})  [{host}]', flush=True)
                return True
            except Exception as e:
                print(f'      attempt {attempt+1}/3 FAILED [{fname}] '
                      f'on {host}: {type(e).__name__}', flush=True)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                if attempt < 2:
                    time.sleep(5)
        print(f'      giving up on {host}, trying next replica...', flush=True)

    print(f'      ALL replicas exhausted for {fname}', flush=True)
    return False


def download_and_concat(fname_url_list, dest_dir, variable):
    """
    Download chunked files into a temp subdir, concatenate with xarray
    (sliced to TRIM_START_YEAR+), save as a single NetCDF, clean up chunks.
    `fname_url_list` is a list of (fname, urls) where urls may be a list or str.
    Returns True on success.
    """
    tmp_dir = os.path.join(dest_dir, '_tmp_chunks')
    os.makedirs(tmp_dir, exist_ok=True)

    chunk_paths = []
    for fname, urls in fname_url_list:
        dest = os.path.join(tmp_dir, fname)
        if os.path.exists(dest):
            print(f"      chunk exists: {fname}", flush=True)
            chunk_paths.append(dest)
            continue
        print(f"      downloading chunk: {fname}", flush=True)
        if download_file(urls, dest):
            chunk_paths.append(dest)
        else:
            print(f"      chunk failed — aborting concat for {variable}", flush=True)
            return False
        time.sleep(0.3)

    if not chunk_paths:
        return False

    print(f"      concatenating {len(chunk_paths)} chunks "
          f"(trimming to {TRIM_START_YEAR}+) ...", flush=True)
    try:
        ds = xr.open_mfdataset(
            sorted(chunk_paths),
            combine='by_coords',
            data_vars='minimal',
            coords='minimal',
            compat='override',
            chunks={'time': 120},
            use_cftime=True,
        )
        if 'time' in ds.dims:
            ds = ds.sel(time=ds.time.dt.year >= TRIM_START_YEAR)

        first    = os.path.basename(sorted(chunk_paths)[0])
        last     = os.path.basename(sorted(chunk_paths)[-1])
        start    = first.rsplit('_', 1)[-1].replace('.nc', '').split('-')[0]
        end      = last.rsplit('_',  1)[-1].replace('.nc', '').split('-')[1]
        stem     = '_'.join(first.rsplit('_', 1)[0].split('_'))
        start    = str(TRIM_START_YEAR) + '01' if int(start[:4]) < TRIM_START_YEAR else start
        out_fname = f"{stem}_{start}-{end}.nc"
        out_path  = os.path.join(dest_dir, out_fname)

        ds.load().to_netcdf(out_path)
        ds.close()
        print(f"      saved → {out_fname}", flush=True)

        for p in chunk_paths:
            os.remove(p)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        return True

    except Exception as e:
        print(f"      concat FAILED: {e}", flush=True)
        return False


def write_wget(model, member, variable, table, hits, out_dir):
    """hits is list of (fname, [url, ...])"""
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(out_dir, f'wget_{model}_{member}_{variable}.sh')
    with open(script, 'w') as fh:
        fh.write('#!/bin/bash\n')
        for fname, urls in hits:
            url = urls[0] if isinstance(urls, list) else urls
            fh.write(f'wget -c --no-check-certificate -P . "{url}"\n')
    os.chmod(script, 0o755)
    return script


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD TASK  (runs inside a thread)
# ─────────────────────────────────────────────────────────────────────────────

def _download_task(task):
    model      = task['model']
    member     = task['member']
    v          = task['variable']
    table      = task['table']
    hits       = task['hits']       # list of (fname, [url, url, ...])
    dest_dir   = task['dest_dir']
    concat_thr = task['concat_threshold']

    print(f"  [{model}/{member}] downloading {v} "
          f"({len(hits)} file(s), table={table})", flush=True)

    if len(hits) > concat_thr:
        print(f"    {len(hits)} chunks — will concatenate after download",
              flush=True)
        ok = download_and_concat(hits, dest_dir, v)
        return dict(model=model, member=member, variable=v,
                    status='concatenated' if ok else 'concat_failed',
                    files=len(hits))
    else:
        n_ok = 0
        for fname, urls in hits:
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest):
                print(f"      already exists: {fname}", flush=True)
                n_ok += 1
                continue
            if download_file(urls, dest):
                _trim_to_start_year(dest, v)
                n_ok += 1
            time.sleep(0.3)
        return dict(model=model, member=member, variable=v,
                    status=f'downloaded {n_ok}/{len(hits)}',
                    files=len(hits))


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def colour_cell(status, src):
    if status == 'ok':
        s = src[0] if src else '?'
        return f"{GREEN} ✓{s[0]} {RESET}"
    elif status == 'missing':
        return f"{RED} ✗  {RESET}"
    else:
        return f"{YELLOW} –  {RESET}"


def print_audit_table(model, audit):
    print(f"\n{'─'*80}")
    print(f"  {model}  ({MODEL_INFO[model]['institution']})"
          f"  primary={MODEL_INFO[model]['primary']}")
    print(f"{'─'*80}")
    hdr = f"  {'Member':<18}"
    for v in ALL_VARS:
        hdr += f" {v[:6]:<6}"
    print(hdr)
    print(f"  {'':─<18}" + "─" * (7 * len(ALL_VARS)))
    for mem, row in sorted(audit.items()):
        line = f"  {mem:<18}"
        for v in ALL_VARS:
            status, src = row[v]
            line += f" {colour_cell(status, src):<6}"
        n_req = sum(1 for v in REQUIRED_VARS if row[v][0] == 'ok')
        n_opt = sum(1 for v in OPTIONAL_VARS if row[v][0] == 'ok')
        line += f"  req:{n_req}/{len(REQUIRED_VARS)}  opt:{n_opt}/{len(OPTIONAL_VARS)}"
        print(line)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def stage_path(inst, model, member, table, variable):
    return os.path.join(STAGE_ROOT, inst, model, 'historical',
                        member, table, variable)


def main():
    parser = argparse.ArgumentParser(
        description='Audit and download CMIP6 data for SIAF pipeline')
    parser.add_argument('--audit-only',    action='store_true')
    parser.add_argument('--dry-run',       action='store_true')
    parser.add_argument('--models',        nargs='+', default=list(MODEL_INFO.keys()))
    parser.add_argument('--members',       nargs='+', default=None,
                        help='Limit to specific members e.g. r12i1p1f2 r13i1p1f2')
    parser.add_argument('--required-only', action='store_true')
    parser.add_argument('--wget-dir',      default=os.path.join(STAGE_ROOT, '_wget_scripts'))
    parser.add_argument('--concat-threshold', type=int, default=CONCAT_THRESHOLD)
    parser.add_argument('--workers',       type=int,   default=N_DOWNLOAD_WORKERS,
                        help=f'Parallel download threads (default: {N_DOWNLOAD_WORKERS})')
    parser.add_argument('--from-priority-csv', default=None,
                        help='Path to esgf_stage_priority.csv — restricts downloads '
                             'to full_T07/approx_T07 targets only (skips rsds_proxy)')
    args = parser.parse_args()

    concat_threshold = args.concat_threshold
    download_vars    = REQUIRED_VARS if args.required_only else ALL_VARS
    n_workers        = args.workers

    _probe_esgf_nodes()

    print("=" * 80)
    print("CMIP6 SIAF DATA AUDIT")
    print(f"  CMIP6_ROOT        : {CMIP6_ROOT}")
    print(f"  STAGE_ROOT        : {STAGE_ROOT}")
    print(f"  Mode              : "
          f"{'audit-only' if args.audit_only else 'dry-run' if args.dry_run else 'download'}")
    if args.from_priority_csv:
        print(f"  Priority CSV      : {args.from_priority_csv}  (full/approx T07 only)")
    print(f"  Variables         : {', '.join(download_vars)}")
    print(f"  Trim start year   : {TRIM_START_YEAR}")
    print(f"  Download workers  : {n_workers}")
    print(f"  Concat threshold  : {concat_threshold} files")
    print(f"  Skipped hosts     : {', '.join(sorted(SKIP_HOSTS))}")
    print("=" * 80)
    print(f"\n  Legend:  {GREEN}✓C{RESET}=CMIP6_ROOT  {GREEN}✓S{RESET}=STAGE_ROOT"
          f"  {RED}✗{RESET}=required/missing  {YELLOW}–{RESET}=optional/missing\n")

    # ── Phase 1: Audit ────────────────────────────────────────────────────────
    all_audits = {}
    for model in args.models:
        if model not in MODEL_INFO:
            print(f"  Unknown model: {model} — skipping")
            continue
        audit = audit_model(model)
        print_audit_table(model, audit)
        all_audits[model] = audit

    if args.audit_only:
        print("\nDONE (audit only).")
        return

    # ── Phase 2: Build target set ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ESGF SEARCH + PARALLEL DOWNLOAD")
    print("=" * 80)

    if args.from_priority_csv:
        if not os.path.exists(args.from_priority_csv):
            print(f"ERROR: priority CSV not found: {args.from_priority_csv}")
            return
        prio = pd.read_csv(args.from_priority_csv)
        prio = prio[prio['target_tier'].isin(['full_T07', 'approx_T07'])].copy()
        print(f"  Loaded {len(prio)} HIGH-priority targets from priority CSV")
        targets = set(zip(prio['model'], prio['member'], prio['variable']))
        pairs   = sorted({(m, r) for m, r, _ in targets})
    else:
        targets = None
        pairs   = [(model, mem)
                   for model, audit in all_audits.items()
                   for mem in sorted(audit)]

    # ── Phase 3: Search + dispatch downloads ──────────────────────────────────
    not_found_rows   = []
    download_futures = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for model, mem in pairs:
            if model not in MODEL_INFO:
                continue
            if model not in all_audits:
                all_audits[model] = audit_model(model)
            if args.members and mem not in args.members:
                continue
            if _skip_member(model, mem):
                continue

            inst = MODEL_INFO[model]['institution']

            if targets is not None:
                missing_vars = sorted({v for m, r, v in targets
                                       if m == model and r == mem})
            else:
                missing_vars = [v for v in download_vars
                                if all_audits[model][mem][v][0] != 'ok']

            if not missing_vars:
                continue

            print(f"\n  Searching {model}/{mem}: {missing_vars}", flush=True)
            for v in missing_vars:
                table, hits = search_esgf(model, mem, v)
                if not hits:
                    print(f"    {RED}NOT FOUND{RESET}: {v}", flush=True)
                    not_found_rows.append(dict(model=model, member=mem,
                                               variable=v,
                                               status='not_on_esgf', files=0))
                    continue

                # Report how many replicas we found for the first file
                n_replicas = len(hits[0][1]) if hits else 0
                print(f"    {GREEN}Found{RESET} {len(hits)} file(s) "
                      f"for {v} (table={table}, "
                      f"up to {n_replicas} replica(s)) — submitting download",
                      flush=True)
                dest_dir = stage_path(inst, model, mem, table, v)

                if args.dry_run:
                    script = write_wget(model, mem, v, table, hits, args.wget_dir)
                    print(f"    wget script → {script}")
                else:
                    task = dict(model=model, member=mem, variable=v,
                                table=table, hits=hits, dest_dir=dest_dir,
                                concat_threshold=concat_threshold)
                    fut = pool.submit(_download_task, task)
                    download_futures[fut] = task

        print(f"\n  All searches submitted. "
              f"Waiting for {len(download_futures)} download(s)...", flush=True)
        grand_summary = list(not_found_rows)
        for fut in as_completed(download_futures):
            task = download_futures[fut]
            try:
                row = fut.result()
                grand_summary.append(row)
                print(f"  ✓ done: {task['model']}/{task['member']}"
                      f"/{task['variable']}", flush=True)
            except Exception as e:
                print(f"  ✗ failed: {task['model']}/{task['member']}"
                      f"/{task['variable']}: {e}", flush=True)
                grand_summary.append(dict(
                    model=task['model'], member=task['member'],
                    variable=task['variable'],
                    status=f'error: {e}', files=0))

    # ── Summary CSV ───────────────────────────────────────────────────────────
    if grand_summary:
        df      = pd.DataFrame(grand_summary)
        out_csv = os.path.join(STAGE_ROOT, 'audit_download_summary.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nDownload summary CSV → {out_csv}")
        print("\nACTION SUMMARY:")
        print(df.to_string(index=False))
    else:
        print("\n✓ All required data is present — nothing to download.")

    print("\nDONE.")


if __name__ == '__main__':
    main()