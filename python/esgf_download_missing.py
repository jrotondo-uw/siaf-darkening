#!/usr/bin/env python
"""
esgf_download_missing.py  (v6 — 2026-05-05)
--------------------------------------------
Downloads missing siconc for CMIP6 members that still need data after
the main SIAF pipeline run.

Changes vs v5:
  - Deduplicate Solr results by filename before queuing: CEDA, DKRZ, and
    BSC each return their own Solr record for the same file, so without
    dedup 35 real files appear as 105 tasks and get downloaded 3x
    simultaneously, causing race conditions on .tmp files and wasted
    bandwidth. Now each filename gets ONE task with all mirror URLs merged.

Changes vs v4:
  - REMOVED start/end temporal filter — breaks ORNL bridge and silently
    times out CEDA/DKRZ. Client-side file_covers_years() used instead.
  - SEARCH_LIMIT=500 so post-1980 files aren't crowded out by pre-1980.

Changes vs v3:
  - Multi-mirror: all HTTP mirror URLs per file tried in order on failure.
  - MPI-ESM1-2-LR r2/r3/r4 removed — already staged.
  - CEDA→DKRZ→LLNL node priority.
  - None-URL filtering before results returned.

Usage:
    python esgf_download_missing.py --dry-run
    python esgf_download_missing.py
    python esgf_download_missing.py --skip-nasa
    python esgf_download_missing.py --node https://esgf-data.dkrz.de/esg-search/search
"""

import os, sys, re, time, hashlib, argparse, requests, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

_HAS_ARIA2 = shutil.which('aria2c') is not None

# ── CONFIG ────────────────────────────────────────────────────────────────────
STAGE_ROOT = '/glade/derecho/scratch/jrotondo/cmip6_esgf_stage'

# CEDA and DKRZ first — LLNL returns null URLs for many EC-Earth3 members
ESGF_NODES = [
    'https://esgf.ceda.ac.uk/esg-search/search',
    'https://esgf-data.dkrz.de/esg-search/search',
    'https://esgf-node.llnl.gov/esg-search/search',
]

NASA_PORTAL_BASE = (
    'https://portal.nccs.nasa.gov/datashare/giss_cmip6/'
    'CMIP/NASA-GISS/GISS-E2-1-G/historical/'
)
NASA_VERSION_PROBES = [
    'v20190815', 'v20180827', 'v20190903', 'v20200115',
    'v20200101', 'v20191206', 'v20200205', 'v20200301',
]
NASA_GRID_PROBES  = ['gn', 'gr', 'gr1']
NASA_SPAN_PROBES  = [
    '185001-201412', '185001-200012', '200101-201412',
    '185001-201512', '195001-201412', '195001-200012',
    '200101-201512',
]
NASA_MEMBER_PROBES = (
    [f'r{i}i1p1f1' for i in range(1, 11)] +
    [f'r{i}i1p1f1' for i in range(101, 111)] +
    [f'r{i}i1p3f1' for i in range(1, 6)]
)

YEAR_MIN          = 1980
YEAR_MAX          = 2015
CONNECT_TIMEOUT   = 10     # seconds — fail fast on dead nodes
READ_TIMEOUT      = 600    # seconds — allow large files to transfer
CHUNK_SIZE        = 1 << 20
SEARCH_LIMIT      = 500    # high enough that post-1980 files aren't crowded out
N_PARALLEL        = 4      # concurrent downloads
ARIA2_CONNECTIONS = 4      # per-file connections if aria2c is available


# ── ESGF TARGETS ─────────────────────────────────────────────────────────────
# MPI-ESM1-2-LR r2/r3/r4 removed — already staged in job 6067582.
ESGF_TARGETS = [

    # MRI-ESM2-0: r11 had no data (may not be on ESGF — will report)
    ('MRI-ESM2-0', 'MRI', 'r11i1p1f1', 'siconc', 'SImon'),

    # EC-Earth3: missing or 0/6 months members
    ('EC-Earth3', 'EC-Earth-Consortium', 'r2i1p1f1',  'siconc', 'SImon'),
    ('EC-Earth3', 'EC-Earth-Consortium', 'r3i1p1f1',  'siconc', 'SImon'),
    ('EC-Earth3', 'EC-Earth-Consortium', 'r4i1p1f1',  'siconc', 'SImon'),
    ('EC-Earth3', 'EC-Earth-Consortium', 'r9i1p1f1',  'siconc', 'SImon'),
    ('EC-Earth3', 'EC-Earth-Consortium', 'r13i1p1f1', 'siconc', 'SImon'),
    ('EC-Earth3', 'EC-Earth-Consortium', 'r15i1p1f1', 'siconc', 'SImon'),

    # GISS-E2-1-H: multiple members missing siconc
    ('GISS-E2-1-H', 'NASA-GISS', 'r4i1p1f1',  'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r9i1p1f1',  'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r11i1p1f1', 'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r13i1p1f1', 'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r15i1p1f1', 'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r21i1p1f1', 'siconc', 'SImon'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r22i1p1f1', 'siconc', 'SImon'),
    # SIday fallback for members where SImon may be absent
    ('GISS-E2-1-H', 'NASA-GISS', 'r11i1p1f1', 'siconc', 'SIday'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r13i1p1f1', 'siconc', 'SIday'),
    ('GISS-E2-1-H', 'NASA-GISS', 'r15i1p1f1', 'siconc', 'SIday'),

    # NorESM2-LM: r1 had 0/6 months (staged file only covers 1980-1989)
    ('NorESM2-LM', 'NCC', 'r1i1p1f1', 'siconc', 'SImon'),
]

NASA_GISS_MODEL = 'GISS-E2-1-G'
NASA_GISS_INST  = 'NASA-GISS'
NASA_GISS_VAR   = 'siconc'
NASA_GISS_TABLE = 'SImon'


# ── SESSION ───────────────────────────────────────────────────────────────────

def _make_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) wget/1.21'
    retry = Retry(
        total=3, backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s

SESSION = _make_session()


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def file_covers_years(fn, ymin=YEAR_MIN, ymax=YEAR_MAX):
    """Client-side filter — exclude pre-1980 and post-2015 files."""
    m = re.search(r'_(\d{4,6})-(\d{4,6})\.nc$', fn)
    if not m:
        return True   # can't parse date — include and let pipeline handle it
    return int(m.group(1)[:4]) <= ymax and int(m.group(2)[:4]) >= ymin


def stage_path(inst, model, member, table, variable, filename):
    return os.path.join(STAGE_ROOT, inst, model,
                        'historical', member, table, variable, filename)


def verify_checksum(path, cs, cs_type):
    if not cs:
        return True
    h = hashlib.md5() if (cs_type or 'MD5').upper() == 'MD5' else hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest() == cs.lower()


def download_file(url_or_urls, dest, checksum=None, cs_type='MD5',
                  dry_run=False):
    """
    Download a file, trying each mirror URL in order until one succeeds.
    Accepts a single URL string or a list of mirror URLs.
    Returns: 'ok' | 'skip' | 'skip_null' | 'fail'
    """
    urls = ([url_or_urls] if isinstance(url_or_urls, str)
            else list(url_or_urls))
    urls = [u for u in urls
            if u and 'None' not in str(u) and u.startswith('http')]
    if not urls:
        return 'skip_null'

    if os.path.exists(dest):
        if verify_checksum(dest, checksum, cs_type or 'MD5'):
            print(f'    [skip]  {os.path.basename(dest)}', flush=True)
            return 'skip'
        print(f'    [redownload] bad checksum: {os.path.basename(dest)}',
              flush=True)
        os.remove(dest)

    if dry_run:
        print(f'    [dry]   {urls[0]}', flush=True)
        return 'ok'

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for i, url in enumerate(urls):
        host  = url.split('/')[2]
        label = f'mirror {i+1}/{len(urls)} ({host})'

        # ── aria2c path (multi-connection, resumable) ─────────────────────
        if _HAS_ARIA2:
            cmd = [
                'aria2c',
                f'--split={ARIA2_CONNECTIONS}',
                f'--max-connection-per-server={ARIA2_CONNECTIONS}',
                '--min-split-size=4M',
                '--connect-timeout=10',
                '--timeout=300',
                '--max-tries=2',
                '--retry-wait=3',
                '--dir',  os.path.dirname(dest),
                '--out',  os.path.basename(dest),
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(dest):
                sz = os.path.getsize(dest)
                print(f'    [OK/a2] {os.path.basename(dest)}  '
                      f'({sz/1e6:.1f} MB)  via {label}', flush=True)
                return 'ok'
            print(f'    [a2 fail] {label}', flush=True)

        # ── requests fallback ─────────────────────────────────────────────
        tmp = dest + '.tmp'
        try:
            with SESSION.get(url, stream=True,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                done  = 0
                with open(tmp, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            done += len(chunk)
            if total and done < total * 0.99:
                raise IOError(f'Short read {done}/{total}')
            if checksum and not verify_checksum(tmp, checksum, cs_type or 'MD5'):
                os.remove(tmp)
                print(f'    [FAIL]  checksum mismatch via {label}', flush=True)
                continue   # try next mirror
            os.replace(tmp, dest)
            print(f'    [OK]    {os.path.basename(dest)}  '
                  f'({os.path.getsize(dest)/1e6:.1f} MB)  via {label}',
                  flush=True)
            return 'ok'
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f'    [FAIL]  {label}: {e}', flush=True)
            # fall through to next mirror

    print(f'    [FAIL]  all {len(urls)} mirrors exhausted: '
          f'{os.path.basename(dest)}', flush=True)
    return 'fail'


# ── ESGF SEARCH ───────────────────────────────────────────────────────────────

def esgf_search(model, inst, member, variable, table, node=None):
    """
    Search ESGF for files matching the given facets.
    No server-side temporal filter — not universally supported.
    Client-side file_covers_years() handles pre-1980 exclusion.
    SEARCH_LIMIT=500 ensures post-1980 files are not crowded out.
    Returns list of dicts: urls, url, filename, checksum, checksum_type.
    """
    nodes  = [node] if node else ESGF_NODES
    params = dict(
        type='File', project='CMIP6', activity_id='CMIP',
        experiment_id='historical',
        source_id=model, institution_id=inst, member_id=member,
        variable_id=variable, table_id=table,
        latest='true', distrib='true',
        limit=SEARCH_LIMIT,
        format='application/solr+json',
    )
    for n in nodes:
        try:
            r = SESSION.get(n, params=params,
                            timeout=(CONNECT_TIMEOUT, 60))
            r.raise_for_status()
            results = []
            for doc in r.json()['response']['docs']:
                urls_raw = doc.get('url', [])
                # Collect ALL valid HTTP mirror URLs for this file
                http = [u.split('|')[0] for u in urls_raw
                        if '|HTTPServer' in u]
                if not http:
                    http = [u.split('|')[0] for u in urls_raw
                            if u.endswith('.nc')]
                # Drop null / broken URLs right here
                http = [u for u in http
                        if u and 'None' not in u and u.startswith('http')]
                if not http:
                    continue
                results.append({
                    'urls'         : http,   # all mirrors in ESGF rank order
                    'url'          : http[0],
                    'checksum'     : doc.get('checksum',      [None])[0],
                    'checksum_type': doc.get('checksum_type', ['MD5'])[0],
                    'filename'     : os.path.basename(http[0].split('?')[0]),
                })
            if results:
                short = n.split('/')[2]
                print(f'    [search] {short}: {len(results)} valid URLs',
                      flush=True)
                return results
        except Exception as e:
            print(f'    [search] {n.split("/")[2]}: {e}', flush=True)
    return []


# ── ESGF DOWNLOAD SECTION ─────────────────────────────────────────────────────

def run_esgf_section(dry_run=False, node=None):
    counts       = dict(ok=0, fail=0, skip=0, notfound=0)
    already_have = set()

    for (model, inst, member, variable, table) in ESGF_TARGETS:
        key = (model, inst, member, variable)
        if key in already_have:
            print(f'\n── {model}/{member}/{table}/{variable}  '
                  f'[already satisfied by earlier table]', flush=True)
            continue

        print(f'\n── ESGF  {model}/{member}/{table}/{variable}', flush=True)

        results  = esgf_search(model, inst, member, variable, table,
                               node=node)
        relevant = [r for r in results if file_covers_years(r['filename'])]

        # ── Deduplicate by filename, merging mirror URLs ───────────────────
        # CEDA, DKRZ, and BSC each publish their own Solr record for the
        # same file. Without dedup, 35 real files appear as 105 tasks and
        # get downloaded simultaneously, causing .tmp race conditions.
        seen = {}
        for rec in relevant:
            fn = rec['filename']
            if fn not in seen:
                seen[fn] = dict(rec, urls=list(rec['urls']))
            else:
                for u in rec['urls']:
                    if u not in seen[fn]['urls']:
                        seen[fn]['urls'].append(u)
        relevant = list(seen.values())
        # ── End dedup ──────────────────────────────────────────────────────

        if not relevant:
            print(f'    [not-found] Nothing on ESGF for this target.',
                  flush=True)
            counts['notfound'] += 1
            continue

        print(f'    Queuing {len(relevant)} unique file(s)  '
              f'(parallel={N_PARALLEL}, aria2={_HAS_ARIA2})', flush=True)

        any_ok = False
        with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
            futures = {
                pool.submit(
                    download_file,
                    rec['urls'],   # all mirrors — fallback handled inside fn
                    stage_path(inst, model, member, table,
                               variable, rec['filename']),
                    rec['checksum'],
                    rec['checksum_type'],
                    dry_run,
                ): rec
                for rec in relevant
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f'    [thread-err] {e}', flush=True)
                    res = 'fail'

                if res == 'ok':
                    counts['ok']   += 1;  any_ok = True
                elif res == 'skip':
                    counts['skip'] += 1;  any_ok = True
                elif res == 'skip_null':
                    pass   # silently ignore — not a failure
                else:
                    counts['fail'] += 1

        if any_ok:
            already_have.add(key)

    return counts


# ── NASA NCCS PORTAL ─────────────────────────────────────────────────────────

def _html_hrefs(url, timeout=20):
    try:
        r = SESSION.get(url, timeout=(CONNECT_TIMEOUT, timeout))
        if r.status_code != 200:
            return None, r.status_code
        if _HAS_BS4:
            soup  = BeautifulSoup(r.text, 'html.parser')
            hrefs = [a['href'] for a in soup.find_all('a', href=True)
                     if not a['href'].startswith(('?', '/', 'http'))]
        else:
            hrefs = re.findall(r'href="(?!http|/)([^"?#]+)"', r.text)
        return hrefs, 200
    except Exception as e:
        return None, str(e)


def _head_ok(url, timeout=12):
    try:
        r = SESSION.head(url, timeout=(CONNECT_TIMEOUT, timeout),
                         allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def _discover_nasa_files(member):
    base      = (f'{NASA_PORTAL_BASE}{member}/'
                 f'{NASA_GISS_TABLE}/{NASA_GISS_VAR}/')
    hrefs, st = _html_hrefs(base)

    if hrefs is not None:
        found     = []
        grid_dirs = [h.rstrip('/') for h in hrefs if re.match(r'^g', h)]
        if not grid_dirs:
            grid_dirs = [h.rstrip('/') for h in hrefs if h.endswith('/')]
        for grid in grid_dirs:
            ver_hrefs, _ = _html_hrefs(f'{base}{grid}/')
            if not ver_hrefs:
                continue
            for ver in [h.rstrip('/') for h in ver_hrefs
                        if re.match(r'^v\d{8}', h)]:
                fh, _ = _html_hrefs(f'{base}{grid}/{ver}/')
                if not fh:
                    continue
                for fn in fh:
                    if fn.endswith('.nc') and file_covers_years(fn):
                        found.append((f'{base}{grid}/{ver}/{fn}', fn))
        if found:
            return found

    # Brute-force probe when directory listing returns 403
    if st == 403:
        print(f'      [nasa] 403 on listing — probing filenames', flush=True)
    found = []
    for grid in NASA_GRID_PROBES:
        for ver in NASA_VERSION_PROBES:
            for span in NASA_SPAN_PROBES:
                fn  = (f'{NASA_GISS_VAR}_{NASA_GISS_TABLE}_{NASA_GISS_MODEL}'
                       f'_historical_{member}_{grid}_{span}.nc')
                url = f'{base}{grid}/{ver}/{fn}'
                if _head_ok(url):
                    found.append((url, fn))
                    print(f'      [nasa] hit: {fn}', flush=True)
    return found


def _accessible_members():
    hrefs, status = _html_hrefs(NASA_PORTAL_BASE)
    if hrefs is not None:
        members = [h.rstrip('/') for h in hrefs
                   if re.match(r'^r\d+i\d+p\d+f\d+$', h.rstrip('/'))]
        if members:
            print(f'    [nasa] dir listing: {len(members)} members',
                  flush=True)
            return members
    print(f'    [nasa] listing status={status}; probing...', flush=True)
    live = []
    for mem in NASA_MEMBER_PROBES:
        url = f'{NASA_PORTAL_BASE}{mem}/'
        try:
            r = SESSION.head(url, timeout=(CONNECT_TIMEOUT, 8),
                             allow_redirects=True)
            if r.status_code in (200, 301, 302, 403):
                live.append(mem)
                print(f'      [nasa] {mem} → {r.status_code}', flush=True)
        except Exception:
            pass
        time.sleep(0.08)
    return live


def run_nasa_section(dry_run=False):
    print('\n' + '═' * 60, flush=True)
    print(f'NASA NCCS portal  —  {NASA_GISS_MODEL} siconc', flush=True)
    print(f'  {NASA_PORTAL_BASE}', flush=True)
    print('═' * 60, flush=True)

    members = _accessible_members()
    if not members:
        print('  [nasa] No accessible members found.\n'
              '  Manual fallback:\n'
              f'    wget -r -np -nH --cut-dirs=6 -A "siconc*.nc" \\\n'
              f'      {NASA_PORTAL_BASE}', flush=True)
        return dict(ok=0, fail=0, skip=0)

    counts = dict(ok=0, fail=0, skip=0)
    for member in members:
        print(f'\n── NASA  {NASA_GISS_MODEL}/{member}/'
              f'{NASA_GISS_TABLE}/{NASA_GISS_VAR}', flush=True)
        files = _discover_nasa_files(member)
        if not files:
            print(f'    [nasa] no files for {member}', flush=True)
            continue
        for url, fn in files:
            dp  = stage_path(NASA_GISS_INST, NASA_GISS_MODEL, member,
                             NASA_GISS_TABLE, NASA_GISS_VAR, fn)
            res = download_file(url, dp, dry_run=dry_run)
            counts[res if res in counts else 'fail'] += 1

    return counts


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Download missing CMIP6 siconc files for SIAF pipeline.')
    ap.add_argument('--dry-run',   action='store_true',
                    help='Search only; do not write files.')
    ap.add_argument('--node',      default=None,
                    help='Force a specific ESGF search node URL.')
    ap.add_argument('--skip-nasa', action='store_true',
                    help='Skip the NASA NCCS portal section.')
    ap.add_argument('--skip-esgf', action='store_true',
                    help='Skip the ESGF section.')
    args = ap.parse_args()

    print('=' * 68, flush=True)
    print('CMIP6 missing siconc downloader  v6  (2026-05-05)', flush=True)
    print(f'Stage root   : {STAGE_ROOT}', flush=True)
    print(f'Dry run      : {args.dry_run}', flush=True)
    print(f'aria2c       : {_HAS_ARIA2}', flush=True)
    print(f'N_PARALLEL   : {N_PARALLEL}', flush=True)
    print(f'Search limit : {SEARCH_LIMIT}  (client-side year filter: '
          f'{YEAR_MIN}–{YEAR_MAX})', flush=True)
    print(f'Node order   : CEDA → DKRZ → LLNL', flush=True)
    print(f'ESGF targets : {len(ESGF_TARGETS)} entries', flush=True)
    print('=' * 68, flush=True)

    ec = dict(ok=0, fail=0, skip=0, notfound=0)
    nc = dict(ok=0, fail=0, skip=0)

    if not args.skip_esgf:
        print(f'\n{"ESGF":─^68}', flush=True)
        ec = run_esgf_section(dry_run=args.dry_run, node=args.node)

    if not args.skip_nasa:
        nc = run_nasa_section(dry_run=args.dry_run)

    print('\n' + '=' * 68, flush=True)
    print('DONE', flush=True)
    print(f'  ESGF : ok={ec["ok"]}  skip={ec["skip"]}  '
          f'fail={ec["fail"]}  not-on-esgf={ec.get("notfound", 0)}',
          flush=True)
    print(f'  NASA : ok={nc["ok"]}  skip={nc["skip"]}  fail={nc["fail"]}',
          flush=True)

    if ec['fail'] or nc['fail']:
        sys.exit(1)


if __name__ == '__main__':
    main()