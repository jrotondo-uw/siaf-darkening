#!/usr/bin/env python
"""
make_core.py
============
Create `siaf_obs_core.py`: the importable head of python/siaf_obs_analysis.py,
truncated immediately before the ERA5-kernel load and the main analysis loop.

Everything above the cut is definitions plus grid construction — constants,
the T07 inversion, the CERES/SIAL/PSN25 loaders, the scatter indices, AREA_W,
DOMAIN_PSN, domain_mean, compute_IS and regrid_sial_to_psn. Importing it
reproduces the analysis environment exactly, with no side effects beyond
building those grids, so any figure rebuilt against it rests on the same
masks, weights and conventions as the published numbers.

The cut is verified byte-for-byte: this script fails rather than write a file
that is not a verbatim prefix of the analysis module.
"""
import hashlib
import pathlib

SRC = pathlib.Path('/glade/work/jrotondo/siaf_drk_paper_final/python/'
                   'siaf_obs_analysis.py')
OUT = pathlib.Path('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/'
                   'revision_20261028/siaf_obs_core.py')
CUT = 588          # last line kept: the close of the CERES -> PSN25 index build
SENTINEL = "    PSN_LAT.ravel(),   PSN_LON.ravel())"
NEXT_LINE = "print('  Loading ERA5 kernels ...')"

lines = SRC.read_text().splitlines(True)

# the cut must land exactly where we think it does
assert lines[CUT - 1].rstrip('\n') == SENTINEL, \
    f'line {CUT} is not the end of the CERES index build: {lines[CUT - 1]!r}'
assert lines[CUT + 1].rstrip('\n') == NEXT_LINE, \
    f'line {CUT + 2} is not the kernel load: {lines[CUT + 1]!r}'

body = ''.join(lines[:CUT])

header = '''"""
siaf_obs_core.py — GENERATED, DO NOT EDIT. Rebuild with make_core.py.

Verbatim first %d lines of python/siaf_obs_analysis.py: every definition and
grid the published SIAF numbers were produced with, cut immediately before the
ERA5-kernel load and the main loop so that it can be imported. Figures rebuilt
against this module therefore use the published masks, area weights and
conventions rather than a reimplementation of them.

sha256 of the copied body: %s
"""
'''

digest = hashlib.sha256(body.encode()).hexdigest()
OUT.write_text((header % (CUT, digest)) + body)

# the written body must still be a verbatim prefix of the source
written = OUT.read_text()
assert written.endswith(body), 'written body is not verbatim'
assert SRC.read_text().startswith(body), 'body is not a prefix of the source'

print(f'wrote {OUT}')
print(f'  {CUT} lines copied verbatim, sha256 {digest[:16]}...')
print(f'  cut before: {NEXT_LINE}')
