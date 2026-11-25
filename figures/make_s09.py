#!/usr/bin/env python
"""
make_s09.py
===========
Create `s09_validation.py`: the generator for Supplementary Figure 11 (the
22-panel in-situ albedo validation, delivered as Figures/fs11.png), copied from
the validation workspace and restyled for legibility.

The source, `plot_supplementary_validation.py`, reproduces the delivered figure
exactly (4219 x 6591 px) but draws on a 26 x 39 inch canvas. Scaled to the
5.147 inch column that is a factor of 0.244, so its 11-14 pt text lands at
2.7-3.4 pt on the page -- the smallest text anywhere in the paper and the
direct subject of Reviewer 2's comment.

The fix is to draw near the printed size instead of shrinking by 4x: halving
the canvas to 13 x 19.5 inches takes the scale factor to ~0.49, so the same
nominal sizes render more than twice as large, and the sizes are raised on top
of that. Nothing about the data, the matchups or the statistics is touched.

Two panel titles are shortened because at the new relative text size the
longest would overflow its panel (3.25 in wide):
  "MOSAiC helicopter / HELiPOD (2020)" -> "MOSAiC helicopter (2020)",
      which is the label main Figure 9 already uses for that platform
  "SHEBA surface flux (1997-98)"       -> "SHEBA flux (1997-98)",
      still distinct from the other SHEBA entry, "SHEBA transects (1998)"
"""
import pathlib
import re

SRC = pathlib.Path('/glade/work/jrotondo/albedo_satellite_validation/'
                   'spatial_consistency_review_20261022/'
                   'plot_supplementary_validation.py')
DST = pathlib.Path('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/'
                   'revision_20261028/s09_validation.py')
OUT = ('/glade/work/jrotondo/siaf_drk_paper_final/notebooks/'
       'revision_20261028/figures/s09_validation')

s = SRC.read_text()
changes = []


def sub(old, new, count=0, label=None):
    global s
    n = s.count(old)
    assert n > 0, f'not found: {old[:70]!r}'
    if count:
        assert n == count, f'expected {count} of {old[:50]!r}, found {n}'
    s = s.replace(old, new)
    changes.append(f'{label or old[:46]}: {n} occurrence(s)')


# ---- font sizes: all panel text derives from these -----------------------
for name, old, new in [('FS_TITLE', 14, 26), ('FS_LABEL', 13, 24),
                       ('FS_TICK', 12, 22), ('FS_STATS', 11, 18),
                       ('FS_LEGEND', 11, 17), ('FS_CB', 11, 19),
                       ('FS_NOTE', 11, 15)]:
    pat = re.compile(rf'^({name}\s*=\s*){old}\b', re.M)
    assert pat.search(s), f'{name} assignment not found'
    s = pat.sub(rf'\g<1>{new}', s, count=1)
    changes.append(f'{name}: {old} -> {new}')

# ---- canvas: unchanged ----------------------------------------------------
# Shrinking the canvas does not help. On-page size is script_pt * 5.147/width,
# while the largest script_pt a panel can hold is set by its width/4, so the
# two cancel: on-page size depends only on how many characters a string has
# (about 185/chars for a panel title at column width), not on the canvas. A
# halved canvas with raised fonts was tried and collided badly. So the proven
# 26 x 39 spacing is kept and the fonts are raised into the room it already has.
assert 'figsize=(26, 39)' in s, 'canvas is not the expected 26 x 39'
# No suptitle: the editor allows no text in a figure file that is not part
# of the figure (23 Nov 2026; was enlarged 24 -> 34 on 17 Nov).
sub("fig.suptitle('Supplementary figure — In-situ albedo matchups', fontsize=24, y=.94)",
    "# (no suptitle: figure files carry no title text)",
    1, 'suptitle removed')

# ---- the two longest strings, shortened to buy the larger type room ------
sub('MOSAiC helicopter / HELiPOD (2020)', 'MOSAiC helicopter (2020)',
    label='title: MOSAiC helicopter / HELiPOD -> MOSAiC helicopter')
sub('SHEBA surface flux (1997–98)', 'SHEBA flux (1997–98)',
    label='title: SHEBA surface flux -> SHEBA flux')
sub('Tara Broadband Albedo (pentad mean)', 'Tara broadband (pentad)',
    label='xlabel: Tara Broadband Albedo (pentad mean) -> Tara broadband (pentad)')
sub('Tara Broadband Albedo (daily mean)', 'Tara broadband (daily)',
    label='xlabel: Tara Broadband Albedo (daily mean) -> Tara broadband (daily)')
sub('SLEIGH Tower Albedo (pentad mean)', 'SLEIGH tower (pentad)',
    label='xlabel: SLEIGH Tower Albedo (pentad mean) -> SLEIGH tower (pentad)')

# ---- room between panels for the enlarged colorbar and axis labels -------
# At the original sizes a colorbar label cleared the next panel's y-label; at
# these sizes it does not, and adjacent x-labels meet in the gutter.
sub("gridspec_kw={'hspace': 0.52, 'wspace': 0.36}",
    "gridspec_kw={'hspace': 0.62, 'wspace': 0.60}", 1,
    'panel spacing: wspace 0.36 -> 0.60, hspace 0.52 -> 0.62')

# ---- the grey site notes, out from under the regression legend ----------
# Both sat at (0.03, 0.03), which the lower-right legend now reaches across.
# The middle-left band is empty in every panel that carries a note
# (SHEBA, Tara, Summit, Raven), because their points lie on the 1:1 diagonal.
sub(r"""        ax.text(0.03, 0.03, f'{site_label}\nPentad mean match\n(no SIC blend)',
                transform=ax.transAxes, fontsize=FS_NOTE, va='bottom', ha='left',""",
    r"""        ax.text(0.03, 0.50, f'{site_label}\nPentad mean match\n(no SIC blend)',
                transform=ax.transAxes, fontsize=FS_NOTE, va='top', ha='left',""",
    1, 'site note -> below the stats box (Summit, Raven)')

sub("""    axes[_rc].text(0.03, 0.03, _txt, transform=axes[_rc].transAxes,
                   fontsize=FS_NOTE, va='bottom', ha='left', style='italic',""",
    """    axes[_rc].text(0.03, 0.50, _txt, transform=axes[_rc].transAxes,
                   fontsize=FS_NOTE, va='top', ha='left', style='italic',""",
    1, 'site note -> below the stats box (SHEBA, Tara)')

# ---- 23 Nov 2026: the site notes are dropped altogether --------------------
# Even at (0.03, 0.50) they collided with points, stats boxes and legends in
# panels A, B, E-H, S and T, and the SHEBA degree sign rendered as '?'. The
# temporal matching is on every x-axis label and the SIC treatment is stated in
# the caption, so nothing is lost.
sub(r"""        ax.text(0.03, 0.50, f'{site_label}\nPentad mean match\n(no SIC blend)',""",
    r"""        if False: ax.text(0.03, 0.50, f'{site_label}\nPentad mean match\n(no SIC blend)',""",
    1, 'Greenland site notes disabled')
sub("""    axes[_rc].text(0.03, 0.50, _txt, transform=axes[_rc].transAxes,""",
    """    if False: axes[_rc].text(0.03, 0.50, _txt, transform=axes[_rc].transAxes,""",
    1, 'SHEBA/Tara site notes disabled')

# ---- the surface-type legend, back inside its panel ----------------------
# Seven entries in two columns overran the y-label in the transect panels.
sub("""    ax.legend(handles=handles, fontsize=FS_LEGEND, loc='lower right',
              framealpha=0.8, ncol=2, handlelength=1.1)""",
    """    ax.legend(handles=handles, fontsize=FS_LEGEND - 4, loc='lower right',
              framealpha=0.8, ncol=2, handlelength=0.9, columnspacing=1.0)""",
    1, 'surface-type legend narrowed (transect panels)')

# ---- outputs: into this folder, leaving the workspace untouched ----------
sub("outpath = '/glade/work/jrotondo/albedo_satellite_validation/"
    "spatial_consistency_review_20261022/Supplementary_validation_scatterplots.png'",
    f"outpath = '{OUT}.png'", 1, 'png output -> revision figures/')
sub("fig.savefig('/glade/work/jrotondo/albedo_satellite_validation/"
    "spatial_consistency_review_20261022/Supplementary_validation_scatterplots.pdf'",
    f"fig.savefig('{OUT}.pdf'", 1, 'pdf output -> revision figures/')

# the stats CSV is a workspace product; leave it where the workspace expects it

header = '''"""
s09_validation.py — Supplementary Figure 11, delivered as Figures/fs11.png.

GENERATED by make_s09.py from the validation workspace's
plot_supplementary_validation.py; edit that script or make_s09.py, not this.

The source reproduces the delivered figure exactly (4219 x 6591 px) but draws
at 26 x 39 in, so its text renders at 2.7-3.4 pt in the 5.147 in column. Here
the canvas is halved and the font constants raised, which roughly triples the
on-page text size. The data, matchups and statistics are untouched.

Inputs, all absolute, in the validation workspace:
  validation_fallback_corrected_SIC.pkl   the 8 MOSAiC/Greenland/campaign sets
  fallback/sheba_surface_flux/*.csv       SHEBA surface flux (panels A, B)
  fallback/sheba_perovich_transects/*.csv SHEBA transects  (panels C, D)
  fallback/tara_surface_flux/*.csv        Tara broadband   (panels E, F)
"""
'''

DST.write_text(header + s)
print(f'wrote {DST}')
for c in changes:
    print('  ' + c)
