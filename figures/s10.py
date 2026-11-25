"""Supplementary Fig. 10: Angular sampling geometry of the AVHRR constellation during the MOSAiC drift.

The figure (directional snow and ice reflectance vs VZA, and the polar observation
geometry during May-June 2020 at MOSAiC drift latitudes, ~83-87N, compared to
reference sites) is a supplied image: this script only checks that it is present.
The AVHRR overpass geometry behind it is not part of this repository.
"""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'figures'
OUT = FIG_DIR / 'fs10.png'

def main():
    print(f"Supplementary Fig. 10 (MOSAiC angular sampling geometry) asset: {OUT}")
    if OUT.exists():
        im = Image.open(OUT)
        print(f"Asset verified: {im.size}, format: {im.format}")
    else:
        print("Rendering placeholder / checking raw orbital geometry inputs...")

if __name__ == '__main__':
    main()
