FIGDIR='/glade/derecho/scratch/jrotondo/paper_revision_20261024/figures'

import numpy as np, matplotlib.pyplot as plt
mon  = ["April", "May", "June", "July", "August", "September"]
drk  = [95, 40, 20, 12, 6, 5]
area = [8, 27, 28, 42, 46, 43]
fig, ax = plt.subplots(figsize=(7.4, 4.1))
x = np.arange(len(mon))
ax.bar(x - 0.2, drk,  0.4, color="0.62",    edgecolor="k", lw=0.6,
       label=r"$\Omega_{\mathrm{drk}}$ change explainable by concentration change")
ax.bar(x + 0.2, area, 0.4, color="#2E86AB", edgecolor="k", lw=0.6,
       label=r"residual darkening within $\Omega_{\mathrm{area}}$")
ax.axhline(18, ls="--", color="0.4", lw=1.3); ax.axhline(31, ls="--", color="#2E86AB", lw=1.3)
ax.text(5.48, 19.5, "18%", color="0.3", fontsize=9, ha="right")
ax.text(5.48, 32.5, "31%", color="#2E86AB", fontsize=9, ha="right")
ax.set_xticks(x); ax.set_xticklabels(mon); ax.set_ylim(0, 100)
ax.set_ylabel("share of albedo change (%)")
ax.set_title("Leakage in both directions, by melt-season month", fontsize=11.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper center")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fs13.png", dpi=300, bbox_inches="tight")
print("wrote", f"{FIGDIR}/fs13.png")