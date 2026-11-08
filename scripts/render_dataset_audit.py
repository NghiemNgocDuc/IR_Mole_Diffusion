"""Render the measured evaluation subset comparison from saved audit JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dataset_review"
c = json.loads((OUT / "chemical_split_audit.json").read_text())["splits"]
f = json.loads((OUT / "functional_coverage_audit.json").read_text())["groups"]
labels = ["Aromatic molecules", "Acyclic molecules", "Contains N", "Contains F", "Contains O-H"]


def percentages(ckey, fkey):
    group, motif = c[ckey], f[fkey]
    n = group["rows"]
    return [100*group["rows_with_aromatic_atoms"]/n, 100*group["acyclic_rows"]/n,
            100*group["rows_with_each_element"]["N"]/n,
            100*group["rows_with_each_element"]["F"]/n,
            100*motif["motif_counts"]["hydroxyl_O_H"]/n]


fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
y = np.arange(len(labels))
for offset, key, fkey, label, color in [(-.18, "val_first_500", "first500_val", "Evaluated subset (500)", "#D55E00"),
                                     (.18, "val", "val", "Full validation (12,478)", "#0072B2")]:
    vals = percentages(key, fkey)
    bars = ax.barh(y+offset, vals, height=.32, color=color, label=label)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=5, fontsize=11)
ax.set(yticks=y, yticklabels=labels, xlim=(0, 75), xlabel="Percentage of molecules")
ax.invert_yaxis()
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("The existing 500-molecule evaluation is not representative", fontsize=15, pad=18)
fig.text(.01, -.025, "Measured from local input labels; this does not establish accuracy for any chemical class.", fontsize=10)
fig.savefig(OUT / "validation_subset_comparison.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print(OUT / "validation_subset_comparison.png")
