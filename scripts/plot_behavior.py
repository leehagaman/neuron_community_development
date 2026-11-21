"""Plot the an->vowel behavior vs. training step for each Pythia size."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"160m": "#2a78d6", "410m": "#eb6834", "1b": "#1baf7a"}  # validated categorical slots 1-3
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(ROOT / "results" / "behavior.csv")
fig, ax = plt.subplots(figsize=(7.5, 5))
for size, g in df.groupby("model", sort=False):
    g = g.sort_values("step")
    x = g["step"].replace(0, 0.5)
    c = COLORS[size]
    ax.plot(x, g["vowel_frac_an"], "o-", color=c, ms=4, label=f"{size}: after ' an'")
    ax.plot(x, g["vowel_frac_a"], "o--", color=c, ms=4, alpha=0.5, label=f"{size}: after ' a'")

ax.set_title("Share of next-word probability on vowel-initial words", loc="left", fontweight="bold")
ax.set_ylabel("vowel-initial share of word-start probability")
ax.set_xlabel("training step (step 0 plotted at 0.5)")
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
fig.tight_layout()
out = ROOT / "results" / "behavior.png"
fig.savefig(out, dpi=130)
print(out)
