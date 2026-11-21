"""Plot results/top_continuations.csv: probability, median rank, top-1 share vs step.

Writes one figure per model: results/top_continuations_{model}.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def plot(df, size):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for col, art in enumerate(["a", "an"]):
        d = df[df.article == art]
        words = d.drop_duplicates("token").sort_values("corpus_p", ascending=False)
        for color, (_, w) in zip(SLOTS, words.iterrows()):
            g = d[d.token == w.token].sort_values("step")
            kw = dict(color=color, lw=2, marker="o", ms=3.5,
                      label=f"{w.token!r} (corpus {w.corpus_p:.3f})")
            axes[0, col].fill_between(g.x, g.model_p_lo, g.model_p_hi, color=color, alpha=0.2, lw=0)
            axes[0, col].plot(g.x, g.model_p, **kw)
            axes[0, col].axhline(w.corpus_p, color=color, lw=1, ls=":")
            axes[1, col].plot(g.x, g.median_rank, **kw)
            axes[2, col].plot(g.x, g.top1_share, **kw)

        axes[0, col].set_title(f"After ' {art}'", fontweight="bold", fontsize=11)
        axes[0, col].set_ylabel("mean P(word)  (dotted = corpus rate,\nband = 95% bootstrap CI)")
        axes[1, col].set_ylabel("median rank in context\n(1 = top prediction)")
        axes[1, col].set_yscale("log")
        axes[1, col].invert_yaxis()
        axes[2, col].set_ylabel("share of contexts where\nword is top-1")
        axes[2, col].set_xlabel("training step (step 0 plotted at 0.5)")

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Pythia-{size.upper()}: corpus top-3 continuations of ' a' and ' an' over training "
                 f"({df.step.nunique()} checkpoints)", fontweight="bold")
    fig.tight_layout()
    out = RES / f"top_continuations_{size}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    df = pd.read_csv(RES / "top_continuations.csv")
    df["x"] = df["step"].replace(0, 0.5)
    for size in df.model.unique():
        plot(df[df.model == size], size)
