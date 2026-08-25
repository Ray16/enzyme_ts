"""Reaction free-energy level diagram from a TSResult (caption-only: no title)."""
from __future__ import annotations


def energy_diagram(result, out_png, labels=("reactant", "TS", "product")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys = [0.0, result.dG_act, result.dG_rxn]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, (x, y) in enumerate(zip((0, 1, 2), ys)):
        ax.hlines(y, x - 0.32, x + 0.32, lw=2.4, color="#1f3b57")
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9)
    for i in range(2):
        ax.plot([i + 0.32, i + 1 - 0.32], [ys[i], ys[i + 1]], ls=":", color="#8a8a8a", lw=1.2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png
