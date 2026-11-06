"""Plot exploratory adaptive search policies from adaptive_summary.csv."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    with args.summary.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2), sharey=True)
    markers = {"fixed_5": "s", "patience_2": "o", "patience_3": "D", "patience_4": "^"}
    for ax, experiment, title in zip(axes, ("imp_global", "imp_fixed_w512"), ("Tuned", "Fixed MLP")):
        for policy in markers:
            r = next(r for r in rows if r["experiment"] == experiment and r["policy"] == policy)
            x, y = float(r["mean_cost"]), float(r["mean_delta_test_pp"])
            lo, hi = float(r["ci_low_pp"]), float(r["ci_high_pp"])
            ax.errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt=markers[policy],
                        markersize=5, color="black" if policy == "patience_3" else "0.45",
                        capsize=2, label=policy.replace("_", " "))
        ax.axhline(0, color="0.8", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Accounted optimizer updates / dense run")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Test accuracy difference vs 15 rounds (pp)")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
