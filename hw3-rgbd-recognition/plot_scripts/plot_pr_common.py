#!/usr/bin/env python3
import csv
import os

import matplotlib.pyplot as plt


HIGHLIGHT_TAUS = {0.10, 0.50, 0.90}


def plot_pr(input_csv, output_png, title):
    curves = {}
    with open(input_csv, newline="") as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["class"], []).append(
                (float(row["recall"]), float(row["precision"]), float(row["threshold"]))
            )

    plt.figure(figsize=(6, 5))
    for cls, pts in sorted(curves.items()):
        pts = sorted(pts, key=lambda p: p[2])
        recalls = [p[0] for p in pts]
        precisions = [p[1] for p in pts]
        plt.plot(recalls, precisions, marker="o", label=cls)

        for recall, precision, tau in pts:
            if round(tau, 2) in HIGHLIGHT_TAUS:
                plt.scatter([recall], [precision], marker="s",
                            s=50, facecolors="none", edgecolors="black")

    plt.scatter([1], [1], c="black", marker="x", label="ideal")
    plt.text(0.98, 0.04, r"squares: $\tau=0.1, 0.5, 0.9$",
             transform=plt.gca().transAxes,
             ha="right",
             fontsize=8,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=160)
    print("saved", output_png)
