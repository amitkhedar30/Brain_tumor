# ============================================================
#  compare.py  –  Generate comparison tables + plots
# ============================================================
"""
Usage:
    python compare.py

Reads results/all_models_test_results.json and produces:
  • results/comparison_table.txt        (print-ready table)
  • results/dice_comparison.png         (grouped bar chart)
  • results/training_curves_<model>.png (loss + WT-Dice per epoch)
  • results/radar_chart.png             (spider chart across regions)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

from config import RESULTS_DIR, MODELS


# ── helpers ──────────────────────────────────────────────────

COLORS = {"unet": "#4C72B0", "resunet": "#DD8452", "transunet": "#55A868"}
REGIONS = ["WT", "TC", "ET"]
METRICS = ["dice", "iou", "hd95"]


def load_results() -> dict:
    path = os.path.join(RESULTS_DIR, "all_models_test_results.json")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run evaluate.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def load_history(model_name: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── 1. comparison table ──────────────────────────────────────

def print_comparison_table(results: dict):
    header = f"{'Model':<12} {'Region':<6} {'Dice':>7} {'IoU':>7} {'HD95':>9}"
    sep    = "-" * len(header)
    lines  = [sep, header, sep]

    for model in MODELS:
        if model not in results:
            continue
        r = results[model]
        for region in REGIONS:
            d = r[region]["dice"]
            i = r[region]["iou"]
            h = r[region]["hd95"]
            lines.append(f"{model:<12} {region:<6} {d:>7.4f} {i:>7.4f} {h:>9.2f}")
        lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)

    out = os.path.join(RESULTS_DIR, "comparison_table.txt")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\n  Table saved → {out}")


# ── 2. grouped bar chart (Dice) ──────────────────────────────

def plot_dice_comparison(results: dict):
    models  = [m for m in MODELS if m in results]
    x       = np.arange(len(REGIONS))
    width   = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        dice_vals = [results[model][r]["dice"] for r in REGIONS]
        bars = ax.bar(x + i * width, dice_vals, width,
                      label=model.upper(), color=COLORS[model], alpha=0.88)
        for bar, val in zip(bars, dice_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Tumour Region")
    ax.set_ylabel("Dice Score")
    ax.set_title("Dice Score Comparison Across Models and Regions")
    ax.set_xticks(x + width)
    ax.set_xticklabels(REGIONS)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dice_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Dice bar chart saved → {out}")


# ── 3. training curves ────────────────────────────────────────

def plot_training_curves():
    for model in MODELS:
        history = load_history(model)
        if not history:
            continue

        epochs     = [h["epoch"]        for h in history]
        train_loss = [h["train_loss"]    for h in history]
        val_loss   = [h["val_loss"]      for h in history]
        wt_dice    = [h["val_WT_dice"]   for h in history]
        tc_dice    = [h["val_TC_dice"]   for h in history]
        et_dice    = [h["val_ET_dice"]   for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{model.upper()} – Training Curves", fontsize=13)

        ax1.plot(epochs, train_loss, label="Train Loss", color="#E07B54")
        ax1.plot(epochs, val_loss,   label="Val Loss",   color="#5B8DB8")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.4)

        ax2.plot(epochs, wt_dice, label="WT Dice", color="#4CAF50")
        ax2.plot(epochs, tc_dice, label="TC Dice", color="#FF9800")
        ax2.plot(epochs, et_dice, label="ET Dice", color="#9C27B0")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice Score")
        ax2.set_title("Validation Dice per Region")
        ax2.legend(); ax2.grid(alpha=0.4)

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"training_curves_{model}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Training curves saved → {out}")


# ── 4. radar / spider chart ───────────────────────────────────

def plot_radar_chart(results: dict):
    categories  = ["WT Dice", "TC Dice", "ET Dice",
                   "WT IoU",  "TC IoU",  "ET IoU"]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ["0.2","0.4","0.6","0.8","1.0"], size=7)
    plt.ylim(0, 1)

    for model in MODELS:
        if model not in results:
            continue
        r = results[model]
        values = [
            r["WT"]["dice"], r["TC"]["dice"], r["ET"]["dice"],
            r["WT"]["iou"],  r["TC"]["iou"],  r["ET"]["iou"],
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2,
                linestyle="solid", label=model.upper(), color=COLORS[model])
        ax.fill(angles, values, alpha=0.12, color=COLORS[model])

    ax.set_title("Model Comparison – Radar Chart\n(Dice & IoU per Region)",
                 size=12, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "radar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart saved → {out}")


# ── 5. HD95 comparison bar chart ─────────────────────────────

def plot_hd95_comparison(results: dict):
    models = [m for m in MODELS if m in results]
    x      = np.arange(len(REGIONS))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        hd_vals = [results[model][r]["hd95"] for r in REGIONS]
        bars = ax.bar(x + i * width, hd_vals, width,
                      label=model.upper(), color=COLORS[model], alpha=0.85)
        for bar, val in zip(bars, hd_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Tumour Region")
    ax.set_ylabel("HD95 (pixels)  ← lower is better")
    ax.set_title("Hausdorff Distance 95th Percentile Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(REGIONS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "hd95_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  HD95 bar chart saved → {out}")


# ── main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[compare.py] Generating comparison plots …\n")
    results = load_results()
    if results:
        print_comparison_table(results)
        plot_dice_comparison(results)
        plot_hd95_comparison(results)
        plot_radar_chart(results)
    plot_training_curves()
    print("\n[compare.py] Done. Check results/ directory.")
