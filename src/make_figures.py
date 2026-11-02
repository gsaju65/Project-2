"""Make the figures for the report and presentation.

1. Example frame: one image at each quality level with each model's detections
2. Mean recall by quality level for each model, with 95% CIs (the main result)
3. Recall lost from original to Q15, image by image, for each model
4. Power curve used to choose the sample size
5. Q-Q plot of the primary test's differences (normality check)

Everything is read from data/ and results/; the compressed copies for the example frame are rebuilt in memory, so data/interim/ is not needed.

Usage:
    python src/make_figures.py
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
METRICS = ROOT / "data" / "processed" / "per_image_metrics_main.csv"
DETECTIONS = ROOT / "data" / "raw" / "detections_main.csv"
SAMPLE = ROOT / "data" / "raw" / "sample"
RESULTS = ROOT / "results"

QUALITIES = ["orig", "q50", "q15"]
QUALITY_LABELS = {"orig": "Original", "q50": "JPEG Q50", "q15": "JPEG Q15"}
MODELS = ["sar", "stock"]
MODEL_LABELS = {"sar": "Custom SAR model", "stock": "Stock YOLO11s"}
COLORS = {"sar": "#2a78d6", "stock": "#eb6834"}  # validated categorical slots 1 and 2
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
POWER_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordered blue steps, small to large dz

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIGURES / name, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figures/{name}")


def recall_by_quality():
    """Figure 2: mean recall at each quality level, one line per model, with 95% CIs."""
    desc = pd.read_csv(RESULTS / "descriptives.csv")
    desc = desc[desc["metric"] == "recall"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(QUALITIES))
    for model in MODELS:
        d = desc[desc["model"] == model].set_index("quality").loc[QUALITIES]
        ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]],
                    color=COLORS[model], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                    capsize=0, elinewidth=1.5, label=MODEL_LABELS[model])
        ax.annotate(f"{MODEL_LABELS[model]}\n{d['mean'].iloc[0]:.2f} → {d['mean'].iloc[-1]:.2f}",
                    (x[-1], d["mean"].iloc[-1]), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK_2)
    ax.set_xticks(x, [QUALITY_LABELS[q] for q in QUALITIES])
    ax.set_xlim(-0.3, 2.9)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean recall (share of people found)")
    ax.set_xlabel("Compression level")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    ax.set_title("Compression lowers recall for both models", pad=12)
    fig.text(0.01, -0.04, "Means across 120 images with 95% confidence intervals. "
             "Persons at least 32 px tall; both models at 960 px input, confidence 0.25.", fontsize=9, color=MUTED)
    save(fig, "fig2_recall_by_quality.png")


def recall_drop_per_image(metrics):
    """Figure 3: each image's recall drop from original to Q15, by model."""
    wide = metrics.pivot_table(index="image_id", columns=["model", "quality"], values="recall")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, model in enumerate(MODELS):
        drop = wide[(model, "orig")] - wide[(model, "q15")]
        jitter = rng.uniform(-0.18, 0.18, len(drop))
        ax.scatter(i + jitter, drop, s=22, color=COLORS[model], alpha=0.45, edgecolor="none")
        half = stats.t.ppf(0.975, len(drop) - 1) * drop.std() / np.sqrt(len(drop))
        ax.errorbar(i + 0.32, drop.mean(), yerr=half, color=INK, marker="o", ms=8, mec=SURFACE,
                    mew=2, elinewidth=2, capsize=0)
        ax.annotate(f"mean {drop.mean():.2f}", (i + 0.32, drop.mean()), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK_2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("Recall lost, original → Q15")
    ax.grid(axis="x", visible=False)
    ax.set_title("Recall lost per image under heavy compression", pad=12)
    fig.text(0.01, -0.04, "Each dot is one image (n = 120). Black dot: mean with 95% CI. "
             "Below zero means the image did better after compression.", fontsize=9, color=MUTED)
    save(fig, "fig3_recall_drop_per_image.png")


def power_curve():
    """Figure 4: power of a paired t-test at the strictest Holm alpha, by sample size."""
    curve = pd.read_csv(RESULTS / "power_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, (dz, d) in zip(POWER_RAMP, curve.groupby("dz")):
        ax.plot(d["n"], d["power"], color=color, lw=2, label=f"dz = {dz:.1f}")
    ax.axhline(0.8, color=MUTED, lw=1)
    ax.axvline(120, color=MUTED, lw=1)
    ax.annotate("80% power", (5, 0.8), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK_2)
    ax.annotate("n = 120 (chosen)", (120, 0.05), xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_xlim(0, 205)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of images")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", title="Effect size", title_fontsize=9, fontsize=9)
    ax.set_title("Power of a paired t-test (two-sided, alpha = 0.05 / 3)", pad=12)
    save(fig, "fig4_power_curve.png")


def qq_plot(metrics):
    """Figure 5: Q-Q plot of the per-image interaction contrast D."""
    wide = metrics.pivot_table(index="image_id", columns=["model", "quality"], values="recall")
    d = (wide[("sar", "orig")] - wide[("sar", "q15")]) - (wide[("stock", "orig")] - wide[("stock", "q15")])
    (theory, sample), (slope, intercept, _) = stats.probplot(d)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(theory, slope * theory + intercept, color=MUTED, lw=1)
    ax.scatter(theory, sample, s=30, color=COLORS["sar"], edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("Theoretical quantiles (normal)")
    ax.set_ylabel("Observed D")
    ax.set_title("Normality check for the primary test", pad=12)
    fig.text(0.01, -0.04, "D = SAR drop minus stock drop, one point per image. Shapiro-Wilk p < .001, "
             "so the Wilcoxon test is reported too.", fontsize=9, color=MUTED)
    save(fig, "fig5_qq_plot.png")


def example_frame(metrics):
    """Figure 1: one image at each quality level, with each model's detections."""
    wide = metrics.pivot_table(index="image_id", columns=["model", "quality"], values="recall")
    n_gt = metrics.groupby("image_id")["n_gt"].first()
    gap = (wide[("stock", "orig")] - wide[("stock", "q15")]) - (wide[("sar", "orig")] - wide[("sar", "q15")])
    candidates = gap[(n_gt >= 5) & (n_gt <= 15)]
    image_id = candidates.idxmax()  # clearest example of stock losing people that SAR keeps

    labels = pd.read_csv(SAMPLE / "annotations" / f"{image_id}.txt", header=None,
                         names=["x", "y", "w", "h", "score", "category", "truncation", "occlusion"])
    tall = labels[labels["category"].isin([1, 2]) & (labels["score"] == 1) & (labels["h"] >= 32)]
    pad = 120
    with Image.open(SAMPLE / "images" / f"{image_id}.jpg") as im:
        original = im.convert("RGB")
    left = max(int(tall["x"].min()) - pad, 0)
    top = max(int(tall["y"].min()) - pad, 0)
    right = min(int((tall["x"] + tall["w"]).max()) + pad, original.width)
    bottom = min(int((tall["y"] + tall["h"]).max()) + pad, original.height)

    frames = {"orig": original}
    for q in [50, 15]:
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=q, subsampling="4:2:0")
        frames[f"q{q}"] = Image.open(io.BytesIO(buffer.getvalue()))

    dets = pd.read_csv(DETECTIONS, dtype={"image_id": str})
    dets = dets[dets["image_id"] == image_id]
    rows = metrics[metrics["image_id"] == image_id].set_index(["model", "quality"])

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for r, model in enumerate(MODELS):
        for c, quality in enumerate(QUALITIES):
            ax = axes[r, c]
            ax.imshow(frames[quality].crop((left, top, right, bottom)))
            boxes = dets[(dets["model"] == model) & (dets["quality"] == quality)]
            for b in boxes.itertuples():
                ax.add_patch(Rectangle((b.x1 - left, b.y1 - top), b.x2 - b.x1, b.y2 - b.y1,
                                       fill=False, edgecolor=COLORS[model], lw=2))
            row = rows.loc[(model, quality)]
            ax.set_title(f"{MODEL_LABELS[model]} · {QUALITY_LABELS[quality]}\n"
                         f"found {int(row['tp'])} of {int(row['n_gt'])} people", fontsize=10, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("The same frame at three compression levels", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.0, f"VisDrone image {image_id}, cropped. Boxes are each model's person detections.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "fig1_example_frame.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    metrics = pd.read_csv(METRICS, dtype={"image_id": str, "sequence": str})
    example_frame(metrics)
    recall_by_quality()
    recall_drop_per_image(metrics)
    power_curve()
    qq_plot(metrics)


if __name__ == "__main__":
    main()