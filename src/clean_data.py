"""Score every detection against the VisDrone labels and compute per-image metrics.

Ground truth is every person box (category 1 "pedestrian" or 2 "people" with score 1) at least 32 px tall. A detection counts as a hit when its overlap (IoU)
with an unmatched person box is at least 0.5, taking the most confident detections first. Detections that land on an ignored region (category 0) or on
a person shorter than 32 px are left out instead of being counted as mistakes.

Output: one row per image x model x quality with recall, precision and F1.

Usage:
    python src/clean_data.py --role pilot
    python src/clean_data.py --role main
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "raw" / "sample" / "annotations"
MANIFEST = ROOT / "data" / "raw" / "sample_manifest.csv"
PROCESSED = ROOT / "data" / "processed"

COLUMNS = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]
MODELS = ["stock", "sar"]
QUALITIES = ["orig", "q50", "q15"]
IOU_THRESHOLD = 0.5
MIN_HEIGHT_PX = 32


def load_labels(image_id):
    """Return person boxes, their heights, and ignored-region boxes as x1, y1, x2, y2 arrays."""
    labels = pd.read_csv(LABELS / f"{image_id}.txt", header=None, names=COLUMNS)
    labels["x2"] = labels["x"] + labels["w"]
    labels["y2"] = labels["y"] + labels["h"]
    persons = labels[labels["category"].isin([1, 2]) & (labels["score"] == 1)]
    ignored = labels[labels["category"] == 0]
    corners = ["x", "y", "x2", "y2"]
    return persons[corners].to_numpy(float), persons["h"].to_numpy(), ignored[corners].to_numpy(float)


def iou(box, boxes):
    """Intersection over union of one box with each row of boxes."""
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter)


def score_image(dets, persons, counted, ignored):
    """Greedy one-to-one matching. Returns (true positives, false positives)."""
    matched = np.zeros(len(persons), bool)
    tp = fp = 0
    for box in dets:
        overlaps = iou(box, persons) if len(persons) else np.array([])
        overlaps[matched] = 0
        if len(overlaps) and overlaps.max() >= IOU_THRESHOLD:
            best = overlaps.argmax()
            matched[best] = True
            tp += counted[best]  # a hit on a too-small person is neither right nor wrong
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        in_ignored = ((ignored[:, 0] <= cx) & (cx <= ignored[:, 2]) &
                      (ignored[:, 1] <= cy) & (cy <= ignored[:, 3])).any()
        fp += not in_ignored
    return tp, fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pilot", "main"], required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST, dtype={"image_id": str, "sequence": str})
    images = manifest[manifest["role"] == args.role]
    detections = pd.read_csv(ROOT / "data" / "raw" / f"detections_{args.role}.csv", dtype={"image_id": str})
    detections = detections.sort_values("conf", ascending=False)
    grouped = {key: g[["x1", "y1", "x2", "y2"]].to_numpy() for key, g in
               detections.groupby(["image_id", "model", "quality"])}

    rows = []
    for image_id, sequence in zip(images["image_id"], images["sequence"]):
        persons, heights, ignored = load_labels(image_id)
        counted = heights >= MIN_HEIGHT_PX
        for model in MODELS:
            for quality in QUALITIES:
                dets = grouped.get((image_id, model, quality), np.empty((0, 4)))
                tp, fp = score_image(dets, persons, counted, ignored)
                n_gt = int(counted.sum())
                rows.append({"image_id": image_id, "sequence": sequence, "model": model, "quality": quality,
                             "n_gt": n_gt, "n_det": len(dets), "tp": tp, "fp": fp, "fn": n_gt - tp})

    metrics = pd.DataFrame(rows)
    metrics["recall"] = metrics["tp"] / metrics["n_gt"]
    metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])  # blank if nothing detected
    metrics["f1"] = 2 * metrics["tp"] / (2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
    assert len(metrics) == len(images) * len(MODELS) * len(QUALITIES)
    assert metrics["recall"].between(0, 1).all()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"per_image_metrics_{args.role}.csv"
    metrics.to_csv(out, index=False)

    print(f"Saved {len(metrics)} rows ({len(images)} images x 2 models x 3 qualities) to {out.name}")
    print(f"Persons counted: {metrics.loc[(metrics.model == 'sar') & (metrics.quality == 'orig'), 'n_gt'].sum()}"
          f" (at least {MIN_HEIGHT_PX} px tall)")
    for metric in ["recall", "precision"]:
        print(f"\nMean {metric}:")
        print(metrics.pivot_table(index="model", columns="quality", values=metric)[QUALITIES].round(3).to_string())


if __name__ == "__main__":
    main()