"""Run the stock and custom SAR detectors on every image at every compression level.

Both models run at the same settings (960 px input, confidence 0.25) so the only difference between them is the trained weights. Every predicted person box
is saved; an image with no detections has no rows, and cleaning counts that as zero people found.

Usage:
    python src/run_detection.py --role pilot --sar-weights PATH --stock-weights PATH
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "sample_manifest.csv"
IMAGE_DIRS = {
    "orig": ROOT / "data" / "raw" / "sample" / "images",
    "q50": ROOT / "data" / "interim" / "q50",
    "q15": ROOT / "data" / "interim" / "q15",
}
IMGSZ = 960  # input size in pixels
CONF = 0.25  # minimum confidence to keep a box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pilot", "main"], required=True)
    parser.add_argument("--sar-weights", required=True, help="path to the private SAR model weights")
    parser.add_argument("--stock-weights", default="yolo11s.pt", help="downloaded from Ultralytics if not found")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | {gpu}")

    manifest = pd.read_csv(MANIFEST, dtype={"image_id": str, "sequence": str})
    image_ids = manifest.loc[manifest["role"] == args.role, "image_id"].tolist()
    models = {"stock": YOLO(args.stock_weights), "sar": YOLO(args.sar_weights)}

    rows = []
    for model_name, model in models.items():
        for quality, folder in IMAGE_DIRS.items():
            for image_id in image_ids:
                result = model.predict(folder / f"{image_id}.jpg", imgsz=IMGSZ, conf=CONF,
                                       classes=[0], device=device, verbose=False)[0]
                for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                    x1, y1, x2, y2 = (round(v, 1) for v in box)
                    rows.append({"image_id": image_id, "model": model_name, "quality": quality,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 4)})

    detections = pd.DataFrame(rows)
    out = ROOT / "data" / "raw" / f"detections_{args.role}.csv"
    detections.to_csv(out, index=False)

    print(f"Saved {len(detections)} person boxes for {len(image_ids)} images to {out.name}")
    print("Boxes per model and quality:")
    print(detections.groupby(["model", "quality"]).size().unstack()[["orig", "q50", "q15"]].to_string())


if __name__ == "__main__":
    main()