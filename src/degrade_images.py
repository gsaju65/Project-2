"""Make the compressed copies of each sampled image (the experimental manipulation).

The original image is the control and is used as-is. Each one is re-saved as a JPEG at quality 50 (moderate) and quality 15 (severe), at the same resolution,
using Pillow. File sizes are logged to show how much each level compresses.

Usage:
    python src/degrade_images.py --role pilot
    python src/degrade_images.py --role main
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "raw" / "sample" / "images"
INTERIM = ROOT / "data" / "interim"
MANIFEST = ROOT / "data" / "raw" / "sample_manifest.csv"

QUALITIES = [50, 15]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pilot", "main"], required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST, dtype={"image_id": str, "sequence": str})
    image_ids = manifest.loc[manifest["role"] == args.role, "image_id"]

    rows = []
    for image_id in image_ids:
        source = SAMPLE / f"{image_id}.jpg"
        rows.append({"image_id": image_id, "quality": "orig", "file_bytes": source.stat().st_size})
        with Image.open(source) as im:
            for q in QUALITIES:
                out = INTERIM / f"q{q}" / f"{image_id}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                im.save(out, "JPEG", quality=q, subsampling="4:2:0")
                rows.append({"image_id": image_id, "quality": f"q{q}", "file_bytes": out.stat().st_size})

    log = pd.DataFrame(rows)
    log.to_csv(ROOT / "data" / "raw" / f"compression_{args.role}.csv", index=False)

    median_kb = log.groupby("quality")["file_bytes"].median().reindex(["orig", "q50", "q15"]) / 1024
    print(f"Compressed {len(image_ids)} {args.role} images")
    print("Median file size (KB):")
    print(median_kb.round(1).to_string())


if __name__ == "__main__":
    main()