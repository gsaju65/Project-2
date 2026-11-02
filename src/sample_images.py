"""I picked the pilot and main images at random with a fixed seed.

An image is eligible if it has at least one person (VisDrone category 1 "pedestrian" or 2 "people") that is at least 16 px tall, so recall is defined for every image. VisDrone images come in sequences 
from the same flight, which are not independent, so at most 4 images are kept per sequence. Each kept image gets a random rank: ranks 1-20 are the pilot, and the
main sample is the next n ranks once the power analysis sets n. The chosen originals are copied to data/raw/sample/ so they can be committed.

Usage:
    python src/sample_images.py                # pilot only
    python src/sample_images.py --n-main 120   # pilot plus the first 120 main images
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "VisDrone2019-DET-test-dev"
SAMPLE = ROOT / "data" / "raw" / "sample"
MANIFEST = ROOT / "data" / "raw" / "sample_manifest.csv"

SEED = 42
N_PILOT = 20
MAX_PER_SEQUENCE = 4
PERSON_CATEGORIES = [1, 2]
MIN_HEIGHT_PX = 32
COLUMNS = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]


def count_persons(label_path):
    """Return (all person boxes, person boxes at least MIN_HEIGHT_PX tall) for one image."""
    boxes = pd.read_csv(label_path, header=None, names=COLUMNS)
    persons = boxes[boxes["category"].isin(PERSON_CATEGORIES) & (boxes["score"] == 1)]
    return len(persons), int((persons["h"] >= MIN_HEIGHT_PX).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-main", type=int, default=0, help="main sample size from the power analysis")
    args = parser.parse_args()

    # 1. List eligible images
    rows = []
    for label_path in sorted((SOURCE / "annotations").glob("*.txt")):
        n_persons, n_tall_persons = count_persons(label_path)
        if n_tall_persons > 0:
            rows.append({"image_id": label_path.stem, "n_persons": n_persons, "n_tall_persons": n_tall_persons})
    manifest = pd.DataFrame(rows)
    manifest["sequence"] = manifest["image_id"].str.split("_").str[0]

    # 2. Shuffle with a fixed seed, keep the first 4 images of each sequence, then rank what is left
    rng = np.random.default_rng(SEED)
    manifest["shuffle"] = rng.permutation(len(manifest))
    manifest = manifest.sort_values("shuffle").reset_index(drop=True)
    manifest["rank_in_sequence"] = manifest.groupby("sequence").cumcount() + 1
    kept = manifest["rank_in_sequence"] <= MAX_PER_SEQUENCE
    if args.n_main > kept.sum() - N_PILOT:
        raise SystemExit(f"Only {kept.sum() - N_PILOT} images are available for the main sample.")
    manifest["rank"] = kept.cumsum().where(kept).astype("Int64")

    manifest["role"] = "over_cap"
    manifest.loc[kept, "role"] = "reserve"
    manifest.loc[kept & (manifest["rank"] <= N_PILOT), "role"] = "pilot"
    manifest.loc[kept & manifest["rank"].between(N_PILOT + 1, N_PILOT + args.n_main), "role"] = "main"
    manifest["seed"] = SEED
    manifest = manifest.drop(columns="shuffle")
    manifest.to_csv(MANIFEST, index=False)

    # 3. Copy the chosen originals into data/raw/sample/
        # 3. Make data/raw/sample/ hold exactly the chosen originals
    chosen = set(manifest.loc[manifest["role"].isin(["pilot", "main"]), "image_id"])
    for folder, ext in [("images", ".jpg"), ("annotations", ".txt")]:
        (SAMPLE / folder).mkdir(parents=True, exist_ok=True)
        for old in (SAMPLE / folder).glob(f"*{ext}"):
            if old.stem not in chosen:
                old.unlink()
        for image_id in chosen:
            shutil.copy2(SOURCE / folder / f"{image_id}{ext}", SAMPLE / folder / f"{image_id}{ext}")

    print(f"Eligible images: {len(manifest)} from {manifest['sequence'].nunique()} sequences")
    print(manifest["role"].value_counts().to_string())
    print(f"Sequences in pilot: {manifest.loc[manifest['role'] == 'pilot', 'sequence'].nunique()}")
    print(f"Copied {len(chosen)} images and labels to {SAMPLE}")


if __name__ == "__main__":
    main()