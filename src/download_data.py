"""Download the VisDrone2019-DET test-dev.

Saves images and annotations to data/raw/VisDrone2019-DET-test-dev/.

Usage:
    python src/download_data.py
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET = RAW / "VisDrone2019-DET-test-dev"

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip"
SHA256 = "b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d"
N_IMAGES = 1610


def sha256(path):
    """Return the SHA256 checksum of a file, reading it in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(list((DATASET / "images").glob("*.jpg"))) == N_IMAGES:
        print(f"Already downloaded: {DATASET}")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "VisDrone2019-DET-test-dev.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    # Stop if the file is not byte-for-byte the one this project used
    if sha256(zip_path) != SHA256:
        zip_path.unlink()
        raise SystemExit("Checksum mismatch: the download is not the expected file.")

    # Extract only images and labels, and refuse any path that would land outside data/raw
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith((".jpg", ".txt")):
                continue
            if not (RAW / name).resolve().is_relative_to(RAW.resolve()):
                raise SystemExit(f"Unsafe path in zip: {name}")
            z.extract(name, RAW)
    zip_path.unlink()

    n_images = len(list((DATASET / "images").glob("*.jpg")))
    n_labels = len(list((DATASET / "annotations").glob("*.txt")))
    print(f"Done: {n_images} images and {n_labels} label files in {DATASET}")


if __name__ == "__main__":
    main()