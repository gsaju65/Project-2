# Project-2 - Drone Video Compression and Person Detection

Does JPEG compression from a bandwidth-limited drone video link hurt a custom search-and-rescue (SAR) person detector more than the stock YOLO11s model?

## Repository structure

| Path | Contents |
|---|---|
| `src/` | One script per stage: download, sample, degrade, detect, clean, power, analyze, figures |
| `data/raw/` | Sample manifest, sampled original images and labels, raw detections |
| `data/processed/` | Per-image recall, precision and F1 |
| `results/` | Statistical test output |
| `figures/` | Figures used in the report and presentation |
| `report/` | Written report |

## Data

VisDrone2019-DET test-dev, 1,610 aerial images with pedestrian and people labels.
