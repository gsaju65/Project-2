"""Work out how many images the main run needs (power analysis).

The tests are paired t-tests, so the effect size is Cohen's dz: the mean of the per-image differences divided by their standard deviation. The plan is built
on the smallest effect worth detecting, dz = 0.3, with 80% power, two-sided:
  - alpha = 0.05 for the primary test (does compression hurt one model more?)
  - alpha = 0.05 / 3 for the follow-up comparisons (the strictest Holm step)

  The pilot's own effect sizes are reported as a check. The main sample is the
largest n in the table, capped at the images left after the pilot.

Usage:
    python src/power_analysis.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestPower

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "processed" / "per_image_metrics_pilot.csv"
MANIFEST = ROOT / "data" / "raw" / "sample_manifest.csv"
RESULTS = ROOT / "results"

DZ_TARGET = 0.3
POWER = 0.80
ALPHA = 0.05
ALPHA_HOLM = 0.05 / 3


def n_needed(dz, alpha):
    """Images needed for a two-sided paired t-test to reach POWER at this effect size."""
    return TTestPower().solve_power(effect_size=abs(dz), alpha=alpha, power=POWER, alternative="two-sided")


def main():
    # Per-image recall drops from the pilot (original minus Q15)
    pilot = pd.read_csv(PILOT, dtype={"image_id": str})
    recall = pilot.pivot_table(index="image_id", columns=["model", "quality"], values="recall")
    drop_sar = recall[("sar", "orig")] - recall[("sar", "q15")]
    drop_stock = recall[("stock", "orig")] - recall[("stock", "q15")]
    pilot_effects = [
        ("Interaction: SAR drop minus stock drop", drop_sar - drop_stock, ALPHA),
        ("SAR drop, original vs Q15", drop_sar, ALPHA_HOLM),
        ("Stock drop, original vs Q15", drop_stock, ALPHA_HOLM),
    ]

    rows = [
        {"source": "a priori", "effect": "Smallest effect of interest (primary test)", "dz": DZ_TARGET, "alpha": ALPHA},
        {"source": "a priori", "effect": "Smallest effect of interest (Holm follow-up)", "dz": DZ_TARGET, "alpha": ALPHA_HOLM},
    ]
    for name, diff, alpha in pilot_effects:
        rows.append({"source": f"pilot (n={len(diff)})", "effect": name, "dz": diff.mean() / diff.std(), "alpha": alpha})
    table = pd.DataFrame(rows)
    table["power"] = POWER
    table["n_required"] = [math.ceil(n_needed(dz, a)) for dz, a in zip(table["dz"], table["alpha"])]

    manifest = pd.read_csv(MANIFEST, dtype={"image_id": str})
    available = int(manifest["role"].isin(["reserve", "main"]).sum())
    n_main = min(int(table["n_required"].max()), available)

    # Power at each sample size, for the power curve figure
    curve = pd.DataFrame([{"dz": dz, "n": n, "power": TTestPower().power(effect_size=dz, nobs=n, alpha=ALPHA_HOLM)}
                          for dz in [0.2, 0.3, 0.4, 0.5] for n in range(5, 201, 5)])

    RESULTS.mkdir(exist_ok=True)
    table.round(4).to_csv(RESULTS / "power_analysis.csv", index=False)
    curve.round(4).to_csv(RESULTS / "power_curve.csv", index=False)

    print(table.round(3).to_string(index=False))
    print(f"\nImages available after the pilot: {available}")
    print(f"Main sample size: {n_main}")
    print(f"Next: python src/sample_images.py --n-main {n_main}")


if __name__ == "__main__":
    main()