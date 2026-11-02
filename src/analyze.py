"""Test the hypotheses on the main sample.

Primary test: does compression lower recall more for one model than the other?
Each image gets an interaction contrast
    D = (SAR recall drop) - (stock recall drop), where drop = original - Q15, and a paired t-test checks whether the mean of D is zero. Two checks back it up:
a Wilcoxon signed-rank test (no normality assumption) and a mixed model with a random intercept per flight sequence (frames from one flight are not independent).

Omnibus: two-way repeated-measures ANOVA (model x quality), Greenhouse-Geisser corrected. Follow-ups: paired t-tests between quality levels within each model,
Holm-corrected within each model. Precision and F1 get the same primary test and ANOVA as an exploratory check.

Usage:
    python src/analyze.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "data" / "processed" / "per_image_metrics_main.csv"
RESULTS = ROOT / "results"

MODELS = ["sar", "stock"]
QUALITIES = ["orig", "q50", "q15"]
PAIRS = [("orig", "q50"), ("orig", "q15"), ("q50", "q15")]


def paired_summary(diff):
    """Mean difference with 95% CI, paired t-test and Cohen's dz (one value per image)."""
    n, mean, sd = len(diff), diff.mean(), diff.std()
    t, p = stats.ttest_1samp(diff, 0)
    half_width = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"n": n, "mean_diff": mean, "ci_low": mean - half_width, "ci_high": mean + half_width,
            "t": t, "df": n - 1, "p": p, "dz": mean / sd}


def to_wide(metrics, metric):
    """One row per image, one column per (model, quality); drops images with a missing cell."""
    wide = metrics.pivot_table(index=["image_id", "sequence"], columns=["model", "quality"], values=metric)
    return wide.dropna()


def interaction_contrast(wide):
    drop = {m: wide[(m, "orig")] - wide[(m, "q15")] for m in MODELS}
    return drop["sar"] - drop["stock"]


def rm_anova(metrics, metric, keep):
    data = metrics[metrics["image_id"].isin(keep)]
    table = pg.rm_anova(data=data, dv=metric, within=["model", "quality"], subject="image_id",
                        detailed=True, effsize="np2")
    return table.assign(metric=metric)


def main():
    metrics = pd.read_csv(METRICS, dtype={"image_id": str, "sequence": str})
    RESULTS.mkdir(exist_ok=True)
    lines = []

    # 1. Descriptives: mean recall, precision and F1 per cell with 95% CI
    desc = metrics.melt(id_vars=["image_id", "model", "quality"], value_vars=["recall", "precision", "f1"],
                        var_name="metric").dropna()
    desc = desc.groupby(["metric", "model", "quality"])["value"].agg(["count", "mean", "std"]).reset_index()
    desc["ci_half"] = stats.t.ppf(0.975, desc["count"] - 1) * desc["std"] / np.sqrt(desc["count"])
    desc["ci_low"], desc["ci_high"] = desc["mean"] - desc["ci_half"], desc["mean"] + desc["ci_half"]
    desc.drop(columns="ci_half").round(4).to_csv(RESULTS / "descriptives.csv", index=False)

    # 2. Primary test on recall
    wide = to_wide(metrics, "recall")
    d = interaction_contrast(wide)
    primary = paired_summary(d)
    wilcoxon = pg.wilcoxon(d)
    mixed_data = d.rename("D").reset_index()
    mixed = smf.mixedlm("D ~ 1", mixed_data, groups=mixed_data["sequence"]).fit(reml=True)
    group_var, resid_var = float(mixed.cov_re.iloc[0, 0]), float(mixed.scale)
    ci = mixed.conf_int().loc["Intercept"]
    primary_rows = [
        {"test": "Paired t-test on D", **primary},
        {"test": "Wilcoxon signed-rank on D", "n": len(d), "median_diff": d.median(),
         "p": wilcoxon["p_val"].iloc[0], "rank_biserial_r": wilcoxon["RBC"].iloc[0]},
        {"test": "Mixed model D ~ 1 + (1 | sequence)", "n": len(d), "mean_diff": mixed.params["Intercept"],
         "ci_low": ci[0], "ci_high": ci[1], "p": mixed.pvalues["Intercept"],
         "icc_sequence": group_var / (group_var + resid_var), "n_sequences": mixed_data["sequence"].nunique()},
    ]
    pd.DataFrame(primary_rows).round(4).to_csv(RESULTS / "primary_test.csv", index=False)

    # 3. Omnibus two-way repeated-measures ANOVA on recall
    anova = rm_anova(metrics, "recall", wide.index.get_level_values("image_id"))
    anova.round(4).to_csv(RESULTS / "rm_anova.csv", index=False)

    # 4. Follow-up comparisons between quality levels within each model (Holm within model)
    posthoc = []
    for model in MODELS:
        rows = [{"model": model, "comparison": f"{a} vs {b}", **paired_summary(wide[(model, a)] - wide[(model, b)])}
                for a, b in PAIRS]
        p_holm = multipletests([r["p"] for r in rows], method="holm")[1]
        posthoc += [{**r, "p_holm": ph} for r, ph in zip(rows, p_holm)]
    posthoc = pd.DataFrame(posthoc)
    posthoc.round(4).to_csv(RESULTS / "posthoc.csv", index=False)

    # 5. Assumption checks
    shapiro = stats.shapiro(d)
    spher_inter = pg.sphericity(metrics, dv="recall", subject="image_id", within=["model", "quality"])
    by_quality = metrics.groupby(["image_id", "quality"], as_index=False)["recall"].mean()
    spher_quality = pg.sphericity(by_quality, dv="recall", subject="image_id", within="quality")
    outliers = int((np.abs(d - d.mean()) > 3 * d.std()).sum())
    assumptions = pd.DataFrame([
        {"check": "Shapiro-Wilk normality of D", "statistic": shapiro.statistic, "p": shapiro.pvalue},
        {"check": "Mauchly sphericity, quality", "statistic": spher_quality.W, "p": spher_quality.pval},
        {"check": "Mauchly sphericity, model x quality", "statistic": spher_inter.W, "p": spher_inter.pval},
        {"check": "Images with D more than 3 SD from the mean (kept)", "statistic": outliers, "p": np.nan},
    ])
    assumptions.round(4).to_csv(RESULTS / "assumptions.csv", index=False)

    # 6. Exploratory: the same primary test and ANOVA on precision and F1
    secondary = []
    for metric in ["precision", "f1"]:
        wide_m = to_wide(metrics, metric)
        result = paired_summary(interaction_contrast(wide_m))
        table = rm_anova(metrics, metric, wide_m.index.get_level_values("image_id")).set_index("Source")
        secondary.append({"metric": metric, **result,
                          "anova_p_quality": table.loc["quality", "p_GG_corr"],
                          "anova_p_interaction": table.loc["model * quality", "p_GG_corr"]})
    pd.DataFrame(secondary).round(4).to_csv(RESULTS / "secondary_tests.csv", index=False)

    # Summary printed and saved
    recall_means = metrics.pivot_table(index="model", columns="quality", values="recall")[QUALITIES]
    lines += ["Mean recall (main sample, n = %d images):" % len(wide), recall_means.round(3).to_string(), ""]
    lines += ["Primary test: D = SAR drop - stock drop (original to Q15)",
              f"  mean D = {primary['mean_diff']:.3f}, 95% CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], "
              f"t({primary['df']}) = {primary['t']:.2f}, p = {primary['p']:.4f}, dz = {primary['dz']:.2f}",
              f"  Wilcoxon p = {primary_rows[1]['p']:.4f}, rank-biserial r = {primary_rows[1]['rank_biserial_r']:.2f}",
              f"  Mixed model (sequence random intercept): estimate = {primary_rows[2]['mean_diff']:.3f}, "
              f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p = {primary_rows[2]['p']:.4f}, ICC = {primary_rows[2]['icc_sequence']:.2f}", ""]
    lines += ["Two-way RM ANOVA on recall:", anova.round(4).to_string(index=False), ""]
    lines += ["Follow-ups (Holm within model):",
              posthoc[["model", "comparison", "mean_diff", "ci_low", "ci_high", "t", "p_holm", "dz"]].round(4).to_string(index=False), ""]
    lines += ["Assumption checks:", assumptions.round(4).to_string(index=False), ""]
    lines += ["Exploratory (precision, F1):",
              pd.DataFrame(secondary)[["metric", "n", "mean_diff", "p", "dz", "anova_p_quality", "anova_p_interaction"]]
              .round(4).to_string(index=False)]
    summary = "\n".join(lines)
    (RESULTS / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()