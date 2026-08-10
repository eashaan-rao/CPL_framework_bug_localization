"""Equivalence (TOST) and robustness statistics reported in the journal paper.

Produces, on the 63 shared source->target pairs from obj1_experimental_results.csv:
  1. Wilcoxon signed-rank p-values (one-sided, CPT > WPS) per model/metric,
     with matched-pairs rank-biserial correlation r_rb and Cohen's d
     -> Table "Wilcoxon signed-rank test results" (tab:wilcoxon)
  2. TOST equivalence tests (margin +/-0.05 MRR, Wilcoxon variant):
     - TRANP-CNN CP-transfer vs WP-large
     - BLAZE CP-cold-start vs WP-small
  3. WP-small run-to-run variance per target (internal-validity threat)
  4. Wilcoxon re-test of CP-transfer vs per-target MEDIAN WP-small MRR
     (robustness check quoted in the threats section)
  5. WP-large MRR by target size group (tab:stratified WP-large columns)

Run inside the obj1 virtualenv:
    python Scripts/analysis/equivalence_and_robustness_tests.py
"""

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results.csv"
TOST_MARGIN_MRR = 0.05  # smallest mean CPL gain treated as practically meaningful

METRICS = ["MRR", "MAP", "top-1", "top-5", "top-10"]
MODELS = ["BLAZE", "COOBA", "TRANP-CNN"]

SIZE_GROUP = {
    "docker/compose": "Small", "jupyterlab/jupyterlab": "Small",
    "lightning-ai/lightning": "Small", "ipython/ipython": "Small",
    "prefecthq/prefect": "Medium", "mesonbuild/meson": "Medium",
    "pydata/xarray": "Medium", "numpy/numpy": "Medium", "wagtail/wagtail": "Medium",
    "ansible/ansible": "Large", "scikit-learn/scikit-learn": "Large",
    "qiskit/qiskit": "Large", "localstack/localstack": "Large",
}


def load_shared_pairs():
    df = pd.read_csv(RESULTS_CSV)
    df["pair"] = list(zip(df.source_project, df.target_project))
    shared = set(df[df.model_name == "BLAZE"].pair)  # the 63-pair design set
    return df[df.pair.isin(shared)].copy()


def paired(df, model, scenario_a, scenario_b, metric="MRR"):
    a = df[(df.model_name == model) & (df.scenario == scenario_a)].set_index("pair")[metric]
    b = df[(df.model_name == model) & (df.scenario == scenario_b)].set_index("pair")[metric]
    idx = a.index.intersection(b.index)
    return a.loc[idx].values.astype(float), b.loc[idx].values.astype(float)


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation from Wilcoxon signed ranks."""
    d = x - y
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    w_plus, w_minus = r[d > 0].sum(), r[d < 0].sum()
    return (w_plus - w_minus) / (w_plus + w_minus)


def tost_wilcoxon(diff, margin):
    """Wilcoxon-based two one-sided tests. Returns the TOST p-value."""
    _, p_lower = stats.wilcoxon(diff + margin, alternative="greater")
    _, p_upper = stats.wilcoxon(diff - margin, alternative="less")
    return max(p_lower, p_upper)


def main():
    df = load_shared_pairs()

    print("=== 1. CPT vs WPS: Wilcoxon (one-sided), rank-biserial, Cohen's d ===")
    for m in MODELS:
        for met in METRICS:
            cpt, wps = paired(df, m, "CP-transfer", "WP-small", met)
            _, p = stats.wilcoxon(cpt, wps, alternative="greater")
            d = cpt - wps
            print(f"{m:10s} {met:6s} n={len(cpt)} p={p:.4g} "
                  f"r_rb={rank_biserial(cpt, wps):+.2f} d={d.mean()/d.std(ddof=1):.2f}")
        print()

    print("=== 2. TOST equivalence (margin +/-%.2f MRR) ===" % TOST_MARGIN_MRR)
    for label, (model, a, b) in {
        "TRANP-CNN CPT vs WPL": ("TRANP-CNN", "CP-transfer", "WP-large"),
        "BLAZE CPC vs WPS": ("BLAZE", "CP-cold-start", "WP-small"),
    }.items():
        x, y = paired(df, model, a, b)
        diff = x - y
        _, p_two = stats.wilcoxon(x, y)
        print(f"{label}: mean diff={diff.mean():+.4f}, two-sided p={p_two:.3f}, "
              f"TOST p={tost_wilcoxon(diff, TOST_MARGIN_MRR):.4f}")

    print("\n=== 3. WP-small run-to-run variance per target ===")
    wps_all = df[df.scenario == "WP-small"]
    for m in MODELS:
        g = wps_all[wps_all.model_name == m].groupby("target_project")["MRR"].agg(["count", "min", "max"])
        g["spread"] = g["max"] - g["min"]
        multi = g[g["count"] > 1]
        print(f"{m}: targets with repeats={len(multi)}, "
              f"mean spread={multi['spread'].mean():.3f}, max spread={multi['spread'].max():.3f}")

    print("\n=== 4. CPT vs per-target MEDIAN WPS (robustness re-test, MRR) ===")
    for m in MODELS:
        sub = df[df.model_name == m]
        med = sub[sub.scenario == "WP-small"].groupby("target_project")["MRR"].median()
        cpt = sub[sub.scenario == "CP-transfer"].set_index("pair")["MRR"]
        wps_med = pd.Series({p: med[p[1]] for p in cpt.index})
        _, p = stats.wilcoxon(cpt.values, wps_med.values, alternative="greater")
        print(f"{m:10s}: p={p:.4g}, r_rb={rank_biserial(cpt.values, wps_med.values):+.2f}, "
              f"win%={(cpt.values > wps_med.values).mean()*100:.1f}")

    print("\n=== 5. WP-large MRR by target size group ===")
    df["size"] = df.target_project.map(SIZE_GROUP)
    for m in MODELS:
        sub = df[(df.model_name == m) & (df.scenario == "WP-large")]
        print(m, sub.groupby("size")["MRR"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
