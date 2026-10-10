"""
merge_wp_small_matched_budget.py
──────────────────────────────────
Merges the corrected, budget-matched WP-small results (20% of target bugs,
matching CP-transfer's target-side share) into the primary pair-keyed
results files, replacing the old WP-small rows that were generated with
the 10%-budget bug.

Inputs
──────
results/wp_small_matched_budget.csv       -- rep 1 (13 targets x 3 models)
results/wp_small_matched_budget_rep2.csv  -- rep 2
results/wp_small_matched_budget_rep3.csv  -- rep 3

Each is keyed by (model_name, target_project) only -- WP-small doesn't
depend on the source project, so the same corrected value is broadcast to
every pair that shares a target.

What this does
───────────────
1. Computes the median across the 3 reps per (model, target) for each
   metric. The median (not a single run) is substituted into the primary
   files, giving a less noisy point estimate than any individual rep --
   consistent with how the study already re-tests RQ1 against each
   target's median WP-small MRR elsewhere.
2. Also writes results/wp_small_matched_budget_summary.csv with
   median/min/max/mean/std per (model, target) per metric -- this is the
   REAL run-to-run noise-floor data at the corrected budget (3 clean reps
   per target, all 13 targets, all 3 models), replacing the old noise-floor
   numbers that came from uneven, budget-mismatched accidental repeats.
3. Patches every row with scenario == 'WP-small' in:
     results/paper_results_complete_corrected.csv
     results/obj1_experimental_results_corrected.csv
     results/obj1_experimental_results.csv
   setting top-1/top-5/top-10/MAP/MRR to the rep-median values for that
   row's (model_name, target_project). Originals must already be backed up
   (results/pre_budget_fix_backup/) before running this -- it overwrites
   in place.

This intentionally collapses the per-pair variation WP-small rows used to
have (each pair previously trained its own independent WP-small run on the
same target, differing only by training stochasticity) into a single
median value shared by every pair with that target. That's deliberate: we
now have a real 3-rep estimate per target, which is a better point estimate
than any single accidental repeat was, and downstream scripts that need the
noise floor should read it from wp_small_matched_budget_summary.csv / the
raw rep files directly, not re-derive it from the now-identical pair-file
repeats.

Run inside the obj1 virtualenv:
    python Scripts/analysis/merge_wp_small_matched_budget.py
"""

import pandas as pd

REP_FILES = [
    "/home/cs21d002_eashaan/PhD/Objective1/results/wp_small_matched_budget.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/wp_small_matched_budget_rep2.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/wp_small_matched_budget_rep3.csv",
]
SUMMARY_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/wp_small_matched_budget_summary.csv"

METRICS = ["top-1", "top-5", "top-10", "MAP", "MRR"]

TARGET_FILES = [
    "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results_corrected.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results.csv",
]


def build_summary():
    reps = []
    for n, path in enumerate(REP_FILES, start=1):
        df = pd.read_csv(path)
        df["rep"] = n
        reps.append(df)
    all_reps = pd.concat(reps, ignore_index=True)

    summary = all_reps.groupby(["model_name", "target_project"])[METRICS].agg(
        ["median", "min", "max", "mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  wrote {SUMMARY_CSV} ({len(summary)} rows)")
    return summary


def patch_file(path, summary):
    df = pd.read_csv(path)
    median_cols = {m: f"{m}_median" for m in METRICS}
    merged = df.merge(
        summary[["model_name", "target_project"] + list(median_cols.values())],
        on=["model_name", "target_project"], how="left",
    )
    is_wps = merged["scenario"] == "WP-small"
    has_median = merged["MRR_median"].notna()
    mask = is_wps & has_median
    n_patched = mask.sum()
    for metric, med_col in median_cols.items():
        merged.loc[mask, metric] = merged.loc[mask, med_col]
    merged = merged.drop(columns=list(median_cols.values()))
    merged.to_csv(path, index=False)
    print(f"  patched {n_patched} WP-small rows in {path}")


def main():
    print("Building 3-rep summary (median/min/max/mean/std per model,target)...")
    summary = build_summary()

    print("\nPatching primary results files...")
    for path in TARGET_FILES:
        patch_file(path, summary)


if __name__ == "__main__":
    main()
