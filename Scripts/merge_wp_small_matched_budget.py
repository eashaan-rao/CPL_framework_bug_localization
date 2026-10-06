"""
merge_wp_small_matched_budget.py
─────────────────────────────────
Replaces every WP-small row's top-1/top-5/top-10/MAP/MRR in the primary
results files with the corrected-budget values from
results/wp_small_matched_budget.csv (20% of target bugs, matching
CP-transfer's target-side share, instead of the original buggy 10%).

WP-small doesn't depend on the source project, so the lookup key is
(model_name, target_project) only -- the same corrected values are
broadcast to every row sharing that target, regardless of which source
project the pair-run happened to log.

Only the 5 metric columns are touched. Everything else is left alone:
  - source_project, scenario: identifying columns, unchanged.
  - n_reachable, n_test: candidate-retrieval stats for the (fixed) test
    split -- WP-small's training budget doesn't affect the test split or
    retrieval, verified identical between old and new runs before this
    script was written.
  - denominator_corrected, MAP_uncorrected, MRR_uncorrected: provenance
    flags from an earlier, unrelated correction pass (MRR/MAP denominator
    bug), keyed to the model's code history, not to this run.

Files updated:
  - results/paper_results_complete_corrected.csv  (63-pair canonical, read
    by 8 of the 9 downstream analysis scripts)
  - results/obj1_experimental_results_corrected.csv (superset, read by
    equivalence_and_robustness_tests.py)
  - results/obj1_experimental_results.csv (raw superset; not read by any
    analysis script, but kept consistent since it's what run_expts_ph1.py
    treats as authoritative / checks for already-done work)

Run inside the obj1 virtualenv:
    python Scripts/merge_wp_small_matched_budget.py
"""

import pandas as pd

WSB_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results/wp_small_matched_budget.csv"

TARGET_FILES = [
    "/home/cs21d002_eashaan/PhD/Objective1/results/paper_results_complete_corrected.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results_corrected.csv",
    "/home/cs21d002_eashaan/PhD/Objective1/results/obj1_experimental_results.csv",
]

METRIC_COLS = ['top-1', 'top-5', 'top-10', 'MAP', 'MRR']


def main():
    wsb = pd.read_csv(WSB_PATH)
    lookup = wsb.set_index(['model_name', 'target_project'])[METRIC_COLS].to_dict('index')
    print(f"Loaded {len(lookup)} (model, target) corrected WP-small results from {WSB_PATH}")

    for path in TARGET_FILES:
        df = pd.read_csv(path)
        mask = df['scenario'] == 'WP-small'
        n_rows = mask.sum()

        missing = sorted({
            (m, t) for m, t in df.loc[mask, ['model_name', 'target_project']].itertuples(index=False)
            if (m, t) not in lookup
        })
        if missing:
            raise SystemExit(f"{path}: no corrected-budget data for {missing} -- aborting, nothing written.")

        before = df.loc[mask, METRIC_COLS].copy()
        for idx in df[mask].index:
            key = (df.at[idx, 'model_name'], df.at[idx, 'target_project'])
            vals = lookup[key]
            for col in METRIC_COLS:
                df.at[idx, col] = vals[col]
        changed = (df.loc[mask, METRIC_COLS].values != before.values).any(axis=1).sum()

        df.to_csv(path, index=False)
        print(f"{path}: replaced {n_rows} WP-small rows ({changed} had different values than before)")


if __name__ == '__main__':
    main()
