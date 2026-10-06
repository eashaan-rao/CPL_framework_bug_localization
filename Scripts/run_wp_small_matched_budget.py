"""
run_wp_small_matched_budget.py
───────────────────────────────
Re-runs WP-small ONLY, at the corrected 20%-of-target budget (matching
CP-transfer's target-side share), for all 3 models (BLAZE, TRANP-CNN,
COOBA) across all 13 target projects in the 63-pair study.

Why this script exists
───────────────────────
The original WP-small results in obj1_experimental_results*.csv were
generated with a bug: train_size=0.125 of the 80% train pool (= 10% of
each target's total bugs) instead of the intended 0.25 (= 20%), half of
what CP-transfer's target-side allocation uses. The underlying bug is
fixed in run_expts_ph1.py, blaze_faiss_restricted_sweep_wp_small.py, and
src/tranp_cnn/pipeline.py (TARGET_TRAIN_SIZE), but those results still
reflect the old, unmatched split -- fixing the code does not retroactively
fix already-collected numbers.

WP-small does NOT depend on the source project (source_train_ids=[]), so
unlike the main 63-pair sweep this does not need to run per source->target
PAIR -- only once per (model, target project). That is 13 targets x 3
models = 39 runs, not 13 x 3 x 63.

Each model's run_*_experiment() still unconditionally loads a "source"
project's embedding DB even when source_train_ids=[] (it's just never
used to build training samples). Rather than touching three model
pipelines to special-case an empty source, this script passes
source_project=target_project as a harmless no-op source -- it loads the
target's own DB a second time under a different name, wastes a little
memory, and changes nothing about training (source_train_ids stays []).

Output
──────
results/wp_small_matched_budget.csv -- one row per (model, target_project),
columns: model_name, target_project, scenario, top-1, top-5, top-10, MAP,
MRR, n_target_train, n_target_test, wall_time_sec. This is a SEPARATE file
from the pair-keyed obj1_experimental_results*.csv -- merging the corrected
numbers in (and updating the journal draft's tables/figures) is a follow-up
step once these results exist and have been reviewed.

Resumable: skips any (model, target_project) already present in the output
CSV, so it's safe to stop and restart. Supports the same SHARD/NUM_SHARDS
pattern as run_expts_ph1.py for running multiple shards in parallel across
GPUs.

Repeats for noise-floor estimation
───────────────────────────────────
The data split is deterministic (fixed seed 42), so re-running this script
as-is would reselect the identical target_train_ids -- the only source of
variation between repeats is training stochasticity (weight init, data-
loader order, non-deterministic GPU kernels), same as the rest of the
study. Set WP_SMALL_REP (default "1") to run additional repeats into their
own output file (wp_small_matched_budget_rep{N}.csv) without touching or
re-skipping against rep 1's already-merged file. Set WP_SMALL_REVERSE=1 to
iterate models in reverse order, for staggering two concurrent repeat runs
on the same GPU so they're less likely to hit their heaviest target at the
same wall-clock time.

Transient CUDA OOM (e.g. from a concurrent process, or a second repeat run
sharing the GPU) is retried with backoff rather than crashing the whole
sweep -- see MAX_OOM_RETRIES.

Run inside the obj1 virtualenv:
    python Scripts/run_wp_small_matched_budget.py
    WP_SMALL_REP=2 python Scripts/run_wp_small_matched_budget.py
    WP_SMALL_REP=3 WP_SMALL_REVERSE=1 python Scripts/run_wp_small_matched_budget.py
"""

import fcntl
import gc
import os
import sys
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cooba.pipeline import run_cooba_experiment
from src.blaze.pipeline import run_blaze_experiment
from src.tranp_cnn.pipeline import run_tranp_cnn_experiment

# ── The 13 target projects (same set as run_expts_ph1.py's PROJECTS) ──────────
TARGET_PROJECTS = [
    'jupyterlab/jupyterlab',
    'lightning-ai/lightning',
    'prefecthq/prefect',
    'pydata/xarray',
    'numpy/numpy',
    'scikit-learn/scikit-learn',
    'ipython/ipython',
    'mesonbuild/meson',
    'ansible/ansible',
    'docker/compose',
    'localstack/localstack',
    'wagtail/wagtail',
    'qiskit/qiskit',
]

MODELS = {
    'TRANP-CNN': run_tranp_cnn_experiment,
    'COOBA':     run_cooba_experiment,
    'BLAZE':     run_blaze_experiment,
}

# Corrected target-side budget: 25% of the 80% train pool = 20% of total
# target bugs, matching CP-transfer. (The original bug used 0.125 = 10%.)
WP_SMALL_TRAIN_SIZE = 0.25

BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
RESULT_PATH      = "/home/cs21d002_eashaan/PhD/Objective1/results"

REP = os.environ.get('WP_SMALL_REP', '1')
RESULTS_FILE = os.path.join(
    RESULT_PATH,
    'wp_small_matched_budget.csv' if REP == '1' else f'wp_small_matched_budget_rep{REP}.csv'
)

if os.environ.get('WP_SMALL_REVERSE') == '1':
    MODELS = dict(reversed(list(MODELS.items())))

# Transient OOM (e.g. a concurrent repeat run, or another process sharing
# the GPU) is retried with backoff instead of crashing the whole sweep.
MAX_OOM_RETRIES = 6
OOM_RETRY_BACKOFF_SEC = 90

# Parallelism: set SHARD=0/1/2 in separate terminals/GPUs to split the 13
# targets across processes. NUM_SHARDS=1 runs all 13 sequentially.
SHARD = 0
NUM_SHARDS = 1


def load_bug_ids_for_project(project_name):
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    return df_bugs[df_bugs['repo_name'] == project_name]['bug_id'].unique().tolist()


def get_data_splits(target_bug_ids):
    """Same fixed 80/20 train/test split (seed 42) used everywhere else in
    the study, so the held-out test set here is IDENTICAL to the one
    WP-large/CP-cold-start/CP-transfer were already evaluated against."""
    train_pool, test_set = train_test_split(target_bug_ids, test_size=0.20, random_state=42)
    return {'train_pool': train_pool, 'test_set': test_set}


def main():
    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=[
            'model_name', 'target_project', 'scenario',
            'top-1', 'top-5', 'top-10', 'MAP', 'MRR',
            'n_target_train', 'n_target_test', 'wall_time_sec',
        ]).to_csv(RESULTS_FILE, index=False)

    df_results = pd.read_csv(RESULTS_FILE)

    targets = TARGET_PROJECTS[SHARD::NUM_SHARDS]

    # Model-major order: finish one model across all targets before moving
    # to the next (order set by MODELS above), rather than cycling through
    # all 3 models per target.
    for model_name, model_function in MODELS.items():
        print(f"\n{'#'*20} Model: {model_name} {'#'*20}")

        for target_project in targets:
            is_done = (
                (df_results['model_name'] == model_name) &
                (df_results['target_project'] == target_project)
            ).any()
            if is_done:
                print(f" -> {model_name} | {target_project}: already in results. Skipping.")
                continue

            print(f"\n{'='*20} WP-small (matched budget) | {model_name} | target={target_project} {'='*20}")

            target_bug_ids_all = load_bug_ids_for_project(target_project)
            splits = get_data_splits(target_bug_ids_all)
            target_train_ids = train_test_split(
                splits['train_pool'], train_size=WP_SMALL_TRAIN_SIZE, random_state=42
            )[0]
            target_test_ids = splits['test_set']
            print(f" -> target_train: {len(target_train_ids)} bugs "
                  f"({len(target_train_ids)/len(target_bug_ids_all)*100:.1f}% of total)  "
                  f"target_test: {len(target_test_ids)} bugs")

            run_start = time.time()
            metrics = None
            for attempt in range(1, MAX_OOM_RETRIES + 1):
                try:
                    metrics = model_function(
                        source_project=target_project,   # harmless no-op "source" -- source_train_ids=[] below
                        target_project=target_project,
                        source_train_ids=[],
                        target_train_ids=target_train_ids,
                        target_test_ids=target_test_ids,
                        scenario='WP-small',
                    )
                    break
                except RuntimeError as e:
                    if 'out of memory' not in str(e).lower():
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt == MAX_OOM_RETRIES:
                        print(f" -> {model_name} | {target_project}: OOM on final attempt "
                              f"{attempt}/{MAX_OOM_RETRIES}, giving up on this run, moving on.")
                        break
                    print(f" -> {model_name} | {target_project}: OOM (attempt {attempt}/{MAX_OOM_RETRIES}), "
                          f"retrying in {OOM_RETRY_BACKOFF_SEC}s...")
                    time.sleep(OOM_RETRY_BACKOFF_SEC)

            if metrics is None:
                continue  # exhausted retries; leave for a future run to pick up (not marked done)

            wall_time = time.time() - run_start
            new_result = pd.DataFrame([{
                'model_name': model_name,
                'target_project': target_project,
                'scenario': 'WP-small',
                'top-1': metrics['Top-1'],
                'top-5': metrics['Top-5'],
                'top-10': metrics['Top-10'],
                'MAP': metrics['MAP'],
                'MRR': metrics['MRR'],
                'n_target_train': len(target_train_ids),
                'n_target_test': len(target_test_ids),
                'wall_time_sec': wall_time,
            }])
            with open(RESULTS_FILE, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                new_result.to_csv(f, header=False, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)
            df_results = pd.concat([df_results, new_result], ignore_index=True)

            print(f" -> {model_name} | {target_project} done in {wall_time/3600:.2f} hrs. "
                  f"MRR={metrics['MRR']:.4f}")
            gc.collect()


if __name__ == '__main__':
    main()
