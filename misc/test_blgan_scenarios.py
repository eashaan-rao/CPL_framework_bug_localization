"""
Dry-run test for BL-GAN paper-faithful implementation.
Tests all 4 scenarios with a real project pair (prefect → xarray).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.blgan.pipeline import run_blgan_experiment

BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"

SOURCE_PROJECT = "prefecthq/prefect"
TARGET_PROJECT = "pydata/xarray"

def load_bug_ids(project_name):
    df = pd.read_parquet(BUG_REPORTS_PATH)
    ids = df[df['repo_name'] == project_name]['bug_id'].unique().tolist()
    print(f"  {project_name}: {len(ids)} bug IDs")
    return ids

def main():
    print("Loading bug IDs...")
    source_ids = load_bug_ids(SOURCE_PROJECT)
    target_ids = load_bug_ids(TARGET_PROJECT)

    train_pool, test_set = train_test_split(target_ids, test_size=0.20, random_state=42)

    scenarios = [
        {
            'name': 'WP-small',
            'source_train': [],
            'target_train': train_test_split(train_pool, train_size=0.125, random_state=42)[0],
            'target_test': test_set,
        },
        {
            'name': 'WP-large',
            'source_train': [],
            'target_train': train_pool,
            'target_test': test_set,
        },
        {
            'name': 'CP-cold-start',
            'source_train': source_ids,
            'target_train': [],
            'target_test': test_set,
        },
        {
            'name': 'CP-transfer',
            'source_train': source_ids,
            'target_train': train_test_split(train_pool, train_size=0.25, random_state=42)[0],
            'target_test': test_set,
        },
    ]

    for scenario in scenarios:
        name = scenario['name']
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"  source_train={len(scenario['source_train'])}, "
              f"target_train={len(scenario['target_train'])}, "
              f"target_test={len(scenario['target_test'])}")
        print(f"{'='*60}")
        try:
            metrics = run_blgan_experiment(
                source_project=SOURCE_PROJECT,
                target_project=TARGET_PROJECT,
                source_train_ids=scenario['source_train'],
                target_train_ids=scenario['target_train'],
                target_test_ids=scenario['target_test'],
                scenario=name,
            )
            print(f"  PASS — metrics: {metrics}")
        except Exception as e:
            print(f"  FAIL — {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
