import fcntl
import os
import sys
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import itertools

# Add project root to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Placeholder for the actual training pipelines
from src.tranp_cnn.pipeline import run_tranp_cnn_experiment
from src.cooba.pipeline import run_cooba_experiment
# from src.flim.pipeline import run_flim_experiment
from src.blaze.pipeline import run_blaze_experiment
from src.blgan.pipeline import run_blgan_experiment

# def run_tranp_cnn_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
#     '''
#     Placeholder function for the TRANP-CNN pipeline.
#     '''
#     print(f" -> Predenting to run TRANP-CNN")
#     print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
#     # In reality, this function would trigger the entire training and evaluation process and return a dictionary
#     # of metrics
#     return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

# def run_cooba_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
#     '''
#     Placeholder function for the CooBA pipeline.
#     '''
#     print(f" -> Predenting to run CooBA")
#     print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
#     # In reality, this function would trigger the entire training and evaluation process and return a dictionary
#     # of metrics
#     return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

# def run_blaze_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
#     '''
#     Placeholder function for BLAZE pipeline.
#     '''
#     print(f" -> Predenting to run BLAZE")
#     print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
#     # In reality, this function would trigger the entire training and evaluation process and return a dictionary
#     # of metrics
#     return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

# Configuration Section

# Define the projects to run experiments on
# This can be expanded with more projects later.
# dmwm/wmcore (263)
# rucio/rucio (297)
PROJECTS = [
    # ── Data Science & AI/ML (6 projects) ────────────────────────────────────
    # (see results/PYTHON_PROJECT_SELECTION.md for full selection methodology)
    {'name': 'jupyterlab/jupyterlab',    'language': 'python'},  # DS S1 — 39K LoC,  197 bugs
    {'name': 'lightning-ai/lightning',   'language': 'python'},  # DS S1 — 51K LoC,  366 bugs
    {'name': 'prefecthq/prefect',        'language': 'python'},  # DS S2 — 107K LoC, 404 bugs
    {'name': 'pydata/xarray',            'language': 'python'},  # DS S2 — 142K LoC, 110 bugs
    {'name': 'numpy/numpy',              'language': 'python'},  # DS S4 — 277K LoC, 789 bugs
    {'name': 'scikit-learn/scikit-learn','language': 'python'},  # DS S5 — 376K LoC, 228 bugs

    # ── Developer Tools & DevOps (3 projects) ────────────────────────────────
    {'name': 'ipython/ipython',          'language': 'python'},  # DT S1 — 78K LoC,  832 bugs
    {'name': 'mesonbuild/meson',         'language': 'python'},  # DT S2 — 121K LoC, 937 bugs
    {'name': 'ansible/ansible',          'language': 'python'},  # DT S3 — 348K LoC, 768 bugs

    # ── Systems & Cloud Infrastructure (2 projects) ──────────────────────────
    {'name': 'docker/compose',           'language': 'python'},  # SY S1 — 35K LoC,  572 bugs
    {'name': 'localstack/localstack',    'language': 'python'},  # SY S2 — 572K LoC, 472 bugs

    # ── Applications & Frameworks (2 projects) ───────────────────────────────
    {'name': 'wagtail/wagtail',          'language': 'python'},  # AP S1 — 286K LoC, 404 bugs
    {'name': 'qiskit/qiskit',            'language': 'python'},  # AP S2 — 435K LoC, 1336 bugs
]

# ── Active model config ───────────────────────────────────────────────────
# COOBA: 1 pair remaining (scikit-learn→numpy: WP-large, CP-cold-start, CP-transfer)
# NUM_SHARDS=1 sweeps all pairs; skip logic skips everything already done.
MODELS_TO_RUN = {
    'COOBA': run_cooba_experiment,
}

# ── Legacy configs (all other models complete) ────────────────────────────
# MODELS_TO_RUN = {'TRANP-CNN': run_tranp_cnn_experiment}
# MODELS_TO_RUN = {'BLAZE': run_blaze_experiment}
# MODELS_TO_RUN = {'TRANP-CNN': run_tranp_cnn_experiment, 'BLAZE': run_blaze_experiment}

# Parallelism: set SHARD=0/1/2 in three terminals to run three instances.
# CSV writes are protected by fcntl.flock so all processes can safely append.
SHARD = 0      # single process — sweeps all pairs, skips completed ones
NUM_SHARDS = 1

# ── Legacy sharding configs ────────────────────────────────────────────────
# SHARD = 2; NUM_SHARDS = 3  # TRANP-CNN 3-shard setup
# SHARD = 0; NUM_SHARDS = 2  # BLAZE 2-shard setup

# ── Paper pair set: 63 directional pairs for journal submission ───────────
# Group A (30): all DS×DS bidirectional pairs
# Group B (14): jupyterlab ↔ each of the 7 non-DS projects (both directions)
# Group C (19): strategic cross-domain diversity pairs (all COOBA-complete)
# See results/PAPER_RUN_PLAN.md for full rationale.
_DS = ['jupyterlab/jupyterlab', 'lightning-ai/lightning', 'prefecthq/prefect',
       'pydata/xarray', 'numpy/numpy', 'scikit-learn/scikit-learn']
PAPER_PAIRS = set(itertools.permutations(_DS, 2))           # Group A — 30 pairs
for _p in ['ipython/ipython', 'mesonbuild/meson', 'ansible/ansible',
           'docker/compose', 'localstack/localstack', 'wagtail/wagtail', 'qiskit/qiskit']:
    PAPER_PAIRS.add(('jupyterlab/jupyterlab', _p))          # Group B — 14 pairs
    PAPER_PAIRS.add((_p, 'jupyterlab/jupyterlab'))
PAPER_PAIRS.update([                                         # Group C — 19 pairs
    ('lightning-ai/lightning',    'ansible/ansible'),
    ('ansible/ansible',           'lightning-ai/lightning'),
    ('lightning-ai/lightning',    'localstack/localstack'),
    ('localstack/localstack',     'lightning-ai/lightning'),
    ('numpy/numpy',               'qiskit/qiskit'),
    ('qiskit/qiskit',             'numpy/numpy'),
    ('numpy/numpy',               'localstack/localstack'),
    ('localstack/localstack',     'numpy/numpy'),
    ('prefecthq/prefect',         'ansible/ansible'),
    ('ansible/ansible',          'prefecthq/prefect'),
    ('ipython/ipython',           'docker/compose'),
    ('docker/compose',            'ipython/ipython'),
    ('ipython/ipython',           'qiskit/qiskit'),
    ('wagtail/wagtail',           'docker/compose'),
    ('lightning-ai/lightning',    'ipython/ipython'),
    ('numpy/numpy',               'mesonbuild/meson'),
    ('prefecthq/prefect',         'mesonbuild/meson'),
    ('lightning-ai/lightning',    'wagtail/wagtail'),
    ('prefecthq/prefect',         'wagtail/wagtail'),
])

# Define paths for output
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
RESULTS_FILE = os.path.join(RESULT_PATH, 'obj1_experimental_results.csv')
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLOB_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/tranp_cnn_cache"


# Helper Functions

def load_bug_ids_for_project(project_name):
    '''
    Loads bug IDs for a given project from the main bug report file.
    '''
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    project_bug_ids = df_bugs[df_bugs['repo_name'] == project_name]['bug_id'].unique().tolist()
    print(f"Loaded {len(project_bug_ids)} total bugs IDs for project {project_name}")
    return project_bug_ids

def get_data_splits(target_bug_ids):
    '''
    Creates a fixed 80/20 train/test split for the target project bugs.
    '''
    if len(target_bug_ids) < 10:
        raise ValueError(f"Not enough bug reports ({len(target_bug_ids)}) to create train/test split.")
    
    
    train_train_pool, target_test_set = train_test_split(target_bug_ids, test_size=0.20, random_state=42)
    return {'train_pool': train_train_pool, 'test_set': target_test_set}

# Main Orchestrator

def main():
    '''
    Main function to run all configured experiments.
    '''

    # Initialize results file if it doesn't exist
    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=['model_name', 'source_project', 'target_project', 'scenario', 'top-1', 'top-5', 'top-10', 'MAP', 'MRR']
                     ).to_csv(RESULTS_FILE, index=False)
        
    df_results = pd.read_csv(RESULTS_FILE)

    # Use itertools.combinations to get unique pairs of projects
    project_pairs = list(itertools.combinations(PROJECTS, 2))[SHARD::NUM_SHARDS]
    for proj1_config, proj2_config in project_pairs:
        # Define the two directions for this pair
        directions = [
            {'source': proj1_config, 'target': proj2_config},
            {'source': proj2_config, 'target': proj1_config} # Reversed direction
        ]
        for direction in directions:
            pair_start_time = time.time()
            source_project_config = direction['source']
            target_project_config = direction['target']

            source_project_name = source_project_config['name']
            target_project_name = target_project_config['name']

            if (source_project_name, target_project_name) not in PAPER_PAIRS:
                continue

            print(f"\n{'='*20} Setting up experiments for {source_project_name} -> {target_project_name} {'='*20}")

            # Load bug IDs for BOTH projects
            source_bug_ids_all = load_bug_ids_for_project(source_project_name)
            target_bug_ids_all = load_bug_ids_for_project(target_project_name)
            # Create data splits for the target project
            splits = get_data_splits(target_bug_ids_all)

            # Define the four scenarios based on the splits
            scenarios = [
                # Scenario 1: WP-Small (Train: 10% Target, Test: 20% Target)
                {'name': 'WP-small', 'source_train':[], 'target_train': train_test_split(splits['train_pool'], train_size=0.125, random_state=42)[0], 'target_test': splits['test_set']},

                # Scenario 2: WP-Large (Train: 80% Target, Test: 20% Target)
                {'name': 'WP-large', 'source_train':[], 'target_train': splits['train_pool'], 'target_test': splits['test_set']},

                # Scenario 3: CP (Cold Start) (Train: 100% Source, Test: 20% Target)
                {'name': 'CP-cold-start', 'source_train': source_bug_ids_all, 'target_train':[], 'target_test': splits['test_set']},

                # Scenario 4: CP (Transfer) (Train: 100% Source + 20% Target, Test: 20% Target)
                {'name': 'CP-transfer', 'source_train': source_bug_ids_all, 'target_train': train_test_split(splits['train_pool'], train_size=0.25, random_state=42)[0], 'target_test': splits['test_set']}
            ]

            # Loop through each scenario and model
            for model_name, model_function in MODELS_TO_RUN.items():
                for scenario in scenarios:
                    scenario_name = scenario['name']
                    print(f"\n -- Running Model: {model_name} | Scenario: {scenario_name} --")

                    # Check if this result already exists to avoid re-running
                    is_done = (
                        (df_results['model_name'] == model_name) &
                        (df_results['source_project'] == source_project_name) &
                        (df_results['target_project'] == target_project_name) &
                        (df_results['scenario'] == scenario_name)
                    ).any()

                    if is_done:
                        print(f" -> Experiment already exists in results. Skipping.")
                        continue

                    scenario_start_time = time.time()

                    # Execute the experiment
                    metrics = model_function(
                        source_project=source_project_name,
                        target_project=target_project_name,
                        source_train_ids=scenario['source_train'],
                        target_train_ids=scenario['target_train'],
                        target_test_ids=scenario['target_test'],
                        scenario=scenario_name
                    )

                    # Log the results
                    new_result = {
                        'model_name': model_name,
                        'source_project': source_project_name,
                        'target_project': target_project_name,
                        'scenario': scenario_name,
                        'Top-1': metrics['Top-1'],
                        'Top-5': metrics['Top-5'],
                        'Top-10': metrics['Top-10'],
                        'MAP': metrics['MAP'],
                        'MRR': metrics['MRR']
                    }
                    new_result_df = pd.DataFrame([new_result])
                    with open(RESULTS_FILE, 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        new_result_df.to_csv(f, header=False, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)
                    df_results = pd.concat([df_results, new_result_df], ignore_index=True)

                    scenario_time = time.time() - scenario_start_time
                    print(f" -> Results logged to {RESULTS_FILE}")
                    print(f" -> Scenario {scenario_name} completed in {scenario_time/3600:.2f} hrs")
                    gc.collect()

            pair_time = time.time() - pair_start_time
            print(f"\nAll 4 scenarios for {source_project_name} -> {target_project_name} completed in {pair_time/3600:.2f} hrs")

if __name__ == '__main__':
    main()