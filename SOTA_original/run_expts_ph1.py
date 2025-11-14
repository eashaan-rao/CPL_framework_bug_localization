import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time 
import itertools

# Placeholder for the actual training pipelines
from tranp_cnn_pipeline_ph1 import run_tranp_cnn_experiment
# from cooba_pipeline_import run_cooba_experiment
# from blaze_pipeline_import run_cooba_experiment

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
    {'name': 'scipy/scipy', 'language':'python'},
    {'name': 'sympy/sympy', 'language':'python'},
    {'name': 'matplotlib/matplotlib', 'language':'python'},
    {'name': 'numpy/numpy', 'language':'python'},
    {'name': 'open-mmlab/mmdetection', 'language':'python'},
    {'name': 'ray-project/ray', 'language':'python'},
    {'name': 'scikit-learn/scikit-learn', 'language':'python'},
    {'name': 'google/jax', 'language':'python'},
    {'name': 'jupyterlab/jupyterlab', 'language':'python'},
    {'name': 'lightning-ai/lightning', 'language':'python'},
    {'name': 'prefecthq/prefect', 'language':'python'},
    {'name': 'pydata/xarray', 'language':'python'}
]

# Define the models to be evaluated
MODELS_TO_RUN = {
    'TRANP-CNN': run_tranp_cnn_experiment
    # 'COOBA': run_cooba_experiment,
    # 'BLAZE': run_blaze_experiment
}

# Define paths for output
RESULTS_FILE = 'phase1_experimental_results.csv'
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLOB_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cache"

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
    project_pairs = list(itertools.combinations(PROJECTS, 2))
    for proj1_config, proj2_config in project_pairs:
        # Define the two directions for this pair
        directions = [
            {'source': proj1_config, 'target': proj2_config},
            {'source': proj2_config, 'target': proj1_config} # Reversed direction
        ]
        for direction in directions:
            start_time = time.time()
            source_project_config = direction['source']
            target_project_config = direction['target']
            
            source_project_name = source_project_config['name']
            target_project_name = target_project_config['name']
        
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

                    # Execute the experiment
                    metrics = model_function(
                        source_project=source_project_name,
                        target_project=target_project_name,
                        source_train_ids=scenario['source_train'],
                        target_train_ids=scenario['target_train'],
                        target_test_ids=scenario['target_test'],
                        scenario=scenario_name
                    )

                    # Log the results - FIX: Unpack metrics dictionary
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
                    new_result_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
                    # Also update the in-memory dataframe
                    df_results = pd.concat([df_results, new_result_df], ignore_index=True)
                    print(f" -> Results logged to {RESULTS_FILE}")

                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"\nAll experiments for pair {source_project_name} -> {target_project_name} completed in {total_time/3600:.2f} hrs")

if __name__ == '__main__':
    main()