import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Placeholder for the actual training pipelines
# from tranp_cnn_pipeline_import run_tranp_cnn_experiment
# from cooba_pipeline_import run_cooba_experiment
# from blaze_pipeline_import run_cooba_experiment

def run_tranp_cnn_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
    '''
    Placeholder function for the TRANP-CNN pipeline.
    '''
    print(f" -> Predenting to run TRANP-CNN")
    print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
    # In reality, this function would trigger the entire training and evaluation process and return a dictionary
    # of metrics
    return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

def run_cooba_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
    '''
    Placeholder function for the CooBA pipeline.
    '''
    print(f" -> Predenting to run CooBA")
    print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
    # In reality, this function would trigger the entire training and evaluation process and return a dictionary
    # of metrics
    return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

def run_blaze_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
    '''
    Placeholder function for BLAZE pipeline.
    '''
    print(f" -> Predenting to run BLAZE")
    print(f" -> Source Bugs: {len(source_train_ids)}, Target Train Bugs: {len(target_train_ids)}, Target Test Bugs: {len(target_test_ids)}")
    # In reality, this function would trigger the entire training and evaluation process and return a dictionary
    # of metrics
    return {'Top-1':np.random.rand(), 'Top-5':np.random.rand(), 'Top-10':np.random.rand(), 'MAP':np.random.rand(), 'MRR':np.random.rand()}

# Configuration Section

# Define the projects to run experiments on
# This can be expanded with more projects later.
PROJECTS = [
    {'name': 'huggingface/transformers', 'language':'python'},
    {'name': 'pandas-dev/pandas', 'language':'python'}
]

# Define the models to be evaluated
MODELS_TO_RUN = {
    'TRANP-CNN': run_tranp_cnn_experiment,
    'COOBA': run_cooba_experiment,
    'BLAZE': run_blaze_experiment
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