import os
import pandas as pd
import numpy as np
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- Configuration (Adjust as needed) ---
# Use paths from your pipeline script
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
OUTPUT_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/all_project_domain_gaps.csv"
# --- End Configuration ---

def load_data(project_name):
    """
    Loads bug count and blob embeddings for a single project.
    Returns (bug_count, embeddings_array)
    """
    # 1. Load Bug Count
    bug_count = 0
    bug_db_path = os.path.join(BUG_METADATA_DIR, project_name.replace('/', '_') + '_bug_metadata32.pkl')
    try:
        with open(bug_db_path, 'rb') as f:
            bug_db = pickle.load(f)
            bug_count = len(bug_db)
    except FileNotFoundError:
        print(f"Warning: Bug DB not found for {project_name}. Bug count set to 0.")
    
    # 2. Load Blob Embeddings
    embeddings = None
    blob_db_path = os.path.join(BLOB_DB_DIR, project_name.replace('/', '_') + '_blob_embeddings32.pkl')
    try:
        with open(blob_db_path, 'rb') as f:
            blob_db = pickle.load(f)
            embeddings = np.array(list(blob_db.values())).astype('float32')
            if embeddings.size == 0:
                print(f"Warning: Blob DB for {project_name} is empty.")
                return bug_count, None
    except FileNotFoundError:
        print(f"Error: Blob DB not found for {project_name}. Skipping.")
        return bug_count, None
        
    return bug_count, embeddings

def calculate_domain_gap(embeddings_A, embeddings_B):
    """
    Trains a Logistic Regression classifier to distinguish two
    sets of embeddings. Returns the test accuracy.
    """
    if embeddings_A is None or embeddings_B is None:
        return -1.0
        
    # Create labels (0 for A, 1 for B)
    labels_A = np.zeros(embeddings_A.shape[0])
    labels_B = np.ones(embeddings_B.shape[0])
    
    # Combine
    X = np.concatenate([embeddings_A, embeddings_B])
    y = np.concatenate([labels_A, labels_B])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

def main():
    # Load all 98 projects
    try:
        projects_df = pd.read_parquet(PROJECTS_METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: Project metadata file not found at {PROJECTS_METADATA_PATH}")
        return

    # Create a list of project dicts for iteration
    project_list = projects_df.to_dict('records')
    print(f"Loaded {len(project_list)} projects.")
    
    # Get all unique pairs
    project_pairs = list(itertools.combinations(project_list, 2))
    print(f"Generated {len(project_pairs)} unique project pairs.")
    
    results = []
    
    for proj_A, proj_B in tqdm(project_pairs, desc="Analyzing project pairs"):
        name_A, lang_A = proj_A['repo_name'], proj_A['language']
        name_B, lang_B = proj_B['repo_name'], proj_B['language']
        
        # Load data for both projects
        bugs_A, embeds_A = load_data(name_A)
        bugs_B, embeds_B = load_data(name_B)
        
        # Calculate domain gap
        gap_score = calculate_domain_gap(embeds_A, embeds_B)
        
        if gap_score != -1.0:
            results.append({
                'project_A': name_A,
                'project_B': name_B,
                'lang_A': lang_A,
                'lang_B': lang_B,
                'num_bugs_A': bugs_A,
                'num_bugs_B': bugs_B,
                'domain_gap_accuracy': gap_score
            })
            
    # Save final CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Analysis complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
    
# import pickle
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load the databases
# with open('/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs/spring-projects_spring-data-mongodb_blob_embeddings32.pkl', 'rb') as f:
#     source_blob_db = pickle.load(f)

# with open('/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs/spring-projects_spring-data-jpa_blob_embeddings32.pkl', 'rb') as f:
#     target_blob_db = pickle.load(f)

# # Get all embeddings as NumPy arrays
# source_embeddings = np.array(list(source_blob_db.values()))
# target_embeddings = np.array(list(target_blob_db.values()))

# # Create labels: 0 for source, 1 for target
# source_labels = np.zeros(source_embeddings.shape[0])
# target_labels = np.ones(target_embeddings.shape[0])

# # Combine all embeddings and labels
# all_embeddings = np.concatenate([source_embeddings, target_embeddings])
# all_labels = np.concatenate([source_labels, target_labels])

# # Run t-SNE (it's slow, so you might sample it)
# print("Running t-SNE...")
# tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
# tsne_results = tsne.fit_transform(all_embeddings)

# # Plot the results
# plt.figure(figsize=(10, 7))
# # Plot source (blue)
# plt.scatter(tsne_results[all_labels == 0, 0], tsne_results[all_labels == 0, 1], c='blue', label='Source (Anvil)', alpha=0.5)
# # Plot target (red)
# plt.scatter(tsne_results[all_labels == 1, 0], tsne_results[all_labels == 1, 1], c='red', label='Target (KotlinPoet)', alpha=0.5)
# plt.legend()
# plt.title('t-SNE Visualization of Project Semantic Spaces')
# plt.savefig('spring_mongodb_jpa_domain_gap_visualization.png')
# print("Saved plot to domain_gap_visualization.png")

# print("Training Domain Classifier...")
# # Use the same data: all_embeddings (X) and all_labels (y)
# X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42)

# # Train a simple model
# classifier = LogisticRegression(random_state=42)
# classifier.fit(X_train, y_train)

# # Get accuracy
# predictions = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)

# print(f"Domain Classifier Accuracy: {accuracy * 100:.2f}%")