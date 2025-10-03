import os
import time
import pandas as pd
import git
import re
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Configuration
PROJECTS_METADATA_PATH = '/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet'
BUG_REPORTS_PATH = '/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports.parquet'
REPO_BASE_PATH = '/home/cs21d002_eashaan/PhD/Objective1/data/repos'
RESULT_PATH = '/home/cs21d002_eashaan/PhD/Objective1/results'
NUM_SAMPLE_PROJECTS = 6
# MODEL_NAME = 'microsoft/codebert-base'
MODEL_NAME = 'all-mpnet-base-v2'

LANGUAGE_EXTENSIONS = {
    'c++': ['.c', '.cc', '.cmake', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.in', '.json', '.make', '.py', '.sh', '.xml'],
    'go': ['.go', '.json', '.proto', '.sh', '.yaml', '.yml'],
    'java': ['.gradle', '.groovy', '.java', '.json', '.properties', '.xml', '.yml', '.yaml'],
    'javascript': ['.css', '.html', '.js', '.json', '.jsx', '.mjs', '.scss', '.sh', '.ts', '.tsx', '.yaml', '.yml'],
    'kotlin': ['.gradle', '.json', '.kt', '.kts', '.properties', '.xml', '.yaml', '.yml'],
    'python': ['.bash', '.cfg', '.in', '.ini', '.json', '.py', '.sh', '.toml', '.yaml', '.yml']
}

K_VALUES = [30, 50, 75, 100, 125, 150, 200]

# Part 1: Semantic Search and Git Management
class SemanticSearchManager:
    '''
    Manages embedding generation and Faiss similarity search.
    '''
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, texts: list, chunk_overlap=50):
        '''
        Converts a list of texts into a numpy array of embeddings.
        Handles long documents by splitting them into overlapping chunks and averaging the embeddings.
        '''
        all_final_embeddings = []

        # Get the model's max sequence length to use for chunking
        max_length = self.model.max_seq_length
        # Process each document (bug report or source file)
        for text in texts:
            # Use the model's tokenizer to get the token IDs
            tokens = self.model.tokenizer.encode(text)
            # If the document is short enough, process it directly
            if len(tokens) <= max_length:
                embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
                all_final_embeddings.append(embedding)
                continue
  
            # For long documents, create embeddings for each chunk
            chunk_embeddings = []
            for i in range(0, len(tokens), max_length - chunk_overlap):
                chunk_token_ids = tokens[i:i + max_length]
                # Decode token IDs back to text for the model to encode
                chunk_text = self.model.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                if chunk_text: # Ensure chunk is not empty
                    embedding = self.model.encode(chunk_text, convert_to_numpy=True, show_progress_bar=False)
                    chunk_embeddings.append(embedding)
            
            # Average the embeddings of all chunks to get a single vector for the whole document
            if chunk_embeddings:
                mean_embedding = np.mean(chunk_embeddings, axis=0)
                all_final_embeddings.append(mean_embedding)
            else:
                # Handle rare case of empty text after processing
                embedding_dim = self.model.get_sentence_embedding_dimension()
                all_final_embeddings.append(np.zeros(embedding_dim))

        return np.array(all_final_embeddings)

        # return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def build_faiss_index(self, doc_embeddings: np.ndarray):
        '''
        Builds a Faiss index from document embeddings.
        '''
        # Normalize vectors from cosine similarity search
        faiss.normalize_L2(doc_embeddings)
        embedding_dim = doc_embeddings.shape[1]
        # Using IndexFlatIP for Inner Product, which is equivalent to cosine similarity on normalized vectors
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(doc_embeddings)
        return index
    
    def search(self, query_embedding: np.ndarray, index:faiss.Index, top_k: int):
        '''
        Searches the Faiss index for the top_k most similar vectors.
        '''
        # Faiss search expects a 2D array, so we reshape the query embedding
        query_embedding_2d = np.expand_dims(query_embedding, axis=0)
        # Normalize the query vector as well
        faiss.normalize_L2(query_embedding_2d)
        distances, indices = index.search(query_embedding_2d, top_k)
        return indices[0] # Return the indices for the first (and only) query

    
def get_project_path(repo_name, language):
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))
    
def checkout_commit(repo_path, sha):
    try:
        repo = git.Repo(repo_path)
        repo.git.checkout(sha, f=True)
        repo.git.submodule('update', '--init', '--recursive')
        return True
    except git.exc.GitCommandError as e:
        # Suppress verbose errors for invalid SHAs, we will count them instead
        return False
        
# Part 2: Experiment Logic

def select_sample_projects(df_meta):
    print("Selecting a stratified sample of projects...")
    df_meta['strata'] = df_meta['project_size'] + '_' + df_meta['project_age']
    sample_df = df_meta.groupby('strata').apply(
        lambda x: x.sample(1) if not x.empty else None,
        include_groups = False
    ).reset_index()
    print("Selected Projects:")
    print(sample_df[['repo_name', 'strata', 'language']])
    return sample_df

def run_semantic_search_experiment():
    '''
    Main function to execute the semantic search experiment.
    '''
    semantic_manager = SemanticSearchManager(MODEL_NAME)

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    select_sample_df = select_sample_projects(df_meta)

    # Counters
    total_bugs_processed = 0
    checkout_failures = 0
    empty_checkouts = 0
    hits_per_k = {k: 0 for k in K_VALUES}

    for _, project_row in tqdm(select_sample_df.iterrows(), total=len(select_sample_df), desc="Projects"):
        repo_name = project_row['repo_name']
        language = project_row['language']
        repo_path = get_project_path(repo_name, language)

        if not os.path.exists(repo_path):
            continue

        project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
        if project_bugs.empty:
            continue
        for _, bug in tqdm(project_bugs.iterrows(), total=len(project_bugs), desc=f"Bugs in {repo_name}", leave=False):
            total_bugs_processed += 1
            snapshot_sha = bug['pre_fix_commit_sha']

            if not snapshot_sha or not checkout_commit(repo_path, snapshot_sha):
                checkout_failures += 1
                continue

            # Read all relevant source files from the snapshot
            source_files = []
            all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if any(file.endswith(ext) for ext in all_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            # Store both the relative path and the content
                            relative_path = os.path.relpath(file_path, start=repo_path)
                            source_files.append({"path": relative_path, "content": content})
                        except Exception:
                            continue
            
            if not source_files:
                empty_checkouts += 1
                continue

            # Generate embeddings
            bug_report_embedding = semantic_manager.get_embeddings([bug['bug_report_text']])[0]
            source_content = [sf['content'] for sf in source_files]
            source_embeddings = semantic_manager.get_embeddings(source_content)

            # Build Faiss index and search
            faiss_index = semantic_manager.build_faiss_index(source_embeddings)
            result_indices = semantic_manager.search(bug_report_embedding, faiss_index, max(K_VALUES))

            # Map result indices back to file paths
            candidates_relative = [source_files[i]['path'] for i in result_indices]

            # Calculate hits
            ground_truth = set(bug['ground_truth_files'])
            for k in K_VALUES:
                candidates_at_k = set(candidates_relative[:k])
                if not candidates_at_k.isdisjoint(ground_truth):
                    hits_per_k[k] += 1

    # --- Aggregrate and Analyze Final Results ---
    print("\n\n--- Semantic Search Experiment Complete ---")
    print(f"Total bug reports processed: {total_bugs_processed}")
    print(f"Checkout Failures: {checkout_failures} ({checkout_failures/total_bugs_processed:.2%})")
    print(f"Empty Checkouts (no source files): {empty_checkouts} ({empty_checkouts/total_bugs_processed:.2%})")
    
    print("\n--- Recall @ K Results ---")
    final_recall = {}
    for k, hits in hits_per_k.items():
        recall = hits / total_bugs_processed if total_bugs_processed > 0 else 0
        final_recall[f"Recall@{k}"] = recall
        print(f"Recall@{k}: {recall:.4f} ({hits}/{total_bugs_processed})")

    # Save final results
    df_final_recall = pd.DataFrame([final_recall])
    recall_path = os.path.join(RESULT_PATH, 'semantic_search_recall_results.csv')
    df_final_recall.to_csv(recall_path, index=False)
    print(f"\nFinal recall results saved to {recall_path}")            


if __name__ == '__main__':
    run_semantic_search_experiment()