import os
import time
import pandas as pd
import git
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
import torch
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# CONFIGURATION
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports.parquet"
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
DATABASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_database_sample.pkl"
MODEL_NAME = "BAAI/bge-code-v1" # long-context model
BATCH_SIZE = 32 # We will tune this based on the system performance
NUM_WORKERS = 4 # Start with 16 and we can increase this up to 96 in this machine

LANGUAGE_EXTENSIONS = {
    'c++': ['.c', '.cc', '.cmake', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.in', '.json', '.make', '.py', '.sh', '.xml'],
    'go': ['.go', '.json', '.proto', '.sh', '.yaml', '.yml'],
    'java': ['.gradle', '.groovy', '.java', '.json', '.properties', '.xml', '.yml', '.yaml'],
    'javascript': ['.css', '.html', '.js', '.json', '.jsx', '.mjs', '.scss', '.sh', '.ts', '.tsx', '.yaml', '.yml'],
    'kotlin': ['.gradle', '.json', '.kt', '.kts', '.properties', '.xml', '.yaml', '.yml'],
    'python': ['.bash', '.cfg', '.in', '.ini', '.json', '.py', '.sh', '.toml', '.yaml', '.yml']
}

class SemanticEmbedder:
    '''
    Manages embedding generation for code.
    '''
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        # Trust_remote_code = True is required for this model
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        # The model's max seq length is 8192, but we can set a practical limit if needed
        self.model.max_seq_length = 512

    def get_embeddings_for_blob(self, blob_contents: list):
        '''
        Converts a list of texts into a numpy array of embeddings.
        Handles long documents by splitting them into overlapping chunks and averaging 
        '''
        all_final_embeddings = []
        max_length = self.model.max_seq_length
        chunk_overlap = 50 # A larger overlap can be good for long docs

        for text in blob_contents:
            # Use the model's tokenizer to handle text consistently
            tokens = self.model.tokenizer.encode(text)   
            # If the document is short enough, process it directly
            if len(tokens) <= max_length:
                prefixed_text = "search_document: " + text
                embedding = self.model.encode(prefixed_text, convert_to_numpy=True, show_progress_bar=False)
                all_final_embeddings.append(embedding)
                continue

            # For long documents that exceed the model's limit, create embeddings for each chunk
            chunk_embeddings = []
            for i in range(0, len(tokens), max_length - chunk_overlap):
                chunk_token_ids = tokens[i: i+max_length]
                chunk_text = self.model.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                if chunk_text:
                    prefixed_chunk = "search_document: " + chunk_text
                    embedding = self.model.encode(prefixed_chunk, convert_to_numpy=True, show_progress_bar=False)
                    chunk_embeddings.append(embedding)
            
            if chunk_embeddings:
                mean_embedding = np.mean(chunk_embeddings, axis=0)
                all_final_embeddings.append(mean_embedding)
            else:
                all_final_embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
        
        return np.array(all_final_embeddings)

    
def get_project_path(repo_name, language):
        return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def select_sample_projects(df_meta):
    '''
    Selects a stratified sample of projects using existing categorical columns.
    '''
    print("Selecting a stratified sample of projects....")
    df_meta['strata'] = df_meta['project_size'] + '_' + df_meta['project_age']
    sample_df = df_meta.groupby('strata').apply(
        lambda x: x.sample(1) if not x.empty else None,
        include_groups=False
    ).reset_index()
    print("Selected Projects:")
    print(sample_df[['repo_name', 'strata', 'language']])
    return sample_df

# ----------The Worker Function --------
# This function will be executed by each parallel process.

def process_commits_chunk(repo_name, language, shas_chunk, model_name):
    '''
    A single worker's job: process a chunk of SHAs for one repo.
    '''
    # Each worker gets its own private copy of the repository
    worker_id = os.getpid()
    repo_path_original = os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))
    repo_path_clone = f"{repo_path_original}_clone_{worker_id}"

    if not os.path.exists(repo_path_clone):
        # print(f"Worker {worker_id}: Cloning {repo_name} to {repo_path_clone}...")
        # Add symlinks=True to prevent errors on broken shortcuts
        shutil.copytree(repo_path_original, repo_path_clone, symlinks=True)
    
    repo = git.Repo(repo_path_clone)
    embedder = SemanticEmbedder(model_name) # Each process loads its own model

    local_embedding_db = {}
    local_processed_blobs = set()
    all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)

    for sha in shas_chunk:
        try:
            commit = repo.commit(sha)
            # Find blobs in THIS commit that we haven't processed yet
            blobs_in_this_commit = []
            for blob in commit.tree.traverse():
                    if blob.type == 'blob' and any(blob.name.endswith(ext) for ext in all_extensions):
                        if blob.hexsha not in local_processed_blobs:
                            blobs_in_this_commit.append(blob)

            if not blobs_in_this_commit:
                    continue # No new blobs in this commit to process
            
            # Now, process only the new blobs from this commit in batches
            for i in range(0, len(blobs_in_this_commit), BATCH_SIZE):
                batch_blobs = blobs_in_this_commit[i:i + BATCH_SIZE]
                batch_contents = [b.data_stream.read().decode('utf-8', 'ignore') for b in batch_blobs]
                
                if not batch_contents: continue

                batch_embeddings = embedder.get_embeddings_for_blob(batch_contents)
                
                for blob, embedding in zip(batch_blobs, batch_embeddings):
                    local_embedding_db[blob.hexsha] = embedding
                    # Add hexsha to our global set so we never process it again
                    local_processed_blobs.add(blob.hexsha)
                
                # Manually clear the CUDA cache to release unused memory
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Worker {worker_id}: Skipping commit {sha} due to error: {e}")
            continue

    # Clean up th private clone
    shutil.rmtree(repo_path_clone)

    return local_embedding_db
    
# The main orchestrator
def build_database():
    start_time = time.time()
    # --- FIX for Rate Limiting ---
    # Instantiate the embedder once in the main process BEFORE starting the pool.
    # This downloads and caches the model, so workers load from the local disk.
    print("Pre-loading model into cache...")
    _ = SemanticEmbedder(MODEL_NAME)
    print("Model is cached.")

    # embedder = SemanticEmbedder(MODEL_NAME)

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    sample_projects_df = select_sample_projects(df_meta)
    embedding_db = {} # Move this here

    # Create a process pool with a specific number of workers
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        future_to_repo = {}
        for _, project_row in sample_projects_df.iterrows():
            repo_name = project_row['repo_name']
            language = project_row['language']

            project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
            shas_to_process = list(project_bugs['pre_fix_commit_sha'].dropna().unique())

            if not shas_to_process:
                continue

            # Split the list of SHAs into chunks for each worker
            sha_chunks = np.array_split(shas_to_process, NUM_WORKERS)

            print(f"Submitting {len(shas_to_process)} commits for {repo_name} to {NUM_WORKERS} workers...")
            for chunk in sha_chunks:
                if len(chunk) > 0:
                    # Submit each chunk as a separate job to the pool
                    future = executor.submit(process_commits_chunk, repo_name, language, list(chunk), MODEL_NAME)
                    future_to_repo[future] = repo_name
        
        # Collect results as they are completed
        for future in tqdm(as_completed(future_to_repo), total=len(future_to_repo), desc="Processing Chunks"):
            try:
                # Get the dictionary of embeddings from the worker process
                result_db = future.result()
                # Merge it into our main database
                embedding_db.update(result_db)
            except Exception as e:
                print(f"A chunk failed to process: {e}")

    # Save the final database
    print(f"\n Completed embedding {len(embedding_db)} unique files.")
    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(embedding_db, f) 

    end_time = time.time()
    total_time_hours = (end_time - start_time) / 3600
    db_size_mb = os.path.getsize(DATABASE_PATH) / (1024 * 1024)

    print("\n --- Build Complete ---")
    print(f"Total time taken: {total_time_hours:.2f} hours")
    print(f"Database size: {db_size_mb:.2f} MB")
    print(f"Database saved to: {DATABASE_PATH}")
                    
if __name__ == '__main__':
    # --- FIX for CUDA Multiprocessing ---
    # Set the start method to 'spawn' to avoid CUDA errors.
    # This must be inside the __name__ == '__main__' block.
    mp.set_start_method('spawn', force=True)
    build_database()
        


