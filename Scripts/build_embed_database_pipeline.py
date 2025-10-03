import os
import time
import pandas as pd
import git
import shutil
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import logging
import numpy as np
import pickle
from tqdm import tqdm
import torch
import multiprocessing as mp
import re
import queue



# This is the "official" way to handle the TF32 warning. It tells PyTorch to use the high-performance TF32 mode for any 
# remaining FP32 operations, which silences the warning.
torch.set_float32_matmul_precision('high')

# This suppresses the informational warnings from the transformers library, including the token sequence length warning.
logging.set_verbosity_error()

# Configuration
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports.parquet"
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
DATABASE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs" # for blobs
# BUG_DB_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_embedding_database.pkl"
DATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data"
MODEL_NAME = "BAAI/bge-code-v1" # long-context model
CONSUMER_BATCH_SIZE = 256 # We will tune this based on the system performance
NUM_CPU_WORKERS = 8 # Start with 16 and we can increase this up to 96 in this machine
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


LANGUAGE_EXTENSIONS = {
    'c++': ['.ac', '.alpine', '.am', '.arm64', '.armv7', '.arrow', '.asar', '.avro', '.bash', '.bat', '.bazel', '.bin', '.bzl', '.c', '.cc', '.cfg', '.clj', '.cmake', '.conf', '.config', '.cpp', '.cs', '.csproj', '.css', '.dat', '.data', '.def', '.dict', '.expect', '.expected', '.fragment', '.gd', '.gemspec', '.glsl', '.gn', '.gni', '.go', '.grdp', '.gyp', '.gypi', '.h', '.hpp', '.html', '.ico', '.in', '.inc', '.include', '.init', '.install', '.j2', '.java', '.js', '.json', '.json5', '.kt', '.lib', '.limits', '.lock', '.m', '.m4', '.make', '.manifest', '.md', '.mjs', '.mk', '.mm', '.mp4', '.npy', '.out', '.patch', '.pb', '.pbxproj', '.php', '.plist', '.png', '.postinst', '.postinstall', '.preinst', '.prerm', '.props', '.proto', '.ps1', '.py', '.python', '.queries', '.rb', '.reference', '.scm', '.scss', '.sh', '.sigs', '.so', '.sql', '.supp', '.svg', '.targets', '.templates', '.ts', '.txt', '.ubuntu', '.ui', '.vcxproj', '.xib', '.xml', '.yaml', '.yml', '.zst'],
    'go': ['.bash', '.bats', '.conf', '.css', '.cue', '.dockerfile', '.ex', '.exs', '.fragment', '.go', '.graphqls', '.gtpl', '.hbs', '.ignore', '.js', '.json', '.key', '.lock', '.md', '.mdx', '.mjs', '.mod', '.mts', '.nightly', '.pem', '.png', '.py', '.rs', '.scss', '.sh', '.sum', '.tmpl', '.toml', '.ts', '.txt', '.win64', '.xml', '.yaml', '.yml'],
    'java': ['.asciidoc', '.bat', '.bazel', '.bz2', '.bzl', '.cer', '.clusterfilter', '.cmd', '.commandstep', '.conf', '.config', '.cpp', '.crt', '.crx', '.cs', '.css', '.csv', '.csv-spec', '.db', '.desktop', '.enc', '.executor', '.factories', '.filter', '.gemspec', '.gradle', '.groovy', '.gz', '.h', '.html', '.importorder', '.ini', '.install4j', '.iss', '.jar', '.java', '.jj', '.jpg', '.js', '.json', '.key', '.kt', '.less', '.liquibasedatatype', '.lock', '.md', '.meshenvlistenerfactory', '.mustache', '.nuspec', '.ods', '.plist', '.png', '.policy', '.properties', '.protocol', '.ps1', '.py', '.rb', '.rs', '.rst', '.scss', '.sh', '.sha1', '.snapshotgenerator', '.sql', '.sql-spec', '.sqlgenerator', '.st', '.svg', '.targets', '.toml', '.tpl', '.ts', '.tsv', '.tsx', '.txt', '.typebuilder', '.validation', '.vm', '.vt', '.vue', '.xls', '.xml', '.xsd', '.yaml', '.yml', '.zip'],
    'javascript': ['.avif', '.cjs', '.coffee', '.css', '.cts', '.dockerfile', '.example', '.graphql', '.graphqls', '.html', '.ico', '.jpg', '.js', '.jsm', '.json', '.jsx', '.link', '.lock', '.map', '.md', '.mdx', '.mjs', '.mts', '.opts', '.pdf', '.png', '.properties', '.rb', '.rs', '.scss', '.sh', '.snap', '.sqlite', '.stderr', '.svelte', '.svg', '.toml', '.ts', '.tsx', '.ttf', '.txt', '.wasm', '.webp', '.xml', '.yaml', '.yml'],
    'kotlin': ['.java', '.kt', '.py'],
    'python': ['.0', '.1', '.acl', '.ambr', '.base', '.bat', '.build', '.c', '.cfg', '.ci', '.cmd', '.cnf', '.conf', '.cs', '.csproj', '.css', '.css_t', '.csv', '.db', '.dockerfile', '.example', '.expected', '.g4', '.gif', '.h', '.html', '.in', '.ini', '.interp', '.inv', '.inventory', '.ipynb', '.j2', '.jar', '.java', '.jinja2', '.js', '.json', '.json5', '.jsx', '.kubernetes-helm-yaml', '.less', '.lock', '.manifest', '.md', '.mdx', '.mkv', '.mp4', '.nodejs14x', '.php', '.pip', '.png', '.pot', '.ps1', '.psm1', '.pxd', '.pxi', '.py', '.pyi', '.pyx', '.r', '.rdb', '.rst', '.run', '.scss', '.sh', '.sha256', '.sln', '.stderr', '.stdout', '.svg', '.template', '.tf', '.tgz', '.toml', '.tpl', '.ts', '.tsx', '.txt', '.xml', '.yaml', '.yml', '.zip'],
}


# The worker function (must be at the top level)
def producer_worker(task_queue, chunk_queue, model_name, shared_file_contents):
    '''
    CPU-bound worker. Processes tasks (either bug reports or commits).
    Its job is to read text, files, tokenizes, and puts Individual chunks into a queue for the GPU
    IT Performs no disk I/O.
    '''
    # Each worker loads its own lightwright tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)
    worker_id = os.getpid()

    while True:
        task = task_queue.get()
        if task == "STOP":
            break # Exit signal
    
        task_type, data = task

        if task_type == 'BUG':
            bug_id, bug_text = data
            try:
                tokens = tokenizer.encode(bug_text)
                # Send chunks to consumer
                # Add the 'BUG' type to the tuple
                if len(tokens) <= CHUNK_SIZE:
                    chunk_queue.put(('BUG', bug_id, tokens))
                else:
                    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
                        chunk_queue.put(('BUG', bug_id, tokens[i:i + CHUNK_SIZE]))
            except Exception as e:
                print(f"Worker {worker_id}: Failed to process bug {bug_id}. Error: {e}")

        elif task_type == 'BLOB':
            # The data is now just the blob's unique hash (hexhsa)
            blob_sha = data
            # Retrieve the file's content from the shared RAM dictionary.
            # This is an extremely fast memory acess, not a slow disk read.
            content = shared_file_contents[blob_sha]

            tokens = tokenizer.encode(content)

            try: 
                if len(tokens) <= CHUNK_SIZE:
                    chunk_queue.put(('BLOB', blob_sha, tokens))
                else:
                    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
                        chunk_queue.put(('BLOB', blob_sha, tokens[i:i + CHUNK_SIZE]))

            except Exception as e:
                print(f"Worker {worker_id}: Failed to process task {task}. Error: {e}")

def consumer_worker(chunk_queue, final_db_queue, model_name):
    '''
    GPU-bound worker. Pulls chunks, batches them, embeds them, and re-assembles them into final file vectors.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the model with float 16 precision
    model = SentenceTransformer(
        model_name, 
        device=device,
        model_kwargs={'torch_dtype': torch.float32}
        )
    
    # This adds a small one-time "warm-up" cost on the first batch, then it's much faster
    if device == 'cuda':
        # The main transformer model is stroed in the '0' attribute of the SentenceTransformer object
        model[0].auto_model = torch.compile(model[0].auto_model)

        # Check the effective sequence length used by the SentenceTransformer library
        # print(f"SentenceTransformer effective max_seq_length: {model.max_seq_length}")

        # Check the true, underlying architectural limit from the model's config file
        #  (We access the core transformer model inside the SentenceTransformer object)
        # print(f"Model's architectural max_position_embeddings: {model[0].auto_model.config.max_position_embeddings}")
    
    print(f"Model is loaded with data type: {model[0].auto_model.dtype} \n")
    # Set the sequence length to match the CHUNK_SIZE
    model.max_seq_length = 512 

    # Internal dictionary to gather all chunks for each blob
    reassembled_embeddings = {}

    # Adding an inactivity counter
    consecutive_timeouts = 0
    MAX_TIMEOUTS = 3 # Stop after 3 consecutive 60-second timeouts (3 minutes of inactivity)

    while True:
        chunks_to_process = []
        
        try:
            # # Pull the first item, waiting if necessary. Timeout prevents permanent blocking.
            item = chunk_queue.get(timeout=60) 
            chunks_to_process.append(item)

            # If we successfully get data, rest the timeout counter
            consecutive_timeouts = 0

            # Greedily pull more items from the queue until we have a full batch.
            # get_nowait() is non-blocking, raising queue.Empty if the queue is empty.
            while len(chunks_to_process) < CONSUMER_BATCH_SIZE:
                item = chunk_queue.get_nowait() # Doesn't block
                chunks_to_process.append(item)
        except queue.Empty:
            #  This is an expected when the queue is temporarily empty
            # Increment the timeout counter
            consecutive_timeouts += 1
            # print(f"Consumer timed out waiting for chunks. Timeout count: {consecutive_timeouts}/{MAX_TIMEOUTS}")
            pass

        if chunks_to_process:
            # Unzip the list of tuples into separate lists
            item_types, unique_ids, chunk_token_ids = zip(*chunks_to_process)
            # The model's tokenizer can efficiently decode a batch of token ID lists into strings
            chunk_texts = model.tokenizer.batch_decode(chunk_token_ids, skip_special_tokens=True)

            # Apply model-specific prefixes for optimal embedding performance (BGE model)
            prefixed_texts = []
            for i, text in enumerate(chunk_texts):
                if item_types[i] == 'BUG':
                    prefixed_texts.append("search_query: " + text) # BGE prefix for query
                else: # BLOB
                    prefixed_texts.append("search_document: " + text) # BGE prefix for source file

            # Embed the full, efficient batch
            chunk_embeddings = model.encode(prefixed_texts, 
                                            convert_to_numpy=True, 
                                            show_progress_bar=False, 
                                            batch_size=len(chunk_texts),
                                            normalize_embeddings=True 
                                            )

            # Accumulate the chunk embeddings for each blob
            # for item_type, unique_id, embedding in zip(item_types, unique_ids, chunk_embeddings):
            for i, embedding in enumerate(chunk_embeddings):
                key = (item_types[i], unique_ids[i])
                if key not in reassembled_embeddings:
                    reassembled_embeddings[key] = []
                reassembled_embeddings[key].append(embedding)
        
        # If we have timed out too many times in a row, assume producers are done and exit.
        if consecutive_timeouts >= MAX_TIMEOUTS:
            print("Consumer has been idle for too long. Shutting down.")
            break
    
    # After the loop finishes, average all the collected chunks
    for (item_type, unique_id), embeddings in reassembled_embeddings.items():
        final_embedding = np.mean(embeddings, axis=0)
        final_db_queue.put((item_type, unique_id, final_embedding))
    
    final_db_queue.put("STOP") # Signal that the consumer is finisged

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

# The main orchestrator, modifies to separate I/O from computation
def build_database_pipeline():
    start_time = time.time()

    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)

    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    # Following code is for extracting sample of 6 projects from our dataset
    # sample_projects_df = select_sample_projects(df_meta)
    # sample_repo_names = sample_projects_df['repo_name'].tolist()
    
    # This is for single project
    target_repos = [
        "elastic/elasticsearch",
        "apache/dolphinscheduler",
        "apache/dubbo",
        "liquibase/liquibase",
        "openrefine/openrefine",
        "seleniumhq/selenium",
        "ccxt/ccxt"
    ]

    print(f"\n--- Starting Source Code Blob and Bug Report Embedding Per project ---")
    # The following code process all projects from the metadata file
    # sample_project_df = df_meta[df_meta['repo_name'] == TARGET_REPO].copy()
    sample_project_df = df_meta[df_meta['repo_name'].isin(target_repos)].reset_index(drop=True)
    projects_to_process = sample_project_df

    for _, project_row in tqdm(projects_to_process.iterrows(), total=len(projects_to_process), desc="Total Projects"):
        repo_name = project_row['repo_name']
        language = project_row['language']

        # Save the project embeddings
        project_blob_db_name = repo_name.replace('/', '_') + '_blob_embeddings32.pkl'
        project_blob_db_path = os.path.join(DATABASE_DIR, project_blob_db_name)

        # Save the bug report embeddings this project
        project_bug_db_name = repo_name.replace('/', '_') + '_bug_metadata32.pkl'
        project_bug_db_path = os.path.join(DATABASE_DIR, project_bug_db_name)

         # Skip if this project's database already exists
        if os.path.exists(project_blob_db_path) and os.path.exists(project_bug_db_path):
            print(f"Databases for {repo_name} already exists. Skipping.")
            continue

        print(f"\n------Starting Project: {repo_name} --------------")
        project_start_time = time.time()

        # Filter bugs for only the current project
        project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
        if project_bugs.empty:
            print(f"No bug reports found for {repo_name}. Skipping....")
            continue

        # 1. I/O Phase: Pre-load all required file contents into memory
        # This is done in the main process to avoid disk contention among CPU Workers
        print(f"I/O Phase: Reading source files for {repo_name} into memory....")
        file_contents_local = {}
        unique_commit_shas = project_bugs['pre_fix_commit_sha'].dropna().unique()
        repo_path = os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

        try:
            repo = git.Repo(repo_path)
            for sha in tqdm(unique_commit_shas, desc="Reading Commits"):
                commit = repo.commit(sha)
                for blob in commit.tree.traverse():
                    if blob.type == 'blob' and blob.hexsha not in file_contents_local and any(blob.name.endswith(ext) for ext in all_extensions):
                        try:
                            # Read file content and store in a standard Python dictionary
                            content = blob.data_stream.read().decode('utf-8', 'ignore')
                            file_contents_local[blob.hexsha] = content
                        except Exception:
                            continue # Skip files that can't be read/decoded
        except Exception as e:
            print(f"Error reading repo for {repo_name}. Skipping project. Error: {e}")
            continue

        print(f"Loaded {len(file_contents_local)} unique source files into memory")

        # 2. Prepare tasks
        # Create Task Lists for this project
        bug_tasks = [('BUG', (row.bug_id, row.bug_report_text)) for _, row in project_bugs.drop_duplicates(subset=['bug_id']).iterrows()]
        # Tasks for blobs are now just their IDs, as the content is already in memory
        blob_tasks = [('BLOB', blob_sha) for blob_sha in file_contents_local.keys()]
        all_tasks = bug_tasks + blob_tasks
        
        if not all_tasks:
            print(f"No valid commits and bugs tasks found for {repo_name}. Skipping.")
            continue

        # 3. Setup MULTIPROCESSING
        # Setup and Run Pipeline for this project
        # Create the Queues for Communication
        # USe a manager for queues in a spawn context
        manager = mp.Manager()
        # Create the shared dictionary and populate it from our local dictionary.
        shared_file_contents = manager.dict()
        shared_file_contents.update(file_contents_local)
        
        # Queues for inter-process communication
        task_queue = manager.Queue()
        chunk_queue = manager.Queue(maxsize = NUM_CPU_WORKERS * 10) # Buffer for GPU
        final_db_queue = manager.Queue()

        # 4. Execute PIPELIN
        # Start the Workers and pass the shared dictionary to the producer workers
        producers = [mp.Process(target=producer_worker, args=(task_queue, chunk_queue, MODEL_NAME, shared_file_contents)) for _ in range(NUM_CPU_WORKERS)]
        consumer = mp.Process(target=consumer_worker, args=(chunk_queue, final_db_queue, MODEL_NAME))

        try:
            for p in producers:
                p.start()
            consumer.start()

            # Feed all tasks for this project to the producers
            print(f"Feeding {len(bug_tasks)} bug reports and {len(blob_tasks)} blobs as {len(all_tasks)} total tasks to {NUM_CPU_WORKERS} CPU workers...")
            for task in all_tasks:
                task_queue.put(task)

            # Signal Producers to Stop once all tasks are fed
            for _ in range(NUM_CPU_WORKERS):
                task_queue.put("STOP")

            # Wait for all producer processes to finish their work
            # print("Waiting for CPU workers (producers) to finish...")
            # for p in producers:
            #     p.join()
            # print("ALL CPU workers have finished.")

            # Now that producers are done, signal the consumer to stop
            # chunk_queue.put("STOP")


            # Aggregate Final Results until the consumer signals it's done
            bug_embedding_db = {}
            blob_embedding_db = {}

            # Track expected unique items for progress
            expected_bugs = len(set(row.bug_id for _, row in project_bugs.drop_duplicates(subset=['bug_id']).iterrows()))
            expected_blobs = len(file_contents_local)
            total_expected = expected_bugs + expected_blobs
            
            with tqdm(total=total_expected, desc=f"\nAggregating embeddings for {repo_name}") as pbar:
                for _ in range(total_expected):
                    try:
                        # Get an item, use a long timeout for safety
                        item = final_db_queue.get()  # no timeout - not recommened.
                        # The consumer might have sent "STOP" early if there was an error.
                        # We can check for it, but the loop will end anyway.
                        if item == "STOP":
                            print("WARNING: Received premature STOP signal from consumer. Results may be incomplete.")
                            break

                        item_type, unique_id, embedding = item
                        
                        # Process the embedding
                        if item_type == 'BUG' and unique_id not in bug_embedding_db:
                            bug_embedding_db[unique_id] = embedding
                            pbar.update(1)
                        elif item_type == 'BLOB' and unique_id not in blob_embedding_db:
                            blob_embedding_db[unique_id] = embedding
                            pbar.update(1)
                    except queue.Empty:
                        print(f"\nERROR: Timed out waiting for results from the final queue.")
                        print(f"Received {pbar.n} of {total_expected} expected items. The consumer may have crashed.")
                        break # Exit the loop if the consumer is silent
            
            print("All results aggregated. Cleaning up worker processes...")
            for p in producers:
                p.join()
            consumer.join()
            print("All worker processes have been cleanly shut down.")

            # Verify we got all expected items
            actual_bugs = len(bug_embedding_db)
            actual_blobs = len(blob_embedding_db)
            print(f"Final counts: {actual_bugs}/{expected_bugs} bugs, {actual_blobs}/{expected_blobs} blobs")
            
            if actual_bugs < expected_bugs or actual_blobs < expected_blobs:
                print(f"WARNING: Missing embeddings! Expected {total_expected}, got {actual_bugs + actual_blobs}")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n--- Initiating shutdown sequence. This may take a moment. ---")
        
            # Terminate any still-running producer processes
            for p in producers:
                if p.is_alive():
                    print(f"Terminating producer process {p.pid}...")
                    p.terminate()
                    p.join() # Wait for termination to complete
            
            # Terminate the consumer process
            if consumer.is_alive():
                print(f"Terminating consumer process {consumer.pid}...")
                consumer.terminate()
                consumer.join()

        print("All worker processes have been terminated.")

        # 5. Save Results
        # Save Bug DB and Blob DB for each repo
        bug_metadata_db = {}
        for _, row in project_bugs.drop_duplicates(subset=['bug_id']).iterrows():
            if row.bug_id in bug_embedding_db:
                bug_metadata_db[row.bug_id] = {
                    'embedding': bug_embedding_db[row.bug_id],
                    'commit_sha': row['pre_fix_commit_sha'],
                    'ground_truth_files': row['ground_truth_files']
                }
        with open(project_bug_db_path, 'wb') as f: 
            pickle.dump(bug_metadata_db, f)

        # Save Blob DB
        with open(project_blob_db_path, 'wb') as f: 
            pickle.dump(blob_embedding_db, f)

        # print(f"\n Completed {len(bug_embedding_db)} Bug reports embeddings & blob embeddings {len(blob_embedding_db)} unique files for {repo_name}.")
        
        project_end_time = time.time()
        total_time_hours = (project_end_time - project_start_time) / 3600
        db_size_mb = os.path.getsize(project_blob_db_path) / (1024 * 1024)

        print(f"\n --- Project {repo_name} Embedding Complete ---")
        print(f"Total time taken: {total_time_hours:.2f} hours")
        print(f"Database size: {db_size_mb:.2f} MB")
        print(f"Saved {len(bug_metadata_db)} bug embeddings and {len(blob_embedding_db)} blob embeddings.")
        print(f"Database saved to: {project_blob_db_path}")

    end_time = time.time()
    print(f"Total time taken for all 98 repos embeddings to be complete: {(end_time - start_time )/ 3600} hrs")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    build_database_pipeline()
    


