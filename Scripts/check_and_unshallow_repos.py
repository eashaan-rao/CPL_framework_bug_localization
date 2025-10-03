import pandas as pd
import os 
import subprocess

PROJECTS_METADATA_PATH = '/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet'
REPO_BASE_PATH = 'data/repos'

def get_project_path(repo_name, language):
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def check_and_fix_repos():
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    print(f"Checking {len(df_meta)} repositories...")

    for _, row in df_meta.iterrows():
        repo_name = row['repo_name']
        language = row['language']
        repo_path = get_project_path(repo_name, language)

        if not os.path.exists(repo_path):
            print(f"SKIPPING: Directory not found for {repo_name} at {repo_path}")
            continue

        try:
            # Check for shallow clone
            is_shallow = subprocess.check_output(
                ['git', 'rev-parse', '--is-shallow-repository'],
                cwd=repo_path, text=True
            ).strip()

            if is_shallow == 'true':
                print(f"INFO: Repository {repo_name} is a shallow clone. Fetching full history...")
                subprocess.run(['git', 'fetch', '--unshallow'], cwd=repo_path, check=True)
                print(f"SUCCESS: {repo_name} has been unshallowed.")
            else:
                print(f"OK: {repo_name} is not a shallow clone.")
        except Exception as e:
            print(f"ERROR: Could not process repository {repo_name}. Reason {e}")

if __name__ == '__main__':
    check_and_fix_repos()