import pandas as pd
import os
import subprocess
import shutil
from tqdm import tqdm

# ---- Configuration -----
# 1. Path to the input CSV file - it should have columns: 'repo_name', 'repo_link', 'language'
INPUT_CSV_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/repo_list_rd2.csv'

# 2. Path for the output CSV File.
OUTPUT_CSV_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/repo_loc_results.csv'

# 3.The directory where all repos will be cloned and stored in language subfolders
CLONE_DIR = 'temp_repos/'
LANGUAGE_EXTENSIONS = {
    'c++': ['.c', '.cc', '.cmake', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.in', '.json', '.make', '.py', '.sh', '.xml'],  # Includes C/C++, build, and scripting files
    'go': ['.go', '.json', '.proto', '.sh', '.yaml', '.yml'],
    'java': ['.gradle', '.groovy', '.java', '.json', '.properties', '.xml', '.yml', '.yaml'],  # Includes source, build, and config files
    'javascript': ['.css', '.html', '.js', '.json', '.jsx', '.mjs', '.scss', '.sh', '.ts', '.tsx', '.yaml', '.yml'],  # Includes source, markup, styles, and config files
    'kotlin': ['.gradle', '.json', '.kt', '.kts', '.properties', '.xml', '.yaml', '.yml'],  # Includes source, build, and config files
    'python': ['.bash', '.cfg', '.in', '.ini', '.json', '.py', '.sh', '.toml', '.yaml', '.yml']
}

def count_lines_in_file(file_path):
    '''
    Counts the number of lines in a single file, handling potential encoding errors.
    '''
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception:
        return 0

def calculate_loc_for_repo(repo_path, language):
    '''
    Calculates the total lines of code for a given language in a repository.
    '''
    loc_count = 0
    extensions_list = LANGUAGE_EXTENSIONS.get(language.lower())

    if not extensions_list:
        print(f" No extension defined for language: {language}")
        return 0
    
    # Conver the list to a tuple for the .endswith() method
    extensions_tuple = tuple(extensions_list)
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(extensions_tuple):
                file_path = os.path.join(root, file)
                loc_count += count_lines_in_file(file_path)

    return loc_count



def main():
    '''
    Main function to orchestrate the clone and LoC calculation process.
    '''
    print("---- Starting Lines of Code (LoC) Calculation and Cloning ----")

    try:
        repos_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: INput file not found at '{INPUT_CSV_PATH}'")
        return
    
    results = []

    # Wrapping the main loop with tqdm for a progress bar
    for index, row in tqdm(repos_df.iterrows(), total=len(repos_df), desc="Processing Repos"):
        repo_name = row['repo_name']
        repo_link = row['repo_link']
        language = row['language']

        # Create a language-specific path for cloning
        lang_clone_dir = os.path.join(CLONE_DIR, language.lower())
        os.makedirs(lang_clone_dir, exist_ok=True) # Ensuring the subfolder exists
        local_repo_path = os.path.join(lang_clone_dir, repo_name.replace('/','_'))

        # Skip if already cloned
        if os.path.exists(local_repo_path):
            print(f"\nSkipping {repo_name}, already cloned")
        else:
            try:
                # Clone the repository
                subprocess.run(['git', 'clone', repo_link, local_repo_path], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f" FAILED to clone {repo_name}.  Error: {e.stderr.decode()}")
                results.append({'repo_name': repo_name, 'LoC': 'Clone Failed'})
                continue # Move to the next repo

        # Calculate LoC
        loc = calculate_loc_for_repo(local_repo_path, language)
        results.append({'repo_name': repo_name, 'LoC': loc})

    # Save results to output CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"---- Process Complete. Results saved to '{OUTPUT_CSV_PATH}' ---")
    else:
        print("No repositories were processed.")


if __name__ == '__main__':
    main()