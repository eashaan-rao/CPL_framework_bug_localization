import google.generativeai as genai
import os
import pandas as pd
from tqdm import tqdm
# from dotenv import load_dotenv

# Load variables from the .env file
# load_dotenv()

MY_API_KEY = "AIzaSyCOOXUvdcvYr6HR1FoEgfBmZk3k_o5Cnl4"

# Path to your input CSV with all other metadata
INPUT_CSV_PATH = "/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/obj1_rd3_dataset.csv"

# Path for the final output CSV with the new domain column
OUTPUT_CSV_PATH = "/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/obj1_rd3_domain.csv"

# Path to the parent directory where repos are cloned
CLONE_DIR = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/temp_repos/'

# Load API Key and Configure Model
# The script will look for the GOOGLE_API_KEY environment variable
try:
    genai.configure(api_key=MY_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have set the GOOGLE_API_KEY environment variable")
    exit()

def find_readme_file(repo_path):
    '''
    Finds the README file in a repository, case-insensitively.
    '''
    for filename in os.listdir(repo_path):
        if filename.lower().startswith('readme'):
            return os.path.join(repo_path, filename)
    return None

def get_domain_from_llm(readme_content):
    '''
    Sends README content to the Gemini API and asks for domain classification.
    '''
    # This promp is designed to get a concise, parsable answer.
    prompt = f"""
    Based on the following README file content, classify the software repository's primary domain.

    Provide a comma-separated list of 1 to 3 relevant domains.
    Example response: Web Framework, API, Backend

    README Content:
    ---
    {readme_content[:15000]}
    ---

    Domains:
    """

    try:
        response = model.generate_content(prompt)
        # Clean up the response to be a simple string
        return response.text.strip()
    except Exception as e:
        print(f" Gemini API call failed: {e}")
        return "API_Error"
    
def main():
    '''
    Main function to read repos, process READMEs, and get domains from the LLM.
    '''
    print(" Starting Domain Detection for Repositories")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
    
    domains = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying Domains"):
        repo_name = row['repo_name']
        language = row['language']
        repo_path = os.path.join(CLONE_DIR, language.lower(), repo_name.replace('/', '_'))

        domain_result = "Not_Processed"
        if os.path.exists(repo_path):
            readme_path = find_readme_file(repo_path)
            if readme_path:
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_text = f.read()
                        if readme_text.strip():
                            # Call the LLM to get the domain
                            domain_result = get_domain_from_llm(readme_text)
                        else:
                            domain_result = "README_Empty"
                except Exception as e:
                    domain_result = f" File_Read_Error: {e}"
            else:
                domain_result = "README_Not_Found"
        else:
            domain_result = "Repo_Not_Found"
        
        domains.append(domain_result)
    
    # Add the new column to the DataFrame
    df['domain_by_llm'] = domains

    # Save the final results
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"----- Domain detection complete. Results saved to '{OUTPUT_CSV_PATH}'---")
    print("Sample of the final output:")
    print(df[['repo_name', 'domain_by_llm']].head().to_string())

if __name__ == '__main__':
    main()

