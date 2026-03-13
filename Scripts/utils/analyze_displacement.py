import pandas as pd
import numpy as np
import os

def analyze_rank_displacement(source_project, target_project, results_path):
    '''
    Analyzes the rank displacement diagnostics for a given project pair across the four standard scenarios.

    Args:
        source_project (str): Name of the source project.
        target_project (str): Name of the target project.
        results_path (str): The path to the directory containing the diagnostic CSV files.
    '''

    scenarios = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']

    # Sanitize project names for filenames
    src_safe = source_project.replace('/', '_')
    tgt_safe = target_project.replace('/', '_')

    print(f"\n--Analysis for {source_project} -> {target_project}--")

    for scenario in scenarios:
        # Construct the expected filename
        diag_filename = os.path.join(results_path, f"{src_safe}_{tgt_safe}_{scenario}_diagnostics.csv")
        print(f"\nScenario: {scenario}")
        try: 
            # Load the CSV file
            df = pd.read_csv(diag_filename)

            if df.empty:
                print(" -> Diagnostic file is empty. Skipping.")
                continue

            # Ensure displacement column is numeric, handle cases where rank was -1
            # We only consider displacements where both ranks were valid (> 0)
            valid_displacements = df[df['faiss_rank'] > 0 & (df['model_rank'] > 0)]['rank_displacement']

            if valid_displacements.empty:
                print(" -> No valid displacements found (likely no ground truth files ranked by both). Skipping.")
                continue

            total_valid = len(valid_displacements)

            # 1. Percentage of Improvements
            improvements = (valid_displacements > 0).sum()
            percent_improved = (improvements / total_valid) * 100 if total_valid > 0 else 0

            # 2. Median Rank Displacement
            median_displacement = valid_displacements.median()

            # 3. Average Rank Displacement
            average_displacement = valid_displacements.mean()

            print(f"  - Total Ground Truth Files Analyzed: {total_valid}")
            print(f"  - Percentage Rank Improved by Model: {percent_improved:.2f}%")
            print(f"  - Median Rank Displacement: {median_displacement:.1f} positions")
            print(f"  - Average Rank Displacement: {average_displacement:.1f} positions")
            # 4. Min and Max Rank Displacement
            print(f"  - Min Displacement: {valid_displacements.min()}")
            print(f"  - Max Displacement: {valid_displacements.max()}")

        except FileNotFoundError:
            print(f" -> Diagnostic file not found. {diag_filename}.")
        except Exception as e:
            print(f" -> Error processing file {diag_filename}: {e}")

if __name__ == "__main__":
    RESULTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"

    # Example 1: Analyze the small Python pair
    analyze_rank_displacement(
        source_project='square/anvil',
        target_project='square/kotlinpoet',
        results_path=RESULTS_PATH
    )

    # Example 2: Analyze the larger Java pair
    analyze_rank_displacement(
        source_project='spring-projects/spring-data-mongodb',
        target_project='spring-projects/spring-data-jpa',
        results_path=RESULTS_PATH
    )