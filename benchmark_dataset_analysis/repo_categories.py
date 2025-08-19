import pandas as pd

# Path to the input CSV file with the LoC Data. 
# This file should have columns: 'repo_name', 'language', 'LoC'
INPUT_CSV_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/repo_LOC.csv'

# Path for the output CSV file.
OUTPUT_CSV_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/benchmark_dataset_analysis/repo_category.csv'
# Dictionary to store the calculated tercile values for each language
tercile_values = {}

def categorize_by_tercile(group):
    '''
    Calculates tercile boundaries for a group and assings size categories.
    This function is applied to each language group in the DataFrame.
    '''
    # Calculate the 1/3 and 2/3 quantiles (the tercile boundaries)
    tercile_1 = group['LoC'].quantile(1/3)
    tercile_2 = group['LoC'].quantile(2/3)

    language_name = group.name
    tercile_values[language_name] = {
        'small_medium_boundary': tercile_1,
        'medium_high_boundary': tercile_2
    }

    # function to assign a category based on the LoC and terciles
    def assign_category(loc):
        if loc <= tercile_1:
            return 'small'
        elif loc<= tercile_2:
            return 'medium'
        else:
            return 'high'
        
    # Apply the function to the 'LoC' column to create the new category column
    group['size_category'] = group['LoC'].apply(assign_category)
    return group

def main():
    '''
    Main function to load data, perform categorization, and save results.
    '''
    print("Categorizing repositories by size (LoC terciles)")
    
    try:
        # Load the input CSV
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
    
    print(f"Loaded {len(df)} repositories for analysis.")

    # Group by language and apply the categorization function
    # This ensures terciles are calculated independently for each language
    categorized_df = df.groupby('language', group_keys=False).apply(categorize_by_tercile)

    # Save the final dataframe to a new CSV
    categorized_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Process Complete. Results with size categories saved to '{OUTPUT_CSV_PATH}'")

    print("\n\n--- Calculated Tercile Boundaries (LoC) per Language ---")
    for language, values in tercile_values.items():
        print(f"\nLanguage: {language}")
        print(f"  - 'small' <= {values['small_medium_boundary']:,.0f}")
        print(f"  - 'medium' <= {values['medium_high_boundary']:,.0f}")
        print(f"  - 'high' > {values['medium_high_boundary']:,.0f}")


if __name__ == '__main__':
    main()