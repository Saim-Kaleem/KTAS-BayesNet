import pandas as pd

def merge_files(xlsx_file, csv_file, output_file):
    try:
        # Load the .xlsx file and the .csv file as DataFrames
        xlsx_df = pd.read_excel(xlsx_file, header=None)
        csv_df = pd.read_csv(csv_file)

        # Ensure the .xlsx file has at least 3 columns
        if xlsx_df.shape[1] < 3:
            raise ValueError("The .xlsx file does not have at least 3 columns.")

        # Extract the 3rd column from the .xlsx DataFrame
        third_column = xlsx_df.iloc[:, 2]
        print(f"Extracted 3rd column from .xlsx file: {third_column}")

        # replace rows with no values with 0
        third_column.fillna(0, inplace=True)

        # Add the 3rd column by the name 'Complain index' after 'Chief_complain' column in .csv DataFrame
        csv_df.insert(csv_df.columns.get_loc('Chief_complain') + 1, 'Complain index', third_column)

        # Save the resulting DataFrame to a new .csv file
        csv_df.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
xlsx_file_path = 'data.xlsx'  # Replace with your .xlsx file path
csv_file_path = 'data_cleaned.csv'    # Replace with your .csv file path
output_file_path = 'data_cleaned3.csv'  # Output file name

merge_files(xlsx_file_path, csv_file_path, output_file_path)