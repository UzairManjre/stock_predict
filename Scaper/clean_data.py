import os
from utils.data_cleaner import DataCleaner

def main():
    # Define input and output paths
    input_file = 'full_nse_market_data.csv'
    output_dir = 'data'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
        
    print(f"Starting data cleaning process...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Clean the data
    DataCleaner.clean_market_data(input_file, output_dir)

if __name__ == "__main__":
    main()
