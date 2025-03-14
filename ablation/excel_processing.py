import os
import pandas as pd
import glob

def process_excel_files(input_folder, output_folder, pattern='*.xlsx'):
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each Excel file found in the input folder
    for file_path in glob.glob(os.path.join(input_folder, pattern)):
        try:
            print(f"Processing file: {file_path}")
            # Read in the Excel file (all values as strings)
            df = pd.read_excel(file_path, dtype=str)
            
            # Only apply the replacement if the respective column exists
            if 'ns1:XValues' in df.columns:
                df['ns1:XValues'] = df['ns1:XValues'].apply(replace_commas)
            if 'ns1:YValues' in df.columns:
                df['ns1:YValues'] = df['ns1:YValues'].apply(replace_commas)
            
            # Create the output path (same file name in the output folder)
            file_name = os.path.basename(file_path)
            output_file = os.path.join(output_folder, file_name)
            
            # Save the modified file
            df.to_excel(output_file, index=False)
            print(f"Saved modified file to: {output_file}\n")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def replace_commas(value):

    if pd.isna(value):
        return value
    parts = value.split(',')
    new_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            new_parts.append(part)
        else:
            new_parts[-1] += '.' + part
    return ','.join(new_parts)