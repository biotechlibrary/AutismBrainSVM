import os
import shutil
import pandas as pd

# Load the CSV metadata
metadata_file = 'Phenotypic_V1_0b_preprocessed1.csv'
metadata = pd.read_csv(metadata_file)

# Define source directories
source_dirs = [
    '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/ccs',
    '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/func_mean',
    '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/func_preproc',
]

# Define destination directories
dest_dirs = {
    'autistic': '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/autistic',
    'control': '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/control',
}

# Process files and move them to appropriate directories
for src_dir in source_dirs:
    for file_name in os.listdir(src_dir):
        file_parts = file_name.split('_')
        
        # Assuming the subject ID is the second part of the file name
        subject_id = file_parts[1]
        
        try:
            # Find the matching row in the metadata
            matching_row = metadata.loc[metadata['subject'] == int(subject_id)]
            
            if not matching_row.empty:
                # Determine the group (autistic or control) based on the 'DX_GROUP' column
                group = 'autistic' if matching_row['DX_GROUP'].values[0] == 1 else 'control'
                
                # Move the file to the appropriate destination directory
                src_file_path = os.path.join(src_dir, file_name)
                dest_file_path = os.path.join(dest_dirs[group], file_name)
                shutil.move(src_file_path, dest_file_path)
        except ValueError:
            print(f"Skipping file with non-integer subject ID: {file_name}")
