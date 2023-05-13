import os
import shutil
import re
import pandas as pd

def find_matching_row(file_id, metadata):
     """
    This function finds and returns the metadata row that matches a given file ID.
    It raises a ValueError if no matching metadata is found.

    Parameters
    ----------
    file_id : str
        The ID of the file to match.
    metadata : pd.DataFrame
        The metadata DataFrame to search for the matching file ID.

    Returns
    -------
    row : pd.Series
        The matching row in the metadata DataFrame.

    Raises
    ------
    ValueError
        If no matching metadata is found for the file ID.
    """
    for _, row in metadata.iterrows():
        if file_id in row.astype(str).apply(lambda x: x.lstrip('0')).values:
            return row

    for _, row in metadata.iterrows():
        if file_id in row.astype(str).apply(lambda x: x.lstrip('0')).values:
            return row

    raise ValueError(f"No matching metadata found for file ID {file_id}. This should not happen.")

metadata_file = '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Phenotypic_V1_0b_preprocessed1.csv'
metadata = pd.read_csv(metadata_file)

source_dirs = [
    '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/filt_noglobal/func_mean/func_mean',
    '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/filt_noglobal/func_preproc/func_preproc',
]

dest_dirs = {
    'autistic': '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/autistic',
    'control': '/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/control',
}

print("Starting to process files...")

for src_dir in source_dirs:
    print(f"Processing source directory: {src_dir}")
    
    for file_name in os.listdir(src_dir):
        if file_name.endswith('.nii.gz'):
            print(f"Processing file: {file_name}")
            
            file_id = re.findall(r'\d+', file_name)[0]
            matching_row = find_matching_row(file_id.lstrip('0'), metadata)

            group = 'autistic' if matching_row['DX_GROUP'] == 1 else 'control'
                
            src_file_path = os.path.join(src_dir, file_name)
            dest_file_path = os.path.join(dest_dirs[group], file_name)
                      
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved {file_name} to {group} directory.")

print("Finished processing files.")
