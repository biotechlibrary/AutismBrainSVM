'''
This script utilizes the CSV file containing metadata for each patient. The script organizes the data into seperate folders 
for autistic and control groups. 

Replace 'path/to/your/metadata.csv' with the path to your metadata CSV file. This script will create two folders data/autistic and data/control, and move the respective .1D and .nii.gz files into the appropriate group folders.

Before running this script, make sure to create a backup of your data as the script moves the files from the original folders to the new group folders.
'''
import os
import shutil
import pandas as pd

# Load the metadata CSV file
metadata_file = 'path/to/your/metadata.csv' #INSERT ACTUAL FILE PATH
metadata = pd.read_csv(metadata_file)

# Set the paths to the original data and the destination folders
data_path = 'Outputs'
dest_path = 'data'

autistic_path = os.path.join(dest_path, 'autistic')
control_path = os.path.join(dest_path, 'control')

# Create the destination folders if they don't exist
os.makedirs(autistic_path, exist_ok=True)
os.makedirs(control_path, exist_ok=True)

# Define a mapping from pipeline names to their respective data subdirectories
pipelines = {
    'ccs': 'rois_ho',
    'cpac': 'func_mean',
    'cpac': 'func_preproc'
}

# Iterate through the metadata, moving files to the appropriate group folders
for _, row in metadata.iterrows():
    patient_id = row['SUB_ID']
    diagnosis_group = row['DX_GROUP']

    # Determine the destination folder based on the diagnosis group
    if diagnosis_group == 1:  # autistic
        group_folder = autistic_path
    else:  # control
        group_folder = control_path

    # Move files from each pipeline to the corresponding group folder
    for pipeline, subdirectory in pipelines.items():
        source_file = os.path.join(
            data_path, pipeline, 'filt_noglobal', subdirectory, f'source_{patient_id}_{pipeline}'
        )
        
        if subdirectory == 'rois_ho':
            source_file += '.1D'
        else:
            source_file += '.nii.gz'
        
        if os.path.exists(source_file):
            dest_file = os.path.join(group_folder, os.path.basename(source_file))
            shutil.move(source_file, dest_file)
        else:
            print(f"File not found: {source_file}")
