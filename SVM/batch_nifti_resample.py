import os
import sys
import glob
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from nilearn import datasets, image
import os
import resource
import nilearn

def resample_file(input_file, reference_file, output_file, target_resolution=2, interp_method='trilinear', flirt_path="/home/pau/fsl/bin/flirt"):
  
    flirt_cmd = f"{flirt_path} -in {input_file} -ref {reference_file} -out {output_file} -applyisoxfm {target_resolution} -interp {interp_method}"
    subprocess.run(flirt_cmd, shell=True, check=True)

    return f"Resampled: {input_file}"

def resample_file_unpack(args):
    return resample_file(*args)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resample_sequential.py <dir1> <dir2> <dir3>")
        sys.exit(1)

    directories = sys.argv[1:]
    nii_files = []

    for directory in directories:
        nii_files.extend(glob.glob(os.path.join(directory, "*.nii.gz")))

    target_resolution = 2
    interp_method = 'trilinear'
    reference_file = "/home/pau/Documents/Research/thesis/RSN_atlas_3mm.nii.gz"  # Replace with the path to your reference file
    output_dir = "/home/pau/micromamba/envs/extraction/AutismBrainSVM/SVM/parsed_data/autistic"  # Replace with the path to your output directory

    # Set the number of cores to be used for parallel processing
    num_cores = 3  # You can set this to a specific number if you want to limit the cores used

    # Use a process pool to parallelize the resampling
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(nii_file, reference_file, os.path.join(output_dir, os.path.basename(nii_file))) for nii_file in nii_files]
        results = list(tqdm(executor.map(resample_file_unpack, args_list), total=len(nii_files), desc="Resampling files"))

    for result in results:
        print(result)
