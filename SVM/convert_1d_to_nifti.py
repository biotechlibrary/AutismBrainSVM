import os
import sys
import numpy as np
import nibabel as nib

"""This script converts a 1D text file of fMRI time series into a NIfTI image file.

Usage: python convert_1d_to_nifti.py <input_file> <output_file> <tr>

Arguments:
    <input_file>: The path to a 1D text file containing fMRI time series data (e.g., output from FSL's `fslmeants` command).
    <output_file>: The desired path and filename of the output NIfTI image file (.nii or .nii.gz format).
    <tr>: The repetition time (TR) of the fMRI data in seconds.

Output:
    A NIfTI image file of the fMRI time series data, with voxel-wise time courses represented as image volumes.

The input file should contain a single row of comma-separated numbers representing the fMRI signal intensity values over time for a single voxel or region of interest. The script creates a 4D NIfTI image file with voxel-wise time courses represented as image volumes, where the x, y, and z dimensions correspond to the voxel coordinates, and the time dimension corresponds to the fMRI time series. The TR value is used to specify the temporal resolution of the output NIfTI image.
"""

def convert_1d_to_nifti(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('.1D'):
            print(f"Processing file: {file}")
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file.replace('.1D', '.nii.gz'))

            data = np.loadtxt(input_file)
            data = data.reshape(data.shape[0], 1, 1, -1)
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, output_file)
            print(f"Saved file: {output_file}")

            # Remove the '.1D' file after successful conversion
            os.remove(input_file)
            print(f"Removed file: {input_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_1d_to_nifti.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    convert_1d_to_nifti(input_dir, output_dir)
