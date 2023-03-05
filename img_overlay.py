import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Open and overlay imgaging data with its corresponding CSV file.')
parser.add_argument('nii.gz_file', help='The path to the nii.gz file.')
parser.add_argument('csv_file', help='The path to the CSV file.')
args = parser.parse_args()

# Load the NIfTI file
nii_img = nib.load(args.nii.gz_file)
nii_data = nii_img.get_fdata()

# Load the CSV file
csv_data = pd.read_csv(args.csv_file)

# Overlay the CSV data onto the NIfTI image
fig, ax = plt.subplots()
ax.imshow(nii_data[:,:,0], cmap='gray')
ax.scatter(csv_data['x'], csv_data['y'], c=csv_data['label'], s=10, cmap='viridis')
plt.show()
