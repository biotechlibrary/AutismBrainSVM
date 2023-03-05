import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# LOAD IMAGING DATA

img = nib.load('ho_roi_atlas.nii.gz')
data = img.get_fdata()

# LOAD ANNOTATIONS
df = pd.read_csv('ho_labels.csv')

# Extract the relevant data columns
x = df['x'].to_numpy()
y = df['y'].to_numpy()
z = df['z'].to_numpy()

# PLOT AND OVERLAY
fig, ax = plt.subplots()
ax.imshow(data[:, :, 0])
ax.scatter(x, y, s=10, c=z, cmap='jet')
plt.show()
