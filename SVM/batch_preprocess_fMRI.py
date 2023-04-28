import os
import sys
import glob
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def preprocess_file(input_file, output_dir, target_resolution=2, flirt_path="/path/to/fsl/bin/flirt", afni_path="/path/to/afni/bin/"):
    # Prepare paths and file names
    file_name = os.path.basename(input_file)
    file_name_no_ext = os.path.splitext(os.path.splitext(file_name)[0])[0]
    output_file = os.path.join(output_dir, file_name)

    # 1. Motion correction
    mc_output_file = os.path.join(output_dir, f"{file_name_no_ext}_mc.nii.gz")
    mc_cmd = f"{afni_path}3dvolreg -prefix {mc_output_file} -base 0 -zpad 4 -1Dfile {output_dir}/{file_name_no_ext}_motion.1D {input_file}"
    subprocess.run(mc_cmd, shell=True, check=True)

    # 2. Slice timing correction
    stc_output_file = os.path.join(output_dir, f"{file_name_no_ext}_mc_stc.nii.gz")
    stc_cmd = f"{afni_path}3dTshift -prefix {stc_output_file} -tpattern alt+z2 {mc_output_file}"
    subprocess.run(stc_cmd, shell=True, check=True)

    # 3. Spatial normalization
    mni152_template = "/path/to/MNI152_T1_2mm_brain.nii.gz"  # Replace with the path to your MNI152 template
    norm_output_file = os.path.join(output_dir, f"{file_name_no_ext}_mc_stc_norm.nii.gz")
    flirt_cmd = f"{flirt_path} -in {stc_output_file} -ref {mni152_template} -out {norm_output_file} -omat {output_dir}/{file_name_no_ext}_mc_stc_norm.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear"
    subprocess.run(flirt_cmd, shell=True, check=True)

    # 4. Spatial smoothing
    smooth_output_file = os.path.join(output_dir, f"{file_name_no_ext}_mc_stc_norm_smooth.nii.gz")
    smooth_cmd = f"{afni_path}3dmerge -1blur_fwhm 6 -doall -prefix {smooth_output_file} {norm_output_file}"
    subprocess.run(smooth_cmd, shell=True, check=True)

    # 5. Temporal filtering
    filter_output_file = os.path.join(output_dir, f"{file_name_no_ext}_mc_stc_norm_smooth_filtered.nii.gz")
    filter_cmd = f"{afni_path}3dFourier -lowpass 0.08 -highpass 0.01 -prefix {filter_output_file} {smooth_output_file}"
    subprocess.run(filter_cmd, shell=True, check=True)

    # 6. Regression of confounding variables (if necessary)
    # Add this step if you need to regress out confounding variables such as motion parameters, CSF, or white matter signals.

    # 7. Save the preprocessed data
    preprocessed_output_file = os.path.join(output_dir, file_name)
    os.rename(filter_output_file, preprocessed_output_file)

    # 6. Regression of confounding variables (if necessary)
    # Add this step if you need to regress out confounding variables such as motion parameters, CSF, or white matter signals.

    # 7. Save the preprocessed data
    preprocessed_output_file = os.path.join(output_dir, file_name)
    os.rename(filter_output_file, preprocessed_output_file)

    # 8. Extract time series from ROIs
    atlas_file = "/path/to/atlas.nii.gz"  # Replace with the path to your atlas file
    roi_ts_output_file = os.path.join(output_dir, f"{file_name_no_ext}_roi_ts.1D")
    extract_cmd = f"{afni_path}3dROIstats -mask {atlas_file} -quiet -1Dformat {preprocessed_output_file} > {roi_ts_output_file}"
    subprocess.run(extract_cmd, shell=True, check=True)

    # 9. Generate correlation matrix
    corr_matrix_output_file = os.path.join(output_dir, f"{file_name_no_ext}_corr_matrix.1D")
    corr_cmd = f"{afni_path}1dCorrelate -pearson -full_first {roi_ts_output_file} {roi_ts_output_file} > {corr_matrix_output_file}"
    subprocess.run(corr_cmd, shell=True, check=True)

    # 10. Apply Fisher's r-to-z transformation
    fisher_z_output_file = os.path.join(output_dir, f"{file_name_no_ext}_fisher_z.1D")
    fisher_cmd = f"{afni_path}1dcalc -expr 'log((1+a)/(1-a))/2' -a {corr_matrix_output_file} > {fisher_z_output_file}"
    subprocess.run(fisher_cmd, shell=True, check=True)

    # 11. Save the final result
    final_output_file = os.path.join(output_dir, f"{file_name_no_ext}_final_result.1D")
    os.rename(fisher_z_output_file, final_output_file)

    return f"Preprocessed: {input_file}"
