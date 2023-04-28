
import os
import subprocess

def analyze_preprocessed_fmri(input_file, output_dir, afni_path="/path/to/afni/bin/"):
    """
    Analyzes preprocessed resting-state fMRI data. This script extracts time series from ROIs,
    generates correlation matrices, and applies Fisher's r-to-z transformation.

    Args:
        input_file (str): Path to the input preprocessed fMRI file.
        output_dir (str): Path to the output directory to save the analysis results.
        afni_path (str): Path to the AFNI command-line tools (default="/path/to/afni/bin/").

    Returns:
        str: A message indicating that the input file has been analyzed.

    Raises:
        subprocess.CalledProcessError: If any of the analysis steps fail.

    Example:
        >>> analyze_preprocessed_fmri('/path/to/input_file.nii.gz', '/path/to/output_dir/')
    """

    # Prepare paths and file names
    file_name = os.path.basename(input_file)
    file_name_no_ext = os.path.splitext(os.path.splitext(file_name)[0])[0]

    # 1. Extract time series from ROIs
    atlas_file = "/path/to/atlas.nii.gz"  # Replace with the path to your atlas file
    roi_ts_output_file = os.path.join(output_dir, f"{file_name_no_ext}_roi_ts.1D")
    extract_cmd = f"{afni_path}3dROIstats -mask {atlas_file} -quiet -1Dformat {input_file} > {roi_ts_output_file}"
    subprocess.run(extract_cmd, shell=True, check=True)

    # 2. Generate correlation matrix
    corr_matrix_output_file = os.path.join(output_dir, f"{file_name_no_ext}_corr_matrix.1D")
    corr_cmd = f"{afni_path}1dCorrelate -pearson -full_first {roi_ts_output_file} {roi_ts_output_file} > {corr_matrix_output_file}"
    subprocess.run(corr_cmd, shell=True, check=True)

    # 3. Apply Fisher's r-to-z transformation
    fisher_z_output_file = os.path.join(output_dir, f"{file_name_no_ext}_fisher_z.1D")
    fisher_cmd = f"{afni_path}1dcalc -expr 'log((1+a)/(1-a))/2' -a {corr_matrix_output_file} > {fisher_z_output_file}"
    subprocess.run(fisher_cmd, shell=True, check=True)

    return f"Analyzed: {input_file}"

  
