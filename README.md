# **Autism Brain Classification using Support Vector Machines**

This project aims to create a classification algorithm to identify anatomical differences in autistic brains compared to control conditions using support vector machines (SVM) in scikit-learn. The input data is preprocessed functional MRI (fMRI) files in NIfTI format (.nii.gz), which will be downloaded from the ABIDE dataset.

**Special thanks to the Autism Brain Imaging Data Exchange, an open-source initiative that made this project   possible:** 

>Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.

##**Setting up a Conda Virtual Environment**

1. Install Miniconda or Anaconda if you haven't already. You can download it from [HERE](https://docs.conda.io/en/latest/miniconda.html) 

2. Create a new virtual environment called 'extraction':

```
conda create -n extraction python=3.8
```

3. Activate the virtual environment:

```
conda activate extraction
```

4. Install the required packages in the virtual environment:

```
pip install -r requirements.txt
```

## **Downloading ABIDE Data**

1. Download the 'download_abide_preproc.py' script from the [Preprocessed Connectomes Project's ABIDE repository](https://github.com/preprocessed-connectomes-project/abide)

2. Download the data using the download_abide_preproc.py script as mentioned in the 'data_retrieval.txt' file in the project directory ABIDE_data/data_retrieval.txt. Make sure to download the data into the 'Outputs' folder.

## **Dependencies**

To install the required packages, run the following command:

```
pip install -r requirements.txt
```
## **Data Preparation and Preprocessing**

1. Organize the data into separate folders for autistic and control groups using the provided Python script:

```
python data_sorter.py
```

This script will create two folders `parsed_data/autistic` and `parsed_data/control`, and move the respective `.nii.gz` files into the appropriate group folders.

2. Verify that the data is properly sorted into the respective folders. The data should be split into the following directories:

- /home/usr/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/ccs
- /home/usr/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/func_mean
- /home/usr/micromamba/envs/extraction/AutismBrainSVM/SVM/Outputs/cpac/func_preproc

3. Process the data using the appropriate atlases and preprocessing methods. In the `brain_classifier.py` script, we use the following steps:

- If your datasets have been preprocessed using different pipelines or atlases, you need to ensure that they are compatible before merging them. For this, you may need to perform additional preprocessing steps such as spatial smoothing, intensity normalization, or resampling to a common atlas. Consult the documentation of the preprocessing tools used for each dataset to understand the specific preprocessing steps and how they can be aligned.

- In the `brain_classifier.py` script, the feature extraction is performed using the Harvard-Oxford cortical atlas, and the features are standardized using z-score normalization to ensure that all features have the same scale.

## **Running the Classification Algorithm**

Once you have preprocessed your data and ensured it is compatible, run the `brain_classifier.py` script to train and evaluate the SVM classifier:

```
python brain_classifier.py
```

This script will load the data from the 'parsed_data/autistic' and 'parsed_data/control' folders, perform feature extraction, and train a Support Vector Machine classifier to distinguish between autistic and control conditions. The results will be displayed as classification accuracy and confusion matrix.


## **Troubleshooting**

If you encounter any issues or have questions, please open an issue on this repository, and we will try to address it as soon as possible. 
