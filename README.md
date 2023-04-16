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
## **Data Preparation**

Organize the data into separate folders for autistic and control groups using the provided Python script:

```
python organize_data.py
```

This script will create two folders `data/autistic` and `data/control`, and move the respective `.1D` and `.nii.gz` files into the appropriate group folders. 

## **Running the Classification Algorithm**

Once you have prepared the data, you can run the classification algorithm using the provided script:

```
python classify_brain.py
```

This script will load the data from the 'data/autistic' and 'data/control' folders, perform feature extraction, and train a Support Vector Machine classifier to distinguish between autistic and control conditions. The results will be displayed as classification accuracy, precission, recall, and F1-score.

## **Troubleshooting**

If you encounter any issues or have questions, please open an issue on this repository, and we will try to address it as soon as possible. 
