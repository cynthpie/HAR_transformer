# transformer_HAR


## Introduction
This repository contains the pre-processing, model training, and hyperparameter tuning codes for the Two-Stream Convolution Augmented Transformer (THAT). See the following instructions for models and pre-processing method for specific dataset.

## Datasets
In order to train or run any of the files in this model repository, the corresponding data files must be located in your directory. 
 
### Public Dataset:
The public dataset has been provided by Yousefi et al. [1] and can be obtained online via https://github.com/ermongroup/Wifi_Activity_Recognition. 
 
### JCP3A Dataset:
Due to the size of the JCP3A dataset, it has not been provided in the repositories. However, the dataset can be accessed by contacting Julie McCann at Imperial College London. 
 
## Raw Data Preprocessing:
Prior to running any of the individual files in this repository, the raw data preprocessing files located in “collected_data_preprocessing” repository (https://gitlab.doc.ic.ac.uk/g22mai03/collected_data_preprocessing) must be run on the public or JCP3A datasets. Please ensure that any data paths defined in the python files are changed to be compatible with your local directory. A more detailed description on running these files can be found in this repository’s README file. 
 
### Public Raw Data Preprocessing:
To run the raw data preprocessing on the public dataset, run the respective python file:
         Public Data: “raw_data_processing.py” followed by "public_dataset_preprocessing.py"
 
### JCP3A Raw Data Preprocessing:
To run the raw data preprocessing on the JCP3A dataset, run the respective python files:
-  1 Person Clean: “process_clean_data_1.py”
-  2 Person Clean: “process_clean_data_2.py”
-  1 Person Noisy: “process_noisy_data_1.py”
-  2 Person Noisy: “process_noisy_data_2.py”

## Preprocessing
All the preprocessing python files are located in the folder named "Preprocessing". For each dataset, run the respective python files: 
-  Public dataset: “data_pre_processing.py”
-  1 Person Clean: “collected_data_processing.py”
-  2 Person Clean: “multi_label_collected_data_processing.py”
-  1 Person Noisy: “collected_data_processing.py”
-  2 Person Noisy: “multi_label_collected_data_processing.py”

To change the proposed input length for the transformer model, change the variable "input_length" located in line 16 of “collected_data_processing.py” and “multi_label_collected_data_processing.py”. 


## Model Training
All the model pythons are located in the folder named "Model". For each dataset, run the respective python files: 
-  Public dataset: “transformer-csi.py”
-  1 Person Clean: “transformer-1p_clean.py”
-  2 Person Clean: “transformer-2p_clean.py”
-  1 Person Noisy: “transformer-1p_noisy.py”
-  2 Person Noisy: “transformer-2p_noisy.py”
-  Combined: "transformer-all.py"

The hyperparameters in these files are set to the best hyperparameters stated in the report. For each run, a log file or a cross_validation log file can be saved when adjusting the corresponding variable. 

## Hyperparameter Tuning
We use manual hyperparameter tuning. All the previous trials are saved in the folder named "Archives". 

## Archives
Previous code versions and old logs have been stored in the archives folder of this repository for record-purposes

## References
[1] Siamak Yousefi, Hirokazu Narui, Sankalp Dayal, Stefano Ermon, and Shahrokh Valaee. A survey on behavior recognition using wifi channel state information. IEEE Communications Magazine, 55(10):98–104, 2017.