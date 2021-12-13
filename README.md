# CS7650 NLP: Final Project
Authors: Abrar Ahmed, Rachit Bhargava, Joseph Miano

## Repo Directory Structure
* data: directory in which the raw data should be placed to run our code locally. The data can be downloaded from here: https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018
* dev_notebooks: directory containing our Jupyter notebooks for exploratory analyses, model training and evaluation, and attention visualization
* models: model architecture file; also, model .pt files are saved here when running our code to train models
* plots: the PNG plots from our paper
* utils: utility .py files to preprocess data and train/evaluate models

## Description of Specific Files
#### data
* README.md: contains instructions and a link to download the data

#### dev_notebooks
* BERT_Model_Development.ipynb: notebook containing development code for initial HuggingFace model development. This notebook does not contain our most current models or results
* Condition_Classification_With_Saliency.ipynb: notebook containing model training and attention visualization for the condition DistilBERT text model; adapted from this tutorial: https://github.com/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb
* Evaluate_All_Models.ipynb: notebook to evaluate all models and generate the model performance scores reported in our paper
* Exploratory_Analysis.ipynb: notebook to explore the data and relationships between variables; contains the plots used in Figure 1 of our report
* Generate_Model_Plots.ipynb: contains code to generate our model evaluation plots, including those used in Figure 2 of our report
* Train_All_Classification_Models.ipynb: code to train all of our useful-score classification-based models
* Train_All_Regression_Models.ipynb: code to train regression-based models (linear metadata, neural metadata, DistilBERT with text only, and DistilBERT with text + metadata)
* Usefulness_Detector_Baselines.ipynb: code to train baseline models for usefulness detection, including linear BOW
* ktrain_usefulness_attention_visualization.ipynb: model training and attention visualization for the useful score binary classifier DistilBERT text model; adapted from this tutorial: https://github.com/amaiya/ktrain/blob/master/tutorials/tutorial-A3-hugging_face_transformers.ipynb

#### models
* transformer_models.py: file containing our specific model architectures, including text-only DistilBERT, text+meta DistilBERT, neural metadata baseline, and linear metadata baseline. The models in this file are coded such that they can be used for regression or classification by specifying the number of outputs as a parameter.

#### plots
* fig1_eda_slim.png: Figure 1 plot from our paper
* fig2_model_eval.png: Figure 2 plot from our paper
* fig3_combined_attention.png: Figure 3 plot from our paper

#### utils
* evaluation.py: contains functions to get predictions and evaluate the classification, regression, and ordinal regression models
* preprocessing.py: contains functions to preprocess our data, including specifying a year range, useful count quantile cap, and useful count split for usefulness classification; this code also cleans the review text, removes duplicates, filters to include only the top 10 conditions, performs one-hot encoding of the condition column, and computes other columns like the age score and useful score
* training.py: functions to train text-only, text + metadata, and metadata-only models
* transformer_dataset.py: code specifying a ReviewDataset class, which facilitates training and evaluation of our models

## Other Notes
#### How to use Usefulness Detection Baseline Models Notebook
The usefulness detection baseline models notebook can be run, as is, from top to bottom. While executing, the user must not change the order of cell execution. Also, please note that this is a Google Colab notebook, meaning that it uses certain features available only on a Google Colab notebook. As a result, it must be uploaded directly to the platform and run over there. Finally, this notebook interacts directly with the Kaggle API to download the latest version of the dataset and then uses it. The user may need a Kaggle API token file to execute it. The notebook prompts the user to obtain it in the beginning (while also providing a link to an article that details exactly how to obtain it) and then asks the user to upload the same. This API token file is saved only in the Google Colab notebook's local session and is deleted when the session is terminated.
