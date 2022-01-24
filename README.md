# umwpl2021-hepatoxicity-prediction

## Overview

A project created for the Machine Learning in Drug Desing (MLDD) course. The aim of the project is to create a model that predicts the parameters ALT and TD50

## RESULTS
For the data provided for the ALT parameter, it was possible to train models that predict the value of the ALT parameter with satisfactory efficiency

![ALL MODELS RESULTS](/results/all_models_results.jpg "ALL MODELS RESULTS")

For the dataset related to the TD50 parameter, there was too little data - only 54 records - the results are not worth analysing.

## Project structure
* [`data`](/data) : zawiera dane wyrazone za pomoca fingerprintu MACCSFP, z ktorych skorzystano do obliczenia hepatoksycznosci - dane są zapisane w postaci przed i po preprocessingu
* [`notebooks`](/notebooks) : Contains python notebooks that show the results of the experiment - you can see the preprocessing, best model analysis, alt parameter prediction (in a subfolder you can see other predictions) and data extraction process
* [`src`](/src) : It contains .py files that contain the functions used in the project. It is very easy to make predictions for models other than those used in the project.
* There are also files from data extraction and pre-processing containing tests, as this is a key stage in the project.
* [`presentations`](/presentations) : Files with project presentations in Polish
* [`results`](/results) : Contains results for all models as csv files and graphs for all models
* [`models`](/models) : Contains all the saved models as a pickle, so that the results of the project can be easily retrieved
* [`best_model_analysis`](/best_model_analysis) : Contains the files used to analyse the best model.
Please note the html files. These are detailed explanations made using the LIME method, which show how a given fingerprint bit has affected a given prediction.


## Installation

$ git clone https://github.com/DawidSzczerba/umwpl2021-hepatoxicity-prediction

Then you need to install all the required libraries

To be able to run scripts:

$ pip install -r requirements.txt

To be able to run jupyter notebooks:

$ conda env create -f environment-versions.yml
  conda activate mldd

The file environment-versions.yml was taken from https://github.com/gmum/umwpl2021 and contains most of the libraries needed to implement machine learning in drug design

