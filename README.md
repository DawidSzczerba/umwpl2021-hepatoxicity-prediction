# umwpl2021-hepatoxicity-prediction

## Overview

A project created for the Machine Learning in Drug Desing (MLDD) course. The aim of the project is to create a model that predicts the parameters ALT and TD50

## RESULTS
For the data provided for the ALT parameter, it was possible to train models that predict the value of the ALT parameter with satisfactory efficiency

![ALL MODELS RESULTS](/results/all_models_results.jpg "ALL MODELS RESULTS")

For the dataset related to the TD50 parameter, there was too little data - only 54 records - the results are not worth analysing.

## RESULTS ANALYSIS

LIME(Local Interpretable Model-Agnostic Explanations) analysis was used for the project. 
In blue we see the highlighted bits of the MACCSFP fingerprint that have the effect of lowering the value that is predicted for a given chemical. In orange we see the bits that increase the predicted value.

![LIME ANALYSIS](/best_model_analysis/lime_analysis_example.PNG "LIME ANALYSIS")

The LIME Framework allows us to use the generation of html reports and images. However, for data analysis it is most useful to return raw results. 

By analysing the 15 best predictions for the best model and the 15 worst predictions, I checked whether any particular bits of the fingerprint bias the prediction of the ALT parameter value.
It turned out not to - probably the reason why the score is not high enough is because we have little data.

From 25 randomly selected predictions I determined the top 10 most and least significant bits of the MACSFP fingerprint for predicting the value of the ALT parameter.

![TOP 10](/best_model_analysis/top10%20rank.PNG "TOP 10")


## Project structure
* [`data`](/data) : Contains the MACCSFP fingerprint data used to calculate hepatotoxicity - data are recorded as before and after preprocessing
* [`notebooks`](/notebooks) : Contains python notebooks that show the results of the experiment - you can see the preprocessing, best model analysis, alt parameter prediction (in a subfolder you can see other predictions) and data extraction process
* [`src`](/src) : It contains .py files that contain the functions used in the project. It is very easy to make predictions for models other than those used in the project. There are also files from data extraction and pre-processing containing tests, as this is a key stage in the project.
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

