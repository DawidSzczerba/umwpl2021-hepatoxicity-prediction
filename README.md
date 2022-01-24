# umwpl2021-hepatoxicity-prediction

## Overview

A project created for the Machine Learning in Drug Desing (MLDD) course. The aim of the project is to create a model that predicts the parameters ALT and TD50

## RESULTS
For the data provided for the ALT parameter, it was possible to train models that predict the value of the ALT parameter with satisfactory efficiency

![ALL MODELS RESULTS](/results/all_models_results.jpg "ALL MODELS RESULTS")

For the dataset related to the TD50 parameter, there was too little data - only 54 records - the results are not worth analysing.



## Installation

$ git clone https://github.com/DawidSzczerba/umwpl2021-hepatoxicity-prediction

Then you need to install all the required libraries

To be able to run scripts:

$ pip install -r requirements.txt

To be able to run jupyter notebooks:

$ conda env create -f environment-versions.yml
  conda activate mldd

The file environment-versions.yml was taken from https://github.com/gmum/umwpl2021 and contains most of the libraries needed to implement machine learning in drug design

