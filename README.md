# CMPT 353 Stock Market Predictor

This README document details what each folder, and file contains as well as what they do and how to run them.

## neuralNetwork.ipynb

This file contains the code that tunes the parameters of the chosen model on the stock data for Forward Industries.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.

## neuralNetworkAnalysis.ipynb

This file contains the code that analyses the results of the model when applied on Forward Industries, and the subsequent boosting.

## {companyName}Model.ipynb

Each of these Jupyter notebooks contains the fitting of the LSTM Neural Network for each respective company.  The code that actually fits a model has been commented out and replaced with a code that reads in the model.  This is because actually fitting a model takes a significant amount of time.


## app.py

This python file is the dashboard that has been created to vizualize the results of the model.  To run this, run the python file using "python app.py", and the following will appear in the command line:

![]("./images/appStartup.PNG")
